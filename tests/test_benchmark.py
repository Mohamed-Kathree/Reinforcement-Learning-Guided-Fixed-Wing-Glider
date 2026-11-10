"""
test_benchmark.py
=================
SPEC V20 Phase 6 — the single evaluator (training/benchmark.py).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import sys

import numpy as np
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from env.curriculum import OUTCOME_BUCKETS, outcome_bucket  # noqa: E402
from training import benchmark as bm  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return bm.load_config()


@pytest.fixture(scope="module")
def env(cfg):
    return bm.make_env(cfg, 0)


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

# Terminal info dicts as env/reward.py produces them.
INFOS = {
    "success":      {"success": True,  "crash": False, "outcome": "landed"},
    "soft_landing": {"success": False, "crash": False, "outcome": "landed"},
    "crash":        {"success": False, "crash": True,  "outcome": "crashed"},
    "timeout":      {"success": False, "crash": False, "outcome": None},
}


def test_buckets_mutually_exclusive(env):
    for expected, info in INFOS.items():
        assert outcome_bucket(info) == expected
    # Real episodes: every one lands in exactly one bucket.
    for actor in (bm.BaselineActor(), bm.RandomActor(), bm.ZeroActor()):
        for seed in (1, 2):
            ep = bm.run_episode(env, actor, seed)
            assert [b for b in OUTCOME_BUCKETS if ep["outcome"] == b] == [ep["outcome"]]
    summary = bm.summarise([{"outcome": b, "quality": 0.0, "miss_distance_m": 0.0, "steps": 1}
                            for b in OUTCOME_BUCKETS])
    assert sum(v["n"] for v in summary["buckets"].values()) == summary["n"] == 4


def test_soft_land_not_counted_as_timeout(env):
    """The bug being retired: training/evaluate.py's `timeout = not success
    and not crash` counted every soft landing as a timeout."""
    assert outcome_bucket(INFOS["soft_landing"]) == "soft_landing"
    # A real soft landing: touched down (terminated), not truncated.
    for seed in range(20):
        ep = bm.run_episode(env, bm.ZeroActor(), seed)
        if ep["outcome"] == "soft_landing":
            assert ep["terminated"] and not ep["truncated"]
            break
    else:
        pytest.fail("no soft landing in 20 zero-policy episodes")
    assert not os.path.exists(os.path.join(PROJ_ROOT, "training", "evaluate.py"))


def test_single_evaluator():
    """Exactly one evaluator: the scripts it replaced are gone."""
    for rel in ("training/evaluate.py", "scratch/final_benchmark.py",
                "scratch/final_benchmark_policy.py", "scratch/bench_rtl.py",
                "scratch/bench_all_stages.py", "scratch/v20_baseline_benchmark.py"):
        assert not os.path.exists(os.path.join(PROJ_ROOT, rel)), rel


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_wilson_interval_contains_estimate():
    # Hand-computed Wilson 95% interval for 8 / 100 (z = 1.959964):
    #   centre = (p + z²/2n) / (1 + z²/n), half = z sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
    k, n, z = 8, 100, 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = bm.wilson(k, n)
    assert lo == pytest.approx(centre - half, abs=1e-12)
    assert hi == pytest.approx(centre + half, abs=1e-12)
    assert (lo, hi) == pytest.approx((0.04108, 0.15004), abs=1e-4)
    for k in (0, 1, 50, 99, 100):
        lo, hi = bm.wilson(k, 100)
        assert lo <= k / 100 <= hi and 0.0 <= lo and hi <= 1.0


def test_mcnemar_on_known_table():
    # Discordant pairs b = 5 (policy only), c = 1 (baseline only): exact
    # two-sided binomial p = 2 * P(X <= 1 | n=6, p=0.5) = 2 * (1 + 6) / 64.
    assert bm.mcnemar_exact([[10, 5], [1, 4]]) == pytest.approx(14 / 64, abs=1e-12)
    # No discordant pairs: no evidence of a difference.
    assert bm.mcnemar_exact([[7, 0], [0, 3]]) == pytest.approx(1.0)


def test_paired_table_layout():
    ok = lambda s, o: {"seed": s, "outcome": o}
    pol = [ok(1, "success"), ok(2, "success"), ok(3, "crash"), ok(4, "soft_landing")]
    base = [ok(1, "success"), ok(2, "crash"), ok(3, "success"), ok(4, "timeout")]
    assert bm.paired_table(pol, base) == [[1, 1], [1, 1]]
    with pytest.raises(AssertionError):
        bm.paired_table(pol, base[::-1])


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_paired_seeds_identical(cfg):
    """Policy and baseline see the same initial conditions and the same
    sampled airframe, bit for bit, and an episode is reproducible from its
    seed regardless of what ran before it."""
    a = bm.evaluate_seed(cfg, 0, bm.RandomActor(), 4242)
    b = bm.evaluate_seed(cfg, 0, bm.BaselineActor(), 4242)
    assert a["launch_state"] == b["launch_state"]
    assert a["dr_params"] == b["dr_params"]
    bm.evaluate_seed(cfg, 0, bm.ZeroActor(), 7)          # something else in between
    assert bm.evaluate_seed(cfg, 0, bm.BaselineActor(), 4242) == b
    other = bm.evaluate_seed(cfg, 0, bm.BaselineActor(), 4243)
    assert other["dr_params"] != b["dr_params"]


def test_reused_env_drift_is_tiny(env):
    """Why evaluate_seed() builds a fresh env: JSBSim carries floating-point
    state across run_ic(), so two resets with one seed on the same env give
    launch states that differ in the last bits. If this ever starts passing
    exactly, the fresh-env cost could be dropped. Bounded here so a large
    carry-over (a real reset bug) would fail."""
    env.reset(seed=4242)
    first = env._fdm.state.copy()
    env.reset(seed=4242)
    assert np.abs(env._fdm.state - first).max() < 1e-12


def test_benchmark_pairs_every_episode(cfg):
    rec = bm.benchmark(bm.ZeroActor(), cfg, stages=[1], n=3, seed=5)["1"]
    pol, base = rec["episodes"]["policy"], rec["episodes"]["baseline"]
    assert [e["seed"] for e in pol] == [e["seed"] for e in base]
    for p, b in zip(pol, base):
        assert p["dr_params"] == b["dr_params"] and p["launch_state"] == b["launch_state"]
    assert sum(map(sum, rec["paired"]["table"])) == 3


# ---------------------------------------------------------------------------
# Floor controls, end to end
# ---------------------------------------------------------------------------

def _run_main(tmp_path, *args):
    out = tmp_path / "bench.json"
    bm.main([*args, "--stages", "0", "--episodes", "2", "--out", str(out)])
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def test_random_policy_runs(tmp_path):
    rec = _run_main(tmp_path, "--policy", "random")
    st = rec["stages"]["0"]
    assert st["policy"]["n"] == st["baseline"]["n"] == 2
    assert "mcnemar_exact_p" in st["paired"]


def test_zero_policy_runs(tmp_path):
    rec = _run_main(tmp_path, "--policy", "zero")
    for ep in rec["stages"]["0"]["episodes"]["policy"]:
        assert ep["outcome"] in OUTCOME_BUCKETS
        assert set(ep) >= {"seed", "dr_params", "outcome", "miss_distance_m", "quality"}


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def _obs_contract() -> dict[str, int]:
    """Index contract parsed from GliderEnv._build_obs()'s docstring."""
    from env.glider_env import GliderEnv
    doc = inspect.getdoc(GliderEnv._build_obs)
    return {name: int(i) for i, name in re.findall(r"^\s*\[(\d+)\]\s+(\w+)", doc, re.M)}


def test_ablation_zeroes_correct_channels():
    contract = _obs_contract()
    assert len(contract) == bm.OBS_DIM
    expected = tuple(sorted(contract[k] for k in ("dx_home", "dy_home", "course_angle")))
    assert bm.HOME_CHANNELS == expected == (0, 1, 3)

    obs = np.arange(1.0, 13.0, dtype=np.float32)
    seen = []

    class Spy:
        name = "spy"
        def reset(self, seed): pass
        def act(self, o):
            seen.append(o)
            return np.zeros(2, dtype=np.float32)

    bm.AblateHome(Spy()).act(obs)
    out = seen[0]
    for i in range(bm.OBS_DIM):
        assert out[i] == (0.0 if i in bm.HOME_CHANNELS else obs[i]), i
    assert obs[0] == 1.0, "ablation must not modify the caller's observation"


# ---------------------------------------------------------------------------
# Trained-policy path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, cfg):
    """An untrained PPO built on the training venv (frame-stacked), saved
    with VecNormalize statistics from a few hundred real steps."""
    from stable_baselines3 import PPO
    from training.train import build_venv
    c = json.loads(json.dumps(cfg))
    c["ppo"]["n_envs"] = 1
    venv = build_venv(c, seed=0, force_dummy=True)
    venv.reset()
    for _ in range(300):    # populate obs_rms with non-trivial statistics
        venv.step(np.random.default_rng(0).uniform(-1, 1, (1, 2)).astype(np.float32))
    d = tmp_path_factory.mktemp("model")
    model = PPO("MlpPolicy", venv, n_steps=8, batch_size=8, device="cpu", seed=0)
    model.save(d / "model.zip")
    venv.save(str(d / "vecnorm.pkl"))
    venv.close()
    return str(d / "model.zip"), str(d / "vecnorm.pkl")


def test_policy_actor_matches_training_pipeline(tiny_model, cfg):
    """PolicyActor's observations equal SB3's own
    VecNormalize(VecFrameStack(DummyVecEnv)) at frozen statistics, at every
    step of an episode, including across the reset fill of the stack."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    model, vecnorm = tiny_model
    actor = bm.PolicyActor(model, vecnorm)
    assert actor.n_stack == int(cfg["ppo"]["n_stack"]) == 8

    ref = VecNormalize.load(vecnorm, VecFrameStack(DummyVecEnv([lambda: bm.make_env(cfg, 0)]),
                                                   n_stack=actor.n_stack))
    ref.training = False
    raw_env = bm.make_env(cfg, 0)

    ref.seed(77)
    ref_obs = ref.reset()
    raw_obs, _ = raw_env.reset(seed=77)
    rng = np.random.default_rng(1)
    for step in range(60):
        mine = actor.observe(raw_obs, first=(step == 0))
        np.testing.assert_allclose(mine, ref_obs[0], rtol=0, atol=1e-6, err_msg=f"step {step}")
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        ref_obs, _, done, _ = ref.step(a[None])
        raw_obs, _, term, trunc, _ = raw_env.step(a)
        assert bool(done[0]) == (term or trunc)
        if term or trunc:
            break
    ref.close()


def test_provenance_recorded(tmp_path, tiny_model):
    model, vecnorm = tiny_model
    rec = _run_main(tmp_path, "--policy", "trained", "--model", model, "--vecnorm", vecnorm)
    prov = rec["provenance"]
    assert re.fullmatch(r"[0-9a-f]{64}", prov["policy"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", prov["config"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_sha"])
    assert prov["utc_timestamp"].endswith("+00:00")
    assert prov["vecnorm"]["sha256"]
    assert rec["n_stack"] == 8


def test_trained_ablation_runs(tmp_path, tiny_model):
    model, vecnorm = tiny_model
    rec = _run_main(tmp_path, "--policy", "trained", "--model", model, "--vecnorm", vecnorm,
                    "--ablate-home", "--no-baseline")
    assert rec["policy_name"] == "trained+ablate_home"
    assert "baseline" not in rec["stages"]["0"]


def test_unstacked_deployment_model_still_loads(env):
    """The pre-V20 deployment model (12 observations) runs with n_stack = 1."""
    path = os.path.join(PROJ_ROOT, "deployment", "best_model")
    actor = bm.PolicyActor(os.path.join(path, "best_model.zip"),
                           os.path.join(path, "vecnormalize_best.pkl"))
    assert actor.n_stack == 1
    assert bm.run_episode(env, actor, 3)["outcome"] in OUTCOME_BUCKETS

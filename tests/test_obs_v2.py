"""
test_obs_v2.py
==============
SPEC V23 Phase O wiring: observation v2 in the environment, the training
pipeline and the evaluator, with v1 (V20-V22 models) still working.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.curriculum import STAGES  # noqa: E402
from env.nav_features import OBS_DIMS, V2, obs_v2_from_v1  # noqa: E402

CONFIG = os.path.join(ROOT, "training", "configs", "base.yaml")
V22_MODEL = os.path.join(ROOT, "checkpoints_v22", "best_model.zip")
V22_VECNORM = os.path.join(ROOT, "checkpoints_v22", "vecnormalize_best.pkl")
V22_BENCHMARK = os.path.join(ROOT, "results", "benchmark_v22_best_15M.json")
needs_v22 = pytest.mark.skipif(not (os.path.exists(V22_MODEL) and os.path.exists(V22_VECNORM)),
                               reason="V22 checkpoint not present")


def _cfg(version: int) -> dict:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("observation", {})["obs_version"] = version
    cfg["ppo"]["n_envs"] = 1
    return cfg


def _env(version: int, stage: int = 1):
    from env.glider_env import GliderEnv
    from training.train import flat_env_cfg
    env = GliderEnv(cfg=flat_env_cfg(_cfg(version)))
    env.set_stage(dict(STAGES[stage]))
    return env


def test_config_default_is_v1():
    with open(CONFIG) as f:
        assert yaml.safe_load(f)["observation"]["obs_version"] == 1


def test_env_spaces_by_version():
    for version in (1, 2):
        env = _env(version)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (OBS_DIMS[version],)
        assert obs.shape == (OBS_DIMS[version],) and obs.dtype == np.float32
    with pytest.raises(ValueError):
        from env.glider_env import GliderEnv
        GliderEnv(cfg={"obs_version": 3})


def test_v2_is_v1_seen_differently():
    """Same seed, same actions: identical flight and reward under v1 and v2;
    every v2 frame is obs_v2_from_v1 of the v1 frame and the previous action."""
    e1, e2 = _env(1), _env(2)
    o1, _ = e1.reset(seed=5)
    o2, _ = e2.reset(seed=5)
    rng = np.random.default_rng(0)
    prev = np.zeros(2, dtype=np.float32)
    for _ in range(400):
        np.testing.assert_allclose(o2, obs_v2_from_v1(o1, prev), atol=1e-6)
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        o1, r1, t1, tr1, _ = e1.step(a)
        o2, r2, t2, tr2, _ = e2.step(a)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        np.testing.assert_array_equal(e1._fdm.state, e2._fdm.state)
        prev = a
        if t1 or tr1:
            break


def test_prev_action_delay_consistent():
    """The previous-action features are the action applied on the previous
    policy step (zeros at reset), whatever the transport delays."""
    env = _env(2)
    obs, _ = env.reset(seed=9)
    assert obs[V2["prev_bank"]] == 0.0 and obs[V2["prev_speed"]] == 0.0
    rng = np.random.default_rng(1)
    for _ in range(100):
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        obs, _, term, trunc, _ = env.step(a)
        assert obs[V2["prev_bank"]] == pytest.approx(a[0])
        assert obs[V2["prev_speed"]] == pytest.approx(a[1])
        if term or trunc:
            break


def test_stack_and_normalize_shapes(tmp_path):
    """Training's venv and the evaluator's PolicyActor agree on the stacked v2
    shape; VecNormalize statistics save and load; obs_version is saved with
    the model and read back."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from training.benchmark import PolicyActor
    from training.train import build_model, build_venv, checkpoint_obs_version

    cfg = _cfg(2)
    venv = build_venv(cfg, seed=0, force_dummy=True)
    try:
        n_stack = int(cfg["ppo"]["n_stack"])
        assert venv.observation_space.shape == (OBS_DIMS[2] * n_stack,)
        assert venv.reset().shape == (1, OBS_DIMS[2] * n_stack)
        model = build_model(cfg, venv, None, seed=0)
        assert model.obs_version == 2
        model.save(tmp_path / "m.zip")
        venv.save(str(tmp_path / "vn.pkl"))
        assert checkpoint_obs_version(tmp_path / "m.zip") == 2
        assert PPO.load(tmp_path / "m.zip", device="cpu").obs_version == 2
        loaded = VecNormalize.load(str(tmp_path / "vn.pkl"), venv.venv)
        assert loaded.obs_rms.mean.shape == (OBS_DIMS[2] * n_stack,)
    finally:
        venv.close()
    actor = PolicyActor(tmp_path / "m.zip", tmp_path / "vn.pkl")
    assert (actor.obs_version, actor.obs_dim, actor.n_stack) == (2, OBS_DIMS[2], n_stack)
    env = _env(2)
    obs, _ = env.reset(seed=0)
    actor.reset(0)
    assert actor.act(obs).shape == (2,)


@needs_v22
def test_v1_models_still_load():
    """A V22 checkpoint loads as v1 and evaluates, even with a v2 config."""
    from training.benchmark import PolicyActor, evaluate_seed
    actor = PolicyActor(V22_MODEL, V22_VECNORM)
    assert (actor.obs_version, actor.obs_dim, actor.n_stack) == (1, 12, 8)
    rec = evaluate_seed(_cfg(2), 0, actor, seed=1826701615)
    assert rec["outcome"] in ("success", "soft_landing", "crash", "timeout")


@needs_v22
def test_version_mismatch_raises(tmp_path):
    from training.benchmark import PolicyActor
    from training.train import checkpoint_obs_version
    actor = PolicyActor(V22_MODEL, V22_VECNORM)
    env = _env(2)
    obs, _ = env.reset(seed=0)
    actor.reset(0)
    with pytest.raises(ValueError, match="observation version mismatch"):
        actor.act(obs)
    with pytest.raises(ValueError, match="observation version mismatch"):
        env.check_obs_version(actor.obs_version)
    assert checkpoint_obs_version(V22_MODEL) == 1

    # --resume refuses a checkpoint of the other version before touching it
    from training.train import train
    with pytest.raises(ValueError, match="observation version mismatch"):
        cfg = _cfg(2)
        cfg["logging"].update(checkpoint_dir=str(tmp_path / "c"), tensorboard_log=str(tmp_path / "tb"))
        train(cfg, resume=V22_MODEL, seed=0)


@pytest.mark.skipif(not os.path.exists(V22_BENCHMARK), reason="pre-refactor benchmark log absent")
def test_baseline_unchanged():
    """The baseline, now reading through env/nav_features.py, reproduces the
    pre-refactor benchmark (commit ed65989) episode for episode. Five seeds
    per stage here; analysis/v23_baseline_parity.py checks all 100.

    The evaluator flies the baseline on observation v1 whatever the policy's
    version (training/benchmark.py benchmark()), so this is the parity that
    every benchmark's baseline column depends on."""
    from training.benchmark import BaselineActor, evaluate_seed
    with open(V22_BENCHMARK, encoding="utf-8") as f:
        ref = json.load(f)
    cfg = _cfg(2)                     # the config's version must not matter
    for stage in ("0", "1", "2", "3"):
        for old in ref["stages"][stage]["episodes"]["baseline"][:5]:
            new = evaluate_seed(cfg, int(stage), BaselineActor(), old["seed"], 1)
            assert new["outcome"] == old["outcome"], (stage, old["seed"])
            assert new["steps"] == old["steps"], (stage, old["seed"])
            assert new["miss_distance_m"] == pytest.approx(old["miss_distance_m"], abs=1e-6)


def test_baseline_flies_v2_too():
    """Reading v2 frames the baseline flies the same pattern: same outcome,
    touchdown within a metre, in calm air and steady wind (float32 rounding
    of the v2 frame is the only difference)."""
    from training.benchmark import BaselineActor, evaluate_seed, episode_seeds
    cfg = _cfg(1)
    for stage in (0, 1):
        for seed in episode_seeds(5, 0):
            a = evaluate_seed(cfg, stage, BaselineActor(), seed, 1)
            b = evaluate_seed(cfg, stage, BaselineActor(), seed, 2)
            assert a["outcome"] == b["outcome"], (stage, seed)
            assert abs(a["miss_distance_m"] - b["miss_distance_m"]) < 1.0, (stage, seed)

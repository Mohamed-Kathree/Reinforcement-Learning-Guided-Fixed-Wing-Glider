"""
SPEC V24 Phase 1: training/configs/v22_restored.yaml trains exactly V22.

V22 archived no resolved config; its archive is base.yaml at the commit it
trained at (ed65989), rebuilt and hash-recorded in
analysis/negative_results/v22_resolved_config.yaml by build_archive.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import zipfile

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESTORED = os.path.join(ROOT, "training", "configs", "v22_restored.yaml")
ARCHIVE = os.path.join(ROOT, "analysis", "negative_results", "v22_resolved_config.yaml")
V22_MODEL = os.path.join(ROOT, "checkpoints_v22", "best_model.zip")
V22_VECNORM = os.path.join(ROOT, "checkpoints_v22", "vecnormalize_best.pkl")
V22_BENCH = os.path.join(ROOT, "results", "benchmark_v22_best_15M.json")
REWARD_REF = os.path.join(ROOT, "tests", "data", "v22_reward_reference.json")

# Keys that differ by design: where a run writes, not what it trains.
ALLOWLIST = {"logging.checkpoint_dir", "logging.tensorboard_log"}
# Keys introduced after V22, with the value that reproduces V22's behaviour.
POST_V22 = {"reward.path_reward_frame": "ground", "reward.cone_shaping": False,
            "observation.obs_version": 1, "curriculum.stage0_wind_ms": 0.0}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _v22_model_data():
    with zipfile.ZipFile(V22_MODEL) as z:
        return json.loads(z.read("data"))


def test_resolved_config_matches_v22():
    have, want = _flat(_load(RESTORED)), _flat(_load(ARCHIVE))
    diffs = []
    for k in sorted(set(have) | set(want)):
        if k in ALLOWLIST:
            continue
        if k not in want:
            if k in POST_V22 and have[k] == POST_V22[k]:
                continue
            diffs.append(f"{k}: added after V22, {have[k]!r} (pre-V22 behaviour: {POST_V22.get(k)!r})")
        elif k not in have:
            diffs.append(f"{k}: missing (V22 {want[k]!r})")
        elif have[k] != want[k]:
            diffs.append(f"{k}: {have[k]!r} != V22 {want[k]!r}")
    print("\n".join(diffs) or "no differences")
    assert not diffs
    assert set(POST_V22) <= set(have), "every post-V22 key is pinned explicitly"


@pytest.mark.skipif(not os.path.exists(V22_MODEL), reason="V22 checkpoint not present (not in git)")
def test_disabled_features_off():
    from env.glider_env import GliderEnv
    from env.curriculum import STAGES
    from training.train import flat_env_cfg

    cfg = _load(RESTORED)
    data = _v22_model_data()
    assert cfg["ppo"]["gae_lambda"] == data["gae_lambda"] == 0.95
    assert cfg["ppo"]["gamma"] == data["gamma"]
    assert cfg["curriculum"]["stage0_wind_ms"] == 0.0

    env = GliderEnv(cfg=flat_env_cfg(cfg))
    try:
        assert env.obs_version == 1
        assert env.observation_space.shape[0] * cfg["ppo"]["n_stack"] == \
            int(data["observation_space"]["_shape"][0])
        assert env.cfg["path_reward_frame"] == "ground"
        assert env._stage0_wind_ms == 0.0
        env.set_stage(STAGES[1])
        env.reset(seed=3)
        for _ in range(200):
            _, _, term, trunc, info = env.step(np.array([0.3, 0.0], dtype=np.float32))
            assert info["r_shaping"] == 0.0
            if term or trunc:
                break
    finally:
        env.close()


def test_reward_identical_to_v22():
    """Open-loop trajectories recorded in the ed65989 tree
    (analysis/v22_reward_reference.py) replay with identical rewards."""
    from analysis.v22_reward_reference import run_case

    with open(REWARD_REF) as f:
        ref = json.load(f)
    cfg = _load(RESTORED)
    for case in ref["cases"]:
        now = run_case(cfg, case["stage"], case["seed"])
        assert len(now["rewards"]) == len(case["rewards"]), (case["stage"], case["seed"])
        np.testing.assert_allclose(now["rewards"], case["rewards"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(now["obs"], case["obs"], rtol=0, atol=1e-12)
        assert now["outcome"] == case["outcome"]


def test_override_changes_one_key():
    from training.train import apply_overrides

    base = _flat(_load(RESTORED))
    over = _flat(apply_overrides(_load(RESTORED), ["ppo.gae_lambda=0.995"]))
    changed = {k for k in base if base[k] != over[k]} | (set(over) ^ set(base))
    assert changed == {"ppo.gae_lambda"}
    assert over["ppo.gae_lambda"] == 0.995


# --- V22 reproduces its own committed stage-0 benchmark, seed by seed --------

_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from training.benchmark import PolicyActor
    _W["actor"] = PolicyActor(V22_MODEL, V22_VECNORM)
    _W["cfg"] = _load(RESTORED)


def _episode(seed):
    from training.benchmark import evaluate_seed
    return evaluate_seed(_W["cfg"], 0, _W["actor"], seed)


@pytest.mark.skipif(not os.path.exists(V22_MODEL), reason="V22 checkpoint not present")
def test_v22_checkpoint_reproduces_benchmark():
    with open(V22_BENCH) as f:
        committed = json.load(f)["stages"]["0"]["episodes"]["policy"]
    seeds = [e["seed"] for e in committed]
    with cf.ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1),
                                mp_context=mp.get_context("spawn"),
                                initializer=_init) as ex:
        now = list(ex.map(_episode, seeds, chunksize=5))
    mismatches = [(a["seed"], a["outcome"], b["outcome"], a["steps"], b["steps"])
                  for a, b in zip(committed, now)
                  if (a["outcome"], a["steps"]) != (b["outcome"], b["steps"])
                  or abs(a["return"] - b["return"]) > 1e-9
                  or abs(a["miss_distance_m"] - b["miss_distance_m"]) > 1e-9]
    print(f"{len(mismatches)} of {len(seeds)} episodes differ: {mismatches[:5]}")
    assert not mismatches
    assert sum(e["outcome"] == "success" for e in now) == 59

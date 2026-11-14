"""
test_stage_mix.py
=================
SPEC V24 run 5: run 4's best model continued on a fixed stage mix.

  1. curriculum.stage_weights -- off (null) by default, so every earlier
     configuration trains exactly as before; when set, each episode's stage
     is drawn from the mix and the episode runs that stage's full config;
  2. --init-from              -- a new run carrying a checkpoint's weights and
     VecNormalize statistics, with the config's hyperparameters and a step
     counter of zero;
  3. v24_run5_mix.yaml        -- differs from v22_restored.yaml in exactly
     the run 5 keys.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.curriculum import (STAGES, validate_stage_weights,  # noqa: E402
                            weighted_episode_stage)
from env.glider_env import GliderEnv  # noqa: E402

CONFIGS = os.path.join(ROOT, "training", "configs")
RUN4_BEST = os.path.join(ROOT, "checkpoints_v24_run4", "best_model.zip")


def _load(name):
    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# --- 1. the stage mix -----------------------------------------------------

def test_weights_are_normalised():
    assert validate_stage_weights([1, 1, 1, 1]) == (0.25, 0.25, 0.25, 0.25)
    w = validate_stage_weights([0.2, 0.2, 0.3, 0.3])
    assert sum(w) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[1, 1, 1], [1, 1, 1, 1, 1], [-0.1, 0.5, 0.3, 0.3], [0, 0, 0, 0]])
def test_bad_weights_rejected(bad):
    with pytest.raises(ValueError):
        validate_stage_weights(bad)


def test_draws_follow_the_weights():
    w = validate_stage_weights([0.2, 0.2, 0.3, 0.3])
    rng = np.random.default_rng(0)
    n = 40_000
    counts = np.bincount([weighted_episode_stage(w, rng) for _ in range(n)], minlength=4)
    for s in range(4):
        se = np.sqrt(w[s] * (1 - w[s]) / n)
        assert abs(counts[s] / n - w[s]) < 5 * se, (s, counts)


def test_zero_weight_stage_never_drawn():
    w = validate_stage_weights([0.0, 0.0, 0.0, 1.0])
    rng = np.random.default_rng(1)
    assert {weighted_episode_stage(w, rng) for _ in range(2000)} == {3}


def test_mixed_episode_runs_its_stage_config():
    """Wind, gusts, success radius, launch height and dr_scale all follow the
    drawn stage, so Stage 3 episodes are real Stage 3 episodes."""
    env = GliderEnv()
    env.set_stage(dict(STAGES[3]))
    env.set_curriculum(3, 0.25, None, [0.25, 0.25, 0.25, 0.25])
    seen = set()
    for seed in range(60):
        env.reset(seed=seed)
        s = env._episode_stage
        seen.add(s)
        for key, value in STAGES[s].items():
            assert env.cfg[key] == value, (s, key)
    assert seen == {0, 1, 2, 3}
    env.close()


def test_stage_draw_does_not_shift_the_episode():
    """The draw uses the replay stream, so a Stage 3 episode under the mix is
    the same flight as a Stage 3 episode pinned with set_stage."""
    mixed = GliderEnv()
    mixed.set_stage(dict(STAGES[3]))
    mixed.set_curriculum(3, 0.0, None, [0.0, 0.0, 0.0, 1.0])
    pinned = GliderEnv()
    pinned.set_stage(dict(STAGES[3]))
    for seed in (3, 17):
        o1, _ = mixed.reset(seed=seed)
        o2, _ = pinned.reset(seed=seed)
        assert np.array_equal(o1, o2)
        for _ in range(200):
            a = np.array([0.3, -0.2], dtype=np.float32)
            r1 = mixed.step(a)
            r2 = pinned.step(a)
            assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]
            if r1[2] or r1[3]:
                break
    mixed.close()
    pinned.close()


def test_off_by_default_everywhere():
    """No config before run 5 turns the mix on, and set_curriculum without it
    behaves exactly as before (replay as in test_curriculum_v21.py)."""
    assert _load("base.yaml")["curriculum"]["stage_weights"] is None
    assert "stage_weights" not in _load("v22_restored.yaml")["curriculum"]
    env = GliderEnv()
    env.set_curriculum(2, 0.25)
    assert env._stage_weights is None
    env.close()


# --- 2. --init-from -------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(RUN4_BEST), reason="run 4 checkpoint not present")
def test_init_from_carries_weights_not_hyperparameters():
    import torch
    from stable_baselines3 import PPO
    from training.train import build_venv, init_model_from, load_config

    cfg = load_config(os.path.join(CONFIGS, "v24_run5_mix.yaml"))
    cfg["ppo"]["n_envs"] = 1
    cfg["ppo"]["learning_rate"] = 1.234e-4          # a value the checkpoint does not have
    venv = build_venv(cfg, seed=0, force_dummy=True)
    try:
        model, venv2 = init_model_from(cfg, venv, RUN4_BEST)
        ref = PPO.load(RUN4_BEST, device="cpu")
        a = model.policy.state_dict()
        b = ref.policy.state_dict()
        assert a.keys() == b.keys()
        for k in a:
            assert torch.equal(a[k].cpu(), b[k].cpu()), k
        assert model.num_timesteps == 0
        assert model.learning_rate == pytest.approx(1.234e-4)
        assert model.gae_lambda == cfg["ppo"]["gae_lambda"]
        assert venv2.obs_rms.count > 1000            # loaded statistics, not fresh ones
        assert venv2.training is True
    finally:
        venv.close()


def test_resume_and_init_from_are_exclusive(tmp_path):
    from training.train import load_config, train
    cfg = load_config(os.path.join(CONFIGS, "v24_run5_mix.yaml"))
    cfg["logging"]["checkpoint_dir"] = str(tmp_path / "c")
    with pytest.raises(ValueError):
        train(cfg, resume="x", init_from="y")


# --- 3. the run 5 config ----------------------------------------------------

def test_run5_config_differs_only_in_run5_keys():
    a = _flatten(_load("v22_restored.yaml"))
    b = _flatten(_load("v24_run5_mix.yaml"))
    changed = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert changed == {"curriculum.stage_weights", "ppo.total_timesteps",
                       "logging.checkpoint_dir", "logging.tensorboard_log"}
    assert b["curriculum.stage_weights"] == [0.2, 0.2, 0.3, 0.3]
    assert b["ppo.total_timesteps"] == 10_000_000

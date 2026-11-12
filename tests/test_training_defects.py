"""
test_training_defects.py
========================
Two defects in the training setup (not in env/reward.py) that the V20 and
V21 runs exposed, each pinned here:

  D. Reward clipping. VecNormalize divided rewards by a running return std
     (13-26 in those runs) and clipped at +/-10, so raw rewards beyond
     +/-128..256 were flattened against a landing reward of up to +650.
     Every clean landing inside 14-38 m paid the same; from 1.6M steps on
     that disc was wider than the success radius.

  E. Unbounded updates. With no KL limit and the action std collapsing from
     1.0 to 0.08, the PPO update's KL rose from 0.016 to 0.19-0.30 and both
     policies degraded from their 2-3.5M-step peak.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.curriculum import STAGES  # noqa: E402
from env.reward import DEFAULT_REWARD_CFG, terminal_reward  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "training", "configs", "base.yaml")

# Smallest running return std seen in the V20/V21 runs was 12.8; anything
# the pipeline does must hold with room to spare below that.
WORST_RETURN_STD = 10.0


def _cfg() -> dict:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["ppo"]["n_envs"] = 1
    return cfg


def _venv(cfg):
    from training.train import build_venv
    return build_venv(cfg, seed=0, force_dummy=True)


def _landing_rewards(stage: int, distances) -> np.ndarray:
    cfg = {**DEFAULT_REWARD_CFG, **STAGES[stage]}
    return np.array([terminal_reward(float(d), 1.0, cfg) for d in distances])


# ---------------------------------------------------------------------------
# D. Reward clipping
# ---------------------------------------------------------------------------

def test_the_old_pipeline_paid_the_same_inside_the_success_radius():
    """Reproduces the defect, so the check below is known to detect it."""
    from training.train import learner_reward
    cfg = _cfg()
    cfg["ppo"]["normalize_reward"] = True
    cfg["ppo"]["reward_scale"] = 1.0
    venv = _venv(cfg)
    try:
        # Stage 1, return std as measured at 3-4M steps in the V21 run.
        seen = learner_reward(venv, _landing_rewards(1, [2.0, 10.0, 20.0, 30.0]),
                              return_std=14.6)
        assert np.allclose(seen, 10.0), seen       # all four clipped to the same value
    finally:
        venv.close()


@pytest.mark.parametrize("stage", range(len(STAGES)))
def test_learner_is_paid_more_for_every_metre_closer(stage):
    from training.train import learner_reward
    venv = _venv(_cfg())
    try:
        d = np.arange(0.0, 300.5, 0.5)
        seen = learner_reward(venv, _landing_rewards(stage, d), return_std=WORST_RETURN_STD)
        assert np.all(np.diff(seen) < 0), "landing reward is flat or rising with distance"
        # ...and by a margin that survives float32: on target vs the success radius.
        r_home = float(STAGES[stage]["R_home_m"])
        on_target, at_radius = learner_reward(
            venv, _landing_rewards(stage, [0.0, r_home]), return_std=WORST_RETURN_STD)
        assert on_target > 1.5 * at_radius
    finally:
        venv.close()


def test_learner_sees_the_full_crash_penalty():
    from training.train import learner_reward
    venv = _venv(_cfg())
    try:
        best = DEFAULT_REWARD_CFG["w_land"] + DEFAULT_REWARD_CFG["w_bullseye"]
        crash = -DEFAULT_REWARD_CFG["w_crash"]
        seen_best, seen_crash = learner_reward(venv, [best, crash],
                                               return_std=WORST_RETURN_STD)
        # A linear transform keeps the ratio; any clipping breaks it.
        assert seen_best / seen_crash == pytest.approx(best / crash)
    finally:
        venv.close()


def test_base_config_uses_a_fixed_reward_scale():
    ppo = _cfg()["ppo"]
    assert ppo["normalize_reward"] is False
    assert 0.0 < ppo["reward_scale"] <= 1.0


def test_scaled_rewards_reach_ppo_and_raw_returns_reach_the_logs():
    cfg = _cfg()
    scale = cfg["ppo"]["reward_scale"]
    venv = _venv(cfg)
    try:
        venv.reset()
        total = 0.0
        for _ in range(5000):
            _, r, done, infos = venv.step(np.zeros((1, 2), dtype=np.float32))
            total += float(r[0])
            if done[0]:
                break
        assert done[0]
        raw_return = float(infos[0]["episode"]["r"])        # Monitor, inside the scaling
        assert total == pytest.approx(raw_return * scale, rel=1e-4, abs=1e-4)
    finally:
        venv.close()


def test_terminal_reward_is_what_the_env_pays():
    from env.glider_env import GliderEnv
    env = GliderEnv()
    env.set_stage(dict(STAGES[1]))
    env.reset(seed=3)
    for _ in range(5000):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        if term or trunc:
            break
    assert term
    assert info["r_terminal"] == pytest.approx(
        terminal_reward(info["dist_home"], info["quality"], env.cfg), rel=1e-12)


def test_terminal_reward_range():
    cfg = dict(DEFAULT_REWARD_CFG)
    assert terminal_reward(0.0, 1.0, cfg) == pytest.approx(cfg["w_land"] + cfg["w_bullseye"])
    assert terminal_reward(50.0, 0.0, cfg) == pytest.approx(-cfg["w_crash"])


def test_resume_takes_reward_normalisation_from_the_config():
    """A V20/V21 checkpoint's VecNormalize was saved with norm_reward on."""
    import inspect

    from training import train
    src = inspect.getsource(train.train)
    assert "venv.norm_reward = bool(ppo_cfg['normalize_reward'])" in src


# ---------------------------------------------------------------------------
# E. Unbounded updates
# ---------------------------------------------------------------------------

def test_ppo_is_built_with_the_configured_kl_limit():
    from training.train import build_model
    cfg = _cfg()
    assert 0.0 < cfg["ppo"]["target_kl"] <= 0.05
    venv = _venv(cfg)
    try:
        model = build_model(cfg, venv, None, seed=0)
        assert model.target_kl == pytest.approx(cfg["ppo"]["target_kl"])
        no_limit = copy.deepcopy(cfg)
        no_limit["ppo"]["target_kl"] = None
        assert build_model(no_limit, venv, None, seed=0).target_kl is None
    finally:
        venv.close()


def test_action_std_floor_holds_the_noise_up():
    import torch
    from stable_baselines3.common.logger import configure

    from training.train import ActionStdFloorCallback, build_model
    cfg = _cfg()
    floor = cfg["ppo"]["min_action_std"]
    assert 0.05 <= floor <= 0.5
    venv = _venv(cfg)
    try:
        model = build_model(cfg, venv, None, seed=0)
        model.set_logger(configure(None, []))
        cb = ActionStdFloorCallback(floor)
        cb.init_callback(model)
        with torch.no_grad():
            model.policy.log_std[:] = torch.tensor([np.log(0.05), np.log(0.6)])
        cb.on_rollout_start()
        std = model.policy.log_std.detach().exp().numpy()
        assert std[0] == pytest.approx(floor)        # raised to the floor
        assert std[1] == pytest.approx(0.6)          # above the floor: untouched
    finally:
        venv.close()


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_bad_floor_and_scale_are_rejected(bad):
    from training.train import ActionStdFloorCallback, VecScaleReward
    with pytest.raises(ValueError):
        ActionStdFloorCallback(bad)
    with pytest.raises(ValueError):
        VecScaleReward(None, bad)


# ---------------------------------------------------------------------------
# The run itself must not die on I/O, or at start-up
# ---------------------------------------------------------------------------

def test_retry_io_retries_then_succeeds_and_never_raises():
    from training.train import retry_io
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Permission denied")

    assert retry_io(flaky, "flaky write", attempts=4, wait_s=0.0) is True
    assert calls["n"] == 3

    def locked():
        raise PermissionError(13, "Permission denied")

    assert retry_io(locked, "locked write", attempts=3, wait_s=0.0) is False


def test_training_loop_end_to_end_smoke(tmp_path, monkeypatch):
    """train() itself, ~1,000 steps, every callback attached, into tmp_path.

    The first V20 run died after one PPO iteration (a callback could not
    serialise np.float32) and again at 18.6M steps (a locked log file). This
    runs the same code path far enough to save a checkpoint, evaluate, pick
    a best model and finish, so a start-up crash fails here instead.
    """
    import json
    import pickle

    from stable_baselines3 import PPO

    import backend.config as backend_config
    from training.train import train

    monkeypatch.setattr(backend_config, "RUNS_DIR", tmp_path / "runs_stream")
    monkeypatch.setattr(backend_config, "DATA_DIR", tmp_path)

    cfg = _cfg()
    cfg["ppo"].update(n_envs=1, n_steps=256, batch_size=64, n_epochs=2, total_timesteps=1024)
    cfg["logging"].update(checkpoint_dir=str(tmp_path / "ckpt"),
                          tensorboard_log=str(tmp_path / "tb"),
                          checkpoint_freq=512, verbose=0)
    cfg["eval"].update(eval_freq=512, n_episodes_per_stage=1, stages=[0])

    train(cfg, resume=None, seed=0)

    ckpt = tmp_path / "ckpt"
    assert (ckpt / "ppo_glider_final.zip").exists()
    assert (ckpt / "ppo_glider_512_steps.zip").exists()
    assert (ckpt / "vecnormalize_512.pkl").exists()
    assert (ckpt / "curriculum_512.json").exists()
    assert (ckpt / "best_model.zip").exists() and (ckpt / "vecnormalize_best.pkl").exists()
    evals = (ckpt / "eval_logs" / "multistage.jsonl").read_text().splitlines()
    assert len(evals) == 2 and all("score" in json.loads(e) for e in evals)

    with open(ckpt / "vecnormalize_final.pkl", "rb") as f:
        assert pickle.load(f).norm_reward is False
    model = PPO.load(ckpt / "ppo_glider_final.zip", device="cpu")
    assert model.target_kl == pytest.approx(cfg["ppo"]["target_kl"])

    stream = list((tmp_path / "runs_stream").glob("*.jsonl"))
    assert len(stream) == 1 and stream[0].read_text().strip()

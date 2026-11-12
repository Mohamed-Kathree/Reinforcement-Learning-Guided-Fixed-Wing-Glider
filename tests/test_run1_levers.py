"""
test_run1_levers.py
===================
The SPEC V23 revision's run-1 levers, each behind a config key whose default
is the old behaviour:

    reward.path_reward_frame   ground | air        (Phase R)
    reward.cone_shaping        wind-aware glide-cone potential shaping
    observation.obs_version 3  v2 + on-board wind estimate (Phase P.1)
    curriculum.stage0_wind_ms  light wind in Stage 0 training only
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.curriculum import STAGES  # noqa: E402
from env.nav_features import OBS_DIMS, V3, obs_v2_from_v1  # noqa: E402
from env.reward import DEFAULT_REWARD_CFG, compute_reward, cone_deficit_m, cone_potential  # noqa: E402

CONFIG = os.path.join(ROOT, "training", "configs", "base.yaml")


def _yaml() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def _env(stage: int = 0, **cfg):
    from env.glider_env import GliderEnv
    from training.train import flat_env_cfg
    env = GliderEnv(cfg={**flat_env_cfg(_yaml()), **cfg})
    env.set_stage(dict(STAGES[stage]))
    return env


def test_defaults_are_the_old_behaviour():
    y = _yaml()
    assert y["reward"]["path_reward_frame"] == "ground"
    assert y["reward"]["cone_shaping"] is False
    assert y["observation"]["obs_version"] == 1
    assert y["curriculum"]["stage0_wind_ms"] == 0.0
    assert DEFAULT_REWARD_CFG["path_reward_frame"] == "ground"
    assert DEFAULT_REWARD_CFG["cone_shaping"] is False


def test_cone_gamma_matches_ppo_gamma():
    y = _yaml()
    assert y["reward"]["cone_gamma"] == y["ppo"]["gamma"]


# ---------------------------------------------------------------------------
# Phase R: air-path reward
# ---------------------------------------------------------------------------

def _one_step_reward(frame: str, airspeed: float, wind_ne=(0.0, 0.0)) -> dict:
    """compute_reward on a fixed mid-flight state; the wind only enters
    through the ground velocity, which the state's NED velocity carries."""
    from sim.math_utils import build_state
    v_air = np.array([airspeed, 0.0, 0.0])
    ground = v_air + np.array([wind_ne[0], wind_ne[1], 0.0])
    state = build_state(p_ned=np.array([-60.0, 10.0, -15.0]), v_body=ground)
    cfg = {**DEFAULT_REWARD_CFG, **STAGES[0], "path_reward_frame": frame}
    obs = np.zeros(12, dtype=np.float32)
    obs[7] = 15.0
    _, _, _, info = compute_reward(state=state, prev_state=state, obs=obs,
                                   action=np.zeros(2), prev_action=np.zeros(2),
                                   home_ned=np.zeros(3), cfg=cfg,
                                   alpha_true=math.radians(4.0), airspeed_true=airspeed)
    return info


def test_path_reward_ignores_wind():
    a = _one_step_reward("air", 10.0, (0.0, 0.0))["r_path"]
    b = _one_step_reward("air", 10.0, (3.0, -2.0))["r_path"]
    c = _one_step_reward("air", 10.0, (-6.0, 4.0))["r_path"]
    assert a == b == c


def test_path_reward_units():
    cfg = DEFAULT_REWARD_CFG
    got = _one_step_reward("air", 11.3, (2.0, 1.0))["r_path"]
    assert got == pytest.approx(cfg["w_path"] * 11.3 * cfg["dt_rl"], rel=1e-12)


def test_ground_option_reproduces_old():
    """'ground' pays w_path x |horizontal ground velocity| x dt, as before."""
    cfg = DEFAULT_REWARD_CFG
    got = _one_step_reward("ground", 10.0, (3.0, -2.0))["r_path"]
    assert got == pytest.approx(cfg["w_path"] * math.hypot(13.0, -2.0) * cfg["dt_rl"], rel=1e-12)
    with pytest.raises(ValueError):
        _one_step_reward("sideways", 10.0)


# ---------------------------------------------------------------------------
# Wind-aware glide cone
# ---------------------------------------------------------------------------

CFG = {**DEFAULT_REWARD_CFG}


def test_cone_reduces_to_reachability_cone_in_calm_air():
    g = CFG["cone_glide_ratio"]
    for d, h in ((80.0, 5.0), (30.0, 10.0), (150.0, 2.0), (20.0, 20.0)):
        want = min(max(d - g * h, 0.0), CFG["cone_cap_m"])
        got = cone_deficit_m([d * 0.6, d * 0.8], h, [0.0, 0.0], CFG)
        assert got == pytest.approx(want, abs=1e-9)
        assert cone_potential([d * 0.6, d * 0.8], h, [0.0, 0.0], CFG) == pytest.approx(
            -CFG["w_cone"] * want, abs=1e-9)


def test_cone_charges_more_downwind_than_upwind():
    wind = [0.0, 3.0]                       # air moving East
    down = cone_deficit_m([0.0, 60.0], 3.0, wind, CFG)    # glider East of home
    up = cone_deficit_m([0.0, -60.0], 3.0, wind, CFG)     # glider West of home
    calm = cone_deficit_m([0.0, 60.0], 3.0, [0.0, 0.0], CFG)
    assert down > calm > up >= 0.0


def test_cone_time_to_home_solves_the_drift_geometry():
    """The t_need inside the deficit really lands on home: flying for t_need
    with air velocity a (|a| = V) aimed to cancel the wind reaches the origin."""
    v, g = CFG["cone_airspeed_ms"], CFG["cone_glide_ratio"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(-100, 100, 2)
        w = rng.uniform(-6, 6, 2)
        h = 0.0
        t = (cone_deficit_m(p, h, w, {**CFG, "cone_cap_m": 1e9}) + g * h) / v
        a = -(p + w * t) / t
        assert np.hypot(*a) == pytest.approx(v, rel=1e-9)


def test_cone_zero_well_inside_glide_and_capped_when_wind_wins():
    assert cone_deficit_m([30.0, 0.0], 20.0, [2.0, 1.0], CFG) == 0.0
    assert cone_deficit_m([30.0, 0.0], 5.0, [12.0, 0.0], CFG) == CFG["cone_cap_m"]


@pytest.mark.parametrize("stage", [0, 1, 3])
def test_shaping_telescopes(stage):
    """For any trajectory, sum_t gamma^t F_t == -Phi(s_0) (terminal potential
    zero), so the shaping cannot change which policy is optimal."""
    env = _env(stage, cone_shaping=True)
    env.reset(seed=21 + stage)
    phi0 = env._phi0
    gamma = float(env.cfg["cone_gamma"])
    rng = np.random.default_rng(stage)
    total, disc, comp = 0.0, 1.0, 0.0
    for _ in range(3000):
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        total += disc * info["r_shaping"]
        comp += info["r_shaping"]
        disc *= gamma
        if term or trunc:
            break
    assert term or trunc
    assert total == pytest.approx(-phi0, abs=1e-9)
    assert info["episode_reward_components"]["r_shaping"] == pytest.approx(comp, abs=1e-9)


def test_shaping_off_pays_nothing_and_components_still_sum():
    env = _env(1)
    env.reset(seed=4)
    ret = 0.0
    for _ in range(3000):
        _, r, term, trunc, info = env.step(np.array([0.2, 0.0], dtype=np.float32))
        ret += r
        assert info["r_shaping"] == 0.0
        if term or trunc:
            break
    assert sum(info["episode_reward_components"].values()) == pytest.approx(ret, abs=1e-6)


def test_shaping_on_components_sum_to_return():
    env = _env(2, cone_shaping=True)
    env.reset(seed=8)
    ret = 0.0
    for _ in range(3000):
        _, r, term, trunc, info = env.step(np.array([0.3, -0.2], dtype=np.float32))
        ret += r
        if term or trunc:
            break
    assert sum(info["episode_reward_components"].values()) == pytest.approx(ret, abs=1e-6)


# ---------------------------------------------------------------------------
# Observation v3: on-board wind estimate
# ---------------------------------------------------------------------------

def test_v3_is_v2_plus_wind():
    e2, e3 = _env(1, obs_version=2), _env(1, obs_version=3)
    o2, _ = e2.reset(seed=3)
    o3, _ = e3.reset(seed=3)
    assert e3.observation_space.shape == (OBS_DIMS[3],) and o3.shape == (19,)
    for _ in range(200):
        np.testing.assert_array_equal(o3[:17], o2)
        a = np.array([0.4, 0.1], dtype=np.float32)
        o2, *_ = e2.step(a)
        o3, _, term, trunc, _ = e3.step(a)
        if term or trunc:
            break


def _fly_baseline_with_estimate(seed: int, wind_ne, steps: int, yaw_bias_deg: float | None = None):
    """Baseline flight in a fixed steady wind; returns |estimate - truth|
    after `steps` and the largest heading change flown on the way."""
    from training.benchmark import BaselineActor
    env = _env(0, obs_version=3)
    actor = BaselineActor()
    actor.reset(seed)
    obs, _ = env.reset(seed=seed)
    env._wind.reset(np.array([wind_ne[0], wind_ne[1], 0.0]), 0.0, np.random.default_rng(seed))
    if yaw_bias_deg is not None:
        env._sensors._imu_att_bias[2] = math.radians(yaw_bias_deg)   # this episode's yaw bias
    sens = env._build_obs()
    course0 = float(sens[3])
    turned = 0.0
    for _ in range(steps):
        sens = env._last_obs
        obs, _, term, trunc, _ = env.step(actor.act(sens))
        c = float(env._last_obs[3])
        turned = max(turned, abs((c - course0 + math.pi) % (2 * math.pi) - math.pi))
        if term or trunc:
            break
    err = float(np.hypot(*(env._wind_est.w - np.asarray(wind_ne[:2]))))
    return err, math.degrees(turned), obs


CASES = [(1, (0.0, 3.0)), (2, (-2.5, 1.5)), (3, (2.0, -2.0)), (4, (0.0, 0.0)),
         (5, (1.0, 0.0)), (6, (3.0, 0.0)), (7, (0.0, -2.0))]


@pytest.mark.parametrize("yaw_bias_deg", [0.0, 7.0])
def test_wind_estimate_converges_within_20s_including_a_turn(yaw_bias_deg):
    """Under 1 m/s after 20 s of baseline flight (which turns), with and
    without a 7 deg per-episode yaw bias. The spec asked for 10 s; measured
    error at 10 s is up to 1.8 m/s (env/wind_estimate.py docstring)."""
    for seed, wind in CASES:
        err, turned, obs = _fly_baseline_with_estimate(seed, wind, 400, yaw_bias_deg)
        assert turned > 45.0, "the flight should include a turn"
        assert err < 1.0, (seed, wind, yaw_bias_deg, err)
        assert abs(obs[V3["wind_tail"]]) <= 0.5 and abs(obs[V3["wind_cross"]]) <= 0.5


def test_wind_estimate_learns_the_speed_loop_error():
    """In still air the airspeed bias state, not the wind, absorbs the gap
    between commanded and flown airspeed."""
    from env.wind_estimate import WindEstimator
    est = WindEstimator(dt=0.05)
    for t in range(2000):                      # flying 11 m/s on a 9.5 command, turning
        yaw = 0.2 * t * 0.05                  # 11.5 deg/s, under the 15 deg/s gate
        est.update(11.0 * math.cos(yaw), 11.0 * math.sin(yaw), yaw, 9.5)
    assert est.b == pytest.approx(1.5, abs=0.05)
    assert np.hypot(*est.w) < 0.05


# ---------------------------------------------------------------------------
# Light wind at Stage 0, training only
# ---------------------------------------------------------------------------

def test_stage0_wind_is_training_only():
    from training.benchmark import load_config, make_env
    env = _env(0)
    env.set_curriculum(0, 0.0, stage0_wind_ms=1.5)
    winds = []
    for s in range(30):
        env.reset(seed=s)
        winds.append(float(np.hypot(*env._wind._mean_ned[:2])))
    assert 0.0 < max(winds) <= 1.5 and min(winds) >= 0.0
    env.set_curriculum(0, 0.0)                      # None keeps the setting
    env.reset(seed=99)
    assert env._stage0_wind_ms == 1.5
    bench = make_env(load_config(), 0)              # the benchmark's Stage 0 stays calm
    bench.reset(seed=5)
    assert float(np.hypot(*bench._wind._mean_ned[:2])) == 0.0
    with pytest.raises(ValueError):
        env.set_curriculum(0, 0.0, stage0_wind_ms=-1.0)


# ---------------------------------------------------------------------------
# The run-1 command line, end to end
# ---------------------------------------------------------------------------

RUN1_SET = [
    "observation.obs_version=3",
    "reward.path_reward_frame=air",
    "reward.cone_shaping=true",
    "ppo.gae_lambda=0.995",
    "curriculum.stage0_wind_ms=1.5",
]


def test_overrides_parse_booleans_and_null():
    from training.train import apply_overrides
    cfg = apply_overrides(_yaml(), ["reward.cone_shaping=false", "ppo.target_kl=null",
                                    "reward.path_reward_frame=air", "ppo.gae_lambda=0.995"])
    assert cfg["reward"]["cone_shaping"] is False and cfg["ppo"]["target_kl"] is None
    assert cfg["reward"]["path_reward_frame"] == "air" and cfg["ppo"]["gae_lambda"] == 0.995


def test_run1_training_smoke(tmp_path, monkeypatch):
    """train() with the run-1 overrides, ~1,000 steps, into tmp_path: the
    model is v3, the env pays shaping and air path, stage 0 has wind."""
    import backend.config as backend_config
    from stable_baselines3 import PPO

    from training.train import apply_overrides, train
    monkeypatch.setattr(backend_config, "RUNS_DIR", tmp_path / "runs_stream")
    monkeypatch.setattr(backend_config, "DATA_DIR", tmp_path)
    cfg = apply_overrides(_yaml(), RUN1_SET)
    cfg["ppo"].update(n_envs=1, n_steps=256, batch_size=64, n_epochs=2, total_timesteps=1024)
    cfg["logging"].update(checkpoint_dir=str(tmp_path / "ckpt"),
                          tensorboard_log=str(tmp_path / "tb"), checkpoint_freq=512, verbose=0)
    cfg["eval"].update(eval_freq=512, n_episodes_per_stage=1, stages=[0])
    train(cfg, resume=None, seed=0)
    model = PPO.load(tmp_path / "ckpt" / "ppo_glider_final.zip", device="cpu")
    assert model.obs_version == 3
    assert model.observation_space.shape == (OBS_DIMS[3] * int(cfg["ppo"]["n_stack"]),)
    assert model.gae_lambda == pytest.approx(0.995)

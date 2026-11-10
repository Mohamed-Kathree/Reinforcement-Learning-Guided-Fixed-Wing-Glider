"""
test_domain_rand.py
===================
SPEC V20 Phase 4 — domain-randomisation table, sampling, frame stacking.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from env.curriculum import DR_ONESIDED, DR_RANGES, DR_SCALE, STAGES, stage_ranges  # noqa: E402
from sim.mass_properties import load_mass_properties  # noqa: E402

ALL_KEYS = set(DR_RANGES) | set(DR_ONESIDED)

# The two independent estimates, hard-coded on purpose: narrowing a range in
# mass_properties.yaml (or DR_RANGES) so that it excludes either one must
# fail loudly here, not pass quietly.
ESTIMATES = {
    #           V19 build, CAD rebuild
    "mass_kg": (1.64, 1.62),
    "ixx":     (0.28, 0.16),
    "iyy":     (0.22, 0.12),
}


@pytest.fixture(scope="module")
def env():
    from env.glider_env import GliderEnv
    return GliderEnv(cfg=dict(STAGES[0]))


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_ranges_bracket_both_estimates():
    for key, (v19, cad) in ESTIMATES.items():
        lo, hi = DR_RANGES[key]
        assert lo <= v19 <= hi, f"{key}: V19 estimate {v19} outside {DR_RANGES[key]}"
        assert lo <= cad <= hi, f"{key}: CAD estimate {cad} outside {DR_RANGES[key]}"


def test_dr_ranges_come_from_mass_properties_yaml():
    mp = load_mass_properties()
    assert DR_RANGES["mass_kg"] == tuple(mp["mass"]["flying_kg"]["range"])
    assert DR_RANGES["cg_x_m"] == tuple(mp["cg"]["residual_m"]["range"])
    assert DR_RANGES["ixx"] == tuple(mp["inertia"]["ixx"]["range"])
    iyy = mp["inertia"]["iyy"]
    assert DR_RANGES["iyy"] == tuple(iyy.get("dr_range", iyy["range"]))
    area = mp["geometry"]["wing_area_m2"]
    assert DR_RANGES["sw_mult"] == pytest.approx(tuple(r / area["value"] for r in area["range"]))
    # Deliberately not randomised (docs/SPEC_V20_decisions.md).
    assert not {"izz", "izz_factor", "ixz"} & ALL_KEYS


def test_stage_scaling_monotone():
    widths = [{k: hi - lo for k, (lo, hi) in stage_ranges(s).items()} for s in DR_SCALE]
    for key in ALL_KEYS:
        w = [stage[key] for stage in widths]
        assert all(a <= b for a, b in zip(w, w[1:])), f"{key} widths not monotone: {w}"
    full = stage_ranges(DR_SCALE[-1])
    assert DR_SCALE[-1] == 1.0
    for key, rng in {**DR_RANGES, **DR_ONESIDED}.items():
        assert full[key] == pytest.approx(rng), key
    # Two-sided ranges shrink about the midpoint.
    for key, (lo, hi) in DR_RANGES.items():
        slo, shi = stage_ranges(DR_SCALE[0])[key]
        assert 0.5 * (slo + shi) == pytest.approx(0.5 * (lo + hi)), key


def test_stages_carry_dr_scale():
    assert [s["dr_scale"] for s in STAGES] == DR_SCALE
    for s in STAGES:
        assert "aero_scale_range" not in s and "mass_range" not in s


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", range(len(STAGES)))
def test_sampled_values_in_range(env, stage):
    env.set_stage(dict(STAGES[stage]))
    env._rng = np.random.default_rng(1000 + stage)
    ranges = stage_ranges(DR_SCALE[stage])
    samples = [env._sample_domain_rand() for _ in range(1000)]
    for key, (lo, hi) in ranges.items():
        vals = np.array([s[key] for s in samples])
        assert vals.min() >= lo and vals.max() <= hi, key
        if hi > lo:   # actually spread across the range, not stuck
            assert vals.max() - vals.min() > 0.8 * (hi - lo), key


def test_cd0_one_sided(env):
    for stage in range(len(STAGES)):
        env.set_stage(dict(STAGES[stage]))
        env._rng = np.random.default_rng(stage)
        vals = [env._sample_domain_rand()["cd0_add"] for _ in range(1000)]
        assert min(vals) >= 0.0
        assert stage_ranges(DR_SCALE[stage])["cd0_add"][0] == 0.0


def _run_episode(env, seed):
    from baseline.deterministic_rtl import DeterministicRTL
    ctrl = DeterministicRTL()
    obs, _ = env.reset(seed=seed)
    infos = []
    while True:
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        infos.append(info)
        yield env, info
        if terminated or truncated:
            return


def test_dr_params_in_info(env):
    env.set_stage(dict(STAGES[1]))
    *_, (_, last) = _run_episode(env, seed=7)
    assert "dr_params" in last
    assert set(last["dr_params"]) == ALL_KEYS
    assert last["dr_params"] == env._dr


def test_resample_per_episode_not_per_step(env):
    env.set_stage(dict(STAGES[3]))
    props = ("aero/cl-mult", "aero/clmax-scale", "aero/ail-eff-mult", "metrics/Sw-sqft",
             "inertia/mass-slugs", "inertia/ixx-slugs_ft2", "inertia/iyy-slugs_ft2",
             "fcs/trim-bias-roll")
    episodes = []
    for seed in (11, 12):
        seen = []
        for e, _ in _run_episode(env, seed):
            seen.append(tuple(e._fdm._fdm[p] for p in props))
        assert len(set(seen)) == 1, "an airframe parameter changed within an episode"
        episodes.append((seen[0], dict(env._dr)))
    assert episodes[0][0] != episodes[1][0]
    assert all(episodes[0][1][k] != episodes[1][1][k] for k in ALL_KEYS)


def test_dr_params_reach_jsbsim():
    """Every sampled value lands on its aircraft-file hook."""
    from sim.jsbsim_fdm import DR_PROPERTY_MAP, JSBSimFDM
    from sim.mass_solver import read_mass_properties
    fdm = JSBSimFDM()
    state = np.zeros(13)
    state[2], state[3], state[6] = -50.0, 12.0, 1.0
    fdm.reset(state)
    sw0 = fdm._fdm["metrics/Sw-sqft"]
    rng = np.random.default_rng(3)
    params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in stage_ranges(1.0).items()}
    fdm.apply_domain_rand(params)
    for key, prop in DR_PROPERTY_MAP.items():
        if key in params:
            assert fdm._fdm[prop] == pytest.approx(params[key], rel=1e-12), key
    assert fdm._fdm["metrics/Sw-sqft"] == pytest.approx(sw0 * params["sw_mult"], rel=1e-12)
    a = read_mass_properties(fdm._fdm)
    for key in ("mass_kg", "ixx", "iyy"):
        assert a[key] == pytest.approx(params[key], rel=0.01), key
    assert a["cg_x_m"] == pytest.approx(params["cg_x_m"], abs=2e-4)

    # The next reset restores every default and empties the point masses.
    fdm.reset(state)
    assert fdm._fdm["metrics/Sw-sqft"] == sw0
    assert read_mass_properties(fdm._fdm)["mass_kg"] == pytest.approx(fdm.empty_weight_kg, rel=1e-4)


def test_apply_domain_rand_rejects_bad_keys():
    from sim.jsbsim_fdm import JSBSimFDM
    fdm = JSBSimFDM()
    with pytest.raises(KeyError):
        fdm.apply_domain_rand({"cmde_mult": 1.0})       # pre-rename name
    with pytest.raises(KeyError):
        fdm.apply_domain_rand({"mass_kg": 1.6})          # mass without CG/inertia


def test_all_mass_corners_solvable():
    """All 16 corners of the full mass/CG/Ixx/Iyy box pass the solver's guards."""
    from sim.jsbsim_fdm import JSBSimFDM
    from sim.mass_solver import read_mass_properties
    fdm = JSBSimFDM()
    state = np.zeros(13)
    state[2], state[3], state[6] = -50.0, 12.0, 1.0
    keys = ("mass_kg", "cg_x_m", "ixx", "iyy")
    for corner in itertools.product(*(DR_RANGES[k] for k in keys)):
        fdm.reset(state)
        fdm.apply_domain_rand(dict(zip(keys, corner)))
        a = read_mass_properties(fdm._fdm)
        assert a["ixx"] > 0 and a["iyy"] > 0 and a["izz"] > 0


# ---------------------------------------------------------------------------
# Envelope: command floor and truncation, from the model itself
# ---------------------------------------------------------------------------

def test_airspeed_floor_above_stall():
    """The command floor exceeds the stall speed of the worst airframe:
    heaviest mass, lowest lift PEAK (clmax_scale; cl_mult sets slope only and
    is checked at both ends to confirm it leaves the peak alone), smallest wing."""
    from analysis.flight_envelope import polar, worst_stall_airframe
    from env.glider_env import SPEED_CMD_CENTRE_MS, SPEED_CMD_MIN_MS, SPEED_CMD_SCALE_MS
    from sim.jsbsim_fdm import JSBSimFDM

    assert SPEED_CMD_CENTRE_MS - SPEED_CMD_SCALE_MS == pytest.approx(SPEED_CMD_MIN_MS)
    fdm = JSBSimFDM()
    stalls = {m: polar({**worst_stall_airframe(), "cl_mult": m}, fdm).v_stall_ms
              for m in (DR_RANGES["cl_mult"][0], 1.0, DR_RANGES["cl_mult"][1])}
    assert max(stalls.values()) - min(stalls.values()) < 0.05, stalls
    assert SPEED_CMD_MIN_MS > max(stalls.values()), (
        f"command floor {SPEED_CMD_MIN_MS} m/s not above worst-corner stall {stalls}"
    )
    # Lowering the peak must raise the stall speed (the semantics, not just the floor).
    nominal = polar({**worst_stall_airframe(), "clmax_scale": 1.0}, fdm).v_stall_ms
    assert max(stalls.values()) > nominal


def test_truncation_covers_longest_flight():
    from analysis.flight_envelope import compute
    from env.glider_env import (DT_RL, FLIGHT_TIME_EXPECTED_S, FLIGHT_TIME_LONGEST_S,
                                FLIGHT_TIME_SHORTEST_S, MAX_STEPS, SPEED_CMD_MAX_MS,
                                SPEED_CMD_MIN_MS)
    env = compute(SPEED_CMD_MIN_MS, SPEED_CMD_MAX_MS)
    assert MAX_STEPS * DT_RL >= 1.15 * env.longest_t_start_s
    # The recorded constants are the computed ones, not stale copies.
    assert FLIGHT_TIME_LONGEST_S == pytest.approx(env.longest_t_start_s, abs=0.1)
    assert FLIGHT_TIME_SHORTEST_S == pytest.approx(env.shortest_t30_s, abs=0.1)
    assert FLIGHT_TIME_EXPECTED_S == pytest.approx(env.nominal_t30_s, abs=0.1)


# ---------------------------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------------------------

def test_frame_stack_shapes_match():
    import yaml
    from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
    from training.train import build_venv

    with open(os.path.join(PROJ_ROOT, "training", "configs", "base.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["ppo"]["n_envs"] = 2
    n_stack = int(cfg["ppo"]["n_stack"])
    assert n_stack == 8

    train_venv = build_venv(cfg, seed=0, force_dummy=True)
    eval_venv = build_venv(cfg, seed=10000, force_dummy=True)
    try:
        assert train_venv.observation_space == eval_venv.observation_space
        assert train_venv.observation_space.shape == (12 * n_stack,)
        # VecNormalize outermost, so eval_venv.training = False reaches it.
        for v in (train_venv, eval_venv):
            assert isinstance(v, VecNormalize) and isinstance(v.venv, VecFrameStack)
        obs = eval_venv.reset()
        assert obs.shape == (2, 12 * n_stack)

        # A policy built on the training env acts on the eval env's
        # observations (no learning: construction and inference only).
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", train_venv, n_steps=8, batch_size=8, device="cpu")
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (2, 2)
    finally:
        train_venv.close()
        eval_venv.close()

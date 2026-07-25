"""
test_dynamics.py
================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Pytest unit-test suite covering sensor models, actuator dynamics, and an
end-to-end integration check that the baseline achieves success.

Core physics correctness (energy conservation, glide ratio, stall, trim
stability, attitude integrity) is validated against the real JSBSim FDM by
validate_glide.py's 5 gates -- run that before this suite.

Tests:
    test_lidar_agl_correction      -- slant-range-corrected AGL formula is correct
    test_lidar_update_rate         -- ultrasonic refreshes at 20 Hz (every 10 physics steps)
    test_gps_update_rate           -- GPS refreshes at 5 Hz (every 40 physics steps)
    test_imu_update_rate           -- IMU refreshes at 200 Hz (every physics step)
    test_baro_update_rate          -- Baro refreshes at 25 Hz (every 8 physics steps)
    test_lidar_dropout_probability -- dropout_prob=1.0 makes every reading invalid
    test_servo_rate_limit          -- JSBSim elevator actuator obeys rlglider.xml rate_limit
    test_baseline_reaches_home     -- DeterministicRTL returns home in Stage 0 conditions

Coordinate frames: NED inertial, BODY (FRD), WIND (stability).
Quaternion convention: [q0, q1, q2, q3], q0 scalar.
Units: SI throughout (m, m/s, rad, rad/s, kg, N, N·m)
"""

from __future__ import annotations

import numpy as np

from sim.math_utils import build_state
from sim.jsbsim_fdm import JSBSimFDM, ELEVATOR_MAX_RAD
from sim.sensor_models import SensorSuite, _lidar_noise
from env.curriculum import STAGES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DT_PHYS    = 0.005   # 200 Hz physics step
DT_RL      = 0.050   # 20 Hz policy step
N_SUBSTEPS = int(DT_RL / DT_PHYS)


def _trim_state(alt_m: float = 100.0, speed_ms: float = 10.0) -> np.ndarray:
    gamma = np.radians(15.0)
    return build_state(
        p_ned  = np.array([0.0, 0.0, -alt_m]),
        v_body = np.array([speed_ms * np.cos(gamma), 0.0, -speed_ms * np.sin(gamma)]),
        pitch  = gamma,
    )


# ---------------------------------------------------------------------------
# 1. LiDAR AGL correction formula
# ---------------------------------------------------------------------------

def test_lidar_agl_correction():
    """Mean of 500 slant-corrected readings must match h*cos(roll)*cos(pitch)."""
    rng      = np.random.default_rng(42)
    roll     = np.radians(10.0)
    pitch    = np.radians(5.0)
    true_agl = 3.0   # within ultrasonic range [0.2, 4.5] m

    readings = []
    for _ in range(500):
        h, valid = _lidar_noise(true_agl, roll, pitch, rng,
                                dropout_prob=0.0, noise_scale=1.0)
        if valid:
            readings.append(h)

    assert len(readings) > 400, "Too many dropouts at dropout_prob=0"
    mean_h   = float(np.mean(readings))
    expected = true_agl * np.cos(roll) * np.cos(pitch)
    assert abs(mean_h - expected) < 0.2, (
        f"Ultrasonic AGL correction: mean={mean_h:.3f}  expected={expected:.3f}"
    )


# ---------------------------------------------------------------------------
# 2–5. Sensor update rates
# ---------------------------------------------------------------------------

def _count_sensor_updates(sensor_attr: str, n_physics_steps: int, alt_m: float = 20.0) -> int:
    suite = SensorSuite()
    rng   = np.random.default_rng(7)
    state = _trim_state(alt_m=alt_m, speed_ms=10.0)
    suite.reset(state, rng, noise_scale=1.0, dropout_prob=0.0)

    prev  = np.asarray(getattr(suite, sensor_attr)).copy()
    count = 0
    for _ in range(n_physics_steps):
        suite.step(state, rng)
        curr = np.asarray(getattr(suite, sensor_attr)).copy()
        if not np.array_equal(prev, curr):
            count += 1
            prev = curr.copy()
    return count


def test_lidar_update_rate():
    """Ultrasonic at 20 Hz: over 200 physics steps (1 s) expect ~20 updates.

    alt_m=3.0 keeps the reading inside the ultrasonic's [0.2, 4.5] m range
    gate so it refreshes instead of holding at 0.0/invalid.
    """
    n = _count_sensor_updates('lidar_agl', 200, alt_m=3.0)
    assert 15 <= n <= 25, f"Ultrasonic update count wrong: {n} (expected ~20)"


def test_gps_update_rate():
    """GPS at 5 Hz: over 200 physics steps (1 s) expect ~5 updates."""
    n = _count_sensor_updates('gps_pos', 200)
    assert 3 <= n <= 7, f"GPS update count wrong: {n} (expected ~5)"


def test_imu_update_rate():
    """IMU at 200 Hz: nearly every physics step produces a new reading."""
    n = _count_sensor_updates('imu_euler', 200)
    assert n >= 190, f"IMU update count too low: {n} (expected ~200)"


def test_baro_update_rate():
    """Baro at 25 Hz: over 200 physics steps (1 s) expect ~25 updates."""
    suite = SensorSuite()
    rng   = np.random.default_rng(7)
    state = _trim_state(alt_m=20.0, speed_ms=10.0)
    suite.reset(state, rng, noise_scale=1.0, dropout_prob=0.0)

    prev  = suite.baro_alt
    count = 0
    for _ in range(200):
        suite.step(state, rng)
        curr = suite.baro_alt
        if curr != prev:
            count += 1
            prev = curr
    assert 15 <= count <= 35, f"Baro update count wrong: {count} (expected ~25)"


# ---------------------------------------------------------------------------
# 6. LiDAR dropout probability
# ---------------------------------------------------------------------------

def test_lidar_dropout_probability():
    """With dropout_prob=1.0 every LiDAR reading must be invalid."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        h, valid = _lidar_noise(10.0, 0.0, 0.0, rng, dropout_prob=1.0)
        assert not valid, "Expected invalid reading at dropout_prob=1.0"
        assert h == 0.0, f"h should be 0.0 when invalid, got {h}"


# ---------------------------------------------------------------------------
# 7. Servo rate limit (real JSBSim FCS, rlglider.xml <rate_limit> = 200 deg/s)
# ---------------------------------------------------------------------------

def test_servo_rate_limit():
    """Elevator surface position must not move faster than 200 deg/s (rlglider.xml)."""
    dyn   = JSBSimFDM()
    state = _trim_state(alt_m=100.0, speed_ms=12.0)
    dyn.reset(state)

    zero = {'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}
    for _ in range(40):   # settle at zero command before stepping the input
        dyn.step(zero, np.zeros(3), DT_PHYS)

    full = {'aileron': 0.0, 'elevator': ELEVATOR_MAX_RAD, 'rudder': 0.0}
    prev_pos = dyn._fdm["fcs/elevator-pos-rad"]
    max_rate = 0.0

    for _ in range(40):
        dyn.step(full, np.zeros(3), DT_PHYS)
        curr_pos = dyn._fdm["fcs/elevator-pos-rad"]
        max_rate = max(max_rate, abs(curr_pos - prev_pos) / DT_PHYS)
        prev_pos = curr_pos

    limit_rad_s = np.radians(200.0)
    assert max_rate <= limit_rad_s + 1e-2, (
        f"Elevator actuator exceeded rate limit: {np.degrees(max_rate):.1f} deg/s "
        f"> 200 deg/s"
    )


# ---------------------------------------------------------------------------
# 8. Baseline reaches home
# ---------------------------------------------------------------------------

def test_baseline_reaches_home():
    """DeterministicRTL must succeed in at least 15% of Stage 0 episodes.

    Threshold history: an earlier P-only heading controller measured ~29%
    here, but that number was produced while sim/jsbsim_fdm.py's
    position_ned had a since-fixed bug (it reported unsigned displacement
    magnitude, not signed -- see position_ned's docstring). With that fixed,
    the P-only controller's TRUE success rate was ~4-6% -- tracing episodes
    showed a genuine, non-decaying heading limit cycle (bank saturated
    ~99% of steps), not a subtle mistuning. DeterministicRTL is now a PD
    heading controller (kp_bank=0.3, kd_bank=5.0, damped by IMU yaw rate --
    see baseline/deterministic_rtl.py's module docstring for the full
    derivation), which reaches ~20-22% over large samples (n=80+). 15% here
    leaves margin for this test's smaller n=20 sample.
    """
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    cfg  = dict(STAGES[0])
    env  = GliderEnv(cfg=cfg)
    ctrl = DeterministicRTL()
    rng  = np.random.default_rng(0)

    n_episodes = 20
    successes  = 0

    for _ in range(n_episodes):
        seed = int(rng.integers(0, 2**31))
        ctrl.reset()   # clear yaw-rate derivative state from the previous episode
        obs, _ = env.reset(seed=seed)
        while True:
            action = ctrl.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get('success'):
                    successes += 1
                break

    success_rate = successes / n_episodes
    assert success_rate >= 0.15, (
        f"DeterministicRTL Stage 0 success rate too low: "
        f"{success_rate:.0%} ({successes}/{n_episodes})"
    )

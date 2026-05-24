"""
test_dynamics.py
================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Pytest unit-test suite covering physics, sensor models, actuator dynamics,
and an end-to-end integration check that the baseline achieves success.

Tests:
    test_energy_no_drag            -- total mechanical energy conserved when CD=0
    test_quat_norm_preserved       -- quaternion stays within 1e-6 of 1.0 over 200 steps
    test_lidar_agl_correction      -- slant-range-corrected AGL formula is correct
    test_lidar_update_rate         -- LiDAR refreshes at 50 Hz (every 4 physics steps)
    test_gps_update_rate           -- GPS refreshes at 5 Hz (every 40 physics steps)
    test_imu_update_rate           -- IMU refreshes at 200 Hz (every physics step)
    test_baro_update_rate          -- Baro refreshes at 25 Hz (every 8 physics steps)
    test_lidar_dropout_probability -- dropout_prob=1.0 makes every reading invalid
    test_servo_rate_limit          -- servo position cannot change faster than rate_limit
    test_baseline_reaches_home     -- DeterministicRTL returns home in Stage 0 conditions

Coordinate frames: NED inertial, BODY (FRD), WIND (stability).
Quaternion convention: [q0, q1, q2, q3], q0 scalar.
Units: SI throughout (m, m/s, rad, rad/s, kg, N, N·m)
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.glider_dynamics import GliderDynamics, GliderParams, build_state, ZERO_CONTROLS
from sim.sensor_models import SensorSuite, _lidar_noise
from sim.actuator_models import ServoModel, ServoParams
from env.curriculum import STAGES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DT_PHYS    = 0.005   # 200 Hz physics step
DT_RL      = 0.050   # 20 Hz policy step
N_SUBSTEPS = int(DT_RL / DT_PHYS)

_PARAMS = GliderParams()


def _trim_state(alt_m: float = 100.0, speed_ms: float = 10.0) -> np.ndarray:
    gamma = np.radians(15.0)
    return build_state(
        p_ned  = np.array([0.0, 0.0, -alt_m]),
        v_body = np.array([speed_ms * np.cos(gamma), 0.0, -speed_ms * np.sin(gamma)]),
        pitch  = gamma,
    )


# ---------------------------------------------------------------------------
# 1. Energy conservation (CD = 0)
# ---------------------------------------------------------------------------

def test_energy_no_drag():
    """E = 0.5*m*V^2 + m*g*h must not drift by more than 0.1% over 10 s
    when aerodynamic forces AND moments are both zero (pure projectile)."""
    p = GliderParams()

    def _zero_aero(*_args, **_kw):
        return np.zeros(3), np.zeros(3)

    dyn   = GliderDynamics(params=p, aero_fn=_zero_aero)
    state = _trim_state(alt_m=100.0, speed_ms=10.0)
    dyn.reset(state)

    def _energy(s):
        V = float(np.linalg.norm(s[3:6]))
        h = float(-s[2])
        return 0.5 * p.m * V**2 + p.m * p.g * h

    E0 = _energy(dyn.state)
    for _ in range(int(10.0 / DT_PHYS)):
        dyn.step(ZERO_CONTROLS, np.zeros(3), DT_PHYS)

    E1 = _energy(dyn.state)
    drift_pct = abs(E1 - E0) / abs(E0) * 100.0
    assert drift_pct < 0.1, (
        f"Energy drifted {drift_pct:.4f}% over 10 s (limit 0.1%)"
    )


# ---------------------------------------------------------------------------
# 2. Quaternion norm preservation
# ---------------------------------------------------------------------------

def test_quat_norm_preserved():
    """Quaternion norm must stay within 1e-6 of 1.0 over 200 RL steps."""
    dyn   = GliderDynamics()
    state = _trim_state(alt_m=100.0, speed_ms=10.0)
    dyn.reset(state)
    rng   = np.random.default_rng(0)

    for step in range(200):
        controls = {
            'aileron':  float(rng.uniform(-0.1, 0.1)),
            'elevator': float(rng.uniform(-0.05, 0.05)),
            'rudder':   0.0,
        }
        for _ in range(N_SUBSTEPS):
            dyn.step(controls, np.zeros(3), DT_PHYS)
            norm = float(np.linalg.norm(dyn.state[6:10]))
            assert abs(norm - 1.0) < 1e-6, (
                f"Quaternion norm drifted at RL step {step}: |q|={norm:.10f}"
            )
        if dyn.state[2] > 0.0:
            break


# ---------------------------------------------------------------------------
# 3. LiDAR AGL correction formula
# ---------------------------------------------------------------------------

def test_lidar_agl_correction():
    """Mean of 500 slant-corrected readings must match h*cos(roll)*cos(pitch)."""
    rng      = np.random.default_rng(42)
    roll     = np.radians(10.0)
    pitch    = np.radians(5.0)
    true_agl = 15.0   # within LiDAR range [0.2, 40] m

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
        f"LiDAR AGL correction: mean={mean_h:.3f}  expected={expected:.3f}"
    )


# ---------------------------------------------------------------------------
# 4–7. Sensor update rates
# ---------------------------------------------------------------------------

def _count_sensor_updates(sensor_attr: str, n_physics_steps: int) -> int:
    suite = SensorSuite()
    rng   = np.random.default_rng(7)
    state = _trim_state(alt_m=20.0, speed_ms=10.0)
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
    """LiDAR at 50 Hz: over 200 physics steps (1 s) expect ~50 updates."""
    n = _count_sensor_updates('lidar_agl', 200)
    assert 40 <= n <= 60, f"LiDAR update count wrong: {n} (expected ~50)"


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
# 8. LiDAR dropout probability
# ---------------------------------------------------------------------------

def test_lidar_dropout_probability():
    """With dropout_prob=1.0 every LiDAR reading must be invalid."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        h, valid = _lidar_noise(10.0, 0.0, 0.0, rng, dropout_prob=1.0)
        assert not valid, "Expected invalid reading at dropout_prob=1.0"
        assert h == 0.0, f"h should be 0.0 when invalid, got {h}"


# ---------------------------------------------------------------------------
# 9. Servo rate limit
# ---------------------------------------------------------------------------

def test_servo_rate_limit():
    """Servo position increment must not exceed rate_limit * dt per step."""
    params = ServoParams(
        max_rad    = np.radians(25.0),
        rate_rad_s = np.radians(200.0),
        tau_s      = 0.05,
    )
    servo = ServoModel(params)
    servo.reset(pos=0.0)

    cmd      = np.radians(25.0)
    dt       = DT_PHYS
    prev_pos = servo.pos

    for _ in range(50):
        new_pos   = servo.step(cmd, dt)
        delta     = abs(new_pos - prev_pos)
        max_delta = params.rate_rad_s * dt + 1e-9
        assert delta <= max_delta, (
            f"Servo exceeded rate limit: Δpos={np.degrees(delta):.4f}° "
            f"> max {np.degrees(max_delta):.4f}°"
        )
        prev_pos = new_pos


# ---------------------------------------------------------------------------
# 10. Baseline reaches home
# ---------------------------------------------------------------------------

def test_baseline_reaches_home():
    """DeterministicRTL must succeed in at least 25% of Stage 0 episodes."""
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
        obs, _ = env.reset(seed=seed)
        while True:
            action = ctrl.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get('success'):
                    successes += 1
                break

    success_rate = successes / n_episodes
    assert success_rate >= 0.25, (
        f"DeterministicRTL Stage 0 success rate too low: "
        f"{success_rate:.0%} ({successes}/{n_episodes})"
    )

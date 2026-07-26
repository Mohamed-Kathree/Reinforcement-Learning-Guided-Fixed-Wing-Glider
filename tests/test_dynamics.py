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
    test_course_angle_matches_yaw_in_still_air  -- would have caught the quat_to_rotmat transpose bug
    test_vertical_speed_sign                    -- a gliding aircraft must report obs[8] < 0
    test_position_ned_matches_dead_reckoning    -- locks in the position_ned tangent-plane fix
    test_control_surface_directions             -- AVL-transcription guard for rlglider.xml
    test_gust_intensity_matches_config           -- locks in the OU gust sqrt(dt) fix
    test_alpha_matches_jsbsim_in_wind            -- reward's stall penalty must use true alpha
    test_env_passes_sb3_checker                  -- GliderEnv satisfies the SB3 API contract
    test_reward_components_sum_to_total          -- info's breakdown must reconcile with the total
    test_baseline_fixed_seed_regression          -- 20-episode fixed-seed hash regression guard

Coordinate frames: NED inertial, BODY (FRD), WIND (stability).
Quaternion convention: [q0, q1, q2, q3], q0 scalar.
Units: SI throughout (m, m/s, rad, rad/s, kg, N, N·m)
"""

from __future__ import annotations

import numpy as np

from sim.math_utils import build_state, quat_to_rotmat, wrap_pi
from sim.jsbsim_fdm import JSBSimFDM, ELEVATOR_MAX_RAD
from sim.sensor_models import SensorSuite, _lidar_noise
from sim.wind import WindModel
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
    """DeterministicRTL must succeed in at least 85% of Stage 0 episodes.

    Threshold history: this test's threshold was lowered twice (25% -> 15%)
    to accommodate a controller that was broken both times -- a threshold
    that ratchets downward to match observed behaviour is not a test, it's
    a record of surrender. The actual root cause was a sign bug in
    sim/sensor_models.py::_refresh_gps() (quat_to_rotmat(q).T instead of
    quat_to_rotmat(q); see that function's corrected docstring), which
    mirrored the GPS course-angle observation and made every "correction"
    this controller issued push the glider further off course -- the
    "limit cycle" and "20-22% is as good as it gets" narratives in prior
    revisions were both downstream of this bug, not evidence the P-only law
    needed detuning. With the GPS fix in place, a plain P-only law
    (kp_bank=1.5, no derivative term) reaches ~100% at Stage 0 (n=40, see
    baseline/deterministic_rtl.py's module docstring). If this ever drops
    again, suspect the GPS/IMU sign conventions before touching the gain --
    do NOT lower this threshold to make a broken controller pass.
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
        ctrl.reset()
        obs, _ = env.reset(seed=seed)
        while True:
            action = ctrl.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get('success'):
                    successes += 1
                break

    success_rate = successes / n_episodes
    assert success_rate >= 0.85, (
        f"DeterministicRTL Stage 0 success rate too low: "
        f"{success_rate:.0%} ({successes}/{n_episodes}) -- if this regresses, "
        f"suspect the GPS/IMU sign conventions before the controller gain."
    )


# ---------------------------------------------------------------------------
# 9. Course angle matches yaw in still air (the quat_to_rotmat transpose guard)
# ---------------------------------------------------------------------------

def test_course_angle_matches_yaw_in_still_air():
    """obs[3] (GPS course) must equal obs[6]-equivalent (IMU yaw) when flying
    wings-level with no wind and no sideslip.

    This is the test that would have caught the quat_to_rotmat transpose bug
    (sim/sensor_models.py::_refresh_gps used to compute v_ned via R.T instead
    of R, mirroring course about North -- see quat_to_rotmat's docstring).
    """
    rng = np.random.default_rng(0)
    for yaw_deg in (0, 45, 90, 135, 180, -45, -90, -135):
        state = build_state(
            p_ned  = np.array([0.0, 0.0, -50.0]),
            v_body = np.array([10.0, 0.0, 0.0]),
            yaw    = np.radians(yaw_deg),
        )
        suite = SensorSuite()
        suite.reset(state, rng, noise_scale=0.0, dropout_prob=0.0)

        course = float(np.arctan2(suite.gps_vel[1], suite.gps_vel[0]))
        yaw    = float(suite.imu_euler[2])
        err_deg = np.degrees(abs(wrap_pi(course - yaw)))
        assert err_deg < 5.0, (
            f"yaw={yaw_deg} deg: course={np.degrees(course):.1f} deg, "
            f"IMU yaw={np.degrees(yaw):.1f} deg (mismatch {err_deg:.1f} deg)"
        )


# ---------------------------------------------------------------------------
# 10. Vertical speed sign
# ---------------------------------------------------------------------------

def test_vertical_speed_sign():
    """A gliding (unpowered) aircraft in steady flight must report obs[8]
    (vertical speed, positive = climbing) < 0 -- it is always sinking.

    Would have caught the same quat_to_rotmat transpose bug from the other
    direction: the bug also inverted the sign of GPS vertical speed.
    """
    from env.glider_env import GliderEnv

    env = GliderEnv(cfg=dict(
        wind_speed=0.0, gust_intensity=0.0, sensor_noise=0.0, dropout_prob=0.0,
        alt0_m=200.0, launch_offset_min_m=50.0, launch_offset_max_m=50.0,
        launch_speed_min_ms=10.0, launch_speed_max_ms=10.0,
        launch_pitch_min_deg=0.0, launch_pitch_max_deg=0.0,
    ))
    obs, _ = env.reset(seed=0)
    for _ in range(150):   # 7.5 s: enough to settle past the launch transient
        obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert obs[8] < 0.0, f"Expected a gliding aircraft to be sinking (obs[8]<0), got {obs[8]:.3f}"


# ---------------------------------------------------------------------------
# 11. Dead-reckoning position check (locks in the position_ned tangent-plane fix)
# ---------------------------------------------------------------------------

def test_position_ned_matches_dead_reckoning():
    """Integrate v_body through the body->NED DCM and compare to
    JSBSimFDM.position_ned over a hard, sustained turn. Locks in the
    tangent-plane position fix (see position_ned's docstring) so it cannot
    silently regress back to using JSBSim's unsigned distance-from-start
    properties.
    """
    fdm   = JSBSimFDM(dt_phys=DT_PHYS)
    state = _trim_state(alt_m=150.0, speed_ms=12.0)
    fdm.reset(state)

    p_dead_reckon = fdm.state[0:3].copy()
    n_steps = 1000   # 5 s of sustained hard turn
    for _ in range(n_steps):
        s = fdm.state
        v_ned = quat_to_rotmat(s[6:10]) @ s[3:6]
        p_dead_reckon = p_dead_reckon + v_ned * DT_PHYS
        fdm.step({'aileron': np.radians(10.0), 'elevator': 0.0, 'rudder': 0.0},
                 np.zeros(3), DT_PHYS)

    p_true = fdm.state[0:3]
    err = float(np.linalg.norm(p_dead_reckon[0:2] - p_true[0:2]))
    assert err < 1.0, f"Dead-reckoning position mismatch over a 5 s hard turn: {err:.3f} m"


# ---------------------------------------------------------------------------
# 12. Control surface directions (AVL-transcription guard)
# ---------------------------------------------------------------------------

def test_control_surface_directions():
    """+aileron -> +roll rate; +elevator -> nose down; +rudder -> yaw LEFT.

    Run after ANY change to rlglider.xml's aero/moment functions or control
    derivatives -- this is the guard against a sign transcription error from
    AVL (rlglider.xml has already had two: rudder coordination and ground-
    contact geometry).
    """
    fdm   = JSBSimFDM(dt_phys=DT_PHYS)
    state = _trim_state(alt_m=200.0, speed_ms=10.0)

    fdm.reset(state)
    for _ in range(40):   # 0.2 s: short window, before lateral coupling dynamics develop
        fdm.step({'aileron': np.radians(15.0), 'elevator': 0.0, 'rudder': 0.0},
                 np.zeros(3), DT_PHYS)
    assert fdm.state[10] > 0.0, "positive aileron must produce positive (right-wing-down) roll rate"

    fdm.reset(state)
    pitch0 = fdm.euler_angles()[1]
    for _ in range(40):
        fdm.step({'aileron': 0.0, 'elevator': np.radians(15.0), 'rudder': 0.0},
                 np.zeros(3), DT_PHYS)
    pitch1 = fdm.euler_angles()[1]
    assert pitch1 < pitch0, "positive elevator must pitch the nose down"

    fdm.reset(state)
    for _ in range(40):
        fdm.step({'aileron': 0.0, 'elevator': 0.0, 'rudder': np.radians(15.0)},
                 np.zeros(3), DT_PHYS)
    assert fdm.state[12] < 0.0, "positive rudder must yaw the nose LEFT (negative r)"


# ---------------------------------------------------------------------------
# 13. Gust intensity matches configured std-dev (locks in the sqrt(dt) fix)
# ---------------------------------------------------------------------------

def test_gust_intensity_matches_config():
    """Realised OU gust standard deviation must be within 20% of the
    configured gust_intensity (locks in sim/wind.py's sqrt(2*dt/tau) fix --
    the old dt-scaled version was ~14x too weak)."""
    rng = np.random.default_rng(0)
    w = WindModel(tau=2.0, dt=DT_PHYS)
    w.reset(np.zeros(3), 1.0, rng)
    samples = np.array([w.step() for _ in range(50000)])[5000:]
    std = float(samples.std(axis=0)[0])
    assert abs(std - 1.0) / 1.0 < 0.2, f"gust std {std:.3f} not within 20% of configured 1.0 m/s"


# ---------------------------------------------------------------------------
# 14. Alpha used for the stall penalty must be true (JSBSim) alpha, not
#     inertial-velocity-derived alpha, which is wrong in wind.
# ---------------------------------------------------------------------------

def test_alpha_matches_jsbsim_in_wind():
    """JSBSimFDM.alpha must equal aero/alpha-rad exactly (the property this
    exposes), and must diverge meaningfully in wind from the inertial-
    velocity-derived arctan2(state[5], state[3]) that env/reward.py's stall
    penalty used to use -- locking in the fix that routes true alpha into
    compute_reward() via GliderEnv.step()'s alpha_true kwarg."""
    fdm = JSBSimFDM(dt_phys=DT_PHYS)
    state = build_state(p_ned=np.array([0.0, 0.0, -25.0]), v_body=np.array([10.0, 0.0, 0.0]))
    fdm.reset(state)
    wind = np.array([9.0, 0.0, 0.0])
    for _ in range(400):
        fdm.step({'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}, wind, DT_PHYS)

    assert abs(fdm.alpha - fdm._fdm['aero/alpha-rad']) < 1e-9

    alpha_inertial = float(np.arctan2(fdm.state[5], fdm.state[3]))
    mismatch_deg = abs(np.degrees(alpha_inertial - fdm.alpha))
    assert mismatch_deg > 5.0, (
        f"expected inertial-derived alpha to diverge meaningfully from true "
        f"alpha in a 9 m/s wind; only {mismatch_deg:.1f} deg apart"
    )


# ---------------------------------------------------------------------------
# 15. GliderEnv satisfies the Gymnasium/SB3 API contract
# ---------------------------------------------------------------------------

def test_env_passes_sb3_checker():
    from stable_baselines3.common.env_checker import check_env
    from env.glider_env import GliderEnv
    check_env(GliderEnv())


# ---------------------------------------------------------------------------
# 16. Reward components reconcile with the returned scalar
# ---------------------------------------------------------------------------

def test_reward_components_sum_to_total():
    """info's per-component breakdown must sum to the returned scalar reward
    for the common (non-terminal) case."""
    from env.reward import compute_reward, DEFAULT_REWARD_CFG

    state      = build_state(p_ned=np.array([50.0, 0.0, -50.0]), v_body=np.array([10.0, 0.0, 0.0]))
    prev_state = build_state(p_ned=np.array([55.0, 0.0, -50.0]), v_body=np.array([10.0, 0.0, 0.0]))
    obs         = np.zeros(11, dtype=np.float32)   # roll=0, lidar_agl=0, lidar_valid=0 (no AGL penalty)
    action      = np.array([0.1, 0.0], dtype=np.float32)
    prev_action = np.array([0.0, 0.0], dtype=np.float32)

    r, terminated, truncated, info = compute_reward(
        state=state, prev_state=prev_state, obs=obs, action=action,
        prev_action=prev_action, home_ned=np.zeros(3), cfg=dict(DEFAULT_REWARD_CFG),
        alpha_true=np.radians(3.0), airspeed_true=10.0,
    )

    assert not terminated and not truncated
    component_sum = (info['r_progress'] + info['r_airtime'] + info['penalty_agl']
                      + info['penalty_stall'] + info['penalty_bank'] + info['penalty_smooth'])
    assert abs(r - component_sum) < 1e-9, (
        f"reward {r} does not reconcile with component sum {component_sum}"
    )


# ---------------------------------------------------------------------------
# 17. Fixed-seed regression: 20 baseline episodes must match a stored
#     reference to high precision, so any future change to physics/sensors/
#     control trips a visible test rather than silently drifting.
# ---------------------------------------------------------------------------

def test_baseline_fixed_seed_regression():
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    expected_final_dist = np.array([
        24.84167786, 24.7809178,  24.95905871, 24.73729723, 24.89262113,
        24.59941355, 24.70637026, 24.87729638, 24.65468331, 24.62146575,
        24.96790705, 24.72530198, 24.78938522, 24.78997205, 24.73732294,
        24.68687485, 24.99141709, 24.76786655, 24.99820917, 24.66255639,
    ])

    env  = GliderEnv(cfg=dict(STAGES[0]))
    ctrl = DeterministicRTL()
    rng  = np.random.default_rng(12345)

    final_dist = []
    for _ in range(20):
        seed = int(rng.integers(0, 2**31))
        ctrl.reset()
        obs, _ = env.reset(seed=seed)
        while True:
            obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
            if terminated or truncated:
                final_dist.append(info['dist_home'])
                break

    np.testing.assert_allclose(
        np.array(final_dist), expected_final_dist, atol=1e-6,
        err_msg=(
            "Final-distance regression mismatch -- an unintended change to "
            "physics, sensors, or control has altered baseline behaviour. "
            "If this change is deliberate, regenerate the reference array."
        ),
    )

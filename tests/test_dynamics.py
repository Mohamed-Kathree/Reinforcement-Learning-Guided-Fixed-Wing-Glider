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
    test_no_open_loop_flare              -- Phase 4: straight glide lands clean without the shield
    test_vertical_wind_bounded            -- Phase 4: sustained vertical wind never exceeds min sink
    test_landing_quality_lexicographic    -- Phase 4: clean-but-further beats close-but-stalled
    test_episode_terminates_on_touchdown  -- Phase 4: crossing R_home_m alone doesn't end the episode
    test_baseline_fixed_seed_regression          -- 20-episode fixed-seed hash regression guard
    test_barrier_inert_below_flare        -- Phase 5: reachability barrier never fires below barrier_min_agl_m
    test_barrier_penalty_capped           -- Phase 5: barrier penalty clips to w_unreach_cap
    test_precision_reward_monotone_and_has_gradient -- Phase 5: hyperbolic r_precision has a real gradient
    test_curriculum_force_advance                    -- Phase 5: max_episodes_at_stage forces an advance
    test_curriculum_force_advance_disabled_by_default -- Phase 5: default behaviour unchanged
    test_reward_cfg_config_parity                     -- Phase 5: DEFAULT_REWARD_CFG keys mirrored in base.yaml

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
    """DeterministicRTL must reliably LAND clean at Stage 0, and hit the full
    precision "success" bar (quality > 0.5 and centred) often enough to be a
    meaningful reference point.

    Phase 4 redefinition (RLGlider_Phase4_Landing_Task_Spec.md): "success" no
    longer means "crossed R_home_m" -- it means a graded, continuous landing
    QUALITY (sink rate / roll / speed / alpha at touchdown) combined with
    centre precision (see env/reward.py's module docstring). The three-phase
    pattern controller (baseline/deterministic_rtl.py) is deliberately a
    simple scripted heuristic that the spec expects to "leave clear headroom"
    for a learned policy -- so a modest success rate here is not a
    regression, and this test does NOT re-litigate the old 85%-crossing bar.
    What WOULD be a regression: crashes/stalls (a fundamental control or
    physics break) or a collapse in landing quality (the wings-level flare
    blend / trim schedule breaking). Those are the properties asserted here;
    if this test ever needs its thresholds loosened to pass, suspect an
    actual physics/control regression before touching the numbers -- do NOT
    silently ratchet them down to match broken behaviour (see git history for
    why that specific failure mode is called out explicitly).

    Empirically (n=20, seed=0, post-Phase-4 tuning): crash_rate=0%,
    mean_quality=1.0, success_rate=35%.
    """
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    cfg  = dict(STAGES[0])
    env  = GliderEnv(cfg=cfg)
    ctrl = DeterministicRTL()
    rng  = np.random.default_rng(0)

    n_episodes = 20
    successes  = 0
    crashes    = 0
    qualities: list[float] = []

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
                if info.get('crash'):
                    crashes += 1
                qualities.append(float(info.get('quality', 0.0)))
                break

    success_rate = successes / n_episodes
    crash_rate   = crashes / n_episodes
    mean_quality = float(np.mean(qualities))

    assert crash_rate <= 0.10, (
        f"DeterministicRTL Stage 0 crash rate too high: {crash_rate:.0%} "
        f"({crashes}/{n_episodes}) -- suspect a control or physics regression "
        f"(trim schedule, wings-level flare blend, ground-effect model)."
    )
    assert mean_quality >= 0.7, (
        f"DeterministicRTL Stage 0 mean landing quality too low: "
        f"{mean_quality:.2f} -- suspect a regression in the flare/trim "
        f"schedule (sink/roll/speed/alpha at touchdown)."
    )
    assert success_rate >= 0.15, (
        f"DeterministicRTL Stage 0 success rate too low: "
        f"{success_rate:.0%} ({successes}/{n_episodes}) -- suspect a "
        f"centre-precision regression in the pattern controller's Phase "
        f"2->3 energy-management transition."
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
    for the common (non-terminal, no ground contact) case."""
    from env.reward import compute_reward, DEFAULT_REWARD_CFG

    state      = build_state(p_ned=np.array([50.0, 0.0, -50.0]), v_body=np.array([10.0, 0.0, 0.0]))
    prev_state = build_state(p_ned=np.array([55.0, 0.0, -50.0]), v_body=np.array([10.0, 0.0, 0.0]))
    obs         = np.zeros(12, dtype=np.float32)   # roll=0 (no bank penalty)
    action      = np.array([0.1, 0.0], dtype=np.float32)
    prev_action = np.array([0.0, 0.0], dtype=np.float32)

    r, terminated, truncated, info = compute_reward(
        state=state, prev_state=prev_state, obs=obs, action=action,
        prev_action=prev_action, home_ned=np.zeros(3), cfg=dict(DEFAULT_REWARD_CFG),
        alpha_true=np.radians(3.0), airspeed_true=10.0,
        touched_down=False, fault=False,
    )

    assert not terminated and not truncated
    assert info['r_terminal'] == 0.0   # not a ground-contact step
    component_sum = (info['r_path'] + info['penalty_unreach'] + info['penalty_stall']
                      + info['penalty_bank'] + info['penalty_smooth'] + info['r_terminal'])
    assert abs(r - component_sum) < 1e-9, (
        f"reward {r} does not reconcile with component sum {component_sum}"
    )


# ---------------------------------------------------------------------------
# Phase 4 precision-landing task tests (RLGlider_Phase4_Landing_Task_Spec.md)
# ---------------------------------------------------------------------------

def test_no_open_loop_flare():
    """A straight glide, calm air, must touch down clean (below SINK_BAD, not
    stalled) now that the open-loop flare shield is deleted -- the aircraft's
    own trimmed glide already flares reasonably on its own (see
    RLGlider_Phase4_Landing_Task_Spec.md §0.2's measurements: with the shield
    removed, touchdowns were 0.43-0.78 m/s sink at 4-8.5 deg alpha, well
    inside a clean landing, whereas the deleted shield's fixed nose-up pull
    caused 4/5 tested speed commands to arrive stalled at 3+ m/s sink)."""
    from env.glider_env import GliderEnv
    from env.reward import DEFAULT_REWARD_CFG

    env = GliderEnv(cfg=dict(
        wind_speed=0.0, gust_intensity=0.0, sensor_noise=0.0, dropout_prob=0.0,
        alt0_m=25.0, launch_offset_min_m=50.0, launch_offset_max_m=50.0,
        launch_speed_min_ms=10.0, launch_speed_max_ms=10.0,
        launch_pitch_min_deg=0.0, launch_pitch_max_deg=0.0,
    ))
    obs, _ = env.reset(seed=0)
    action = np.array([0.0, -0.5], dtype=np.float32)   # ~7 m/s command, spec's cleanest case
    while True:
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert not truncated, "straight glide must reach touchdown, not time out"
    assert abs(info['vs_td']) < DEFAULT_REWARD_CFG['sink_bad'], (
        f"touchdown sink {info['vs_td']:.2f} m/s at/above SINK_BAD -- "
        f"suspect a regression in the ground-effect model or trim schedule"
    )
    assert info['alpha_td_deg'] < 12.0, (
        f"touchdown alpha {info['alpha_td_deg']:.1f} deg at/above the 12 deg stall angle"
    )


def test_vertical_wind_bounded():
    """Sustained per-episode vertical wind must never exceed the ~0.39 m/s
    measured minimum sink, even at Stage 3's maximum wind speed -- otherwise
    a path-length reward lets PPO discover a free, unbounded-airtime thermal
    (RLGlider_Phase4_Landing_Task_Spec.md §0.3)."""
    from env.glider_env import GliderEnv

    env = GliderEnv(cfg=dict(STAGES[3]))
    max_abs = 0.0
    for seed in range(200):
        env.reset(seed=seed)
        max_abs = max(max_abs, abs(float(env._wind.mean_ned[2])))

    assert max_abs <= 0.25 + 1e-9, (
        f"sustained vertical wind {max_abs:.3f} m/s exceeds the 0.25 m/s cap"
    )


def test_landing_quality_lexicographic():
    """A clean-but-further-from-centre landing must always score higher than
    a closer-but-stalled/hard one -- the multiplicative quality gate
    (quality * (r_precision + r_bullseye)) exists specifically to guarantee
    this; a plain additive sum would let the optimiser trade a hard landing
    for a slightly better position, which is exactly the failure mode the
    Phase 4 reward redesign exists to avoid."""
    from env.reward import compute_reward, DEFAULT_REWARD_CFG

    def touchdown_state(d_home, vs, roll_deg):
        return build_state(
            p_ned=np.array([d_home, 0.0, 0.0]),
            v_body=np.array([10.0, 0.0, vs]),
            roll=np.radians(roll_deg),
        )

    obs = np.zeros(12, dtype=np.float32)
    zero_action = np.zeros(2, dtype=np.float32)

    close_but_bad = touchdown_state(2.0, vs=3.0, roll_deg=40.0)
    far_but_clean = touchdown_state(15.0, vs=0.5, roll_deg=5.0)

    r_bad, _, _, info_bad = compute_reward(
        state=close_but_bad, prev_state=close_but_bad, obs=obs,
        action=zero_action, prev_action=zero_action, home_ned=np.zeros(3),
        cfg=dict(DEFAULT_REWARD_CFG), alpha_true=np.radians(13.5),
        airspeed_true=10.0, touched_down=True,
    )
    r_clean, _, _, info_clean = compute_reward(
        state=far_but_clean, prev_state=far_but_clean, obs=obs,
        action=zero_action, prev_action=zero_action, home_ned=np.zeros(3),
        cfg=dict(DEFAULT_REWARD_CFG), alpha_true=np.radians(3.0),
        airspeed_true=10.0, touched_down=True,
    )

    assert info_bad['quality'] < info_clean['quality']
    assert r_clean > r_bad, (
        f"lexicographic property violated: close-but-bad ({r_bad:.1f}, "
        f"quality={info_bad['quality']:.2f}) beat far-but-clean "
        f"({r_clean:.1f}, quality={info_clean['quality']:.2f})"
    )


def test_episode_terminates_on_touchdown():
    """Crossing R_home_m no longer ends the episode by itself -- only ground
    contact (touched_down) or a JSBSim fault does. Run the baseline past the
    point where dist_home first drops below R_home_m and confirm the episode
    is still running there."""
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    cfg = dict(STAGES[0])
    env = GliderEnv(cfg=cfg)
    ctrl = DeterministicRTL()
    ctrl.reset()
    obs, _ = env.reset(seed=0)

    r_home = float(cfg['R_home_m'])
    crossed_radius_while_airborne = False
    while True:
        action = ctrl.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        if info['dist_home'] < r_home and not terminated:
            crossed_radius_while_airborne = True
        if terminated or truncated:
            break

    assert crossed_radius_while_airborne, (
        "expected the glider to fly inside R_home_m for at least one step "
        "before touchdown, without that alone ending the episode"
    )


# ---------------------------------------------------------------------------
# 17. Fixed-seed regression: 20 baseline episodes must match a stored
#     reference to high precision, so any future change to physics/sensors/
#     control trips a visible test rather than silently drifting.
# ---------------------------------------------------------------------------

def test_baseline_fixed_seed_regression():
    """Reference array regenerated for the Phase 4 precision-landing task
    (touchdown-based termination, three-phase pattern controller with the
    glide_slope=10/margin=1/r_pattern_m=40 tuning and wings-level flare
    blend -- see baseline/deterministic_rtl.py's tuning notes). The old
    ~24.7-25.0 m values were all clustered near R_home_m because they were
    the position at the moment the episode ended by CROSSING that radius;
    under the new task the episode runs to touchdown, so final distances
    vary with wherever the controller actually lands."""
    from env.glider_env import GliderEnv
    from baseline.deterministic_rtl import DeterministicRTL

    expected_final_dist = np.array([
        47.12624594, 39.17500110, 43.15555920, 25.45874363, 11.64544601,
        41.07336285, 26.22067733, 48.45388996, 21.37594191,  9.86971285,
         1.81280433, 36.07644867, 41.39158878, 21.42766124, 48.03224165,
        16.71123792,  3.52328116, 11.81277227,  5.59599839, 18.44716598,
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


# ---------------------------------------------------------------------------
# Phase 5 reward-rescaling tests (RLGlider_Phase5_Reward_Rescaling_Spec.md)
# ---------------------------------------------------------------------------

def test_barrier_inert_below_flare():
    """The reachability barrier must never fire below barrier_min_agl_m --
    pre-fix it fired on 79-186 steps/episode, 72% of them below 3 m AGL,
    degenerating into a landing-accuracy penalty on geometry the policy can
    no longer change (Defect A, RLGlider_Phase5_Reward_Rescaling_Spec.md
    §1)."""
    from env.glider_env import GliderEnv
    from env.reward import DEFAULT_REWARD_CFG
    from baseline.deterministic_rtl import DeterministicRTL

    cfg = dict(STAGES[0])
    env = GliderEnv(cfg=cfg)
    ctrl = DeterministicRTL()
    ctrl.reset()
    obs, _ = env.reset(seed=0)

    barrier_min_agl = float(DEFAULT_REWARD_CFG['barrier_min_agl_m'])
    while True:
        action = ctrl.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        if float(env._fdm.altitude) < barrier_min_agl:
            assert not info['unreach_violation'], (
                f"barrier fired at {env._fdm.altitude:.2f} m AGL, below the "
                f"{barrier_min_agl} m gate"
            )
        if terminated or truncated:
            break


def test_barrier_penalty_capped():
    """A synthetic state far outside glide range must clip to exactly
    w_unreach_cap PER SECOND (Phase 6: the cap is applied before the dt_rl
    scaling that makes the barrier share units with r_path), not the
    unbounded linear penalty -- guards against the cap being dropped in a
    future edit."""
    import pytest
    from env.reward import compute_reward, DEFAULT_REWARD_CFG

    cfg = dict(DEFAULT_REWARD_CFG)
    state = build_state(p_ned=np.array([5000.0, 0.0, -15.0]), v_body=np.array([10.0, 0.0, 0.0]))
    obs = np.zeros(12, dtype=np.float32)
    zero_action = np.zeros(2, dtype=np.float32)

    r, terminated, truncated, info = compute_reward(
        state=state, prev_state=state, obs=obs, action=zero_action,
        prev_action=zero_action, home_ned=np.zeros(3), cfg=cfg,
        alpha_true=np.radians(3.0), airspeed_true=10.0,
        touched_down=False, fault=False,
    )

    expected = -cfg['w_unreach_cap'] * cfg['dt_rl']
    assert info['unreach_violation']
    assert info['penalty_unreach'] == pytest.approx(expected), (
        f"expected penalty_unreach == {expected}, got {info['penalty_unreach']}"
    )


def test_precision_reward_monotone_and_has_gradient():
    """r_terminal must be strictly decreasing in dist_home with a
    non-vanishing gradient across the whole range the airframe actually
    achieves -- under the old Gaussian, the 100->150 m finite difference was
    ~1e-5 (Defect B); the hyperbolic replacement must clear 1.0 point there."""
    from env.reward import compute_reward, DEFAULT_REWARD_CFG

    def terminal_reward(dist_home):
        state = build_state(p_ned=np.array([dist_home, 0.0, 0.0]), v_body=np.array([9.0, 0.0, 0.5]))
        obs = np.zeros(12, dtype=np.float32)
        zero_action = np.zeros(2, dtype=np.float32)
        _, _, _, info = compute_reward(
            state=state, prev_state=state, obs=obs, action=zero_action,
            prev_action=zero_action, home_ned=np.zeros(3), cfg=dict(DEFAULT_REWARD_CFG),
            alpha_true=np.radians(3.0), airspeed_true=9.0,
            touched_down=True, fault=False,
        )
        assert info['quality'] == 1.0
        return info['r_terminal']

    dists = [0, 1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
    rewards = [terminal_reward(d) for d in dists]

    assert all(rewards[i] > rewards[i + 1] for i in range(len(rewards) - 1)), (
        f"r_terminal not strictly decreasing in dist_home: {list(zip(dists, rewards))}"
    )

    grad_100_150 = abs(rewards[dists.index(150)] - rewards[dists.index(100)])
    assert grad_100_150 > 1.0, (
        f"gradient between 100-150 m is {grad_100_150:.6f}, expected > 1.0 "
        f"(this is the check that catches Defect B -- under the Gaussian it was ~1e-5)"
    )


def test_curriculum_force_advance():
    """max_episodes_at_stage must force an advance once the cap is hit, even
    if the success-rate threshold is never met -- guards a run from stalling
    at one stage for its whole timestep budget (eval_venv is pinned at
    STAGES[-1], so a stall means best_model.zip is selected by a policy that
    never trained against wind/gusts/noise/domain-randomisation). Also checks
    that an EARNED advance sets last_advance_was_forced=False, so the flag
    isn't simply always true after any advance."""
    from env.curriculum import CurriculumScheduler

    # --- forced advance: never meets the (deliberately unreachable) threshold
    forced = CurriculumScheduler(advance_threshold=0.99, rolling_window=10, max_episodes_at_stage=25)
    advanced_on = None
    for i in range(25):
        if forced.record_episode(success=False):
            advanced_on = i
    assert advanced_on == 24, f"expected the 25th episode (index 24) to force the advance, got {advanced_on}"
    assert forced.current_stage == 1
    assert forced.last_advance_was_forced is True
    assert forced.episodes_in_window == 0, "rolling buffer must be cleared on advance"
    assert forced.episodes_at_stage == 0, "at-stage counter must be reset on advance"

    # --- earned advance: threshold trivially met, so last_advance_was_forced must be False
    earned = CurriculumScheduler(advance_threshold=0.5, rolling_window=10, max_episodes_at_stage=1000)
    for _ in range(10):
        earned.record_episode(success=True)
    assert earned.current_stage == 1
    assert earned.last_advance_was_forced is False
    assert earned.episodes_in_window == 0
    assert earned.episodes_at_stage == 0


def test_curriculum_force_advance_disabled_by_default():
    """max_episodes_at_stage defaults to None (disabled) -- the pre-Phase-5
    behaviour must be unchanged: a scheduler that never earns an advance
    stays at Stage 0 indefinitely."""
    from env.curriculum import CurriculumScheduler

    scheduler = CurriculumScheduler()
    for _ in range(500):
        scheduler.record_episode(success=False)

    assert scheduler.current_stage == 0
    assert scheduler.last_advance_was_forced is False


def test_curriculum_state_roundtrip():
    """state_dict()/load_state_dict() must round-trip enough scheduler state
    that a --resume restores the curriculum stage instead of silently
    regressing to Stage 0 (training runbook Phase B.1). Also checks that
    load_state_dict({}) on a fresh scheduler is a tolerant no-op, so an
    older checkpoint without a curriculum JSON does not crash a resume."""
    from env.curriculum import CurriculumScheduler

    src = CurriculumScheduler(advance_threshold=0.5, rolling_window=10, max_episodes_at_stage=1000)
    for _ in range(10):
        src.record_episode(success=True)   # earns the advance to Stage 1
    assert src.current_stage == 1
    for success in (True, False, True):
        src.record_episode(success=success)

    state = src.state_dict()

    dst = CurriculumScheduler(advance_threshold=0.5, rolling_window=10, max_episodes_at_stage=1000)
    dst.load_state_dict(state)

    assert dst.current_stage == src.current_stage
    assert dst.episodes_at_stage == src.episodes_at_stage
    assert dst.episodes_seen == src.episodes_seen
    assert dst.success_rate == src.success_rate
    assert list(dst._buffer) == list(src._buffer)

    fresh = CurriculumScheduler()
    fresh.load_state_dict({})   # must not raise
    assert fresh.current_stage == 0


def test_reward_cfg_config_parity():
    """Every key in DEFAULT_REWARD_CFG must also appear in training/configs/
    base.yaml's reward: section -- Phases 1 and 2 both added keys to both
    places, and a silent divergence between them means training and tests
    optimise different reward functions."""
    import pathlib
    import yaml
    from env.reward import DEFAULT_REWARD_CFG

    base_yaml_path = pathlib.Path(__file__).parent.parent / 'training' / 'configs' / 'base.yaml'
    with open(base_yaml_path) as f:
        base_cfg = yaml.safe_load(f)

    yaml_reward_keys = set(base_cfg['reward'].keys())
    missing = set(DEFAULT_REWARD_CFG.keys()) - yaml_reward_keys
    assert not missing, (
        f"keys present in DEFAULT_REWARD_CFG but missing from base.yaml's reward: "
        f"section: {missing} -- training would silently use the DEFAULT_REWARD_CFG "
        f"value for these while tests use whatever DEFAULT_REWARD_CFG also uses, "
        f"diverging the moment either is changed independently"
    )

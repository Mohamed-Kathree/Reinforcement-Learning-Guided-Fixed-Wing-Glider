"""
test_dynamics.py
================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Pytest unit-test suite for the physics simulation and key subsystems.

Tests (CLAUDE.md Section 18):
    test_energy_no_drag         -- total mechanical energy conserved when CD=0
    test_stall_cl_drop          -- CL decreases past stall angle (no hard jump)
    test_quat_norm_preserved    -- quaternion stays within 1e-6 of 1.0 over 200 steps
    test_lidar_agl_correction   -- slant-range-corrected AGL formula is correct
    test_servo_rate_limit       -- servo position cannot change faster than rate_limit
    test_baseline_reaches_home  -- DeterministicRTL returns home in Stage 0 conditions

Additional coverage:
    test_wind_model_zero_gust   -- zero gust intensity -> total wind == mean wind
    test_wind_model_ou_bounded  -- OU gust stays bounded (|gust| < 5*sigma at 3-sigma noise)
    test_sensor_gps_update_rate -- GPS refreshes at 5 Hz (every 40 physics steps)
    test_sensor_lidar_update_rate -- LiDAR refreshes at 50 Hz (every 4 physics steps)
    test_sensor_lidar_range_gate  -- LiDAR reads invalid outside [0.2, 40] m
    test_curriculum_advance     -- scheduler advances at 80% success over 100 episodes
    test_curriculum_no_advance  -- scheduler does NOT advance below threshold
    test_curriculum_full_window -- scheduler requires a full window before advancing
    test_reward_success         -- reaching home triggers terminal bonus
    test_reward_crash           -- ground impact triggers crash penalty
    test_reward_progress        -- closing distance gives positive progress reward
    test_reward_agl_penalty     -- AGL below minimum with valid LiDAR gives penalty

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.glider_dynamics import GliderDynamics, GliderParams, build_state, ZERO_CONTROLS
from sim.aerodynamics import CL, CD, forces_moments
from sim.wind import WindModel
from sim.sensor_models import SensorSuite, _lidar_noise
from sim.actuator_models import ServoModel, ServoParams, ActuatorSuite
from sim.math_utils import euler_from_quat, euler_to_quat
from env.curriculum import CurriculumScheduler, STAGES
from env.reward import compute_reward, DEFAULT_REWARD_CFG
from baseline.deterministic_rtl import DeterministicRTL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT_PHYS  = 0.005   # 200 Hz physics step
DT_RL    = 0.050   # 20 Hz policy step
N_STEPS_RL = int(DT_RL / DT_PHYS)  # 10 substeps per RL step

_PARAMS = GliderParams()


def _trim_state(alt_m: float = 100.0, speed_ms: float = 10.0) -> np.ndarray:
    """Build a plausible near-trim launch state at the given altitude."""
    gamma = np.radians(15.0)
    return build_state(
        p_ned  = np.array([0.0, 0.0, -alt_m]),
        v_body = np.array([speed_ms * np.cos(gamma), 0.0, -speed_ms * np.sin(gamma)]),
        pitch  = gamma,
    )


def _zero_wind() -> np.ndarray:
    return np.zeros(3)


# ---------------------------------------------------------------------------
# Physics: energy conservation
# ---------------------------------------------------------------------------

class TestEnergyNoRag:
    """Total mechanical energy must be conserved when drag is zero."""

    def test_energy_no_drag(self):
        """E = 0.5*m*V^2 + m*g*h must not drift by more than 0.1% over 10 s
        when both aerodynamic forces AND moments are zero (pure projectile).

        We inject a zero-aero function so that Cm0 (pitch trim offset) does
        not build angular rate through the omega×v_body coupling term.  With
        zero moments the quaternion stays constant, gravity acts as a fixed
        body-frame force, and RK4 conserves energy to numerical precision.
        """
        p = GliderParams()

        def _zero_aero(V, alpha, beta, omega, controls, params):
            return np.zeros(3), np.zeros(3)

        dyn   = GliderDynamics(params=p, aero_fn=_zero_aero)
        state = _trim_state(alt_m=100.0, speed_ms=10.0)
        dyn.reset(state)

        def _energy(s):
            V = float(np.linalg.norm(s[3:6]))
            h = float(-s[2])
            return 0.5 * p.m * V**2 + p.m * p.g * h

        E0 = _energy(dyn.state)
        n_steps = int(10.0 / DT_PHYS)   # 10 s at 200 Hz

        for _ in range(n_steps):
            dyn.step(ZERO_CONTROLS, _zero_wind(), DT_PHYS)

        E1 = _energy(dyn.state)
        drift_pct = abs(E1 - E0) / abs(E0) * 100.0
        assert drift_pct < 0.1, (
            f"Energy drifted {drift_pct:.4f}% over 10 s (limit 0.1%); "
            "check RK4 integrator or gravity term sign."
        )


# ---------------------------------------------------------------------------
# Aerodynamics: stall
# ---------------------------------------------------------------------------

class TestStallCLDrop:
    """CL must decrease past the stall angle, with no hard discontinuity."""

    def test_stall_cl_drop(self):
        """CL at alpha=15° must be strictly less than CL at alpha=12° (stall)."""
        p          = GliderParams()
        cl_at_stall = CL(p.a_stall, p)
        cl_past_stall = CL(np.radians(15.0), p)
        assert cl_past_stall < cl_at_stall, (
            f"CL did not drop past stall: "
            f"CL(stall)={cl_at_stall:.4f}  CL(15°)={cl_past_stall:.4f}"
        )

    def test_cl_positive_pre_stall(self):
        """CL must be positive for positive pre-stall alpha."""
        p = GliderParams()
        for alpha_deg in [1, 3, 5, 8, 10, 11]:
            cl = CL(np.radians(alpha_deg), p)
            assert cl > 0.0, f"CL <= 0 at alpha={alpha_deg}°: {cl}"

    def test_cl_continuous_at_stall(self):
        """CL must be continuous at the stall angle (no hard jump)."""
        p = GliderParams()
        eps = 1e-4
        cl_just_before = CL(p.a_stall - eps, p)
        cl_just_after  = CL(p.a_stall + eps, p)
        assert abs(cl_just_before - cl_just_after) < 0.05, (
            f"CL discontinuity at stall: "
            f"CL(stall-ε)={cl_just_before:.4f}  CL(stall+ε)={cl_just_after:.4f}"
        )

    def test_cd_never_below_cd0(self):
        """CD must always be at least CD0 (parabolic polar lower bound)."""
        p = GliderParams()
        for alpha_deg in range(-20, 21):
            cl = CL(np.radians(alpha_deg), p)
            cd = CD(cl, p)
            assert cd >= p.CD0 - 1e-9, (
                f"CD < CD0 at alpha={alpha_deg}°: CD={cd:.6f} CD0={p.CD0}"
            )


# ---------------------------------------------------------------------------
# Physics: quaternion norm
# ---------------------------------------------------------------------------

class TestQuatNorm:
    """Quaternion norm must stay within 1e-6 of 1.0 throughout integration."""

    def test_quat_norm_preserved(self):
        """Run 200 RL steps (2000 RK4 substeps) and check quaternion norm each step."""
        dyn   = GliderDynamics()
        state = _trim_state(alt_m=100.0, speed_ms=10.0)
        dyn.reset(state)
        rng   = np.random.default_rng(0)

        for step in range(200):
            # Random controls to exercise attitude changes
            controls = {
                'aileron':  float(rng.uniform(-0.1, 0.1)),
                'elevator': float(rng.uniform(-0.05, 0.05)),
                'rudder':   0.0,
            }
            for _ in range(N_STEPS_RL):
                dyn.step(controls, _zero_wind(), DT_PHYS)
                norm = float(np.linalg.norm(dyn.state[6:10]))
                assert abs(norm - 1.0) < 1e-6, (
                    f"Quaternion norm drifted at step {step}: |q|={norm:.10f}"
                )

            # Stop if glider hits ground (normal termination, not a bug)
            if dyn.state[2] > 0.0:
                break


# ---------------------------------------------------------------------------
# Sensor models: LiDAR AGL correction
# ---------------------------------------------------------------------------

class TestLidarAGLCorrection:
    """Slant-range-corrected AGL formula: h_agl = r_slant * cos(roll) * cos(pitch)."""

    def test_lidar_agl_correction_formula(self):
        """At known roll/pitch and true AGL, verify the corrected reading is correct."""
        rng  = np.random.default_rng(42)
        roll  = np.radians(10.0)
        pitch = np.radians(5.0)
        true_agl = 15.0   # m, within LiDAR range [0.2, 40]

        # Run many samples; with no noise (noise_scale=0 not available, so
        # use a very large sample to average out 0.05 m noise)
        readings = []
        for _ in range(500):
            h, valid = _lidar_noise(
                true_agl, roll, pitch, rng,
                dropout_prob=0.0, noise_scale=1.0
            )
            if valid:
                readings.append(h)

        assert len(readings) > 400, "Too many dropouts at dropout_prob=0"
        mean_h = float(np.mean(readings))
        expected = true_agl * np.cos(roll) * np.cos(pitch)
        assert abs(mean_h - expected) < 0.2, (
            f"LiDAR AGL correction: mean={mean_h:.3f}  expected={expected:.3f}"
        )

    def test_lidar_range_gate_low(self):
        """Reading below 0.2 m must return invalid."""
        rng = np.random.default_rng(0)
        h, valid = _lidar_noise(0.1, 0.0, 0.0, rng, dropout_prob=0.0)
        assert not valid, "Expected invalid for AGL < 0.2 m"
        assert h == 0.0

    def test_lidar_range_gate_high(self):
        """Reading above 40 m must return invalid."""
        rng = np.random.default_rng(0)
        h, valid = _lidar_noise(50.0, 0.0, 0.0, rng, dropout_prob=0.0)
        assert not valid, "Expected invalid for AGL > 40 m"
        assert h == 0.0

    def test_lidar_dropout(self):
        """With dropout_prob=1.0 every reading must be invalid."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            h, valid = _lidar_noise(10.0, 0.0, 0.0, rng, dropout_prob=1.0)
            assert not valid
            assert h == 0.0


# ---------------------------------------------------------------------------
# Sensor models: update rates
# ---------------------------------------------------------------------------

class TestSensorUpdateRates:
    """Each sensor must refresh at its declared Hz (zero-order hold between)."""

    def _count_updates(self, sensor_attr: str, n_physics_steps: int) -> int:
        """Count how many times a sensor reading changes over n_physics_steps."""
        suite = SensorSuite()
        rng   = np.random.default_rng(7)
        # 20 m AGL so LiDAR is in range
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

    def test_gps_update_rate(self):
        """GPS at 5 Hz: over 200 physics steps (1 s) expect ~5 updates."""
        n = self._count_updates('gps_pos', 200)
        assert 3 <= n <= 7, f"GPS update count wrong: {n} (expected ~5)"

    def test_lidar_update_rate(self):
        """LiDAR at 50 Hz: over 200 physics steps (1 s) expect ~50 updates."""
        n = self._count_updates('lidar_agl', 200)
        assert 40 <= n <= 60, f"LiDAR update count wrong: {n} (expected ~50)"

    def test_imu_update_rate(self):
        """IMU at 200 Hz: every physics step should produce a new reading."""
        n = self._count_updates('imu_euler', 200)
        # IMU updates every step; nearly all 200 should differ (noise guarantees it)
        assert n >= 190, f"IMU update count too low: {n} (expected ~200)"


# ---------------------------------------------------------------------------
# Actuator models: servo rate limit
# ---------------------------------------------------------------------------

class TestServoRateLimit:
    """Servo position cannot change faster than rate_limit * dt per step."""

    def test_servo_rate_limit(self):
        """Step-change command: position increment must not exceed rate * dt."""
        params = ServoParams(
            max_rad    = np.radians(25.0),
            rate_rad_s = np.radians(200.0),
            tau_s      = 0.05,
        )
        servo = ServoModel(params)
        servo.reset(pos=0.0)

        cmd = np.radians(25.0)   # command full deflection immediately
        dt  = DT_PHYS

        prev_pos = servo.pos
        for _ in range(50):
            new_pos = servo.step(cmd, dt)
            delta   = abs(new_pos - prev_pos)
            max_delta = params.rate_rad_s * dt + 1e-9   # tiny float tolerance
            assert delta <= max_delta, (
                f"Servo exceeded rate limit: Δpos={np.degrees(delta):.4f}° "
                f"> max {np.degrees(max_delta):.4f}° "
                f"(rate={np.degrees(params.rate_rad_s):.0f}°/s, dt={dt})"
            )
            prev_pos = new_pos

    def test_servo_saturation(self):
        """Servo position must never exceed ±max_rad regardless of command."""
        params = ServoParams(
            max_rad    = np.radians(20.0),
            rate_rad_s = np.radians(300.0),
            tau_s      = 0.01,
        )
        servo = ServoModel(params)
        servo.reset(pos=0.0)

        for cmd_deg in [90.0, -90.0, 45.0, -45.0]:
            for _ in range(100):
                pos = servo.step(np.radians(cmd_deg), DT_PHYS)
                assert abs(pos) <= params.max_rad + 1e-9, (
                    f"Servo exceeded saturation: |pos|={np.degrees(abs(pos)):.2f}° "
                    f"> max {np.degrees(params.max_rad):.1f}°"
                )

    def test_actuator_suite_keys(self):
        """ActuatorSuite.step() must return exactly the three expected keys."""
        suite = ActuatorSuite()
        suite.reset(rng=None)
        result = suite.step({'aileron': 0.1, 'elevator': -0.05, 'rudder': 0.0}, DT_PHYS)
        assert set(result.keys()) == {'aileron', 'elevator', 'rudder'}


# ---------------------------------------------------------------------------
# Wind model
# ---------------------------------------------------------------------------

class TestWindModel:
    """Wind model produces correct mean and bounded gust behaviour."""

    def test_zero_gust(self):
        """With gust_intensity=0, total wind must equal mean_ned exactly."""
        rng  = np.random.default_rng(1)
        mean = np.array([3.0, -2.0, 0.0])
        wm   = WindModel()
        wm.reset(mean_ned=mean, gust_intensity=0.0, rng=rng)

        for _ in range(100):
            w = wm.step()
            np.testing.assert_array_equal(w, mean)

    def test_gust_bounded(self):
        """OU gust magnitude should stay below 5*sigma over 10 s (extremely unlikely)."""
        rng       = np.random.default_rng(2)
        intensity = 1.0   # sigma = 1 m/s per axis
        wm        = WindModel()
        wm.reset(mean_ned=np.zeros(3), gust_intensity=intensity, rng=rng)

        for _ in range(int(10.0 / DT_PHYS)):
            w = wm.step()
            gust_mag = float(np.linalg.norm(wm.gust_ned))
            assert gust_mag < 5.0 * intensity * np.sqrt(3), (
                f"OU gust unexpectedly large: |gust|={gust_mag:.2f} m/s"
            )


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class TestCurriculum:
    """Scheduler advances stages correctly and enforces the full-window guard."""

    def test_advance_at_threshold(self):
        """Feed 100% success for a full window -> should advance to stage 1."""
        s = CurriculumScheduler(advance_threshold=0.80, rolling_window=100)
        assert s.current_stage == 0
        for _ in range(100):
            s.record_episode(success=True)
        assert s.current_stage == 1, (
            f"Expected stage 1 after 100 successes, got {s.current_stage}"
        )

    def test_no_advance_below_threshold(self):
        """70% success (below 80% threshold) must not advance the stage."""
        s = CurriculumScheduler(advance_threshold=0.80, rolling_window=100)
        for i in range(100):
            s.record_episode(success=(i % 10 < 7))   # 70 successes / 100
        assert s.current_stage == 0, (
            f"Should not have advanced at 70%: stage={s.current_stage}"
        )

    def test_requires_full_window(self):
        """Scheduler must NOT advance before the rolling window is full."""
        s = CurriculumScheduler(advance_threshold=0.80, rolling_window=100)
        # Feed 99 successes (window not yet full)
        for _ in range(99):
            advanced = s.record_episode(success=True)
            assert not advanced, "Advanced before window was full"
        assert s.current_stage == 0

    def test_buffer_cleared_on_advance(self):
        """After advancing, the rolling buffer must be empty (fresh start)."""
        s = CurriculumScheduler(advance_threshold=0.80, rolling_window=10)
        for _ in range(10):
            s.record_episode(success=True)
        assert s.current_stage == 1
        assert s.episodes_in_window == 0

    def test_advance_through_all_stages(self):
        """Must be able to advance through all four stages sequentially."""
        s = CurriculumScheduler(advance_threshold=0.80, rolling_window=10)
        for target_stage in range(1, 4):
            for _ in range(10):
                s.record_episode(success=True)
            assert s.current_stage == target_stage, (
                f"Expected stage {target_stage}, got {s.current_stage}"
            )
        # At final stage, further successes do not advance
        advanced = s.record_episode(success=True)
        assert not advanced
        assert s.current_stage == 3

    def test_current_cfg_is_copy(self):
        """current_cfg must return a new dict each call, not the STAGES reference."""
        s = CurriculumScheduler()
        cfg1 = s.current_cfg
        cfg1['_sentinel'] = True
        cfg2 = s.current_cfg
        assert '_sentinel' not in cfg2, "current_cfg returned a mutable reference"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class TestReward:
    """Key reward components behave as documented."""

    def _base_cfg(self) -> dict:
        return dict(DEFAULT_REWARD_CFG)

    def _make_states(self, dist_home: float, prev_dist: float, alt: float = 50.0):
        """Build minimal state vectors for reward tests."""
        state      = np.zeros(13)
        prev_state = np.zeros(13)
        state[0:3]      = [dist_home,  0.0, -alt]
        prev_state[0:3] = [prev_dist,  0.0, -alt]
        # Valid quaternion (identity)
        state[6]      = 1.0
        prev_state[6] = 1.0
        # Plausible airspeed
        state[3]      = 10.0
        prev_state[3] = 10.0
        return state, prev_state

    def _base_obs(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float32)
        obs[10] = 0.0   # lidar_valid = False -> no AGL penalty
        return obs

    def test_reward_success(self):
        """Reaching home (dist < R_home_m) triggers the terminal bonus."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=5.0, prev_dist=20.0)
        home = np.zeros(3)
        r, terminated, _, info = compute_reward(
            state, prev, self._base_obs(),
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert terminated, "Should be terminated on success"
        assert info['success'], "info['success'] must be True"
        assert r > cfg['w_terminal'] * 0.9, (
            f"Reward {r:.1f} too low; expected terminal bonus ~{cfg['w_terminal']}"
        )

    def test_reward_crash(self):
        """Ground impact (state[2] > 0) triggers the crash penalty."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=200.0, prev_dist=200.0, alt=0.0)
        state[2] = 0.5   # below ground in NED (positive p_d)
        home = np.zeros(3)
        r, terminated, _, info = compute_reward(
            state, prev, self._base_obs(),
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert terminated, "Should be terminated on crash"
        assert info['crash'], "info['crash'] must be True"
        assert r < 0, f"Crash reward should be negative; got {r}"

    def test_reward_progress_positive(self):
        """Closing distance to home produces a positive progress reward."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=80.0, prev_dist=100.0)
        home = np.zeros(3)
        r, _, _, info = compute_reward(
            state, prev, self._base_obs(),
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert info['r_progress'] > 0, (
            f"Expected positive progress reward; got {info['r_progress']}"
        )

    def test_reward_progress_negative(self):
        """Moving away from home produces a negative progress reward."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=120.0, prev_dist=100.0)
        home = np.zeros(3)
        _, _, _, info = compute_reward(
            state, prev, self._base_obs(),
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert info['r_progress'] < 0, (
            f"Expected negative progress reward; got {info['r_progress']}"
        )

    def test_reward_agl_penalty(self):
        """AGL below agl_min with valid LiDAR triggers a quadratic penalty."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=200.0, prev_dist=200.0)
        home = np.zeros(3)
        obs = self._base_obs()
        obs[9]  = 4.0    # lidar_agl = 4 m (below agl_min = 8 m)
        obs[10] = 1.0    # lidar_valid = True
        _, _, _, info = compute_reward(
            state, prev, obs,
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert info['penalty_agl'] < 0, (
            f"Expected negative AGL penalty; got {info['penalty_agl']}"
        )
        assert info['agl_violation'], "agl_violation flag must be True"

    def test_reward_agl_no_penalty_when_invalid(self):
        """AGL penalty must be zero when LiDAR reports invalid."""
        cfg = self._base_cfg()
        state, prev = self._make_states(dist_home=200.0, prev_dist=200.0)
        home = np.zeros(3)
        obs = self._base_obs()
        obs[9]  = 2.0    # below agl_min
        obs[10] = 0.0    # lidar_valid = False
        _, _, _, info = compute_reward(
            state, prev, obs,
            np.zeros(2), np.zeros(2), home, cfg
        )
        assert info['penalty_agl'] == 0.0, (
            f"AGL penalty should be 0 when LiDAR invalid; got {info['penalty_agl']}"
        )


# ---------------------------------------------------------------------------
# Deterministic baseline: returns home
# ---------------------------------------------------------------------------

class TestBaselineReachesHome:
    """DeterministicRTL must return home reliably in ideal (Stage 0) conditions."""

    def test_baseline_reaches_home(self):
        """RTL controller must succeed in at least 95% of Stage 0 episodes."""
        from env.glider_env import GliderEnv

        cfg = dict(STAGES[0])   # perfect conditions
        env = GliderEnv(cfg=cfg)
        ctrl = DeterministicRTL()
        rng  = np.random.default_rng(0)

        n_episodes = 20
        successes  = 0

        for ep in range(n_episodes):
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
        assert success_rate >= 0.95, (
            f"DeterministicRTL Stage 0 success rate too low: "
            f"{success_rate:.0%} ({successes}/{n_episodes})"
        )

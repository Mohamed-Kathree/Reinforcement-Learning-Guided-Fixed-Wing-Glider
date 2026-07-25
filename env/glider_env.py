"""
glider_env.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Gymnasium environment that wraps the JSBSim-backed glider simulator and
exposes the standard reset() / step() interface for Stable-Baselines3.

Architecture (matches hardware split described in CLAUDE.md Section 0):
    RL policy          (20 Hz) -> bank_cmd, speed_cmd setpoints
    AttitudeController          -> aileron / elevator / rudder commands (inner loop)
    JSBSimFDM                   -> actuator lag/rate limits + 6-DOF physics at 200 Hz
    SensorSuite                 -> noisy, rate-limited observations

Each env.step() call:
    1. Denormalise action -> setpoints
    2. Run 10 physics substeps at 200 Hz, each substep:
       a. Wind model advances gust state
       b. AttitudeController computes servo commands
       c. Safety shield clips servo commands
       d. JSBSimFDM applies actuator lag/rate limits and integrates one step
       e. SensorSuite refreshes sensor buffers
    3. Build 11-element observation from sensor readings
    4. Call compute_reward() for reward + termination flags
    5. Return (obs, reward, terminated, truncated, info)

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
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces

from sim.jsbsim_fdm import JSBSimFDM
from sim.wind import WindModel
from sim.sensor_models import SensorSuite
from sim.math_utils import build_state, euler_from_quat, wrap_pi
from env.reward import compute_reward, DEFAULT_REWARD_CFG


# ---------------------------------------------------------------------------
# Inner controller PID gains (fixed; match the simulated ESP32 structure)
# ---------------------------------------------------------------------------

# Roll channel: PD on roll error + roll-rate damping
KP_ROLL:  float = 2.0
KD_ROLL:  float = 0.3   # reduced from 1.5: 1.5 caused limit-cycle where derivative cancelled proportional

# Pitch channel: trim-schedule maps speed command to pitch target
KP_PITCH: float = 3.0
KD_PITCH: float = 0.4
KV_PITCH: float = 0.05    # pitch offset per m/s above/below trim speed

# Yaw / coordinated-turn: rudder proportional to roll rate
KR_RUDDER: float = 0.05

# Trim reference
V_TRIM:     float = 10.0           # m/s best-glide speed
ALPHA_TRIM: float = np.radians(2.5)  # reduced from 4.8°: JSBSim trims lower; 4.8° trapped speed at 7 m/s

# Safety shield thresholds
SHIELD_MAX_BANK_RAD:    float = np.radians(45.0)   # increased from 40°: gives headroom during nominal ±30° command
# Landing flare trigger, not a continuous in-flight AGL floor: the only
# ground-proximity sensor on the hardware is a short-range ultrasonic
# (RCWL-1655, valid <4.5 m -- see sim/sensor_models.py), so this can only
# ever fire in the last couple of seconds before touchdown.
SHIELD_MIN_AGL_M:       float = 2.5
SHIELD_PULLUP_ELEV_RAD: float = np.radians(5.0)
SHIELD_MAX_ALPHA_RAD:   float = np.radians(10.0)  # stall warning margin

# Physics / policy rates
DT_PHYS:  float = 0.005   # 200 Hz
DT_RL:    float = 0.050   # 20 Hz
N_SUBSTEPS: int = 10      # DT_RL / DT_PHYS

# Episode limits
MAX_STEPS: int = 2000     # ~100 s of flight

# Action scaling (denormalise ±1 -> physical units)
BANK_CMD_SCALE_RAD:  float = np.radians(45.0)   # ±45° bank
SPEED_CMD_CENTRE_MS: float = 9.0
SPEED_CMD_SCALE_MS:  float = 4.0                # range 5–13 m/s

# Launch state randomisation (fallback defaults; overridable via cfg -- see
# training/configs/base.yaml launch.speed_{min,max}_ms / pitch_{min,max}_deg).
#
# The episode start represents whatever energy state the (not-yet-built)
# ESP32 apex-detection firmware hands control over at. Since that firmware
# doesn't exist yet, the real hand-off state is unknown -- it could be close
# to a true apex (near-level pitch, ~trim speed) or closer to the original
# sim assumption of releasing near launch speed while still pitched up. This
# range is deliberately wide to cover both ends rather than betting on one:
# training on a single fixed start risks an out-of-distribution hand-off
# state once real apex-detection is characterised, whereas this costs
# nothing now (it's the same domain-randomisation strategy already used for
# wind/sensors, just applied to the initial condition instead of the dynamics).
LAUNCH_SPEED_MIN_MS: float = 8.0
LAUNCH_SPEED_MAX_MS: float = 15.0
LAUNCH_PITCH_MIN_DEG: float = 0.0
LAUNCH_PITCH_MAX_DEG: float = 15.0     # nose-up (positive pitch)


# ---------------------------------------------------------------------------
# Inner attitude controller
# ---------------------------------------------------------------------------

class AttitudeController:
    """Simulated ESP32 inner-loop controller.

    Converts bank and speed setpoints into servo deflection commands.
    Runs at every 200 Hz physics substep inside env.step().
    """

    def update(
        self,
        state:        npt.NDArray,
        bank_cmd_rad: float,
        speed_cmd_ms: float,
    ) -> dict:
        """Compute aileron/elevator/rudder commands.

        Args:
            state        : 13-element physics state
            bank_cmd_rad : desired roll angle (rad), ±π/2
            speed_cmd_ms : desired airspeed (m/s)

        Returns:
            dict with keys 'aileron', 'elevator', 'rudder' (rad)
        """
        roll, pitch, _ = euler_from_quat(state[6:10])
        p_rate, q_rate, r_rate = float(state[10]), float(state[11]), float(state[12])

        # Roll PD — wrap_pi ensures shortest-path recovery at any bank angle
        roll_err     = wrap_pi(bank_cmd_rad - roll)
        aileron_cmd  = KP_ROLL * roll_err - KD_ROLL * p_rate

        # Pitch: speed-scheduled pitch-angle controller.
        # Target pitch = ALPHA_TRIM * (V_TRIM/V)^2 / cos(roll).
        # Rationale: CL for level flight scales as 1/V^2, and CL ≈ a0*alpha ≈
        # a0*pitch at small angles in steady glide.  Dividing by cos(roll)
        # restores the reduced vertical lift component in banked turns.
        # Controlling PITCH (not alpha) prevents the pitch-spiral that an
        # alpha-only controller causes due to the Cm0>0 nose-down tendency.
        # Negate sign: Cm_de=-1.2 means positive elevator=nose-down, so a
        # positive pitch_err (need nose-down) drives a negative elevator cmd.
        V_actual     = float(np.linalg.norm(state[3:6]))
        V_clamp      = max(V_actual, 6.0)
        cos_roll     = float(np.cos(roll))
        pitch_target = (ALPHA_TRIM * (V_TRIM / V_clamp) ** 2) / max(abs(cos_roll), 0.5)
        pitch_target = float(np.clip(pitch_target, np.radians(1.0), np.radians(10.0)))
        pitch_target -= KV_PITCH * (speed_cmd_ms - V_TRIM)
        pitch_err    = pitch_target - pitch
        # Use Euler pitch rate θ_dot = q·cos(φ) − r·sin(φ) as the derivative
        # term. Body pitch rate q alone is blind to the turn-coupling disturbance
        # −r·sin(φ) that drives pitch DOWN in banked turns (ZYX kinematics).
        theta_dot_est = q_rate * np.cos(roll) - r_rate * np.sin(roll)
        elevator_cmd = -(KP_PITCH * pitch_err - KD_PITCH * theta_dot_est)

        # Coordinated turn: rudder follows roll rate
        rudder_cmd = KR_RUDDER * p_rate

        return {
            'aileron':  float(np.clip(aileron_cmd,  -np.radians(25), np.radians(25))),
            'elevator': float(np.clip(elevator_cmd, -np.radians(20), np.radians(20))),
            'rudder':   float(np.clip(rudder_cmd,   -np.radians(25), np.radians(25))),
        }


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class GliderEnv(gym.Env):
    """Gymnasium environment for the RL-guided fixed-wing glider RTL task.

    Observation space : Box(11,) float32 — see CLAUDE.md Section 10.2
    Action space      : Box(2,)  float32 — normalised [bank_cmd, speed_cmd]

    Episode ends when:
        - Glider reaches home (dist < R_home_m) -> success, terminated=True
        - Ground impact (state[2] > 0)          -> crash,   terminated=True
        - Step limit (MAX_STEPS)                -> truncated=True

    Args:
        cfg : dict of overrides merged into DEFAULT_REWARD_CFG. Also accepts
              curriculum stage keys (wind_speed, gust_intensity, sensor_noise,
              dropout_prob, R_home_m, alt0_m, launch_offset_min_m,
              launch_offset_max_m, aero_scale_range, mass_range) — these are
              applied at each reset(). launch_speed_min_ms/max_ms and
              launch_pitch_min_deg/max_deg are accepted too but are NOT
              curriculum-stage-dependent (same range at every stage; see
              env/curriculum.py's module docstring).
        seed: passed to reset() if provided at construction time.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg:  dict | None = None,
        seed: int | None  = None,
    ) -> None:
        super().__init__()

        # Merge caller config over defaults
        self.cfg: dict = {**DEFAULT_REWARD_CFG}
        if cfg is not None:
            self.cfg.update(cfg)

        # Ensure timing keys are present
        self.cfg.setdefault('dt_rl', DT_RL)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (11,),
            dtype = np.float32,
        )
        self.action_space = spaces.Box(
            low   = np.array([-1.0, -1.0], dtype=np.float32),
            high  = np.array([ 1.0,  1.0], dtype=np.float32),
            dtype = np.float32,
        )

        # Sub-systems (reinitialised at reset)
        self._fdm     = JSBSimFDM(dt_phys=DT_PHYS)
        self._wind    = WindModel(dt=DT_PHYS)
        self._sensors = SensorSuite()
        self._ctrl    = AttitudeController()

        # Episode state
        self._step_count: int         = 0
        self._prev_state: npt.NDArray = np.zeros(13, dtype=np.float64)
        self._prev_action: npt.NDArray = np.zeros(2, dtype=np.float32)
        self._home_ned: npt.NDArray   = np.zeros(3, dtype=np.float64)

        # Cached last observation (built in _build_obs; exposed for reward fn)
        self._last_obs: npt.NDArray = np.zeros(11, dtype=np.float32)

        # RNG (seeded properly in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    int | None  = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray, dict]:
        """Reset the environment for a new episode.

        Applies domain randomisation from self.cfg (curriculum stage keys).
        Returns the initial observation and an empty info dict.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Allow a one-shot cfg override via options
        if options is not None:
            self.cfg.update(options)

        # Home is always at the NED origin
        self._home_ned = np.zeros(3, dtype=np.float64)

        # --- Domain randomisation ----------------------------------------
        dr = self._sample_domain_rand()

        # Wind: random direction + speed up to curriculum maximum
        wind_dir   = self._rng.uniform(0.0, 2.0 * np.pi)
        wind_speed = self._rng.uniform(0.0, float(self.cfg.get('wind_speed', 0.0)))
        mean_ned   = np.array([
            wind_speed * np.cos(wind_dir),
            wind_speed * np.sin(wind_dir),
            0.0,
        ], dtype=np.float64)
        gust_intensity = float(self._rng.uniform(
            0.0, float(self.cfg.get('gust_intensity', 0.0))
        ))
        self._wind.reset(mean_ned, gust_intensity, self._rng)

        # Sensor noise
        noise_scale  = float(self._rng.uniform(
            0.5, max(float(self.cfg.get('sensor_noise', 0.0)), 0.5)
        ))
        dropout_prob = float(self._rng.uniform(
            0.0, float(self.cfg.get('dropout_prob', 0.0))
        ))

        # --- Launch state ------------------------------------------------
        launch_state = self._build_launch_state()
        self._fdm.reset(launch_state)
        self._fdm.apply_domain_rand(**dr)

        # Seed sensors from initial state
        self._sensors.reset(
            state        = self._fdm.state,
            rng          = self._rng,
            noise_scale  = noise_scale,
            dropout_prob = dropout_prob,
        )

        # Episode bookkeeping
        self._step_count    = 0
        self._prev_state    = self._fdm.state.copy()
        self._prev_action   = np.zeros(2, dtype=np.float32)
        self._has_left_home = False   # must move > R_home_m away before success counts

        obs = self._build_obs()
        self._last_obs = obs.copy()

        return obs, {}

    def step(
        self,
        action: npt.NDArray,
    ) -> tuple[npt.NDArray, float, bool, bool, dict]:
        """Advance the environment by one policy step (DT_RL = 0.05 s).

        Args:
            action : normalised [bank_cmd, speed_cmd] ∈ [-1, 1]²

        Returns:
            obs        : 11-element float32 observation
            reward     : scalar shaped reward
            terminated : True when the episode ends (success or crash)
            truncated  : True when the step limit is reached
            info       : per-component reward breakdown + diagnostics
        """
        action = np.asarray(action, dtype=np.float32)
        bank_cmd_rad  = float(action[0]) * BANK_CMD_SCALE_RAD
        speed_cmd_ms  = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS

        prev_state  = self._fdm.state.copy()

        # --- 10 physics substeps (200 Hz inner loop) --------------------
        for _ in range(N_SUBSTEPS):
            # 1. Advance wind gust
            wind_ned = self._wind.step()

            # 2. Inner controller: bank + speed setpoints -> servo commands
            ctrl_cmds = self._ctrl.update(
                self._fdm.state, bank_cmd_rad, speed_cmd_ms
            )

            # 3. Safety shield: clip dangerous commands
            ctrl_cmds = self._safety_shield(self._fdm.state, ctrl_cmds)

            # 4. JSBSim step: normalises radian commands internally, applies
            #    actuator lag/rate-limit, advances 6-DOF EOM one dt step
            self._fdm.step(ctrl_cmds, wind_ned, DT_PHYS)

            # 5. Abort substep loop on solver failure or ground contact.
            #    touched_down fires ~0.1 m of AGL before p_d>0 (CG altitude)
            #    would -- see JSBSimFDM.touched_down. Stopping here, in the
            #    same physics step contact first occurs, avoids running the
            #    stiff spring-damper contact model through several more
            #    compression/release cycles, which was observed to
            #    numerically diverge (sudden altitude spike -> NaN state).
            if self._fdm.fault or self._fdm.touched_down:
                break

            # 6. Sensor suite: refresh rate-limited buffers
            self._sensors.step(self._fdm.state, self._rng)

        # --- Build observation -------------------------------------------
        obs = self._build_obs()
        self._last_obs = obs.copy()

        # --- Reward + termination ----------------------------------------
        reward, terminated, truncated_r, info = compute_reward(
            state       = self._fdm.state,
            prev_state  = prev_state,
            obs         = obs,
            action      = action,
            prev_action = self._prev_action,
            home_ned    = self._home_ned,
            cfg         = self.cfg,
        )

        # Ground contact (touched_down) or JSBSim solver/state failure (fault):
        # treat as an immediate crash, taking priority over success -- mirrors
        # the "crash checked first" rule in compute_reward's own state[2]>0
        # branch -- so a landing/fault event is never silently left counted
        # as neither success nor crash.
        if self._fdm.fault or self._fdm.touched_down:
            if not info['crash']:
                reward -= float(self.cfg.get('w_crash', DEFAULT_REWARD_CFG['w_crash']))
                info['crash']   = True
                info['success'] = False
            terminated = True

        # Track whether the glider has flown outside R_home_m; success is
        # only valid after the glider has genuinely departed from the launch
        # point — prevents step-0 false success when dist_home_2D == 0.
        r_home = float(self.cfg.get('R_home_m', DEFAULT_REWARD_CFG['R_home_m']))
        if info['dist_home'] > r_home:
            self._has_left_home = True
        if info['success'] and not self._has_left_home:
            reward  -= float(self.cfg.get('w_terminal', DEFAULT_REWARD_CFG['w_terminal']))
            terminated = False
            info['success'] = False

        # Step-limit truncation (handled here, not in reward)
        self._step_count += 1
        truncated = truncated_r or (self._step_count >= MAX_STEPS)

        # Update previous step cache
        self._prev_state  = self._fdm.state.copy()
        self._prev_action = action.copy()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass  # rendering not implemented; use analysis/plot_trajectories.py

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------

    def set_stage(self, stage_cfg: dict) -> None:
        """Apply a curriculum stage config dict.

        Called by CurriculumCallback whenever the scheduler advances.
        Merges stage_cfg keys into self.cfg; the new values take effect
        at the next reset().

        Args:
            stage_cfg : dict from CurriculumScheduler.current_cfg
        """
        self.cfg.update(stage_cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_domain_rand(self) -> dict:
        """Sample per-episode domain-randomisation parameters for JSBSim.

        Returns a dict suitable for **kwargs to JSBSimFDM.apply_domain_rand().
        Ranges match the old GliderParams randomisation ranges for continuity.
        """
        aero_lo, aero_hi = self.cfg.get('aero_scale_range', (1.0, 1.0))
        cd0_scale    = float(self._rng.uniform(aero_lo, aero_hi))
        cl_mult      = float(self._rng.uniform(0.9, 1.1))
        ctrl_eff_mult = float(self._rng.uniform(0.75, 1.10))

        # Convert multiplicative CD0 factor to additive offset (CD0_nominal = 0.025)
        cd0_add = 0.025 * (cd0_scale - 1.0)

        return dict(cl_mult=cl_mult, cd0_add=cd0_add, ctrl_eff_mult=ctrl_eff_mult)

    def _build_launch_state(self) -> npt.NDArray:
        """Construct the 13-element launch state for a new episode.

        alt0_m defaults to the as-built hardware reality: the glider is
        HAND/GROUND-LAUNCHED, not released from a tow at altitude. Realistic
        peak altitude after the launch zoom-climb is ~15-25 m -- see the
        as-built electronics notes. The episode begins at that post-launch
        peak, not at ground level (the ~2-3 s zoom-climb transient itself is
        out of scope for the RTL task).
        """
        alt0 = float(self.cfg.get('alt0_m', 22.0))

        # Launch speed/pitch are randomised per episode across the plausible
        # range of real apex-hand-off states (see LAUNCH_SPEED_MIN_MS etc.
        # above) -- not just a small jitter around one nominal release state.
        speed_lo = float(self.cfg.get('launch_speed_min_ms', LAUNCH_SPEED_MIN_MS))
        speed_hi = float(self.cfg.get('launch_speed_max_ms', LAUNCH_SPEED_MAX_MS))
        pitch_lo = float(self.cfg.get('launch_pitch_min_deg', LAUNCH_PITCH_MIN_DEG))
        pitch_hi = float(self.cfg.get('launch_pitch_max_deg', LAUNCH_PITCH_MAX_DEG))

        V0    = self._rng.uniform(speed_lo, speed_hi)
        gamma = np.radians(self._rng.uniform(pitch_lo, pitch_hi))
        psi   = self._rng.uniform(0.0, 2.0 * np.pi)   # random heading

        # Random horizontal offset ensures dist_home > R_home_m at step 0,
        # making the navigation task well-posed from the first observation.
        # Without an offset the glider starts directly above home (dist_home=0),
        # flies in a random direction, and the 100-second step budget is often
        # exhausted before it can return.
        # Defaults scaled down from the old 80-150 m (which assumed a
        # 100-120 m tow-release) to fit inside the glide range available
        # from a ~20-25 m ground-launch peak (L/D ~= 12-14 per validate_glide.py).
        offset_lo = float(self.cfg.get('launch_offset_min_m', 40.0))
        offset_hi = float(self.cfg.get('launch_offset_max_m', 70.0))
        offset    = self._rng.uniform(offset_lo, offset_hi)
        bearing   = self._rng.uniform(0.0, 2.0 * np.pi)
        p_north   = offset * np.cos(bearing)
        p_east    = offset * np.sin(bearing)

        # Body-frame velocity: v_body = [V, 0, 0] gives zero AoA (velocity
        # aligned with nose). The glider naturally trims to alpha_trim within
        # a few seconds. Using NED components here would give AoA = -15° and
        # a violent pitch-up divergence.
        return build_state(
            p_ned  = np.array([p_north, p_east, -alt0]),
            v_body = np.array([V0, 0.0, 0.0]),
            roll   = 0.0,
            pitch  = gamma,    # positive pitch = nose up
            yaw    = psi,
            omega  = np.zeros(3),
        )

    def _build_obs(self) -> npt.NDArray:
        """Build the 11-element float32 observation from sensor readings.

        Index contract (CLAUDE.md Section 10.2):
            [0]  dx_home        m        (GPS North - home North)
            [1]  dy_home        m        (GPS East  - home East)
            [2]  ground_speed   m/s
            [3]  course_angle   rad      atan2(ve, vn)
            [4]  roll           rad      (IMU)
            [5]  pitch          rad      (IMU)
            [6]  yaw            rad      (IMU)
            [7]  baro_alt       m
            [8]  vertical_speed m/s      (GPS vd, positive down -> negated for obs)
            [9]  lidar_agl      m         (ultrasonic; 0 unless <4.5 m, flare-only)
            [10] lidar_valid    binary    (ultrasonic)

        No airspeed channel: the hardware has no pitot (see as-built
        electronics notes). An earlier revision fed obs[11] from JSBSim's
        ground-truth wind-relative airspeed -- a sim-to-real leak, since no
        real sensor on this airframe can measure that. obs[2] (GPS ground
        speed) is the closest real signal and is kept as-is.
        """
        gps_pos = self._sensors.gps_pos   # [pn, pe, pd]
        gps_vel = self._sensors.gps_vel   # [vn, ve, vd]
        euler   = self._sensors.imu_euler  # [roll, pitch, yaw]

        dx_home = float(gps_pos[0] - self._home_ned[0])
        dy_home = float(gps_pos[1] - self._home_ned[1])

        vn, ve, vd = float(gps_vel[0]), float(gps_vel[1]), float(gps_vel[2])
        ground_speed  = float(np.sqrt(vn**2 + ve**2))
        course_angle  = float(np.arctan2(ve, vn))
        vertical_speed = -vd   # positive = climbing (negate NED down component)

        obs = np.array([
            dx_home,
            dy_home,
            ground_speed,
            course_angle,
            float(euler[0]),              # roll
            float(euler[1]),              # pitch
            float(euler[2]),              # yaw
            float(self._sensors.baro_alt),
            vertical_speed,
            float(self._sensors.lidar_agl),
            float(self._sensors.lidar_valid),
        ], dtype=np.float32)

        return obs

    def _safety_shield(
        self,
        state:     npt.NDArray,
        cmds:      dict,
    ) -> dict:
        """Clip servo commands to enforce hard safety limits.

        Hard bank limit: if |roll| > 50°, zero the aileron (flatten).
        Landing flare:   if ultrasonic valid and AGL < 2.5 m, force elevator up.
                         Only ever active in the last couple of seconds before
                         touchdown -- the ultrasonic is invalid above 4.5 m.
        Stall margin:    if alpha > 10°, prevent further pitch-up.

        The shield operates on a copy so the original dict is not mutated.
        """
        cmds = dict(cmds)   # shallow copy

        roll, _, _ = euler_from_quat(state[6:10])
        alpha = float(np.arctan2(state[5], state[3]))

        # Hard bank limit: actively level wings rather than just zeroing aileron,
        # so angular momentum doesn't carry the glider past inverted.
        p_rate = float(state[10])
        if abs(roll) > SHIELD_MAX_BANK_RAD:
            recovery_cmd = KP_ROLL * (0.0 - roll) - KD_ROLL * p_rate
            cmds['aileron'] = float(np.clip(recovery_cmd, -np.radians(25), np.radians(25)))

        # Landing flare: force nose-UP (negative elevator, since Cm_de=-1.2
        # means positive elevator=nose-down) to arrest sink rate just before
        # touchdown.
        if self._sensors.lidar_valid and self._sensors.lidar_agl < SHIELD_MIN_AGL_M:
            cmds['elevator'] = min(
                float(cmds['elevator']), -SHIELD_PULLUP_ELEV_RAD
            )

        # Stall margin: block nose-UP elevator (negative) to stop alpha growing.
        # Positive elevator is nose-down (recovery direction) so allow it through.
        if alpha > SHIELD_MAX_ALPHA_RAD:
            cmds['elevator'] = max(float(cmds['elevator']), 0.0)

        return cmds

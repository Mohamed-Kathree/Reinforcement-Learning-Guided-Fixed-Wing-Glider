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
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces

from sim.jsbsim_fdm import JSBSimFDM
from sim.wind import WindModel
from sim.sensor_models import SensorSuite
from sim.math_utils import build_state, wrap_pi
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
# Pitch offset per m/s above/below trim speed. Rescaled down from the old
# 0.05 rad/(m/s) (~2.9 deg/(m/s)), which -- combined with the pre-offset
# clip bug (see AttitudeController.update()) -- pushed the sum tens of
# degrees outside the sane pitch-attitude band across the +/-4 m/s command
# range. At this scale the full command range maps to roughly a 10 deg
# spread, verified monotonic and distinct across all 5 held-action values
# with scratch/audit3.py's sweep.
KV_PITCH: float = 0.015

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
# Stall-margin proxy: a nose-up PITCH ATTITUDE ceiling, not a true angle-of-
# attack limit. The physical ESP32 has no AoA vane/pitot (see as-built
# electronics notes) so the shield -- which mirrors ESP32 firmware -- must
# only depend on quantities the IMU can actually supply (pitch + roll).
# True alpha is used for the REWARD's stall penalty (env/reward.py, via
# JSBSimFDM.alpha) since rewards are privileged information; this shield
# deliberately does not switch to it. Set above the AttitudeController's
# normal commanded pitch band so it only fires on genuine upsets (gusts,
# actuator saturation, launch transients), not routine commanded flight.
SHIELD_MAX_PITCH_RAD:   float = np.radians(20.0)

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

# CG placement uncertainty (m), FRD convention (+ = forward of nominal CG).
# Like launch speed/pitch, this is a fixed hardware build-tolerance
# uncertainty, not a per-stage difficulty knob: every curriculum stage
# samples within +/- this range regardless of mass_range (which sets the
# TOTAL mass sampled via the ballast point mass; see _sample_domain_rand()).
CG_OFFSET_RANGE_M: float = 0.01

# Transport latency (sim-to-real gap: zero latency here vs. ~60-150 ms on
# the real GPS-fix-age + Pi<->ESP32 link + servo command path in real life,
# i.e. 1-3 whole policy steps). Sampled once per episode at reset() -- a
# fixed delay for the whole episode, not resampled every step, since real
# transport latency doesn't change step-to-step.
OBS_DELAY_MAX_STEPS:        int = 4   # 0-4 policy steps (0-200 ms @ 20 Hz)
ACTION_DELAY_MIN_SUBSTEPS:  int = 1
ACTION_DELAY_MAX_SUBSTEPS:  int = 3   # 1-3 physics substeps (5-15 ms @ 200 Hz)


# ---------------------------------------------------------------------------
# Inner attitude controller
# ---------------------------------------------------------------------------

class AttitudeController:
    """Simulated ESP32 inner-loop controller.

    Converts bank and speed setpoints into servo deflection commands.
    Runs at every 200 Hz physics substep inside env.step(), reading attitude
    and body rates from the noisy, biased 200 Hz IMU channel (SensorSuite)
    rather than ground-truth physics state -- this is what makes the
    "mirrors ESP32 firmware" claim actually true: the real firmware has no
    way to read exact roll/pitch/rates either. V_actual remains a ground-
    truth quantity (there is no pitot on this airframe -- see
    env/glider_env.py::GliderEnv._build_obs's "No airspeed channel" note);
    closing that gap requires a GPS-groundspeed-based estimate, which is
    Phase 4+ observation-redesign scope, not this fix.
    """

    def update(
        self,
        imu_euler:    npt.NDArray,
        imu_omega:    npt.NDArray,
        V_actual:     float,
        bank_cmd_rad: float,
        speed_cmd_ms: float,
    ) -> dict:
        """Compute aileron/elevator/rudder commands.

        Args:
            imu_euler    : noisy, biased [roll, pitch, yaw] (rad) from SensorSuite.imu_euler
            imu_omega    : noisy, biased [p, q, r] (rad/s) from SensorSuite.imu_omega
            V_actual     : true airspeed (m/s) -- ground truth, see class docstring
            bank_cmd_rad : desired roll angle (rad), ±π/2
            speed_cmd_ms : desired airspeed (m/s)

        Returns:
            dict with keys 'aileron', 'elevator', 'rudder' (rad)
        """
        roll, pitch, _ = float(imu_euler[0]), float(imu_euler[1]), float(imu_euler[2])
        p_rate, q_rate, r_rate = float(imu_omega[0]), float(imu_omega[1]), float(imu_omega[2])

        # Roll PD — wrap_pi ensures shortest-path recovery at any bank angle
        roll_err     = wrap_pi(bank_cmd_rad - roll)
        aileron_cmd  = KP_ROLL * roll_err - KD_ROLL * p_rate

        # Pitch: speed-scheduled pitch-angle controller.
        # Base target pitch = ALPHA_TRIM * (V_TRIM/V_cmd)^2 / cos(roll) -- the
        # trim attitude for the COMMANDED speed, not the current one.
        # Rationale: CL for level flight scales as 1/V^2, and CL ≈ a0*alpha ≈
        # a0*pitch at small angles in steady glide.  Dividing by cos(roll)
        # restores the reduced vertical lift component in banked turns.
        # Controlling PITCH (not alpha) prevents the pitch-spiral that an
        # alpha-only controller causes due to the Cm0>0 nose-down tendency.
        # Negate sign: Cm_de=-1.2 means positive elevator=nose-down, so a
        # positive pitch_err (need nose-down) drives a negative elevator cmd.
        #
        # NOTE on an earlier bug: a previous version keyed this base term to
        # the CURRENT airspeed (not the command) and then added an unclipped
        # speed-command offset on top of an already-clipped base -- two
        # competing feedback paths (one pulling attitude toward whatever
        # holds V_actual near V_TRIM regardless of command, the other an
        # oversized command bias) that collapsed all 5 points of
        # scratch/audit3.py's speed_cmd sweep to nearly the same
        # elevator-saturated, near-stall flight condition instead of 5
        # distinct, monotonically-ordered trim speeds. Keying the schedule
        # directly to speed_cmd_ms removes the competing path: the target IS
        # the trim attitude for the desired speed, and KV_PITCH now supplies
        # only a small closed-loop correction (from the gap between actual
        # and commanded speed) to keep tracking sane once wind/domain-rand
        # perturb the true trim point away from the nominal polar.
        cos_roll     = float(np.cos(roll))
        cos_factor   = max(abs(cos_roll), 0.5)
        V_cmd_clamp  = max(float(speed_cmd_ms), 4.0)
        pitch_base   = (ALPHA_TRIM * (V_TRIM / V_cmd_clamp) ** 2) / cos_factor
        pitch_corr   = -KV_PITCH * (speed_cmd_ms - V_actual)
        pitch_target = float(np.clip(pitch_base + pitch_corr,
                                      np.radians(-8.0), np.radians(12.0)))
        pitch_err    = pitch_target - pitch
        # Use Euler pitch rate θ_dot = q·cos(φ) − r·sin(φ) as the derivative
        # term. Body pitch rate q alone is blind to the turn-coupling disturbance
        # −r·sin(φ) that drives pitch DOWN in banked turns (ZYX kinematics).
        theta_dot_est = q_rate * np.cos(roll) - r_rate * np.sin(roll)
        elevator_cmd = -(KP_PITCH * pitch_err - KD_PITCH * theta_dot_est)

        # Coordinated turn: rudder opposes roll rate.
        # rlglider.xml's Cn_dr = -0.05 (positive rudder -> nose LEFT, verified
        # with scratch/audit1.py TEST 3). A right roll (p>0) needs a nose-
        # RIGHT yaw moment to coordinate, i.e. NEGATIVE rudder. The old
        # +KR_RUDDER*p_rate commanded positive rudder in a right roll --
        # nose-left, adverse yaw, the opposite of coordination.
        rudder_cmd = -KR_RUDDER * p_rate

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

        # Observation/action transport-delay state (sampled per episode in
        # reset(); see OBS_DELAY_MAX_STEPS / ACTION_DELAY_*_SUBSTEPS above).
        self._obs_delay_steps:       int = 0
        self._action_delay_substeps: int = 1
        self._obs_buffer:    deque = deque()
        self._action_buffer: deque = deque()

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
        # Modest vertical component (thermal lift / mechanical sink), scaled
        # by this episode's horizontal wind speed so it's zero in Stage 0's
        # "no wind" condition and grows with the same curriculum knob rather
        # than being a separate one -- an unpowered aircraft operating
        # entirely within 0-25 m AGL sees vertical motion on this order
        # routinely once there's any wind at all (mechanical turbulence off
        # terrain/obstacles, or thermal activity). NED convention: positive
        # wd = downdraft.
        vertical_wind_ms = float(self._rng.uniform(-0.3, 0.3)) * wind_speed
        mean_ned   = np.array([
            wind_speed * np.cos(wind_dir),
            wind_speed * np.sin(wind_dir),
            vertical_wind_ms,
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

        # Sample this episode's fixed transport delays (held constant for the
        # whole episode, not resampled every step -- real transport latency
        # doesn't change step-to-step). Pre-fill each buffer with the initial
        # value so early steps see a defined (if stale) reading rather than
        # an undefined default.
        self._obs_delay_steps = int(self._rng.integers(0, OBS_DELAY_MAX_STEPS + 1))
        self._action_delay_substeps = int(self._rng.integers(
            ACTION_DELAY_MIN_SUBSTEPS, ACTION_DELAY_MAX_SUBSTEPS + 1
        ))

        obs = self._build_obs()
        self._last_obs = obs.copy()
        self._obs_buffer = deque([obs.copy() for _ in range(self._obs_delay_steps)])
        self._action_buffer = deque(
            [(0.0, SPEED_CMD_CENTRE_MS)] * self._action_delay_substeps
        )

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
            # 1. Advance wind gust; log-profile-scale the horizontal mean
            # component by current AGL altitude (see WindModel.step()).
            wind_ned = self._wind.step(self._fdm.altitude)

            # 1b. Action transport delay: push this step's command onto the
            # FIFO and pop the command from action_delay_substeps ago -- a
            # fixed-length delay line modelling the real command-path
            # latency (Pi -> ESP32 link, servo transport lag) that a
            # zero-latency sim would otherwise not have at all. The buffer
            # persists across policy steps (not reset each call), so a delay
            # spanning a step boundary correctly carries commands over from
            # the previous step, same as a real transport delay would.
            self._action_buffer.append((bank_cmd_rad, speed_cmd_ms))
            delayed_bank_cmd_rad, delayed_speed_cmd_ms = self._action_buffer.popleft()

            # 2. Inner controller: bank + speed setpoints -> servo commands.
            # Reads attitude/rates from the 200 Hz IMU channel (noisy +
            # per-episode biased -- see SensorSuite), not ground truth, so
            # the "mirrors ESP32 firmware" claim is actually true. V_actual
            # is the one exception (no pitot on this airframe -- see
            # AttitudeController's class docstring).
            imu_euler = self._sensors.imu_euler
            imu_omega = self._sensors.imu_omega
            V_actual  = float(np.linalg.norm(self._fdm.state[3:6]))
            ctrl_cmds = self._ctrl.update(
                imu_euler, imu_omega, V_actual,
                delayed_bank_cmd_rad, delayed_speed_cmd_ms,
            )

            # 3. Safety shield: clip dangerous commands (also IMU-driven)
            ctrl_cmds = self._safety_shield(imu_euler, imu_omega, ctrl_cmds)

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
        # Observation transport delay: same FIFO pattern as the action delay
        # above, but at policy-step granularity (0-4 steps @ 20 Hz). With
        # obs_delay_steps == 0 the buffer starts empty and this is a no-op
        # (append then immediately popleft the same array).
        fresh_obs = self._build_obs()
        self._obs_buffer.append(fresh_obs.copy())
        obs = self._obs_buffer.popleft()
        self._last_obs = obs.copy()

        # --- Reward + termination ----------------------------------------
        reward, terminated, truncated_r, info = compute_reward(
            state         = self._fdm.state,
            prev_state    = prev_state,
            obs           = obs,
            action        = action,
            prev_action   = self._prev_action,
            home_ned      = self._home_ned,
            cfg           = self.cfg,
            alpha_true    = self._fdm.alpha,
            airspeed_true = self._fdm.airspeed,
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

        # Mass/CG: previously dead config -- STAGES declared mass_range per
        # stage but nothing read it, so mass and CG were fixed at 1.1 kg /
        # nominal for every episode despite being the two largest build-to-
        # build uncertainties on a hand-built airframe. Wired to the BALLAST
        # point mass in rlglider.xml via JSBSimFDM.apply_domain_rand().
        mass_lo, mass_hi = self.cfg.get('mass_range', (1.1, 1.1))
        mass_kg     = float(self._rng.uniform(mass_lo, mass_hi))
        cg_offset_m = float(self._rng.uniform(-CG_OFFSET_RANGE_M, CG_OFFSET_RANGE_M))

        return dict(cl_mult=cl_mult, cd0_add=cd0_add, ctrl_eff_mult=ctrl_eff_mult,
                    mass_kg=mass_kg, cg_offset_m=cg_offset_m)

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
        imu_euler: npt.NDArray,
        imu_omega: npt.NDArray,
        cmds:      dict,
    ) -> dict:
        """Clip servo commands to enforce hard safety limits.

        Hard bank limit: if |roll| > 50°, zero the aileron (flatten).
        Landing flare:   if ultrasonic valid and AGL < 2.5 m, force elevator up.
                         Only ever active in the last couple of seconds before
                         touchdown -- the ultrasonic is invalid above 4.5 m.
        Stall margin:    if pitch attitude exceeds a bank-derated ceiling,
                         prevent further pitch-up. Uses PITCH (IMU-measurable),
                         not true angle-of-attack -- the physical ESP32 has no
                         AoA vane, so this shield must mirror what firmware can
                         actually see (see SHIELD_MAX_PITCH_RAD's docstring).

        Reads attitude/rates from the noisy, biased 200 Hz IMU channel
        (SensorSuite), not ground truth -- same rationale as
        AttitudeController.update().

        The shield operates on a copy so the original dict is not mutated.
        """
        cmds = dict(cmds)   # shallow copy

        roll, pitch = float(imu_euler[0]), float(imu_euler[1])

        # Hard bank limit: actively level wings rather than just zeroing aileron,
        # so angular momentum doesn't carry the glider past inverted.
        p_rate = float(imu_omega[0])
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

        # Stall margin: block nose-UP elevator (negative) once pitch attitude
        # exceeds a ceiling that derates with bank (a banked turn needs more
        # pitch for the same alpha at a given speed -- same reasoning as the
        # nominal pitch schedule in AttitudeController.update()).
        # Positive elevator is nose-down (recovery direction) so allow it through.
        pitch_ceiling = SHIELD_MAX_PITCH_RAD / max(abs(np.cos(roll)), 0.5)
        if pitch > pitch_ceiling:
            cmds['elevator'] = max(float(cmds['elevator']), 0.0)

        return cmds

"""
reward.py
=========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Shaped reward function for the Phase 4 precision-landing task (see
RLGlider_Phase4_Landing_Task_Spec.md; supersedes the earlier "cross R_home_m
and stop" RTL task). The glider must actually LAND at the launch point, as
close to centre as possible, while flying the longest route it can.

Structure -- lexicographic (land safely >> land centred >> fly long), because
a plain weighted sum lets the optimiser trade a hard landing for a longer
track, which is exactly the failure mode to avoid:

    Terminal (on touchdown/fault only):
        r_terminal = quality * (r_precision + r_bullseye) - (1-quality) * w_crash
        where quality in [0,1] is the PRODUCT of four graded landing-quality
        factors (sink rate, roll, airspeed, alpha at touchdown) -- the
        multiplicative form means precision credit is always scaled by how
        clean the touchdown was, so 2 m closer at the cost of a worse
        touchdown can never win.
    Dense (every step):
        r_path     : reward per metre of ACTUAL ground track flown (replaces
                     the old airtime bonus, which rewarded loitering in place
                     rather than route length)
        w_unreach  : one-sided barrier, zero during normal flight, penalising
                     only once the glider has flown itself out of glide range
                     of home (replaces the old progress-toward-home term,
                     which directly opposed "take the longest route"). Scaled
                     by dt_rl like r_path (Phase 6) so the two dense terms
                     share units -- do not reintroduce an unscaled per-step
                     penalty here, it silently becomes 20x per second at 20 Hz.
        w_stall / w_bank / w_smooth : unchanged safety/regularisation terms

Deleted vs. the pre-Phase-4 version: w_progress (opposed the new objective),
w_airtime (rewarded loitering, double-counted against r_path), w_agl /
agl_min (penalised being below the flare altitude, which under the new task
is a MANDATORY part of every successful landing -- the open-loop shield that
used to enforce this was also deleted, see env/glider_env.py::_safety_shield).

Termination: `dist_home < R_home_m` no longer ends the episode -- R_home_m is
now purely a precision-scoring parameter (curriculum-scaled via
`sigma_centre_m`). The episode ends only on ground contact (`touched_down`,
passed in from JSBSimFDM.touched_down via GliderEnv.step()) or a JSBSim
solver/state fault (`fault`, from JSBSimFDM.fault) -- or the legacy
`state[2] > 0.0` raw-position check as a backstop in case the contact model
ever misses the WOW transition.

Observation vector index contract (must match env/glider_env.py):
    obs[4]  : roll           (rad)

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.math_utils import quat_to_rotmat, euler_from_quat


# ---------------------------------------------------------------------------
# Default reward weights (mirrors training/configs/base.yaml reward section)
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CFG: dict = {
    # Terminal landing grade
    'w_land':          500.0,   # centre-precision bonus, scaled by landing quality
    'w_bullseye':      150.0,   # narrow secondary precision bonus (also quality-scaled)
    'w_crash':         200.0,   # penalty on a bad landing / fault, scaled by (1 - quality)
    'sigma_centre_m':   10.0,   # half-value distance for the hyperbolic precision term
                                # (r_precision = w_land / (1 + d/sigma)); curriculum-overridden

    # Landing-quality grading bands: OK = full credit, BAD = zero credit,
    # linear ramp between (see _grade()). Fixed across the curriculum on
    # purpose -- only sigma_centre_m/sink_bad/roll_bad_deg widen at easy
    # stages (env/curriculum.py STAGES), so the SHAPE of "clean landing"
    # never changes, only how strictly distance-from-centre is scored.
    'sink_ok':           0.8,   # m/s; measured clean landings are 0.43-0.78 m/s
    'sink_bad':          2.5,   # m/s; curriculum-overridden (widens at easy stages)
    'roll_ok_deg':      10.0,   # deg; wingtip geometry allows ~6 deg before contact
    'roll_bad_deg':     35.0,   # deg; curriculum-overridden
    'vtd_ok':            9.0,   # m/s; ~1.4x V_stall (6.35 m/s)
    'vtd_bad':          13.0,   # m/s; ~2.0x V_stall
    'alpha_ok_deg':     10.0,   # deg; brackets the 12 deg stall angle
    'alpha_bad_deg':    14.0,   # deg

    # Dense (every step, terminal or not)
    'w_path':            0.10,  # reward per metre of ACTUAL ground track flown
    'w_unreach':         2.0,   # one-sided reachability-barrier penalty
    'glide_ratio_usable': 8.0,  # conservative vs. measured best L/D ~16.7 (12.9 trimmed)
    'barrier_min_agl_m': 10.0,  # AGL below which the reachability barrier is inert.
                                # Above the ultrasonic's 4.5 m validity gate and the
                                # baseline's 3 m flare, below normal cruise. Measured
                                # pre-fix barrier firing peaked at 6.06 m AGL, so this
                                # makes it inert through the entire landing phase.
    'w_unreach_cap':    20.0,   # ceiling on the barrier penalty PER SECOND (applied
                                # before the dt_rl scaling below, so this reads as a
                                # rate, not a per-step or per-episode bound). Bounds
                                # each second's contribution to ~2.5% of a mid-range
                                # terminal reward, keeping the barrier subordinate.
    'dt_rl':             0.05,  # policy step size (s); multiplied by w_path

    # Safety (unchanged from the pre-Phase-4 reward)
    'w_stall':          30.0,   # stall margin violation: linear penalty
    'w_bank':            5.0,   # excess bank: linear penalty above soft limit

    # Smoothness
    'w_smooth':          0.01,  # L2 penalty on Δaction (prevents chattering)

    # Truncation backstop (applied by GliderEnv.step(), not here -- only the
    # env knows about MAX_STEPS) -- listed here so training/configs/base.yaml
    # has one place to tune it, and GliderEnv reads cfg['w_truncate'].
    'w_truncate':      300.0,

    # Thresholds
    'R_home_m':               20.0,             # precision-scoring radius (m); no longer terminal
    'stall_buffer_rad':      np.radians(2.0),   # penalty starts 2 deg before stall
    'bank_soft_limit_rad':   np.radians(30.0),  # penalty starts at 30 deg bank
}

# Stall angle from GliderParams (hard-coded here to avoid a circular import)
_ALPHA_STALL: float = np.radians(12.0)


# ---------------------------------------------------------------------------
# Grading helper
# ---------------------------------------------------------------------------

def _grade(x: float, ok: float, bad: float) -> float:
    """Linear landing-quality ramp: 1.0 at x<=ok, 0.0 at x>=bad, clipped."""
    return float(np.clip((bad - x) / (bad - ok), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public reward function
# ---------------------------------------------------------------------------

def compute_reward(
    state:      NDArray,
    prev_state: NDArray,
    obs:        NDArray,
    action:     NDArray,
    prev_action: NDArray,
    home_ned:   NDArray,
    cfg:        dict,
    alpha_true:    float,
    airspeed_true: float,
    touched_down: bool = False,
    fault:        bool = False,
    sensors_valid: bool = True,
) -> tuple[float, bool, bool, dict]:
    """Compute the shaped reward for one policy step.

    Args:
        state       : current 13-element physics state
        prev_state  : physics state at the previous policy step
        obs         : current 12-element normalised observation vector
                      (obs[4]=roll)
        action      : current normalised action [bank_cmd, speed_cmd] ∈ [-1,1]²
        prev_action : action from the previous step (for smoothness penalty)
        home_ned    : NED position of the home/launch point, shape (3,) or (2,)
        cfg         : reward weight dict (see DEFAULT_REWARD_CFG for keys)
        alpha_true  : true wind-relative angle of attack (rad), from
                      JSBSimFDM.alpha -- ground truth, safety-critical.
        airspeed_true : true wind-relative airspeed (m/s), from
                      JSBSimFDM.airspeed.
        touched_down: True the step the belly-skid contact model first
                      engages (JSBSimFDM.touched_down, read by GliderEnv
                      right after the substep loop exits -- see its
                      docstring for why this fires before state[2]>0).
        fault       : True on a JSBSim solver failure or non-finite state
                      (JSBSimFDM.fault). Forces the worst-case grade
                      regardless of the (possibly corrupted) state, since a
                      fault means the physics can no longer be trusted.
        sensors_valid: overall sensor health flag; reserved for future use.

    Returns:
        r           : scalar reward for this step
        terminated  : True when the episode ends (touchdown or fault)
        truncated   : always False — step limits are handled by the env
        info        : dict with per-component contributions, violation
                      flags, and (on termination) landing-quality diagnostics
                      for TensorBoard logging
    """
    # A fault means the physics state may be non-finite/corrupted -- grade
    # the worst case immediately and do not touch `state` for anything else.
    if fault:
        r = -float(cfg['w_crash'])
        info: dict = {
            'r_path': 0.0, 'penalty_unreach': 0.0,
            'penalty_stall': 0.0, 'penalty_bank': 0.0, 'penalty_smooth': 0.0,
            'stall_violation': False, 'bank_violation': False,
            'unreach_violation': False,
            'success': False, 'crash': True, 'outcome': 'crashed',
            'quality': 0.0, 'r_terminal': -float(cfg['w_crash']),
            'dist_home': float('nan'), 'vs_td': float('nan'),
            'roll_td_deg': float('nan'), 'V_td': float('nan'),
            'alpha_td_deg': float('nan'),
            'alpha_deg': float(np.degrees(alpha_true)),
            'roll_deg': float(np.degrees(float(obs[4]))),
            'airspeed': float(airspeed_true),
        }
        return float(r), True, False, info

    # --- Unpack positions -----------------------------------------------
    pos_ne      = state[0:2]          # North, East (ground truth)
    prev_pos_ne = prev_state[0:2]
    home_ne     = np.asarray(home_ned, dtype=np.float64).ravel()[:2]

    dist_home      = float(np.linalg.norm(pos_ne  - home_ne))
    prev_dist_home = float(np.linalg.norm(prev_pos_ne - home_ne))

    # --- Sensor-derived quantities (from corrupted observation) ---------
    roll = float(obs[4])

    # --- Ground-truth quantities (safety-critical; not sensor-limited) --
    V     = float(airspeed_true)
    alpha = float(alpha_true)

    vel_ned           = quat_to_rotmat(state[6:10]) @ state[3:6]
    ground_speed_true = float(np.linalg.norm(vel_ned[0:2]))

    # --- Reward accumulator --------------------------------------------
    r = 0.0

    # 1. Route-length reward: metres of ACTUAL ground track flown this step.
    #    Replaces the old airtime bonus (which rewarded loitering in place,
    #    not distance) -- see module docstring.
    r_path = float(cfg['w_path']) * ground_speed_true * float(cfg['dt_rl'])
    r += r_path

    # 2. Reachability barrier: a CRUISE-PHASE constraint, not a landing-accuracy
    #    penalty. r_path pays per metre of ground track, so without this the
    #    optimal policy is to fly downwind forever; this is the only thing
    #    bounding that. It deliberately does NOT encode "hurry home" (that was
    #    w_progress, removed in Phase 4 and not to be reintroduced).
    #
    #    Phase 5: gated on altitude. The dist/alt ratio makes the permitted
    #    radius shrink linearly as the glider descends (160 m at 20 m AGL,
    #    16 m at 2 m AGL), so below the flare the barrier degenerates into a
    #    landing-accuracy penalty on geometry the policy can no longer change.
    #    Measured pre-fix, it fired on 79-186 steps/episode, 72% of them below
    #    3 m AGL, contributing ~-3,900/episode against a +81 terminal reward --
    #    i.e. it WAS the reward function. Below barrier_min_agl_m it is now
    #    inert, and above it the per-step magnitude is capped.
    altitude_agl      = float(-state[2])
    barrier_min_agl   = float(cfg['barrier_min_agl_m'])
    glide_usable      = float(cfg['glide_ratio_usable'])
    penalty_unreach   = 0.0
    unreach_violation = False
    if altitude_agl >= barrier_min_agl:
        glide_needed = dist_home / max(altitude_agl, barrier_min_agl)
        if glide_needed > glide_usable:
            # Phase 6: scaled by dt_rl, matching r_path above. Without this the
            # two dense terms are in different units -- r_path is per METRE
            # flown, the barrier was per STEP, so at 20 Hz the barrier accrued
            # 20x per second and nothing bounded the episode total (ceiling
            # -40,000 at MAX_STEPS, against a terminal reward spanning
            # -200..+500). Measured pre-fix: a random policy at Stage 0 saw a
            # reward that was 98.7% barrier. The cap is applied BEFORE the dt
            # scaling, so w_unreach_cap is a ceiling per SECOND, not per step.
            penalty_unreach = min(
                float(cfg['w_unreach']) * (glide_needed - glide_usable),
                float(cfg['w_unreach_cap']),
            ) * float(cfg['dt_rl'])
            unreach_violation = True
            r -= penalty_unreach

    # 3. Stall margin penalty (ground truth alpha, safety-critical)
    penalty_stall = 0.0
    stall_violation = False
    stall_margin = _ALPHA_STALL - alpha
    if stall_margin < cfg['stall_buffer_rad']:
        penalty_stall   = cfg['w_stall'] * (cfg['stall_buffer_rad'] - stall_margin)
        stall_violation = True
        r -= penalty_stall

    # 4. Excess bank penalty (sensor roll from obs to match the Pi's view)
    penalty_bank = 0.0
    bank_violation = False
    if abs(roll) > cfg['bank_soft_limit_rad']:
        penalty_bank   = cfg['w_bank'] * (abs(roll) - cfg['bank_soft_limit_rad'])
        bank_violation = True
        r -= penalty_bank

    # 5. Smoothness regularisation — penalise large action changes
    action_arr      = np.asarray(action,      dtype=np.float64)
    prev_action_arr = np.asarray(prev_action, dtype=np.float64)
    penalty_smooth  = cfg['w_smooth'] * float(np.sum((action_arr - prev_action_arr) ** 2))
    r -= penalty_smooth

    # --- Terminal conditions --------------------------------------------
    # Ground contact (touched_down, or the legacy state[2]>0.0 raw-position
    # check as a backstop) ends the episode with a GRADED landing quality --
    # dist_home < R_home_m no longer terminates anything by itself.
    terminated = False
    success    = False
    crash      = False
    quality    = 0.0
    outcome    = None
    r_terminal = 0.0
    vs_td      = 0.0
    roll_td_deg  = 0.0
    V_td       = V
    alpha_td_deg = float(np.degrees(alpha))

    if touched_down or state[2] > 0.0:
        vs_td = float(vel_ned[2])   # NED down velocity; + = descending
        roll_td, _, _ = euler_from_quat(state[6:10])
        roll_td_deg   = float(np.degrees(roll_td))
        V_td          = V
        alpha_td_deg  = float(np.degrees(alpha))

        q_sink  = _grade(abs(vs_td),      float(cfg['sink_ok']),      float(cfg['sink_bad']))
        q_roll  = _grade(abs(roll_td_deg),float(cfg['roll_ok_deg']),  float(cfg['roll_bad_deg']))
        q_speed = _grade(V_td,            float(cfg['vtd_ok']),      float(cfg['vtd_bad']))
        q_alpha = _grade(alpha_td_deg,    float(cfg['alpha_ok_deg']), float(cfg['alpha_bad_deg']))
        quality = q_sink * q_roll * q_speed * q_alpha

        sigma_centre = float(cfg['sigma_centre_m'])
        # Phase 5: hyperbolic, not Gaussian. The Gaussian's tail vanished long
        # before the distances this airframe actually achieves (measured mean
        # touchdown 32.7 m at Stage 0, 155.8 m at Stage 3; the Gaussian paid
        # 0.97 and ~0 points respectively out of 500), leaving no gradient to
        # learn from. The hyperbolic form is monotone with a non-vanishing
        # derivative at every distance: dR/dd = -w_land*sigma/(sigma+d)^2.
        # dist_home is non-negative by construction (a norm), so no guard
        # against a negative denominator is needed.
        r_precision  = float(cfg['w_land']) * sigma_centre / (sigma_centre + dist_home)
        # r_bullseye stays a NARROW Gaussian -- fine-precision incentive once
        # already close -- and stays quality-gated (not just r_precision) --
        # the spec's formula only shows r_precision scaled, but leaving a
        # narrow bonus unscaled would let a stalled-but-centred landing beat
        # a clean off-centre one, breaking the lexicographic guarantee the
        # whole multiplicative design exists to provide.
        r_bullseye   = float(cfg['w_bullseye']) * float(np.exp(-(dist_home / 2.0) ** 2))
        r_terminal   = quality * (r_precision + r_bullseye) - (1.0 - quality) * float(cfg['w_crash'])
        r += r_terminal

        terminated = True
        crash      = quality <= 0.0
        success    = (quality > 0.5) and (dist_home < float(cfg['R_home_m']))
        outcome    = 'landed' if quality > 0.0 else 'crashed'

    truncated = False   # step-limit truncation is handled by the env

    # --- Info dict for TensorBoard / episode statistics -----------------
    info = {
        # Per-component reward contributions (signed)
        'r_path':          r_path,
        'penalty_unreach': -penalty_unreach,
        'penalty_stall':   -penalty_stall,
        'penalty_bank':    -penalty_bank,
        'penalty_smooth':  -penalty_smooth,
        'r_terminal':       r_terminal,
        # Violation flags (for counting events per episode)
        'unreach_violation': unreach_violation,
        'stall_violation':   stall_violation,
        'bank_violation':    bank_violation,
        # Terminal outcome
        'success': success,
        'crash':   crash,
        'outcome': outcome,
        'quality': quality,
        # Diagnostic scalars
        'dist_home':    dist_home,
        'vs_td':        vs_td,
        'roll_td_deg':  roll_td_deg,
        'V_td':         V_td,
        'alpha_td_deg': alpha_td_deg,
        'alpha_deg':    float(np.degrees(alpha)),
        'roll_deg':     float(np.degrees(roll)),
        'airspeed':     V,
    }

    return float(r), terminated, truncated, info

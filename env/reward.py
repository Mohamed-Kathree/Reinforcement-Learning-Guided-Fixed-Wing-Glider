"""
reward.py
=========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Shaped reward function for the RL-guided glider return-to-launch task.

Structure (applied every policy step at 20 Hz):
    Dense rewards  : progress toward home, airtime bonus
    Safety penalties: AGL too low (lidar-gated), near-stall alpha, excess bank
    Smoothness      : L2 penalty on action change
    Terminal events : large bonus on reaching home, penalty on ground impact

Returns a 4-tuple (r, terminated, truncated, info) where info contains
per-component breakdowns and violation flags for TensorBoard logging.

Default weights come from training/configs/base.yaml (reproduced in
DEFAULT_REWARD_CFG below). Override any key in the cfg dict passed to
compute_reward() to tune the shaping.

Observation vector index contract (must match env/glider_env.py):
    obs[4]  : roll           (rad)
    obs[9]  : lidar_agl      (m,   0 when invalid)
    obs[10] : lidar_valid    (0 or 1)

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
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Default reward weights (mirrors training/configs/base.yaml reward section)
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CFG: dict = {
    # Terminal
    'w_terminal':  500.0,   # bonus on reaching home radius (must dominate)
    'w_crash':     100.0,   # penalty on ground impact

    # Dense
    'w_progress':  1.0,     # reward per metre of progress toward home
    'w_airtime':   0.1,     # reward per second of sustained flight
    'dt_rl':       0.05,    # policy step size (s); multiplied by w_airtime

    # Safety
    'w_agl':       50.0,    # AGL violation: quadratic penalty below agl_min
    'w_stall':     30.0,    # stall margin violation: linear penalty
    'w_bank':      5.0,     # excess bank: linear penalty above soft limit

    # Smoothness
    'w_smooth':    0.01,    # L2 penalty on Δaction (prevents chattering)

    # Thresholds
    'R_home_m':              15.0,              # success radius (m)
    'agl_min':               8.0,               # minimum safe AGL (m)
    'stall_buffer_rad':      np.radians(2.0),   # penalty starts 2 deg before stall
    'bank_soft_limit_rad':   np.radians(30.0),  # penalty starts at 30 deg bank
}

# Stall angle from GliderParams (hard-coded here to avoid a circular import)
_ALPHA_STALL: float = np.radians(12.0)


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
    sensors_valid: bool = True,
) -> tuple[float, bool, bool, dict]:
    """Compute the shaped reward for one policy step.

    Args:
        state       : current 13-element physics state
        prev_state  : physics state at the previous policy step
        obs         : current 12-element normalised observation vector
                      (obs[4]=roll, obs[9]=lidar_agl, obs[10]=lidar_valid)
        action      : current normalised action [bank_cmd, speed_cmd] ∈ [-1,1]²
        prev_action : action from the previous step (for smoothness penalty)
        home_ned    : NED position of the home/launch point, shape (3,) or (2,)
        cfg         : reward weight dict (see DEFAULT_REWARD_CFG for keys)
        sensors_valid: overall sensor health flag; reserved for future use
                      (LiDAR validity is read from obs[10])

    Returns:
        r           : scalar reward for this step
        terminated  : True when the episode ends (success or crash)
        truncated   : always False — step limits are handled by the env
        info        : dict with per-component contributions and violation flags
                      for TensorBoard logging
    """
    # --- Unpack positions -----------------------------------------------
    pos_ne      = state[0:2]          # North, East (ground truth)
    prev_pos_ne = prev_state[0:2]
    home_ne     = np.asarray(home_ned, dtype=np.float64).ravel()[:2]

    dist_home      = float(np.linalg.norm(pos_ne  - home_ne))
    prev_dist_home = float(np.linalg.norm(prev_pos_ne - home_ne))

    # --- Sensor-derived quantities (from corrupted observation) ---------
    roll        = float(obs[4])
    lidar_agl   = float(obs[9])
    lidar_valid = bool(obs[10])

    # --- Ground-truth quantities (safety-critical; not sensor-limited) --
    V     = float(np.linalg.norm(state[3:6]))           # airspeed (m/s)
    alpha = float(np.arctan2(state[5], state[3]))        # angle of attack (rad)

    # --- Reward accumulator --------------------------------------------
    r = 0.0

    # 1. Progress toward home (positive when closing distance)
    r_progress = cfg['w_progress'] * (prev_dist_home - dist_home)
    r += r_progress

    # 2. Airtime bonus (encourages staying aloft)
    r_airtime = cfg['w_airtime'] * cfg['dt_rl']
    r += r_airtime

    # 3. AGL safety penalty — only when LiDAR reports a valid reading
    penalty_agl = 0.0
    agl_violation = False
    if lidar_valid and lidar_agl < cfg['agl_min']:
        penalty_agl   = cfg['w_agl'] * (cfg['agl_min'] - lidar_agl) ** 2
        agl_violation = True
        r -= penalty_agl

    # 4. Stall margin penalty (ground truth alpha, safety-critical)
    penalty_stall = 0.0
    stall_violation = False
    stall_margin = _ALPHA_STALL - alpha
    if stall_margin < cfg['stall_buffer_rad']:
        penalty_stall   = cfg['w_stall'] * (cfg['stall_buffer_rad'] - stall_margin)
        stall_violation = True
        r -= penalty_stall

    # 5. Excess bank penalty (sensor roll from obs to match the Pi's view)
    penalty_bank = 0.0
    bank_violation = False
    if abs(roll) > cfg['bank_soft_limit_rad']:
        penalty_bank   = cfg['w_bank'] * (abs(roll) - cfg['bank_soft_limit_rad'])
        bank_violation = True
        r -= penalty_bank

    # 6. Smoothness regularisation — penalise large action changes
    action_arr      = np.asarray(action,      dtype=np.float64)
    prev_action_arr = np.asarray(prev_action, dtype=np.float64)
    penalty_smooth  = cfg['w_smooth'] * float(np.sum((action_arr - prev_action_arr) ** 2))
    r -= penalty_smooth

    # --- Terminal conditions --------------------------------------------
    terminated = False
    success    = False
    crash      = False

    # Crash is checked first: ground impact takes priority over success so
    # that a glider that hits the ground inside R_home_m is scored as a crash,
    # not a successful return. state[2] = p_d; positive means below the NED
    # origin (ground level), i.e. crashed.
    if state[2] > 0.0:
        r         -= cfg['w_crash']
        terminated = True
        crash      = True
    elif dist_home < cfg['R_home_m']:
        r         += cfg['w_terminal']
        terminated = True
        success    = True

    truncated = False   # step-limit truncation is handled by the env

    # --- Info dict for TensorBoard / episode statistics -----------------
    info: dict = {
        # Per-component reward contributions (signed)
        'r_progress':     r_progress,
        'r_airtime':      r_airtime,
        'penalty_agl':    -penalty_agl,
        'penalty_stall':  -penalty_stall,
        'penalty_bank':   -penalty_bank,
        'penalty_smooth': -penalty_smooth,
        # Violation flags (for counting events per episode)
        'agl_violation':   agl_violation,
        'stall_violation': stall_violation,
        'bank_violation':  bank_violation,
        # Terminal outcome
        'success': success,
        'crash':   crash,
        # Diagnostic scalars
        'dist_home':  dist_home,
        'alpha_deg':  float(np.degrees(alpha)),
        'roll_deg':   float(np.degrees(roll)),
        'airspeed':   V,
        'lidar_agl':  lidar_agl,
    }

    return float(r), terminated, truncated, info

"""
Isolate AttitudeController behaviour with JSBSim.

Part 1: Static table — pitch_target vs speed and roll angle.
Part 2: Dynamic trace — log inner loop at every 200 Hz substep
        for one episode, showing roll/pitch errors, servo commands,
        and whether the controller is saturating.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.glider_env import (
    GliderEnv, AttitudeController,
    KP_ROLL, KD_ROLL, KP_PITCH, KD_PITCH, KV_PITCH,
    KR_RUDDER, V_TRIM, ALPHA_TRIM,
    BANK_CMD_SCALE_RAD, SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS,
)
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL
from sim.math_utils import euler_from_quat
from sim.jsbsim_fdm import JSBSimFDM


# ── Part 1: Static pitch schedule ────────────────────────────────────────────
print("=" * 60)
print("PART 1: Pitch target schedule (degrees)")
print(f"  V_TRIM={V_TRIM} m/s  ALPHA_TRIM={np.degrees(ALPHA_TRIM):.1f}deg")
print(f"  pitch_target = ALPHA_TRIM * (V_TRIM/V_clamp)^2 / max(|cos(roll)|, 0.5)")
print()
print(f"  {'Speed':>7}  {'roll=0°':>9}  {'roll=20°':>9}  {'roll=30°':>9}  {'roll=45°':>9}")
print(f"  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
for v in [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
    v_clamp = max(v, 6.0)
    row = f"  {v:7.1f}"
    for roll_deg in [0, 20, 30, 45]:
        roll_rad = np.radians(roll_deg)
        cos_roll = np.cos(roll_rad)
        pt = (ALPHA_TRIM * (V_TRIM / v_clamp)**2) / max(abs(cos_roll), 0.5)
        pt_deg = np.degrees(np.clip(pt, np.radians(1.0), np.radians(10.0)))
        # check if clipped
        raw = np.degrees(pt)
        clip_flag = " CLIP" if raw > 10.0 else ""
        row += f"  {pt_deg:7.2f}deg{clip_flag}"
    print(row)

print()
print("  Elevator authority: positive elevator_cmd = nose-down (sign convention)")
print(f"  elevator_cmd = -(KP_PITCH * pitch_err - KD_PITCH * theta_dot)")
print(f"  KP_PITCH={KP_PITCH}  KD_PITCH={KD_PITCH}  KV_PITCH={KV_PITCH}")
print(f"  Aileron authority: KP_ROLL={KP_ROLL}  KD_ROLL={KD_ROLL}")

# ── Part 2: Dynamic inner-loop trace ─────────────────────────────────────────
print()
print("=" * 60)
print("PART 2: Inner-loop trace at 200 Hz (first 500 substeps = 2.5 s)")
print()

cfg = dict(STAGES[0])
env = GliderEnv(cfg=cfg)
ctrl_rtl = DeterministicRTL()

# Use the same seed that gives a TIMEOUT
rng = np.random.default_rng(0)
seed = int(rng.integers(0, 2**31))
obs, _ = env.reset(seed=seed)

print(f"Launch: dx={obs[0]:.1f}m  dy={obs[1]:.1f}m  alt={obs[7]:.1f}m  airspeed={obs[11]:.1f}m/s")
print()

header = (f"{'sub':>5}  {'V':>6}  {'roll':>7}  {'roll_cmd':>8}  {'roll_err':>8}  "
          f"{'ail_raw':>8}  {'ail_clip':>8}  "
          f"{'pitch':>7}  {'p_tgt':>7}  {'p_err':>7}  "
          f"{'elv_raw':>8}  {'elv_clip':>8}  {'alpha_fdm':>9}")
print(header)
print("-" * len(header))

MAX_TRACE_SUBSTEPS = 500   # 2.5 s at 200 Hz
substep_count = 0
done = False

# We need to hook into the substep loop manually.
# Reproduce step() logic with extra logging.
action = ctrl_rtl.act(obs)
bank_cmd_rad  = float(action[0]) * BANK_CMD_SCALE_RAD
speed_cmd_ms  = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS

from env.glider_env import (
    DT_PHYS, N_SUBSTEPS, SHIELD_MAX_BANK_RAD, SHIELD_MIN_AGL_M,
    SHIELD_PULLUP_ELEV_RAD, SHIELD_MAX_ALPHA_RAD
)

fdm  = env._fdm
wind = env._wind
ctrl = env._ctrl
rng2 = env._rng

policy_step = 0
while substep_count < MAX_TRACE_SUBSTEPS and not done:
    # Get fresh action from RTL controller each policy step
    if substep_count % N_SUBSTEPS == 0 and substep_count > 0:
        obs = env._build_obs()
        action = ctrl_rtl.act(obs)
        bank_cmd_rad  = float(action[0]) * BANK_CMD_SCALE_RAD
        speed_cmd_ms  = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS
        policy_step += 1

    wind_ned = wind.step()
    state = fdm.state

    roll, pitch, _ = euler_from_quat(state[6:10])
    p_rate = float(state[10])
    q_rate = float(state[11])
    r_rate = float(state[12])

    V_actual = float(np.linalg.norm(state[3:6]))
    V_clamp  = max(V_actual, 6.0)
    cos_roll = float(np.cos(roll))
    pitch_target = (ALPHA_TRIM * (V_TRIM / V_clamp)**2) / max(abs(cos_roll), 0.5)
    pitch_target = float(np.clip(pitch_target, np.radians(1.0), np.radians(10.0)))
    pitch_target -= KV_PITCH * (speed_cmd_ms - V_TRIM)

    roll_err   = float(np.arctan2(np.sin(bank_cmd_rad - roll), np.cos(bank_cmd_rad - roll)))
    ail_raw    = KP_ROLL * roll_err - KD_ROLL * p_rate
    ail_clip   = float(np.clip(ail_raw, -np.radians(25), np.radians(25)))

    pitch_err     = pitch_target - pitch
    theta_dot_est = q_rate * np.cos(roll) - r_rate * np.sin(roll)
    elv_raw       = -(KP_PITCH * pitch_err - KD_PITCH * theta_dot_est)
    elv_clip      = float(np.clip(elv_raw, -np.radians(20), np.radians(20)))

    alpha_fdm = float(np.degrees(fdm._fdm["aero/alpha-rad"]))

    print(f"{substep_count:5d}  {V_actual:6.2f}  {np.degrees(roll):7.2f}  "
          f"{np.degrees(bank_cmd_rad):8.2f}  {np.degrees(roll_err):8.2f}  "
          f"{np.degrees(ail_raw):8.2f}  {np.degrees(ail_clip):8.2f}  "
          f"{np.degrees(pitch):7.2f}  {np.degrees(pitch_target):7.2f}  {np.degrees(pitch_err):7.2f}  "
          f"{np.degrees(elv_raw):8.2f}  {np.degrees(elv_clip):8.2f}  {alpha_fdm:9.2f}")

    # Apply shield + step
    ctrl_cmds = ctrl.update(state, bank_cmd_rad, speed_cmd_ms)
    ctrl_cmds = env._safety_shield(state, ctrl_cmds)
    fdm.step(ctrl_cmds, wind_ned, DT_PHYS)
    env._sensors.step(fdm.state, rng2)

    substep_count += 1

print()
print("Legend: V=airspeed  roll=actual  roll_cmd=commanded  ail_raw=before clip")
print("        p_tgt=pitch_target  p_err=pitch_target-actual_pitch")
print("        elv_raw=before clip  alpha_fdm=JSBSim AoA (deg)")
print("        All angles in degrees.")

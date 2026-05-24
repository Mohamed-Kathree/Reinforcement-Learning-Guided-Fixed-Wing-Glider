"""
diag_ctrl.py
============
Single-episode trace for the DeterministicRTL controller.

Prints one line per policy step showing distance to home, commanded bank
angle, airspeed command, and altitude.  Ends with a SUCCESS / CRASH /
TIMEOUT outcome line.

Usage:
    python scratch/diag_ctrl.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.glider_env import GliderEnv, BANK_CMD_SCALE_RAD, SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL

SEED  = 0
STAGE = 0
DT_RL = 0.05   # policy step size (s)

cfg  = dict(STAGES[STAGE])
env  = GliderEnv(cfg=cfg)
ctrl = DeterministicRTL()

print(f"Episode trace — DeterministicRTL  Stage {STAGE}  seed={SEED}")
obs, _ = env.reset(seed=SEED)

step = 0
while True:
    action = ctrl.act(obs)

    # Commanded bank angle and speed for this step
    bank_deg  = float(action[0]) * np.degrees(BANK_CMD_SCALE_RAD)
    speed_ms  = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS

    obs, _reward, terminated, truncated, info = env.step(action)

    dist_home = float(info.get('dist_home', 0.0))
    # True altitude from JSBSim state (positive up)
    alt_m = float(-env._fdm.state[2])
    time_s = (step + 1) * DT_RL

    if terminated and info.get('success'):
        print(f"Step {step:3d} | dist_home={dist_home:6.1f}m | SUCCESS — reached home in {time_s:.1f}s")
        break
    elif terminated or truncated:
        print(f"Step {step:3d} | dist_home={dist_home:6.1f}m | bank={bank_deg:+6.1f}°  speed={speed_ms:.1f} m/s | agl={alt_m:6.1f}m")
        if info.get('crash'):
            print(f"=> CRASH at step {step}")
        else:
            print(f"=> TIMEOUT after {step + 1} steps")
        break
    else:
        print(f"Step {step:3d} | dist_home={dist_home:6.1f}m | bank={bank_deg:+6.1f}°  speed={speed_ms:.1f} m/s | agl={alt_m:6.1f}m")

    step += 1

"""Trace one failing episode in detail to diagnose the baseline failure."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from env.glider_env import GliderEnv
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL

cfg = dict(STAGES[0])
env = GliderEnv(cfg=cfg)
ctrl = DeterministicRTL()

# Episode 1 from diag_rtl.py — seed that gave TIMEOUT with dist=792m
rng = np.random.default_rng(0)
seed = int(rng.integers(0, 2**31))
obs, _ = env.reset(seed=seed)
print(f"Launch: dx={obs[0]:.1f}m  dy={obs[1]:.1f}m  alt={obs[7]:.1f}m  airspeed={obs[11]:.1f}m/s")

step = 0
while True:
    action = ctrl.act(obs)
    obs, r, terminated, truncated, info = env.step(action)
    step += 1
    # Print every 50 policy steps (2.5 s)
    if step % 50 == 0 or step <= 10:
        state = env._fdm.state
        roll_deg  = float(np.degrees(np.arctan2(2*(state[6]*state[7]+state[8]*state[9]),
                          1-2*(state[7]**2+state[8]**2))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2*(state[6]*state[8]-state[9]*state[7]),-1,1))))
        print(f"  step {step:4d}: dist={info['dist_home']:.1f}m  alt={obs[7]:.1f}m  "
              f"spd={obs[11]:.1f}m/s  roll={roll_deg:.1f}deg  pitch={pitch_deg:.1f}deg  "
              f"act=[{action[0]:.2f},{action[1]:.2f}]")
    if terminated or truncated:
        outcome = 'SUCCESS' if info['success'] else ('CRASH' if info['crash'] else 'TIMEOUT')
        print(f"\nFinal: {outcome}  steps={step}  dist={info['dist_home']:.1f}m")
        break

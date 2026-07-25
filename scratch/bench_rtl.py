"""Benchmark DeterministicRTL over 100 episodes at Stage 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.glider_env import GliderEnv
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL

N = 100
cfg = dict(STAGES[0])
env = GliderEnv(cfg=cfg)
ctrl = DeterministicRTL()
rng = np.random.default_rng(42)

outcomes = {'SUCCESS': 0, 'CRASH': 0, 'TIMEOUT': 0}
final_dists = []

for ep in range(N):
    seed = int(rng.integers(0, 2**31))
    ctrl.reset()   # clear yaw-rate derivative state from the previous episode
    obs, _ = env.reset(seed=seed)
    while True:
        action = ctrl.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info['success']:
                outcomes['SUCCESS'] += 1
            elif info['crash']:
                outcomes['CRASH'] += 1
            else:
                outcomes['TIMEOUT'] += 1
            final_dists.append(info['dist_home'])
            break
    if (ep + 1) % 10 == 0:
        print(f"  [{ep+1:3d}/100] SUCCESS={outcomes['SUCCESS']}  CRASH={outcomes['CRASH']}  TIMEOUT={outcomes['TIMEOUT']}")

print(f"\nFinal: SUCCESS={outcomes['SUCCESS']}%  CRASH={outcomes['CRASH']}%  TIMEOUT={outcomes['TIMEOUT']}%")
print(f"Dist home — mean={np.mean(final_dists):.1f}m  min={np.min(final_dists):.1f}m  max={np.max(final_dists):.1f}m")

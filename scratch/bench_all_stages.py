"""Benchmark DeterministicRTL across ALL curriculum stages (pre-training gate).
Throwaway diagnostic — safe to delete. Does NOT modify bench_rtl.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.glider_env import GliderEnv
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL

N = 100
ctrl = DeterministicRTL()

print(f"{'stage':<6}{'success%':>9}{'crash%':>8}{'timeout%':>9}"
      f"{'mean_d':>9}{'min_d':>8}{'max_d':>8}")
print("-" * 57)

for stage_idx in range(len(STAGES)):
    cfg = dict(STAGES[stage_idx])
    env = GliderEnv(cfg=cfg)
    rng = np.random.default_rng(42)   # fixed seed = comparable across stages
    outcomes = {'SUCCESS': 0, 'CRASH': 0, 'TIMEOUT': 0}
    dists = []
    for ep in range(N):
        seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed)
        while True:
            obs, r, terminated, truncated, info = env.step(ctrl.act(obs))
            if terminated or truncated:
                if info['success']:
                    outcomes['SUCCESS'] += 1
                elif info['crash']:
                    outcomes['CRASH'] += 1
                else:
                    outcomes['TIMEOUT'] += 1
                dists.append(info['dist_home'])
                break
    s = outcomes['SUCCESS'] * 100 // N
    c = outcomes['CRASH'] * 100 // N
    t = outcomes['TIMEOUT'] * 100 // N
    print(f"S{stage_idx:<5}{s:>9}{c:>8}{t:>9}"
          f"{np.mean(dists):>9.1f}{np.min(dists):>8.1f}{np.max(dists):>8.1f}")

print("\nReference: Stage 0 was previously measured at ~29% success / 71% crash / 0% timeout.")

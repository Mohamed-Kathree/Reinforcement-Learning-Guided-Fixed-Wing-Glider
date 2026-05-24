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
rng = np.random.default_rng(0)

for ep in range(8):
    seed = int(rng.integers(0, 2**31))
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    step = 0
    min_alt = 9999.0
    while True:
        action = ctrl.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        step += 1
        min_alt = min(min_alt, float(obs[7]))
        if terminated or truncated:
            outcome = 'SUCCESS' if info['success'] else ('CRASH' if info['crash'] else 'TIMEOUT')
            print(f"ep {ep+1}: {outcome}  steps={step}  dist={info['dist_home']:.1f}m  min_alt={min_alt:.1f}m  last_alt={obs[7]:.1f}m  airspeed={obs[11]:.1f}m/s")
            break

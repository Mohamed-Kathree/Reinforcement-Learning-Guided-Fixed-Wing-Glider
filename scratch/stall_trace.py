"""Read-only diagnostic (Phase F.1 follow-up): step-level trace of the stall
penalty at Stage 3 with best_model, checking whether penalty_stall's missing
dt_rl scaling (env/reward.py line ~284, no `* cfg['dt_rl']`, unlike r_path
and penalty_unreach which got that fix in Phase 6) explains its share of the
reward budget. Not wired into any test; throwaway."""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.glider_env import GliderEnv
from env.curriculum import STAGES

dummy_venv = DummyVecEnv([lambda: GliderEnv(cfg=dict(STAGES[0]))])
vecnorm = VecNormalize.load('checkpoints/vecnormalize_best.pkl', dummy_venv)
vecnorm.training = False
vecnorm.norm_reward = False
model = PPO.load('checkpoints/best_model.zip', device='cpu')

env = GliderEnv(cfg=dict(STAGES[3]))
rng = np.random.default_rng(777)

DT_RL = 0.05
total_pen_raw = 0.0
total_pen_scaled = 0.0
n_violate_steps = 0
n_steps = 0
n_episodes = 15
per_ep_raw = []

for ep in range(n_episodes):
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    ep_raw = 0.0
    while True:
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        n_steps += 1
        pen = info['penalty_stall']  # already negated sign in info (-penalty_stall)
        pen = -pen if pen < 0 else pen
        if pen > 0:
            n_violate_steps += 1
            total_pen_raw += pen
            total_pen_scaled += pen * DT_RL
            ep_raw += pen
        if term or trunc:
            break
    per_ep_raw.append(ep_raw)

print(f"episodes={n_episodes}  total_steps={n_steps}  violating_steps={n_violate_steps} "
      f"({100*n_violate_steps/n_steps:.1f}% of steps)")
print(f"sum(penalty_stall) RAW (as currently applied)  = {total_pen_raw:.1f}  "
      f"({total_pen_raw/n_episodes:.1f}/episode)")
print(f"sum(penalty_stall) IF scaled by dt_rl={DT_RL}    = {total_pen_scaled:.1f}  "
      f"({total_pen_scaled/n_episodes:.1f}/episode)")
print(f"ratio raw/scaled = {total_pen_raw/max(total_pen_scaled,1e-9):.1f}x  (expected ~{1/DT_RL:.0f}x = 1/dt_rl)")
print(f"per-episode raw penalty_stall values: {[round(x,1) for x in per_ep_raw]}")

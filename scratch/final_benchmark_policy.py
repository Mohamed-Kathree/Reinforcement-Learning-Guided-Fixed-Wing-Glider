"""Phase F.1 (training runbook): same benchmark as final_benchmark.py --
three-phase bucket definitions (success / soft-land / crash / timeout),
n=100 episodes/stage -- but driving GliderEnv with a trained SB3 policy
instead of DeterministicRTL, so the trained-policy numbers are directly
comparable to the fixed baseline table in RLGlider_Training_Runbook.md.

training/evaluate.py already does an RL-vs-baseline comparison, but its
'timeout' bucket is `not success and not crash`, which lumps soft-landed
episodes into 'timeout' -- exactly the mislabelling final_benchmark.py's
own docstring says was fixed for the baseline-only path. This script
reuses final_benchmark.bench()'s corrected 4-bucket logic for the policy
side too, stepping the raw GliderEnv (not a VecEnv) so terminated/
truncated stay distinguishable, and normalising observations by hand
with the loaded VecNormalize running stats.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.glider_env import GliderEnv
from env.curriculum import STAGES


def bench_policy(stage_idx, n_ep, policy, vecnorm, seed=0):
    env = GliderEnv(cfg=dict(STAGES[stage_idx]))
    rng = np.random.default_rng(seed)
    succ = crash = soft_land = timed_out = 0
    dists, steps, qualities, tracks = [], [], [], []
    for _ in range(n_ep):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        n = 0
        while True:
            norm_obs = vecnorm.normalize_obs(obs)
            action, _ = policy.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            n += 1
            if term or trunc:
                break
        dists.append(info["dist_home"])
        steps.append(n)
        qualities.append(float(info.get("quality", 0.0)))
        tracks.append(float(info.get("track_length", 0.0)))
        if info.get("success"):
            succ += 1
        elif trunc and not term:
            timed_out += 1
        elif info.get("crash"):
            crash += 1
        else:
            soft_land += 1
    return (succ / n_ep, soft_land / n_ep, crash / n_ep, timed_out / n_ep,
            float(np.mean(dists)), float(np.mean(steps)), float(np.mean(qualities)),
            float(np.mean(tracks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--vecnorm', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    dummy_venv = DummyVecEnv([lambda: GliderEnv(cfg=dict(STAGES[0]))])
    vecnorm = VecNormalize.load(args.vecnorm, dummy_venv)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = PPO.load(args.policy, device='cpu')

    print(f"policy={args.policy}  vecnorm={args.vecnorm}")
    print(f"{'stage':>6} {'success':>8} {'soft-land':>10} {'crash':>7} {'timeout':>8} "
          f"{'mean dist':>10} {'mean steps':>11} {'mean qual':>10} {'mean track':>11}")
    for stage in range(4):
        s, sl, c, t, d, st, q, tr = bench_policy(stage, args.n, model, vecnorm, seed=args.seed)
        print(f"{stage:>6} {s*100:>7.1f}% {sl*100:>9.1f}% {c*100:>6.1f}% {t*100:>7.1f}% "
              f"{d:>9.1f} {st:>11.1f} {q:>10.3f} {tr:>10.1f}")


if __name__ == '__main__':
    main()

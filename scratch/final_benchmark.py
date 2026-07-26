"""Phase 4 precision-landing benchmark: three-phase DeterministicRTL at all
4 curriculum stages, n=100 episodes each. success = quality > 0.5 and
d_td < R_home_m (env/reward.py); mean_quality is the continuous landing
grade (0-1) and is the more informative number under the new task."""
import numpy as np
from env.glider_env import GliderEnv
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL


def bench(stage_idx, n_ep, seed=0):
    env = GliderEnv(cfg=dict(STAGES[stage_idx]))
    ctrl = DeterministicRTL()
    rng = np.random.default_rng(seed)
    succ = crash = soft_land = timed_out = 0
    dists = []
    steps = []
    qualities = []
    tracks = []
    for _ in range(n_ep):
        ctrl.reset()
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        n = 0
        while True:
            obs, r, term, trunc, info = env.step(ctrl.act(obs))
            n += 1
            if term or trunc:
                break
        dists.append(info["dist_home"])
        steps.append(n)
        qualities.append(float(info.get("quality", 0.0)))
        tracks.append(float(info.get("track_length", 0.0)))
        # Four buckets: success (quality>0.5 and centred), soft_land (landed
        # clean-ish but missed the success bar -- NOT a timeout), crash
        # (quality<=0 / fault), timed_out (truncated: never touched down).
        # An earlier version of this script lumped soft_land into "timeout",
        # which badly mislabelled most Stage 0-2 episodes (they land fine,
        # just off-centre or below the quality/distance bar for "success").
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


print(f"{'stage':>6} {'success':>8} {'soft-land':>10} {'crash':>7} {'timeout':>8} "
      f"{'mean dist':>10} {'mean steps':>11} {'mean qual':>10} {'mean track':>11}")
for stage in range(4):
    s, sl, c, t, d, st, q, tr = bench(stage, 100)
    print(f"{stage:>6} {s*100:>7.1f}% {sl*100:>9.1f}% {c*100:>6.1f}% {t*100:>7.1f}% "
          f"{d:>9.1f} {st:>11.1f} {q:>10.3f} {tr:>10.1f}")

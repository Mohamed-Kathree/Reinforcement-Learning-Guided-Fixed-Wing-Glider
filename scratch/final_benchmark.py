"""Final Phase 1-3 benchmark: DeterministicRTL (P-only, kp_bank=1.5) at all
4 curriculum stages, n=100 episodes each, with every Phase 1/2 fix applied."""
import numpy as np
from env.glider_env import GliderEnv
from env.curriculum import STAGES
from baseline.deterministic_rtl import DeterministicRTL


def bench(stage_idx, n_ep, seed=0):
    env = GliderEnv(cfg=dict(STAGES[stage_idx]))
    ctrl = DeterministicRTL()
    rng = np.random.default_rng(seed)
    succ = crash = timeout = 0
    dists = []
    steps = []
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
        if info.get("success"):
            succ += 1
        elif info.get("crash"):
            crash += 1
        else:
            timeout += 1
    return succ / n_ep, crash / n_ep, timeout / n_ep, float(np.mean(dists)), float(np.mean(steps))


print(f"{'stage':>6} {'success':>8} {'crash':>7} {'timeout':>8} {'mean dist':>10} {'mean steps':>11}")
for stage in range(4):
    s, c, t, d, st = bench(stage, 100)
    print(f"{stage:>6} {s*100:>7.1f}% {c*100:>6.1f}% {t*100:>7.1f}% {d:>9.1f} {st:>11.1f}")

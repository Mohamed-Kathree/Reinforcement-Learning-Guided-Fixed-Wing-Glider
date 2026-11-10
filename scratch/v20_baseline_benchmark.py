"""SPEC V20 post-Phase-5 baseline re-measurement on the as-built aircraft.

DeterministicRTL at all 4 curriculum stages, four buckets exactly as in
scratch/final_benchmark.py (success / soft_land / crash / timeout), with
Wilson 95% intervals and every episode's sampled airframe (dr_params) kept
for later analysis. Interim tool: Phase 6's training/benchmark.py replaces it
as the single evaluator. The old 28-29% figure is void (GPS velocity bug).

Run:   python -m scratch.v20_baseline_benchmark [--n 100]
       -> analysis/baseline_benchmark_v20.json
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from statsmodels.stats.proportion import proportion_confint

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from baseline.deterministic_rtl import DeterministicRTL  # noqa: E402
from env.curriculum import STAGES  # noqa: E402
from env.glider_env import GliderEnv  # noqa: E402

BUCKETS = ("success", "soft_land", "crash", "timeout")


def bucket(info, terminated, truncated):
    if info.get("success"):
        return "success"
    if truncated and not terminated:
        return "timeout"
    if info.get("crash"):
        return "crash"
    return "soft_land"


def bench(stage_idx, n_ep, seed=0):
    env = GliderEnv(cfg=dict(STAGES[stage_idx]))
    ctrl = DeterministicRTL()
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_ep):
        ep_seed = int(rng.integers(0, 2**31))
        ctrl.reset()
        obs, _ = env.reset(seed=ep_seed)
        n = 0
        while True:
            obs, _, term, trunc, info = env.step(ctrl.act(obs))
            n += 1
            if term or trunc:
                break
        episodes.append({
            "seed": ep_seed, "outcome": bucket(info, term, trunc), "steps": n,
            "dist_home_m": float(info["dist_home"]), "quality": float(info.get("quality", 0.0)),
            "dr_params": info["dr_params"],
        })
    counts = {b: sum(e["outcome"] == b for e in episodes) for b in BUCKETS}
    summary = {}
    for b in BUCKETS:
        lo, hi = proportion_confint(counts[b], n_ep, alpha=0.05, method="wilson")
        summary[b] = {"n": counts[b], "rate": counts[b] / n_ep, "wilson95": [lo, hi]}
    return {
        "stage": stage_idx, "n": n_ep, "buckets": summary,
        "mean_quality": float(np.mean([e["quality"] for e in episodes])),
        "mean_dist_home_m": float(np.mean([e["dist_home_m"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(PROJ_ROOT, "analysis", "baseline_benchmark_v20.json"))
    args = ap.parse_args()
    stages = []
    for s in range(len(STAGES)):
        r = bench(s, args.n)
        stages.append(r)
        b = r["buckets"]
        print(f"stage {s}: " + "  ".join(
            f"{k} {v['rate'] * 100:5.1f}% [{v['wilson95'][0] * 100:4.1f}, {v['wilson95'][1] * 100:4.1f}]"
            for k, v in b.items()) + f"  quality {r['mean_quality']:.2f}", flush=True)
    rec = {
        "description": "DeterministicRTL baseline, SPEC V20 as-built aircraft, post-Phase-5 ranges",
        "provenance": {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT,
                                      capture_output=True, text=True).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()),
        },
        "stages": stages,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(args.out, PROJ_ROOT)}")


if __name__ == "__main__":
    main()

"""
baseline_distances.py
=====================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

The 20 fixed-seed deterministic-baseline episodes behind
tests/test_dynamics.py::test_baseline_fixed_seed_regression.

The test checks the distribution (mean, spread, every episode lands). The
exact per-episode distances are kept in tests/data/baseline_distances.json as
a change detector to diff by eye, not as a pass/fail assertion: single
episodes are chaotic over a ~30 s flight (a 1e-4 change in CD0 moves one of
them by 11.75 m), so pinning them exactly made the test fail on every
legitimate physics change.

Run:   python -m analysis.baseline_distances    (rewrites the JSON; then git diff it)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

DISTANCES_JSON = os.path.join(PROJ_ROOT, "tests", "data", "baseline_distances.json")
N_EPISODES = 20
SEED = 12345


def run_baseline_episodes(n: int = N_EPISODES, seed: int = SEED) -> tuple[list[float], list[str]]:
    """Final distance from home and outcome for n stage-0 baseline episodes."""
    from baseline.deterministic_rtl import DeterministicRTL
    from env.curriculum import STAGES
    from env.glider_env import GliderEnv

    env = GliderEnv(cfg=dict(STAGES[0]))
    ctrl = DeterministicRTL()
    rng = np.random.default_rng(seed)
    dists, outcomes = [], []
    for _ in range(n):
        ctrl.reset()
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        while True:
            obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
            if terminated or truncated:
                dists.append(float(info["dist_home"]))
                outcomes.append(str(info["outcome"]))
                break
    return dists, outcomes


def main() -> None:
    dists, outcomes = run_baseline_episodes()
    d = np.array(dists)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT,
                         capture_output=True, text=True).stdout.strip()
    record = {
        "description": "Stage-0 deterministic baseline, 20 fixed seeds (rng seed 12345). "
                       "Change detector only; see analysis/baseline_distances.py.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha_parent": sha,
        "mean_m": round(float(d.mean()), 4),
        "std_m": round(float(d.std(ddof=1)), 4),
        "outcomes": sorted(set(outcomes)),
        "final_dist_m": [round(x, 8) for x in dists],
    }
    with open(DISTANCES_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(f"mean {record['mean_m']} m, std {record['std_m']} m, outcomes {record['outcomes']}")
    print(f"wrote {os.path.relpath(DISTANCES_JSON, PROJ_ROOT)}")


if __name__ == "__main__":
    main()

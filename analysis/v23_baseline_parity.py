"""SPEC V23 test_baseline_unchanged, all 100 seeds x 4 stages: the baseline,
now reading through env/nav_features.py, against the pre-refactor benchmark
log (results/benchmark_v22_best_15M.json, commit ed65989), under observation
v1 and v2. tests/test_obs_v2.py runs five seeds per stage of the same check.

Usage: python analysis/v23_baseline_parity.py [--workers 10]
Writes results/v23_baseline_parity.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REF = os.path.join(ROOT, "results", "benchmark_v22_best_15M.json")


def _one(task):
    from training.benchmark import BaselineActor, evaluate_seed, load_config
    stage, seed, version = task
    r = evaluate_seed(load_config(), stage, BaselineActor(), seed, version)
    return stage, seed, version, r["outcome"], r["steps"], r["miss_distance_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    with open(REF, encoding="utf-8") as f:
        ref = json.load(f)
    old = {(int(s), e["seed"]): e for s in ref["stages"] for e in ref["stages"][s]["episodes"]["baseline"]}
    tasks = [(s, seed, v) for v in (1, 2) for (s, seed) in old]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(_one, tasks, chunksize=4)
    out = {"reference": os.path.relpath(REF, ROOT), "by_version": {}}
    for v in (1, 2):
        rows = [r for r in res if r[2] == v]
        diff = [{"stage": s, "seed": seed, "old": [old[(s, seed)]["outcome"], old[(s, seed)]["steps"],
                                                   old[(s, seed)]["miss_distance_m"]],
                 "new": [o, n, m]}
                for s, seed, _, o, n, m in rows
                if o != old[(s, seed)]["outcome"] or n != old[(s, seed)]["steps"]
                or abs(m - old[(s, seed)]["miss_distance_m"]) > 1e-6]
        out["by_version"][str(v)] = {"episodes": len(rows), "identical": len(rows) - len(diff),
                                     "differences": diff}
        print(f"obs_version {v}: {len(rows) - len(diff)}/{len(rows)} episodes identical")
    with open(os.path.join(ROOT, "results", "v23_baseline_parity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

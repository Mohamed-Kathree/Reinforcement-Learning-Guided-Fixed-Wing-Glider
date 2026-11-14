"""
v24_run5_check.py
=================
SPEC V24 run 5 early-stop check, on the DEVELOPMENT seeds
(analysis/seeds/dev_seeds.json, episode_seeds(200, 7)), never the benchmark's.
Nothing trains.

Rule (written before the run, docs/SPEC_V20_decisions.md "V24 run 5"):
at 2M and at 5M steps, STOP if
    Stage 0 (calm) success < 40%, or
    Stage 3 crash rate is not below run 4 best's on the same seeds.

    python analysis/v24_run5_check.py reference                 # run 4 best, once
    python analysis/v24_run5_check.py check --step 2000000      # a run 5 checkpoint

Default 3 workers: a check runs beside training, and 10 workers beside
training exhausted memory on 14 Nov and killed run 4.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "training", "configs", "v24_run5_mix.yaml")
REF_MODEL = ("checkpoints_v24_run4/best_model.zip", "checkpoints_v24_run4/vecnormalize_best.pkl")
REF_PATH = os.path.join(ROOT, "results", "v24_run5_dev_reference.json")
STAGES = (0, 3)
MIN_STAGE0 = 0.40
_W: dict = {}


def _init(model, vecnorm):
    import torch
    torch.set_num_threads(1)
    from training.benchmark import PolicyActor, load_config
    _W["actor"] = PolicyActor(os.path.join(ROOT, model), os.path.join(ROOT, vecnorm))
    _W["cfg"] = load_config(CONFIG)


def _episode(task):
    from training.benchmark import evaluate_seed
    stage, seed = task
    r = evaluate_seed(_W["cfg"], stage, _W["actor"], seed, None)
    return stage, {k: r[k] for k in ("seed", "outcome", "miss_distance_m")}


def evaluate(model, vecnorm, workers):
    with open(os.path.join(ROOT, "analysis", "seeds", "dev_seeds.json")) as f:
        seeds = json.load(f)["seeds"]
    tasks = [(st, s) for st in STAGES for s in seeds]
    with mp.get_context("spawn").Pool(workers, initializer=_init,
                                      initargs=(model, vecnorm)) as pool:
        res = pool.map(_episode, tasks, chunksize=4)
    out = {}
    for st in STAGES:
        eps = [r for s, r in res if s == st]
        n = len(eps)
        out[str(st)] = {"n": n,
                        "success": sum(e["outcome"] == "success" for e in eps) / n,
                        "crash": sum(e["outcome"] == "crash" for e in eps) / n,
                        "episodes": eps}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("reference", "check"))
    ap.add_argument("--run", default="checkpoints_v24_run5")
    ap.add_argument("--step", type=int)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)

    if a.cmd == "reference":
        res = evaluate(*REF_MODEL, a.workers)
        with open(REF_PATH, "w", encoding="utf-8") as f:
            json.dump({"model": REF_MODEL[0], "seed_set": "dev_seeds.json", "stages": res}, f, indent=1)
        print({st: (v["success"], v["crash"]) for st, v in res.items()})
        print(f"wrote {os.path.relpath(REF_PATH, ROOT)}")
        return

    with open(REF_PATH) as f:
        ref = json.load(f)["stages"]
    model = f"{a.run}/ppo_glider_{a.step}_steps.zip"
    vecnorm = f"{a.run}/vecnormalize_{a.step}.pkl"
    res = evaluate(model, vecnorm, a.workers)
    s0, c3, c3_ref = res["0"]["success"], res["3"]["crash"], ref["3"]["crash"]
    ok0, ok3 = s0 >= MIN_STAGE0, c3 < c3_ref
    print(f"step {a.step}: stage 0 success {s0:.0%} (need >= {MIN_STAGE0:.0%}; run 4 best "
          f"{ref['0']['success']:.0%}): {'PASS' if ok0 else 'FAIL'}")
    print(f"           stage 3 crash {c3:.0%}, success {res['3']['success']:.0%} "
          f"(run 4 best crash {c3_ref:.0%}, success {ref['3']['success']:.0%}): "
          f"{'PASS' if ok3 else 'FAIL'}")
    verdict = "continue" if ok0 and ok3 else "STOP"
    print(verdict)
    path = os.path.join(ROOT, "results", f"v24_run5_check_{a.step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "stages": res, "reference": {k: {kk: v[kk] for kk in ("success", "crash")}
                                                                 for k, v in ref.items()},
                   "rule": {"min_stage0": MIN_STAGE0, "stage3_crash_below": c3_ref},
                   "verdict": verdict}, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

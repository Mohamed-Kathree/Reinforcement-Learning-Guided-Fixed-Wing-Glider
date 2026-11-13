"""
v24_early_check.py
==================
SPEC V24 Phase 2: the early-stop check for runs 3 and 4. Nothing trains.

    1. Stage 0 must graduate by 1.5x the steps V22 needed (3,625,600 -> 5,438,400),
       read from the run's curriculum_<steps>.json files.
    2. The first checkpoint saved at or after graduation must score at least 45%
       calm-air (Stage 0) success on the 100 paired benchmark seeds.

STOP if either fails. Otherwise let the run finish and benchmark its best model.

Usage:
    python analysis/v24_early_check.py --run checkpoints_v24_run3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.early_check import V22_GRADUATION_STEPS, graduation  # noqa: E402

LIMIT = 1.5 * V22_GRADUATION_STEPS
MIN_STAGE0 = 0.45
CONFIG = os.path.join(ROOT, "training", "configs", "v22_restored.yaml")

_W: dict = {}


def _init(model, vecnorm):
    import torch
    torch.set_num_threads(1)
    from training.benchmark import BaselineActor, PolicyActor, load_config
    _W["policy"] = PolicyActor(model, vecnorm)
    _W["baseline"] = BaselineActor()
    _W["cfg"] = load_config(CONFIG)


def _episode(task):
    from training.benchmark import evaluate_seed
    who, seed = task
    return who, evaluate_seed(_W["cfg"], 0, _W[who], seed, None if who == "policy" else 1)


def graduation_checkpoint(run_dir: str, grad_step: int) -> tuple[int, str, str] | None:
    steps = sorted(int(f[len("ppo_glider_"):-len("_steps.zip")]) for f in os.listdir(run_dir)
                   if f.startswith("ppo_glider_") and f.endswith("_steps.zip"))
    for s in steps:
        if s >= grad_step:
            return s, os.path.join(run_dir, f"ppo_glider_{s}_steps.zip"), \
                os.path.join(run_dir, f"vecnormalize_{s}.pkl")
    return None


def main(argv=None):
    from training.benchmark import episode_seeds, mcnemar_exact, paired_table, summarise

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's checkpoint directory")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-stage0", type=float, default=MIN_STAGE0,
                    help="calm-air bar at graduation (spec: 0.45; V22 itself scored 0.24 there, "
                         "see docs/SPEC_V20_decisions.md V24). 0 = graduation-only rule")
    a = ap.parse_args(argv)

    g = graduation(a.run)
    out = {"run": a.run, "graduation": g}
    print(f"1. stage 0 graduated at {g['graduated_at']} (limit {LIMIT:.0f}, "
          f"latest checkpoint {g['latest_checkpoint']}): "
          f"{'PASS' if g['pass'] else 'FAIL' if g['fail'] else 'not yet'}")
    if g["fail"]:
        out["verdict"] = "STOP"
    elif not g["pass"]:
        out["verdict"] = "not yet graduated; check again later"
    else:
        step, model, vecnorm = graduation_checkpoint(a.run, g["graduated_at"])
        seeds = episode_seeds(100, 0)
        tasks = [(w, s) for w in ("policy", "baseline") for s in seeds]
        with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                          initargs=(model, vecnorm)) as pool:
            res = pool.map(_episode, tasks, chunksize=5)
        pol = [r for w, r in res if w == "policy"]
        base = [r for w, r in res if w == "baseline"]
        sp, sb = summarise(pol), summarise(base)
        rate = sp["buckets"]["success"]["rate"]
        table = paired_table(pol, base)
        out["checkpoint"] = {"step": step, "model": model, "stage0": sp, "baseline": sb,
                             "paired": table, "mcnemar_exact_p": mcnemar_exact(table),
                             "threshold": a.min_stage0, "pass": rate >= a.min_stage0}
        lo, hi = sp["buckets"]["success"]["wilson95"]
        print(f"2. checkpoint {step}: stage 0 success {rate:.0%} [{lo:.0%}, {hi:.0%}] "
              f"(baseline {sb['buckets']['success']['rate']:.0%}, V22 59%; need >= {a.min_stage0:.0%}): "
              f"{'PASS' if rate >= a.min_stage0 else 'FAIL'}")
        out["verdict"] = "continue" if rate >= a.min_stage0 else "STOP"
    print(out["verdict"])
    path = os.path.join(ROOT, "results", f"v24_early_check_{os.path.basename(a.run.rstrip('/'))}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

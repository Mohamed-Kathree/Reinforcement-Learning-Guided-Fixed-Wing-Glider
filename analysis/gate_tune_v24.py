"""
gate_tune_v24.py
================
SPEC V24 Phase 3.2-3.3: tune the wind gate on the development seeds, then
evaluate the frozen configuration ONCE on the benchmark seeds.

    python analysis/gate_tune_v24.py tune       # development seeds only
    python analysis/gate_tune_v24.py evaluate   # benchmark seeds, values from base.yaml gate:

Tuning shortcut (exact, checked in the run): the gated flight is identical to
V22 flying alone until the handover step, so the handover step of any gate
setting can be read off the |w_est| trace recorded on the solo flight
(baseline/wind_gate.py first_crossing_step / single_check_step). Only the
distinct (stage, seed, handover step) flights are then flown, with a gate
forced to hand over at that step. A random sample of grid points is re-flown
with the real gate and must give the same outcome.

Selection rule (set before looking at results): the highest mean success over
the four stages on the development seeds; ties go to the lower stage-3 crash
rate, then the lower handover share at Stage 0.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

THRESHOLDS = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
T_EARLIEST = [3.0, 5.0, 8.0, 10.0]
T_LATEST = [10.0, 15.0, 20.0]
V22 = ("checkpoints_v22/best_model.zip", "checkpoints_v22/vecnormalize_best.pkl")
_A: dict = {}


def _policy():
    if "policy" not in _A:
        import torch
        torch.set_num_threads(1)
        from training.benchmark import PolicyActor
        _A["policy"] = PolicyActor(*(os.path.join(ROOT, p) for p in V22))
    return _A["policy"]


def _flight(task):
    from baseline.wind_gated import WindGatedActor
    from training.benchmark import BaselineActor, evaluate_seed, load_config
    kind, arg, stage, seed = task
    if kind == "baseline":
        actor = BaselineActor()
    elif kind == "alone":
        actor = WindGatedActor(_policy(), threshold_ms=1e9, t_decide_s=0.0)
    elif kind == "at_step":
        actor = WindGatedActor(_policy(), mode="at_step", at_step=arg)
    elif kind == "window":
        T, a, b = arg
        actor = WindGatedActor(_policy(), threshold_ms=T, mode="window", t_earliest_s=a, t_latest_s=b)
    else:  # "single"
        T, t = arg
        actor = WindGatedActor(_policy(), threshold_ms=T, t_decide_s=t)
    r = evaluate_seed(load_config(), stage, actor, seed, 1)
    out = {"kind": kind, "arg": arg, "stage": stage, "seed": seed, "outcome": r["outcome"],
           "steps": r["steps"], "miss": r["miss_distance_m"]}
    if kind == "alone":
        out["trace"] = [round(w, 6) for w in actor.gate.trace]
    if "gate" in r:
        out["gate"] = r["gate"]
    if kind != "at_step":
        out["record"] = {k: v for k, v in r.items() if k not in ("launch_state", "dr_params")}
    return out


def _pool(tasks, workers):
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_flight, tasks, chunksize=4)


def _summ(eps):
    n = len(eps)
    return {"success": sum(e["outcome"] == "success" for e in eps) / n,
            "crash": sum(e["outcome"] == "crash" for e in eps) / n,
            "handover": sum(e["handover_step"] is not None for e in eps) / n}


def tune(workers: int) -> None:
    from baseline.wind_gate import first_crossing_step, single_check_step

    with open(os.path.join(ROOT, "analysis", "seeds", "dev_seeds.json")) as f:
        seeds = json.load(f)["seeds"]
    stages = range(4)
    print(f"solo and baseline flights: {2 * 4 * len(seeds)}", flush=True)
    base = _pool([(k, None, st, s) for k in ("alone", "baseline") for st in stages for s in seeds], workers)
    solo = {(r["stage"], r["seed"]): r for r in base if r["kind"] == "alone"}
    bl = {(r["stage"], r["seed"]): r for r in base if r["kind"] == "baseline"}

    configs = [("window", (T, a, b)) for T, a, b in itertools.product(THRESHOLDS, T_EARLIEST, T_LATEST)]
    configs += [("single", (T, t)) for T in THRESHOLDS for t in (10.0, 15.0)]

    def hand_step(kind, arg, trace):
        return first_crossing_step(trace, *arg) if kind == "window" else single_check_step(trace, *arg)

    need = sorted({(st, s, hand_step(k, a, solo[(st, s)]["trace"]))
                   for k, a in configs for st in stages for s in seeds} - {(st, s, None)
                   for st in stages for s in seeds})
    need = [(st, s, k) for st, s, k in need if k is not None]
    print(f"distinct handover flights: {len(need)}", flush=True)
    forced = {(r["stage"], r["seed"], r["arg"]): r
              for r in _pool([("at_step", k, st, s) for st, s, k in need], workers)}

    def episodes(kind, arg, st):
        out = []
        for s in seeds:
            k = hand_step(kind, arg, solo[(st, s)]["trace"])
            f = solo[(st, s)] if k is None else forced[(st, s, k)]
            out.append({"seed": s, "outcome": f["outcome"], "handover_step": k})
        return out

    grid = []
    for kind, arg in configs:
        per = {st: _summ(episodes(kind, arg, st)) for st in stages}
        grid.append({"kind": kind, "arg": arg, "stages": per,
                     "mean_success": sum(p["success"] for p in per.values()) / 4})
    ref = {"baseline": {st: _summ([{"outcome": bl[(st, s)]["outcome"], "handover_step": None}
                                   for s in seeds]) for st in stages},
           "v22_alone": {st: _summ([{"outcome": solo[(st, s)]["outcome"], "handover_step": None}
                                    for s in seeds]) for st in stages}}

    def key(g):
        return (g["mean_success"], -g["stages"][3]["crash"], -g["stages"][0]["handover"])

    best_window = max((g for g in grid if g["kind"] == "window"), key=key)
    best_single = max((g for g in grid if g["kind"] == "single"), key=key)

    # Exactness check of the trace shortcut: re-fly 24 random (config, stage, seed).
    rng = random.Random(0)
    sample = [(rng.choice(configs), rng.randrange(4), rng.choice(seeds)) for _ in range(24)]
    sample += [(("window", best_window["arg"]), st, s) for st in (2, 3) for s in seeds[:4]]
    real = _pool([(k, a, st, s) for (k, a), st, s in sample], workers)
    mismatch = []
    for ((k, a), st, s), r in zip(sample, real):
        ep = [e for e in episodes(k, a, st) if e["seed"] == s][0]
        if ep["outcome"] != r["outcome"] or (ep["handover_step"] is not None) != r["gate"]["handed_over"]:
            mismatch.append([k, a, st, s, ep["outcome"], r["outcome"]])
    print(f"shortcut check: {len(sample) - len(mismatch)}/{len(sample)} identical", flush=True)

    out = {"seed_set": "analysis/seeds/dev_seeds.json", "n_seeds": len(seeds),
           "selection_rule": "max mean success over stages 0-3; ties: lower stage-3 crash, "
                             "then lower stage-0 handover share",
           "reference": ref, "best_window": best_window, "best_single": best_single,
           "shortcut_check": {"n": len(sample), "mismatches": mismatch}, "grid": grid}
    path = os.path.join(ROOT, "results", "gate_tune_v24_dev.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for name, r in ref.items():
        print(f"{name:10s} " + "  ".join(f"s{st} {r[st]['success']:.2f}" for st in stages))
    for g in sorted(grid, key=key, reverse=True)[:12]:
        print(f"{g['kind']:6s} {str(g['arg']):18s} mean {g['mean_success']:.3f}  " +
              "  ".join(f"s{st} {g['stages'][st]['success']:.2f}/h{g['stages'][st]['handover']:.2f}"
                        for st in stages))
    print(f"best window: {best_window['arg']}  best single: {best_single['arg']}")
    print(f"wrote {os.path.relpath(path, ROOT)}")


def evaluate(workers: int) -> None:
    """One evaluation on the benchmark seeds, gate values frozen in base.yaml."""
    from training.benchmark import (episode_seeds, load_config, mcnemar_exact, paired_table,
                                    summarise)
    gate = load_config()["gate"]
    seeds = episode_seeds(100, 0)
    variants = {"single": ("single", (gate["single"]["threshold_ms"], gate["single"]["t_decide_s"])),
                "window": ("window", (gate["window"]["threshold_ms"], gate["window"]["t_earliest_s"],
                                      gate["window"]["t_latest_s"]))}
    tasks = [(k, a, st, s) for k, a in variants.values() for st in range(4) for s in seeds]
    tasks += [(k, None, st, s) for k in ("alone", "baseline") for st in range(4) for s in seeds]
    raw = os.path.join(ROOT, "results", "gated_v24_benchmark_raw.json")
    if os.path.exists(raw) and os.environ.get("REUSE_RAW") == "1":
        with open(raw) as f:
            res = json.load(f)
        for r in res:
            if isinstance(r["arg"], list):
                r["arg"] = tuple(r["arg"])
    else:
        res = _pool(tasks, workers)
        with open(raw, "w", encoding="utf-8") as f:
            json.dump(res, f)
    out = {"seed_set": "benchmark: episode_seeds(100, 0)", "gate_config": gate, "stages": {}}
    for st in range(4):
        def eps(kind, arg=None):
            return [r["record"]
                    for r in res if r["kind"] == kind and r["stage"] == st
                    and (arg is None or r["arg"] == arg)]
        base = eps("baseline")
        rec = {"baseline": summarise(base), "v22_alone": summarise(eps("alone"))}
        for name, (k, a) in variants.items():
            e = eps(k, a)
            table = paired_table(e, base)
            gates = [x["gate"] for x in e]
            rec[name] = {**summarise(e), "paired_vs_baseline": table,
                         "mcnemar_exact_p": mcnemar_exact(table),
                         "handover_share_of_episodes": sum(g["handed_over"] for g in gates) / len(gates),
                         "baseline_share_of_flight_time":
                             sum(g["baseline_share_of_flight"] * g["steps"] for g in gates)
                             / sum(g["steps"] for g in gates),
                         "episodes": e}
        out["stages"][str(st)] = rec
    path = os.path.join(ROOT, "results", "gated_v24_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for st, rec in out["stages"].items():
        line = f"stage {st}: baseline {rec['baseline']['buckets']['success']['rate']:.2f}  " \
               f"V22 {rec['v22_alone']['buckets']['success']['rate']:.2f}"
        for name in variants:
            v = rec[name]
            b = v["buckets"]
            line += (f"  | {name} {b['success']['rate']:.2f} {[round(x, 2) for x in b['success']['wilson95']]}"
                     f" crash {b['crash']['rate']:.2f} p {v['mcnemar_exact_p']:.3f}"
                     f" handover {v['handover_share_of_episodes']:.2f}"
                     f" time {v['baseline_share_of_flight_time']:.2f}")
        print(line)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tune"
    w = int(os.environ.get("WORKERS", "10"))
    {"tune": tune, "evaluate": evaluate}[cmd](w)

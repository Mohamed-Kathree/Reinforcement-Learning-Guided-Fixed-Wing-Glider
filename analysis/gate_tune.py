"""Tune and benchmark the wind-gated controller (baseline/wind_gated.py).

Per seed and stage, four flights: V22 alone (logging its on-board wind
estimate at 10 and 15 s), V22 with a forced handover to the baseline at 10 s
and at 15 s, and the baseline alone. A gated flight with threshold T and
decision time t is then the V22-alone flight if the estimate at t is <= T,
else the forced-handover-at-t flight: identical up to t, so every threshold
is scored from the same four flights.

    python analysis/gate_tune.py tune        # seeds from default_rng(7): choose (T, t)
    python analysis/gate_tune.py test T t    # benchmark seeds (default_rng(0)), paired vs baseline
"""
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
TIMES = [10.0, 15.0]
_A = {}


def _flight(task):
    from baseline.wind_gated import WindGatedActor
    from training.benchmark import BaselineActor, PolicyActor, evaluate_seed, load_config
    kind, t, stage, seed = task
    if "policy" not in _A:
        _A["policy"] = PolicyActor(os.path.join(ROOT, "checkpoints_v22/best_model.zip"),
                                   os.path.join(ROOT, "checkpoints_v22/vecnormalize_best.pkl"))
    if kind == "baseline":
        actor = BaselineActor()
    elif kind == "alone":
        actor = WindGatedActor(_A["policy"], threshold_ms=1e9, t_decide_s=0.0)
    else:
        actor = WindGatedActor(_A["policy"], threshold_ms=0.0, t_decide_s=t)
    r = evaluate_seed(load_config(), stage, actor, seed, 1)
    est = getattr(actor, "estimates", {})
    return {"kind": kind, "t": t, "stage": stage, "seed": seed, "outcome": r["outcome"],
            "miss": r["miss_distance_m"], "est": {str(k): v for k, v in est.items()},
            "handed": getattr(actor, "handed_over", None)}


def flights(seed_base: int, n: int, workers: int):
    from training.benchmark import episode_seeds
    seeds = episode_seeds(n, seed_base)
    tasks = [(k, t, st, s) for st in range(4) for s in seeds
             for k, t in (("alone", 0.0), ("handover", 10.0), ("handover", 15.0), ("baseline", 0.0))]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_flight, tasks, chunksize=4)
    by = {}
    for r in res:
        by[(r["kind"], r["t"], r["stage"], r["seed"])] = r
    return seeds, by


def gated(by, seeds, stage, T, t):
    out = []
    for s in seeds:
        alone = by[("alone", 0.0, stage, s)]
        est = alone["est"].get(str(t))
        switch = est is not None and est > T
        f = by[("handover", t, stage, s)] if switch else alone
        out.append({"seed": s, "outcome": f["outcome"], "switched": switch, "miss": f["miss"]})
    return out


def rate(eps):
    return sum(e["outcome"] == "success" for e in eps) / len(eps)


def main():
    mode = sys.argv[1]
    workers = 4
    if mode == "tune":
        seeds, by = flights(7, 100, workers)
        rows = []
        base = {st: rate([by[("baseline", 0.0, st, s)] for s in seeds]) for st in range(4)}
        pol = {st: rate([by[("alone", 0.0, st, s)] for s in seeds]) for st in range(4)}
        print("baseline", base, "\nV22 alone", pol)
        for t in TIMES:
            for T in THRESHOLDS:
                r = {st: rate(gated(by, seeds, st, T, t)) for st in range(4)}
                sw = {st: sum(e["switched"] for e in gated(by, seeds, st, T, t)) / len(seeds) for st in range(4)}
                worst = min(r[st] - base[st] for st in range(4))
                mean = sum(r.values()) / 4
                rows.append({"T": T, "t": t, "success": r, "switched": sw, "worst_margin": worst, "mean": mean})
                print(f"t {t:>4} T {T:>4}: success {[round(r[k], 2) for k in range(4)]}  "
                      f"switched {[round(sw[k], 2) for k in range(4)]}  worst vs base {worst:+.2f}  mean {mean:.3f}")
        json.dump({"seed_base": 7, "baseline": base, "v22_alone": pol, "grid": rows},
                  open(os.path.join(ROOT, "results", "gate_tune_v22.json"), "w"), indent=1)
    else:
        from training.benchmark import mcnemar_exact, paired_table, wilson
        T, t = float(sys.argv[2]), float(sys.argv[3])
        seeds, by = flights(0, 100, workers)
        out = {"T": T, "t": t, "seed_base": 0, "stages": {}}
        for st in range(4):
            g = gated(by, seeds, st, T, t)
            b = [{"seed": s, "outcome": by[("baseline", 0.0, st, s)]["outcome"]} for s in seeds]
            a = [{"seed": s, "outcome": by[("alone", 0.0, st, s)]["outcome"]} for s in seeds]
            tab = paired_table(g, b)
            k = sum(e["outcome"] == "success" for e in g)
            crash = sum(e["outcome"] == "crash" for e in g) / len(g)
            rec = {"gated": rate(g), "gated_wilson95": list(wilson(k, len(g))), "gated_crash": crash,
                   "baseline": rate(b), "v22_alone": rate(a), "switched": sum(e["switched"] for e in g) / len(g),
                   "paired_vs_baseline": tab, "mcnemar_p": mcnemar_exact(tab)}
            out["stages"][str(st)] = rec
            print(f"stage {st}: gated {rec['gated']:.2f} [{rec['gated_wilson95'][0]:.2f}, {rec['gated_wilson95'][1]:.2f}] "
                  f"crash {crash:.2f} | baseline {rec['baseline']:.2f} | V22 alone {rec['v22_alone']:.2f} | "
                  f"handed over {rec['switched']:.2f} | p vs baseline {rec['mcnemar_p']:.3f}")
        json.dump(out, open(os.path.join(ROOT, "results", f"gate_test_v22_T{T:g}_t{t:g}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

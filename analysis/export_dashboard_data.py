"""
export_dashboard_data.py
========================
Writes frontend/src/data/validation.json, the dashboard's single source of
truth for benchmark figures. Every number is read from a committed result
file; nothing is typed in by hand. Re-run after any new evaluation:

    python analysis/export_dashboard_data.py

Sources
    results/gated_v24_run4_benchmark.json  selected controller (run 4 + window
                                           gate), run 4 alone, baseline
                                           (one evaluation, 100 benchmark seeds)
    results/gated_v24_benchmark.json       V22 + window gate, same seeds
    results/feasibility_v24.json           per-seed physical reachability
    training/configs/base.yaml gate.run4   frozen gate settings
"""

from __future__ import annotations

import json
import os
import sys
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "frontend", "src", "data", "validation.json")
WIND = {0: "Calm", 1: "Light wind, up to 3 m/s", 2: "Moderate wind, up to 6 m/s",
        3: "Strong wind, up to 9 m/s"}
MAX_WIND = {0: 0, 1: 3, 2: 6, 3: 9}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (same test as
    training/benchmark.py's statsmodels call)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rate(rec, bucket):
    return round(rec["buckets"][bucket]["rate"], 4)


def ci(rec, bucket):
    return [round(x, 4) for x in rec["buckets"][bucket]["wilson95"]]


def main():
    from training.benchmark import load_config

    r4 = json.load(open(os.path.join(ROOT, "results", "gated_v24_run4_benchmark.json")))
    v22 = json.load(open(os.path.join(ROOT, "results", "gated_v24_benchmark.json")))
    feas = json.load(open(os.path.join(ROOT, "results", "feasibility_v24.json")))
    raw = json.load(open(os.path.join(ROOT, "results", "gated_v24_run4_benchmark_raw.json")))
    gate = load_config()["gate"]["run4"]["window"]
    assert r4["gate_config"]["window"] == gate, "benchmark was not run with the frozen gate"

    stages = []
    for st in range(4):
        s = r4["stages"][str(st)]
        w, base, alone = s["window"], s["baseline"], s["v22_alone"]   # 'v22_alone' = the flown policy, run 4
        g22 = v22["stages"][str(st)]["window"]
        base_eps = {r["seed"]: r["outcome"] for r in raw if r["kind"] == "baseline" and r["stage"] == st}
        win_eps = {e["seed"]: e["outcome"] for e in w["episodes"]}
        assert set(base_eps) == set(win_eps)
        crash_more = sum(win_eps[k] == "crash" and base_eps[k] != "crash" for k in win_eps)
        crash_less = sum(win_eps[k] != "crash" and base_eps[k] == "crash" for k in win_eps)
        (both, gated_only), (base_only, neither) = w["paired_vs_baseline"]
        fs = feas["summary"][str(st)]
        crashes = [e for e in w["episodes"] if e["outcome"] == "crash"]
        t_open = gate["t_earliest_s"]
        hand_t = sorted(e["gate"]["handover_time_s"] for e in w["episodes"] if e["gate"]["handed_over"])
        crash_timing = {
            "total": len(crashes),
            "after_handover": sum(e["gate"]["handed_over"] for e in crashes),
            "policy_only_after_window_opened": sum(not e["gate"]["handed_over"] and e["steps"] * 0.05 >= t_open
                                                   for e in crashes),
            "before_window_opened": sum(not e["gate"]["handed_over"] and e["steps"] * 0.05 < t_open
                                        for e in crashes),
        }
        stages.append({
            "stage": st,
            "label": WIND[st],
            "max_wind_ms": MAX_WIND[st],
            "n": w["n"],
            "gated": {"success": rate(w, "success"), "success_ci95": ci(w, "success"),
                      "crash": rate(w, "crash"), "crash_ci95": ci(w, "crash"),
                      "median_miss_m": round(w["median_miss_distance_m"], 1)},
            "baseline": {"success": rate(base, "success"), "crash": rate(base, "crash"),
                         "median_miss_m": round(base["median_miss_distance_m"], 1)},
            "run4_alone": {"success": rate(alone, "success"), "crash": rate(alone, "crash")},
            "v22_gated": {"success": rate(g22, "success"), "crash": rate(g22, "crash")},
            "paired_success": {"both": both, "gated_only": gated_only,
                               "baseline_only": base_only, "neither": neither,
                               "p": float(f"{w['mcnemar_exact_p']:.4g}")},
            "paired_crash": {"gated_only": crash_more, "baseline_only": crash_less,
                             "p": float(f"{mcnemar_exact(crash_more, crash_less):.4g}")},
            "handover_share_of_episodes": round(w["handover_share_of_episodes"], 4),
            "baseline_share_of_flight_time": round(w["baseline_share_of_flight_time"], 4),
            "handover_time_median_s": hand_t[len(hand_t) // 2] if hand_t else None,
            "crash_timing": crash_timing,
            "feasibility": {"reachable": fs["reachable"], "landed_by_any_controller": fs["any_controller_success"]},
            # Every benchmark flight, so the dashboard can open any of them in
            # Flight Replay (backend recorder, benchmark=True, reproduces them).
            "flights": [{"seed": e["seed"], "outcome": e["outcome"],
                         "miss_m": round(e["miss_distance_m"], 1),
                         "handover_s": e["gate"]["handover_time_s"],
                         "baseline_outcome": base_eps[e["seed"]]}
                        for e in w["episodes"]],
        })

    out = {
        "generated_by": "analysis/export_dashboard_data.py",
        "sources": ["results/gated_v24_run4_benchmark.json", "results/gated_v24_benchmark.json",
                    "results/feasibility_v24.json", "training/configs/base.yaml gate.run4"],
        "controller": {
            "name": "Run 4 – Gated Controller",
            "policy_checkpoint": r4["model"],
            "gate": {"threshold_ms": gate["threshold_ms"], "t_earliest_s": gate["t_earliest_s"],
                     "t_latest_s": gate["t_latest_s"], "mode": "window"},
        },
        "evaluation": {"seed_set": r4["seed_set"], "n_seeds": 100, "evaluations": 1,
                       "tuning_seed_set": "analysis/seeds/dev_seeds.json (200 development seeds)",
                       "test": "two-sided exact McNemar, paired on seed", "alpha": 0.05},
        "stages": stages,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for s in stages:
        print(s["stage"], s["gated"], s["baseline"], s["paired_success"]["p"], s["paired_crash"])
    print(f"wrote {os.path.relpath(OUT, ROOT)}")

    research = build_research(feas, stages)
    research_out = os.path.join(os.path.dirname(OUT), "research.json")
    with open(research_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(research, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(research_out, ROOT)}")


# Runs since the V22 reference, in order. Benchmarks are the committed
# results/benchmark_*.json files (100 paired seeds, stages 0-3); findings
# summarise docs/SPEC_V20_decisions.md.
RUNS = [
    {"id": "v22", "name": "V22", "change": "Reference configuration (commit ed65989), seed 0.",
     "verdict": "reference",
     "finding": "Best calm-air policy before V24; falls to 0% calm-air success by 20M steps.",
     "checkpoints": [("best, 15M", "benchmark_v22_best_15M.json"), ("final, 20M", "benchmark_v22_final_20M.json")]},
    {"id": "v23r1", "name": "V23 run 1", "change": "GAE λ 0.995 with glide-cone shaping and other changes bundled.",
     "verdict": "negative",
     "finding": "Worse than V22 at every stage. Bundling made the cause impossible to attribute.",
     "checkpoints": [("best, 6.75M", "benchmark_v23run1_best_6.75M.json"), ("final, 20M", "benchmark_v23run1_final_20M.json")]},
    {"id": "v23r2", "name": "V23 run 2", "change": "As run 1 without cone shaping.",
     "verdict": "negative",
     "finding": "Still worse than V22, so shaping was not the cause.",
     "checkpoints": [("best, 10M", "benchmark_v23run2_best_10M.json"), ("final, 20M", "benchmark_v23run2_final_20M.json")]},
    {"id": "run3", "name": "V24 run 3", "change": "V22 configuration with GAE λ 0.995 only.",
     "verdict": "negative",
     "finding": "Never left stage 0. The critic could not fit near-Monte-Carlo targets (explained variance 0.07–0.59 against V22's 0.84–0.93), so λ 0.995 is what broke runs 1–2.",
     "checkpoints": [("best, 1.75M", "benchmark_v24run3_best_1.75M.json"), ("stopped, 8.6M", "benchmark_v24run3_8.6M.json")]},
    {"id": "run4", "name": "V24 run 4", "change": "V22 configuration unchanged, training seed 1.",
     "verdict": "selected",
     "finding": "Reproduces V22 (63% against 59% calm-air success). Its best model is the policy inside the selected gated controller.",
     "checkpoints": [("best, 15.5M", "benchmark_v24run4_best_15.5M.json"), ("20M", "benchmark_v24run4_20M.json"),
                     ("final, 24.5M", "benchmark_v24run4_final_24.5M.json")]},
    {"id": "run5", "name": "V24 run 5", "change": "Run 4 best continued 10M steps on a fixed stage mix (20/20/30/30%).",
     "verdict": "mixed",
     "finding": "Stage 3 crashes fell from 54% to 37%, but it learned to avoid crashing rather than to get home, and calm-air success fell from 63% to 15%.",
     "checkpoints": [("best, 1.75M", "benchmark_v24run5_best_1.75M.json"), ("5M", "benchmark_v24run5_5M.json"),
                     ("final, 10M", "benchmark_v24run5_final_10M.json")]},
]

FINDINGS = [
    {"title": "Raising GAE λ broke the critic",
     "body": "λ 0.995 was meant to carry the landing outcome back to early decisions. The value network could not fit the resulting near-Monte-Carlo targets, and every run with it failed. Tested on its own in run 3."},
    {"title": "Calm-air skill is forgotten in harder conditions",
     "body": "Under every curriculum tried, calm-air success collapses once training moves to windier stages: V22 59% to 0%, run 4 63% to 1%, run 5 63% to 15%. The best checkpoint is always early; final checkpoints are much worse."},
    {"title": "More training steps did not help",
     "body": "Run 4 spent 9.4M steps at stage 2 at 2–7% training success against a 15% promotion bar, and never reached stage 3. The bars sit at the best any controller has achieved."},
    {"title": "The task is limited by gusts, not energy",
     "body": "Home is within gliding reach on 397 of 400 benchmark launches, usually with 7–9 times the energy needed. With no thrust or airbrakes the glider must shed that surplus precisely in one attempt, and gusts decide most stage-3 outcomes."},
]


def build_research(feas, stages):
    def rates(path):
        d = json.load(open(os.path.join(ROOT, "results", path)))["stages"]
        return [{"success": round(d[str(s)]["policy"]["buckets"]["success"]["rate"], 4),
                 "crash": round(d[str(s)]["policy"]["buckets"]["crash"]["rate"], 4)} for s in range(4)]

    runs = [{**{k: v for k, v in r.items() if k != "checkpoints"},
             "checkpoints": [{"label": lab, "source": f"results/{f}", "stages": rates(f)}
                             for lab, f in r["checkpoints"]]} for r in RUNS]
    baseline = [{"success": s["baseline"]["success"], "crash": s["baseline"]["crash"]} for s in stages]

    rows = feas["rows"]
    gust_bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    wind_bins = [(-10.0, -4.0), (-4.0, -1.5), (-1.5, 1.5), (1.5, 4.0), (4.0, 10.0)]

    def binned(st, key, bins):
        rs = [r for r in rows if r["stage"] == st]
        out = []
        for i, (lo, hi) in enumerate(bins):
            last = i == len(bins) - 1
            inside = [r for r in rs if lo <= (r[key] or 0.0) < hi or (last and (r[key] or 0.0) == hi)]
            out.append({"lo": lo, "hi": hi, "n": len(inside),
                        "landed": sum(bool(r["any_controller_success"]) for r in inside)})
        return out

    feasibility = []
    for st in range(4):
        sm = feas["summary"][str(st)]
        feasibility.append({
            "stage": st,
            "n": sm["n"],
            "reachable": sm["reachable"],
            "landed_by_any_controller": int(sm["any_controller_success"]),
            "gated_success": round(stages[st]["gated"]["success"] * 100),
            "baseline_success": round(stages[st]["baseline"]["success"] * 100),
            "energy_margin_median": round(sm["margin_median"], 2),
            "energy_margin_p10": round(sm["margin_p10"], 2),
            "by_gust": binned(st, "gust_sigma", gust_bins),
            "by_along_wind": binned(st, "w_along", wind_bins),
        })
    return {"generated_by": "analysis/export_dashboard_data.py",
            "sources": ["results/benchmark_*.json", "results/feasibility_v24.json", "docs/SPEC_V20_decisions.md"],
            "baseline": baseline, "runs": runs, "findings": FINDINGS, "feasibility": feasibility}


if __name__ == "__main__":
    main()

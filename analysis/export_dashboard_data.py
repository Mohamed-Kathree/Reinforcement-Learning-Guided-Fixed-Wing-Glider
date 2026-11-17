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


if __name__ == "__main__":
    main()

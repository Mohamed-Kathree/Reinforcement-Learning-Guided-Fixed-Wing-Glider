"""
early_check.py
==============
SPEC V23 revision: the early-stop checks for a training run, on one saved
checkpoint (about 3 minutes, 10 workers). Nothing trains.

    1. Stage 0 graduated within 1.5x the steps V22 needed (3.63M), from the
       run's curriculum_<steps>.json files.
    2. Downwind touchdown bias against the baseline under +10 m, in steady
       wind, 100 paired seeds (V22: +21.5 m).
    3. Mid-flight value-to-return correlation of at least 0.5 (steady wind,
       stochastic policy, the value at 50% of each flight against the
       realised discounted return in the run's own learner units).
    4. The pattern adapts to the wind: slope of the mean along-wind position
       above 8 m against wind speed at most +7 m per m/s. The plan asked for
       a negative slope, but calibration on saved models showed the baseline
       itself at +14.1 and V22 at +14.0 (4M) and +13.7 (best), so the mark is
       half of that: a run passes by drifting half as far as V22 did.

Stop rule from the plan: stop if checks 2 and 4 both fail.

The rollouts use the run's own reward settings (cone shaping, path frame,
reward scale), so pass the same --set overrides the run was started with.

Usage:
    python analysis/early_check.py --model checkpoints_v23_run1/ppo_glider_4000000_steps.zip \\
        --vecnorm checkpoints_v23_run1/vecnormalize_4000000.pkl \\
        --set reward.cone_shaping=true reward.path_reward_frame=air ...
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

V22_GRADUATION_STEPS = 3_625_600
STEADY_WIND_MS = 3.0
SLOPE_PASS = 7.0             # m of downwind pattern position per m/s of wind
DT = 0.05

_A: dict = {}


def _setup(model, vecnorm, overrides):
    from training.benchmark import PolicyActor, load_config
    from training.train import apply_overrides
    _A["policy"] = PolicyActor(model, vecnorm)
    _A["cfg"] = apply_overrides(load_config(), list(overrides))


def _episode(task):
    import torch

    from env.curriculum import STAGES
    from env.glider_env import GliderEnv
    from training.benchmark import BaselineActor
    from training.train import flat_env_cfg

    torch.set_num_threads(1)
    who, seed, stochastic = task
    cfg = _A["cfg"]
    actor = _A["policy"] if who == "policy" else BaselineActor()
    env_cfg = flat_env_cfg(cfg)
    env_cfg["obs_version"] = actor.obs_version if who == "policy" else 1
    env = GliderEnv(cfg=env_cfg)
    st = dict(STAGES[0])
    st["wind_speed"] = STEADY_WIND_MS
    env.set_stage(st)
    actor.reset(seed)
    if stochastic:
        torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    wind = env._wind._mean_ned[:2].astype(float).copy()
    xs, rs, along_pos, first = [], [], [], True
    scale = float(cfg["ppo"].get("reward_scale", 1.0))
    ws = float(np.hypot(*wind))
    u = wind / ws if ws > 0.2 else None
    while True:
        if who == "policy":
            x = actor.observe(obs, first)
            first = False
            action, _ = actor.model.predict(x, deterministic=not stochastic)
            xs.append(x)
        else:
            action = actor.act(obs)
        obs, r, term, trunc, info = env.step(action)
        rs.append(scale * float(r))
        p = env._fdm.state[0:2].astype(float)
        if u is not None and -float(env._fdm.state[2]) > 8.0:
            along_pos.append(float(p @ u))
        if term or trunc:
            break
    env.close()
    rec = {"who": who, "seed": seed, "wind_speed": ws, "outcome": info.get("success") and "success"}
    if u is not None:
        rec["td_along_m"] = float(p @ u)
        rec["pattern_along_m"] = float(np.mean(along_pos)) if along_pos else float("nan")
    if stochastic and who == "policy":
        with torch.no_grad():
            v = actor.model.policy.predict_values(
                torch.as_tensor(np.stack(xs), dtype=torch.float32)).numpy().ravel()
        g, ret = 0.0, np.zeros(len(rs))
        for t in range(len(rs) - 1, -1, -1):
            g = rs[t] + float(actor.model.gamma) * g
            ret[t] = g
        i = len(rs) // 2
        rec["mid_value"], rec["mid_return"] = float(v[i]), float(ret[i])
    return rec


def graduation(ckpt_dir: str) -> dict:
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "curriculum_*.json")):
        m = re.search(r"curriculum_(\d+)\.json$", f)
        if m:
            with open(f) as fh:
                steps.append((int(m.group(1)), int(json.load(fh).get("current_stage", 0))))
    steps.sort()
    first = next((s for s, st in steps if st >= 1), None)
    limit = 1.5 * V22_GRADUATION_STEPS
    latest = steps[-1][0] if steps else 0
    return {"graduated_at": first, "limit": limit, "latest_checkpoint": latest,
            "pass": (first is not None and first <= limit),
            "fail": (first is None and latest > limit) or (first is not None and first > limit)}


def main(argv=None):
    from scipy import stats

    from analysis.wind_diagnostics import _mean_ci
    from training.benchmark import episode_seeds

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    seeds = episode_seeds(a.episodes, 0)
    tasks = [(w, s, False) for w in ("policy", "baseline") for s in seeds]
    tasks += [("policy", s, True) for s in seeds]
    with mp.get_context("spawn").Pool(a.workers, initializer=_setup,
                                      initargs=(a.model, a.vecnorm, a.set)) as pool:
        res = pool.map(_episode, tasks, chunksize=4)
    det = {w: [r for r in res[:2 * len(seeds)] if r["who"] == w] for w in ("policy", "baseline")}
    sto = res[2 * len(seeds):]

    out = {"model": a.model, "overrides": a.set}
    out["1_stage0_graduation"] = graduation(os.path.dirname(os.path.abspath(a.model)))

    diff = [p["td_along_m"] - b["td_along_m"] for p, b in zip(det["policy"], det["baseline"])
            if "td_along_m" in p and "td_along_m" in b]
    bias = _mean_ci(diff)
    out["2_downwind_bias_vs_baseline_m"] = {**bias, "threshold": 10.0, "v22": 21.5,
                                            "pass": bias["mean"] < 10.0}

    mv = np.array([r["mid_value"] for r in sto])
    mr = np.array([r["mid_return"] for r in sto])
    corr = float(np.corrcoef(mv, mr)[0, 1])
    out["3_mid_flight_value_return_corr"] = {"corr": corr, "threshold": 0.5, "v22": 0.12,
                                             "pass": corr >= 0.5}

    for w in ("policy", "baseline"):
        pts = [(r["wind_speed"], r["pattern_along_m"]) for r in det[w]
               if "pattern_along_m" in r and r["pattern_along_m"] == r["pattern_along_m"]]
        x, y = np.array(pts).T
        lr = stats.linregress(x, y)
        h = float(stats.t.ppf(0.975, len(x) - 2)) * lr.stderr
        rec = {"slope_m_per_ms": float(lr.slope), "ci95": [lr.slope - h, lr.slope + h], "n": len(x)}
        if w == "policy":
            rec["pass"] = bool(lr.slope <= SLOPE_PASS)
            out["4_pattern_shifts_upwind"] = rec
        else:
            out["4_baseline_reference"] = rec
    out["success_steady_wind"] = {w: sum(r["outcome"] == "success" for r in det[w]) / len(det[w])
                                  for w in det}
    out["stop"] = (not out["2_downwind_bias_vs_baseline_m"]["pass"]
                   and not out["4_pattern_shifts_upwind"]["pass"])

    g = out["1_stage0_graduation"]
    print(f"1. stage 0 graduation: {g['graduated_at']} (limit {g['limit']:.0f}) "
          f"{'PASS' if g['pass'] else 'FAIL' if g['fail'] else 'not yet'}")
    b = out["2_downwind_bias_vs_baseline_m"]
    print(f"2. downwind bias vs baseline: {b['mean']:+.1f} m [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}] "
          f"(< +10; V22 +21.5) {'PASS' if b['pass'] else 'FAIL'}")
    c = out["3_mid_flight_value_return_corr"]
    print(f"3. mid-flight value/return corr: {c['corr']:.2f} (>= 0.5; V22 0.12) "
          f"{'PASS' if c['pass'] else 'FAIL'}")
    p, q = out["4_pattern_shifts_upwind"], out["4_baseline_reference"]
    print(f"4. pattern along-wind slope: {p['slope_m_per_ms']:+.1f} m per m/s "
          f"[{p['ci95'][0]:+.1f}, {p['ci95'][1]:+.1f}] (<= +7; V22 +14, baseline {q['slope_m_per_ms']:+.1f}) "
          f"{'PASS' if p['pass'] else 'FAIL'}")
    print(f"   steady-wind success: policy {out['success_steady_wind']['policy']:.2f}, "
          f"baseline {out['success_steady_wind']['baseline']:.2f}")
    print("STOP the run (2 and 4 both failed)" if out["stop"] else "continue")
    path = a.out or os.path.join(ROOT, "results",
                                 f"early_check_{os.path.basename(a.model).replace('.zip', '')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

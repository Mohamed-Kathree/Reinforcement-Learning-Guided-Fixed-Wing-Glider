"""
horizon_diagnostics.py
======================
SPEC V23 revision: does the landing outcome reach the decisions that set it
up? Five tests on saved models (no training):

    t1  advantage horizon: GAE recomputed on V22 rollouts at several lambda;
        correlation of the advantage k seconds before touchdown with the
        landing outcome
    t2  home shifted upwind: V22 is told home is tau*wind upwind of where it
        is; does success recover (pure drift) or not (final approach broken)
    t3  touchdown direction relative to the wind and signed along-track miss,
        V22 against the baseline
    t4  fresh value networks on frozen V22 rollouts (mixed stages): current
        inputs; plus a wind estimate; plus true wind and stage
    t5  wind-aware glide margin when the glider passes 8 m altitude

Steady wind is Stage 0 plus Stage 1's mean wind (0-3 m/s), as in
analysis/wind_diagnostics.py, whose rollout conventions this follows.

Run: python analysis/horizon_diagnostics.py all [--workers 10]
Output: results/horizon_t<k>_v22.json
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from analysis.wind_diagnostics import (MODELS, _actor, _fit_mlp, _learner_reward,  # noqa: E402
                                       _mean_ci, _r2, _seeds, _write, paired, stage_cfg)
from env.curriculum import outcome_bucket  # noqa: E402

MODEL = "v22"
DT = 0.05
LAMBDAS = [0.95, 0.98, 0.99, 0.995, 1.0]
SECONDS_BEFORE = [0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
TAUS = [0, 2, 5, 10, 15, 20]
V_AIR = 10.0                  # nominal cruise airspeed for the glide margin (m/s)
X_EVERY = 5                   # t4 keeps every fifth step's inputs


# ---------------------------------------------------------------------------
# One traced episode (worker side)
# ---------------------------------------------------------------------------

def trace(task: dict) -> dict:
    import torch

    from env.glider_env import SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS, GliderEnv
    from env.wind_estimate import WindEstimator
    from training.benchmark import load_config
    from training.train import flat_env_cfg

    torch.set_num_threads(1)
    model, cfg_name, seed = task["model"], task["cfg"], int(task["seed"])
    stochastic, tau = bool(task.get("stochastic", False)), float(task.get("tau", 0.0))
    keep_x = bool(task.get("keep_x", False))
    actor = _actor(model)
    env = GliderEnv(cfg={**flat_env_cfg(load_config()), "obs_version": 1})
    env.set_stage(stage_cfg(cfg_name))
    actor.reset(seed)
    if stochastic:
        torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    wind = env._wind._mean_ned[:2].astype(float).copy()
    est = WindEstimator(DT)
    to_learner = _learner_reward(actor, model) if model else None

    xs, rs, west, pos, alt, first = [], [], [], [], [], True
    prev_speed_cmd = SPEED_CMD_CENTRE_MS
    while True:
        o = np.array(obs, dtype=np.float32, copy=True)
        gs, course, yaw = float(o[2]), float(o[3]), float(o[6])
        west.append(est.update(gs * math.cos(course), gs * math.sin(course), yaw, prev_speed_cmd))
        if tau:
            o[0] += tau * wind[0]
            o[1] += tau * wind[1]
            o[11] = math.hypot(o[0], o[1]) / max(float(o[7]), 1.0) / 15.0
        if model is None:
            action = actor.act(o)
        else:
            x = actor.observe(o, first)
            xs.append(x)
            action, _ = actor.model.predict(x, deterministic=not stochastic)
        first = False
        prev_speed_cmd = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS
        obs, r, term, trunc, info = env.step(action)
        if to_learner:
            rs.append(to_learner(float(r)))
        pos.append(env._fdm.state[0:2].astype(float).copy())
        alt.append(-float(env._fdm.state[2]))
        if term or trunc:
            break
    env.close()

    pos, west = np.array(pos), np.array(west)
    rec = {"model": model or "baseline", "cfg": cfg_name, "seed": seed, "tau": tau,
           "outcome": outcome_bucket(info), "miss_m": float(info.get("dist_home", np.nan)),
           "wind_ne": wind.tolist(), "steps": len(pos), "touchdown_ne": pos[-1].tolist(),
           "r_terminal": float(info["episode_reward_components"]["r_terminal"]),
           "truncated": bool(trunc and not term)}
    ws = float(np.hypot(*wind))
    if ws > 0.2:
        u = wind / ws
        rec["td_along_wind_m"] = float(pos[-1] @ u)
        k = min(20, len(pos) - 1)
        d = pos[-1] - pos[-1 - k]                       # last second of ground track
        t = d / max(np.hypot(*d), 1e-9)
        rec["final_track_cos_downwind"] = float(t @ u)
        rec["td_along_track_m"] = float(pos[-1] @ t)    # + : landed past home along the final
    # Glide margin when first at or below 8 m
    i8 = next((i for i, h in enumerate(alt) if h <= 8.0), None)
    if i8 is not None:
        rec["at_8m"] = {"p": pos[i8].tolist(), "h": alt[i8], "frac": i8 / len(pos)}
    if model is not None:
        with torch.no_grad():
            v = actor.model.policy.predict_values(
                torch.as_tensor(np.stack(xs), dtype=torch.float32)).numpy().ravel()
        rec["values"], rec["rewards"] = v.tolist(), rs
        if keep_x:
            gamma = float(actor.model.gamma)
            g, ret = 0.0, np.zeros(len(rs))
            for t_ in range(len(rs) - 1, -1, -1):
                g = rs[t_] + gamma * g
                ret[t_] = g
            idx = list(range(0, len(xs), X_EVERY))
            rec["x"] = np.stack(xs)[idx].tolist()
            rec["x_west"] = west[idx].tolist()
            rec["x_ret"] = ret[idx].tolist()
            rec["x_frac"] = (np.array(idx) / len(xs)).tolist()
    rec["west_final"] = west[-1].tolist()
    return rec


def run(tasks: list[dict], workers: int) -> list[dict]:
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(trace, tasks, chunksize=4)
    print(f"  {len(tasks)} episodes in {time.time() - t0:.0f} s", flush=True)
    return out


def _strip(eps, drop=("values", "rewards", "x", "x_west", "x_ret", "x_frac")):
    return [{k: v for k, v in e.items() if k not in drop} for e in eps]


# ---------------------------------------------------------------------------
# t1. Advantage horizon
# ---------------------------------------------------------------------------

def gae(rewards, values, gamma: float, lam: float) -> np.ndarray:
    r, v = np.asarray(rewards), np.asarray(values)
    nxt = np.append(v[1:], 0.0)                     # episode ends: no bootstrap
    delta = r + gamma * nxt - v
    adv, a = np.zeros(len(r)), 0.0
    for t in range(len(r) - 1, -1, -1):
        a = delta[t] + gamma * lam * a
        adv[t] = a
    return adv


def t1(n: int, workers: int) -> dict:
    from training.benchmark import PolicyActor
    gamma = float(PolicyActor(os.path.join(ROOT, MODELS[MODEL][0]),
                              os.path.join(ROOT, MODELS[MODEL][1])).model.gamma)
    eps = run([{"model": MODEL, "cfg": "steady_wind", "seed": s, "stochastic": True}
               for s in _seeds(n)], workers)
    eps = [e for e in eps if not e["truncated"]]
    targets = {
        "r_terminal": np.array([e["r_terminal"] for e in eps]),
        "neg_miss_m": -np.array([e["miss_m"] for e in eps]),
        "neg_downwind_m": np.array([-e.get("td_along_wind_m", np.nan) for e in eps]),
    }
    out = {"description": "t1: Pearson r between the GAE advantage k seconds before touchdown "
                          "and the landing outcome, V22 stochastic, steady wind. Higher r = the "
                          "landing result reaches that decision",
           "gamma": gamma, "n_episodes": len(eps), "by_lambda": {}}
    for lam in LAMBDAS:
        advs = [gae(e["rewards"], e["values"], gamma, lam) for e in eps]
        rows = {}
        for s in SECONDS_BEFORE:
            k = int(round(s / DT))
            sel = [i for i, a in enumerate(advs) if len(a) > k]
            a = np.array([advs[i][len(advs[i]) - 1 - k] for i in sel])
            row = {"n": len(sel)}
            for name, y in targets.items():
                yy = y[sel]
                m = np.isfinite(yy)
                row[name] = float(np.corrcoef(a[m], yy[m])[0, 1]) if m.sum() > 5 else float("nan")
            rows[str(s)] = row
        out["by_lambda"][str(lam)] = rows
        print(f"  lambda {lam}: r(adv, r_terminal) by s before touchdown " +
              " ".join(f"{s}s:{rows[str(s)]['r_terminal']:+.2f}" for s in SECONDS_BEFORE), flush=True)
    # The value function alone: V(s_t) against the outcome, for reference
    out["value_vs_r_terminal"] = {}
    for s in SECONDS_BEFORE:
        k = int(round(s / DT))
        sel = [i for i, e in enumerate(eps) if len(e["values"]) > k]
        v = np.array([eps[i]["values"][len(eps[i]["values"]) - 1 - k] for i in sel])
        out["value_vs_r_terminal"][str(s)] = float(np.corrcoef(v, targets["r_terminal"][sel])[0, 1])
    _write("horizon_t1", MODEL, out)
    return out


# ---------------------------------------------------------------------------
# t2/t3/t5 share deterministic steady-wind flights
# ---------------------------------------------------------------------------

def _summ(eps):
    w = [e for e in eps if "td_along_wind_m" in e]
    k = sum(e["outcome"] == "success" for e in eps)
    from training.benchmark import wilson
    return {"n": len(eps), "success": k / len(eps), "success_wilson95": list(wilson(k, len(eps))),
            "median_miss_m": float(np.median([e["miss_m"] for e in eps])),
            "signed_along_wind_m": _mean_ci([e["td_along_wind_m"] for e in w])}


def t2(n: int, workers: int) -> dict:
    seeds = _seeds(n)
    eps = run([{"model": MODEL, "cfg": "steady_wind", "seed": s, "tau": t}
               for t in TAUS for s in seeds], workers)
    calm = run([{"model": MODEL, "cfg": "calm", "seed": s} for s in seeds], workers)
    out = {"description": "t2: V22 told home is tau seconds of wind upwind of where it is "
                          "(dx, dy and the required glide ratio shifted); scored against the "
                          "true home. Steady wind, deterministic",
           "calm_reference": _summ(calm), "by_tau_s": {}}
    ref = [e for e in eps if e["tau"] == 0]
    for t in TAUS:
        sel = [e for e in eps if e["tau"] == t]
        s = _summ(sel)
        if t:
            s["vs_tau0"] = paired(sel, ref)
        out["by_tau_s"][str(t)] = s
        print(f"  tau {t:>2} s: success {s['success']:.2f}, median miss {s['median_miss_m']:.1f} m, "
              f"along {s['signed_along_wind_m']['mean']:+.1f} m", flush=True)
    out["episodes"] = _strip(eps)
    _write("horizon_t2", MODEL, out)
    return out


def _glide_margin(at8: dict, wind, glide: float) -> float:
    """Seconds of glide in hand minus seconds needed to reach home through
    the moving air, at nominal airspeed V_AIR."""
    p, w = np.asarray(at8["p"]), np.asarray(wind)
    a = V_AIR ** 2 - float(w @ w)
    if a <= 0:
        return float("-inf")
    pw = float(p @ w)
    t_need = (pw + math.sqrt(pw * pw + a * float(p @ p))) / a
    return glide * float(at8["h"]) / V_AIR - t_need


def t3_t5(n: int, workers: int) -> dict:
    seeds = _seeds(n)
    eps = run([{"model": m, "cfg": "steady_wind", "seed": s} for m in (MODEL, None) for s in seeds],
              workers)
    out3 = {"description": "t3: direction of the last second of ground track at touchdown "
                           "relative to the wind (cos with the downwind direction: -1 into wind, "
                           "+1 downwind), and touchdown position along that final track "
                           "(+ = landed past home). Steady wind, wind > 0.2 m/s",
            "by_controller": {}}
    out5 = {"description": "t5: wind-aware glide margin when the glider first reaches 8 m: "
                           "glide_ratio*h/V minus the time to reach home through the moving air "
                           f"at V = {V_AIR} m/s, in seconds. Negative = home already out of reach",
            "by_controller": {}}
    for who in (MODEL, "baseline"):
        sel = [e for e in eps if e["model"] == who]
        w = [e for e in sel if "final_track_cos_downwind" in e]
        c = np.array([e["final_track_cos_downwind"] for e in w])
        rec = {"n": len(w),
               "into_wind_frac": float((c < -0.5).mean()), "crosswind_frac": float((abs(c) <= 0.5).mean()),
               "downwind_frac": float((c > 0.5).mean()),
               "along_track_miss_m": _mean_ci([e["td_along_track_m"] for e in w])}
        for lab, cond in (("into_wind", c < -0.5), ("crosswind", abs(c) <= 0.5), ("downwind", c > 0.5)):
            sub = [e for e, k in zip(w, cond) if k]
            rec[f"{lab}_success"] = (sum(e["outcome"] == "success" for e in sub) / len(sub)) if sub else None
            rec[f"{lab}_median_miss_m"] = float(np.median([e["miss_m"] for e in sub])) if sub else None
        out3["by_controller"][who] = rec
        r5 = {}
        for glide in (8.0, 12.0):
            m = {"success": [], "failed": []}
            for e in sel:
                if "at_8m" in e:
                    m["success" if e["outcome"] == "success" else "failed"].append(
                        _glide_margin(e["at_8m"], e["wind_ne"], glide))
            r5[f"glide_{glide:g}"] = {
                k: {"n": len(v), "median_s": float(np.median(v)) if v else None,
                    "negative_frac": float(np.mean(np.array(v) < 0)) if v else None}
                for k, v in m.items()}
        r5["frac_of_flight_at_8m"] = _mean_ci([e["at_8m"]["frac"] for e in sel if "at_8m" in e])
        out5["by_controller"][who] = r5
        print(f"  {who}: into/cross/down {rec['into_wind_frac']:.2f}/{rec['crosswind_frac']:.2f}/"
              f"{rec['downwind_frac']:.2f}, along-track miss {rec['along_track_miss_m']['mean']:+.1f} m; "
              f"glide-12 margin failed {r5['glide_12']['failed']}", flush=True)
    out3["episodes"] = _strip(eps)
    _write("horizon_t3", MODEL, out3)
    _write("horizon_t5", MODEL, out5)
    return {"t3": out3, "t5": out5}


# ---------------------------------------------------------------------------
# t4. What would make the critic work
# ---------------------------------------------------------------------------

def t4(n_per_stage: int, workers: int, epochs: int = 30) -> dict:
    seeds = _seeds(n_per_stage)
    stages = ["stage0", "stage1", "stage2", "stage3"]
    eps = run([{"model": MODEL, "cfg": c, "seed": s, "stochastic": True, "keep_x": True}
               for c in stages for s in seeds], workers)
    X, WE, WT, ST, G, F, EP, V = [], [], [], [], [], [], [], []
    for i, e in enumerate(eps):
        k = len(e["x"])
        X.append(np.asarray(e["x"], dtype=np.float32))
        WE.append(np.asarray(e["x_west"], dtype=np.float32) / 10.0)
        WT.append(np.tile(np.asarray(e["wind_ne"], dtype=np.float32) / 10.0, (k, 1)))
        oh = np.zeros((k, 4), dtype=np.float32)
        oh[:, stages.index(e["cfg"])] = 1.0
        ST.append(oh)
        G.append(np.asarray(e["x_ret"], dtype=np.float32)[:, None])
        F.append(np.asarray(e["x_frac"]))
        EP.append(np.full(k, i))
        V.append(np.asarray(e["values"], dtype=np.float32)[::X_EVERY][:k])
    X, WE, WT, ST, G, F, EP, V = map(np.concatenate, (X, WE, WT, ST, G, F, EP, V))
    STI = ST.argmax(1)
    test = np.isin(EP, np.arange(0, len(eps), 5))
    inputs = {
        "current_inputs": X,
        "plus_wind_estimate": np.concatenate([X, WE], 1),
        "plus_true_wind_and_stage": np.concatenate([X, WT, ST], 1),
        "plus_wind_estimate_and_stage": np.concatenate([X, WE, ST], 1),
    }
    out = {"description": "t4: fresh [128,128] value networks fitted to the realised discounted "
                          "return (learner units) on frozen V22 rollouts, stages 0-3 mixed, "
                          "stochastic policy; held-out episodes. corr_mid = correlation at 40-60% "
                          "of the flight, within each stage",
           "n_episodes": len(eps), "n_train": int((~test).sum()), "n_test": int(test.sum()),
           "wind_estimate_error_ms_at_touchdown": _mean_ci(
               [float(np.hypot(*(np.asarray(e["west_final"]) - np.asarray(e["wind_ne"])))) for e in eps]),
           "fits": {}}

    def score(pred):
        g, p = G[test, 0], pred
        r = {"r2": float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()),
             "corr": float(np.corrcoef(p, g)[0, 1]), "corr_mid_by_stage": {}}
        for s in range(4):
            m = test & (STI == s) & (F >= 0.4) & (F <= 0.6)
            mm = m[test]
            r["corr_mid_by_stage"][stages[s]] = float(np.corrcoef(p[mm], g[mm])[0, 1])
        return r

    out["fits"]["v22_critic"] = score(V[test])
    for name, Xin in inputs.items():
        p = _fit_mlp(Xin[~test], G[~test], Xin[test], G[test], epochs)[:, 0]
        out["fits"][name] = score(p)
    for name, r in out["fits"].items():
        print(f"  {name:30s} R2 {r['r2']:+.2f} corr {r['corr']:.2f} mid " +
              " ".join(f"{k}:{v:.2f}" for k, v in r["corr_mid_by_stage"].items()), flush=True)
    _write("horizon_t4", MODEL, out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=("t1", "t2", "t3", "t4", "t5", "all"))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    todo = ["t1", "t3", "t2", "t4"] if a.test == "all" else [a.test]
    for t in todo:
        print(f"-- {t}", flush=True)
        if t == "t1":
            t1(a.episodes, a.workers)
        elif t == "t2":
            t2(min(a.episodes, 100), a.workers)
        elif t in ("t3", "t5"):
            t3_t5(min(a.episodes, 100), a.workers)
        elif t == "t4":
            t4(75, a.workers)


if __name__ == "__main__":
    main()

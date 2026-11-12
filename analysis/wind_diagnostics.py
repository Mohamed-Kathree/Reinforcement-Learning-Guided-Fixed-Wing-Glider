"""
wind_diagnostics.py
===================
SPEC V23 Phase D: why the policy fails in steady wind. Diagnostics on saved
models only; nothing here trains.

    d1  observation swap: fly the policy with yaw <- course, course <- yaw,
        or yaw biased +15 deg, and see what it was steering by
    d2  input sensitivity: |d bank / d input| x input std, per channel
    d3  success and miss by launch bearing-to-home sector (existing logs)
    d4  signed along-wind touchdown error, r_path against wind speed,
        reward-budget shares per stage
    d5  supervised representability (needs observation v2; see SPEC Phase O)
    d6  critic calibration: V(s_t) against the realised discounted return

Every rollout uses training/benchmark.py's conventions: a freshly built env
per episode, seeded with the benchmark's episode seeds (default_rng(0)), so
the policy and the baseline fly identical episodes.

"Steady wind" is Stage 0 with Stage 1's mean wind (3 m/s maximum, uniform
in speed and direction) and nothing else: the V21 stage-1 ablation's
"meanwind" group.

Run:
    python analysis/wind_diagnostics.py d1 --model v22
    python analysis/wind_diagnostics.py all --model v22 --model v21
Output: results/wind_d<k>_<model>.json, each with a provenance block.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from env.curriculum import STAGES, outcome_bucket  # noqa: E402

MODELS = {
    # name: (model, vecnorm, reward_scale the run trained with)
    "v22": ("checkpoints_v22/best_model.zip", "checkpoints_v22/vecnormalize_best.pkl", 0.02),
    "v21": ("checkpoints_v21/best_model.zip", "checkpoints_v21/vecnormalize_best.pkl", 1.0),
}

CHANNELS = ["dx_home", "dy_home", "ground_speed", "course", "roll", "pitch", "yaw",
            "baro_alt", "vertical_speed", "lidar_agl", "lidar_valid", "req_glide_ratio"]
COURSE, YAW = 3, 6
OBS_DIM = len(CHANNELS)
DT_RL = 0.05
TRACK_ERR_MIN_DIST_M = 30.0     # cruise phase: track error sampled only this far out
TRACK_ERR_SKIP_S = 3.0          # ...and after the launch turn (launch heading is random)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def stage_cfg(name: str) -> dict:
    if name == "calm":
        return dict(STAGES[0])
    if name == "steady_wind":
        c = dict(STAGES[0])
        c["wind_speed"] = STAGES[1]["wind_speed"]
        return c
    if name.startswith("stage"):
        return dict(STAGES[int(name[5:])])
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Observation interventions (D1). Applied to the raw 12-value observation
# before PolicyActor stacks and normalises it, so every stacked frame
# carries the intervention, as a wrapper on the base env would.
# ---------------------------------------------------------------------------

def _yaw_from_course(o):
    o[YAW] = o[COURSE]


def _course_from_yaw(o):
    o[COURSE] = o[YAW]


def _yaw_bias_15(o):
    o[YAW] = wrap(float(o[YAW]) + math.radians(15.0))


INTERVENTIONS = {
    "none": None,
    "yaw_from_course": _yaw_from_course,
    "course_from_yaw": _course_from_yaw,
    "yaw_bias_15": _yaw_bias_15,
}


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _actor(model: str | None):
    from training.benchmark import BaselineActor, PolicyActor
    if model is None:
        return BaselineActor()
    if model not in _CACHE:
        path, vecnorm, _ = MODELS[model]
        _CACHE[model] = PolicyActor(os.path.join(ROOT, path), os.path.join(ROOT, vecnorm))
    return _CACHE[model]


def _learner_reward(actor, model: str):
    """The reward exactly as PPO saw it during that model's training."""
    scale = MODELS[model][2]
    vn = actor.vecnorm
    if getattr(vn, "norm_reward", False):
        std = math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon))
        clip = float(vn.clip_reward)
        return lambda r: float(np.clip(scale * r / std, -clip, clip))
    return lambda r: scale * r


def rollout(task: dict) -> dict:
    """One episode. task: model (None = baseline), cfg, seed, intervention,
    mode ('plain' | 'log_obs' | 'critic')."""
    import torch

    from env.glider_env import GliderEnv
    from training.benchmark import load_config
    from training.train import flat_env_cfg

    torch.set_num_threads(1)
    model, cfg_name, seed = task["model"], task["cfg"], int(task["seed"])
    mode = task.get("mode", "plain")
    fn = INTERVENTIONS[task.get("intervention", "none")]
    actor = _actor(model)

    env_cfg = flat_env_cfg(load_config())
    env_cfg["obs_version"] = int(getattr(actor, "obs_version", 1))   # interventions index v1
    env = GliderEnv(cfg=env_cfg)
    env.set_stage(stage_cfg(cfg_name))
    actor.reset(seed)
    obs, _ = env.reset(seed=seed)
    wind = env._wind._mean_ned[:2].astype(float).copy()
    launch = env._fdm.state[0:2].astype(float).copy()
    prev = launch.copy()

    if mode == "critic":
        torch.manual_seed(seed)
        to_learner = _learner_reward(actor, model)
        xs, rs = [], []
    logged, d5_v1, d5_bank = [], [], []
    track_err, first, steps = [], True, 0
    while True:
        o = np.array(obs, dtype=np.float32, copy=True)
        if fn is not None:
            fn(o)
        if mode == "plain" or model is None:
            action = actor.act(o)
            if mode == "d5":
                d5_v1.append(o.tolist())
                d5_bank.append(float(action[0]) * 45.0)
        else:
            x = actor.observe(o, first)
            if mode == "log_obs":
                logged.append(actor.stacked())
            action, _ = actor.model.predict(x, deterministic=(mode != "critic"))
            if mode == "critic":
                xs.append(x)
        first = False
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        if mode == "critic":
            rs.append(to_learner(float(r)))
        pos = env._fdm.state[0:2].astype(float)
        d = pos - prev
        if (steps * DT_RL > TRACK_ERR_SKIP_S and math.hypot(*pos) > TRACK_ERR_MIN_DIST_M
                and math.hypot(*d) > 1e-3):
            track_err.append(wrap(math.atan2(-pos[1], -pos[0]) - math.atan2(d[1], d[0])))
        prev = pos.copy()
        if term or trunc:
            break
    env.close()

    ws = float(np.hypot(*wind))
    rec = {
        "model": model or "baseline", "cfg": cfg_name, "seed": seed,
        "intervention": task.get("intervention", "none"),
        "outcome": outcome_bucket(info),
        "miss_m": float(info.get("dist_home", float("nan"))),
        "quality": float(info.get("quality", 0.0)),
        "steps": steps,
        "wind_speed": ws,
        "wind_ne": wind.tolist(),
        "launch_ne": launch.tolist(),
        "launch_bearing_home_deg": math.degrees(math.atan2(-launch[1], -launch[0])),
        "touchdown_ne": pos.tolist(),
        "mean_track_err_deg": math.degrees(float(np.mean(track_err))) if track_err else float("nan"),
        "components": {k: float(v) for k, v in info["episode_reward_components"].items()},
    }
    if ws > 0.2:
        u = wind / ws                                   # direction the air moves toward
        rec["td_along_m"] = float(pos @ u)              # + : landed downwind of home
        rec["td_cross_m"] = float(pos[0] * u[1] - pos[1] * u[0])
    if mode == "log_obs":
        rec["stacked_obs"] = np.stack(logged).tolist()
    if mode == "d5":
        rec["stacked_obs"] = {"v1": d5_v1, "bank_deg": d5_bank}
    if mode == "critic":
        with torch.no_grad():
            v = actor.model.policy.predict_values(
                torch.as_tensor(np.stack(xs), dtype=torch.float32)).numpy().ravel()
        gamma = float(actor.model.gamma)
        g, ret = 0.0, np.zeros(len(rs))
        for t in range(len(rs) - 1, -1, -1):
            g = rs[t] + gamma * g
            ret[t] = g
        idx = {f: min(int(f * len(rs)), len(rs) - 1) for f in (0.0, 0.25, 0.5, 0.75)}
        rec["critic"] = {str(f): {"value": float(v[i]), "return": float(ret[i])}
                         for f, i in idx.items()}
        rec["truncated"] = bool(trunc and not term)
    return rec


def run_tasks(tasks: list[dict], workers: int) -> list[dict]:
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        out = pool.map(rollout, tasks, chunksize=4)
    print(f"  {len(tasks)} episodes in {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _wilson(k, n):
    from training.benchmark import wilson
    return list(wilson(k, n))


def _mean_ci(x) -> dict:
    x = np.asarray([v for v in x if v == v], dtype=float)
    if len(x) < 2:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": int(len(x))}
    from scipy import stats
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    h = float(stats.t.ppf(0.975, len(x) - 1)) * se
    return {"mean": m, "ci95": [m - h, m + h], "n": int(len(x))}


def summarise(eps: list[dict]) -> dict:
    n = len(eps)
    k = sum(e["outcome"] == "success" for e in eps)
    out = {
        "n": n,
        "success": k / n, "success_wilson95": _wilson(k, n),
        "crash": sum(e["outcome"] == "crash" for e in eps) / n,
        "median_miss_m": float(np.median([e["miss_m"] for e in eps])),
        "mean_track_err_deg": _mean_ci([e["mean_track_err_deg"] for e in eps]),
    }
    w = [e for e in eps if "td_along_m" in e]     # episodes with a defined wind direction
    if w:
        out["n_with_wind_frame"] = len(w)
        out["mean_abs_cross_m"] = _mean_ci([abs(e["td_cross_m"]) for e in w])
        out["mean_signed_along_m"] = _mean_ci([e["td_along_m"] for e in w])
        out["mean_abs_along_m"] = _mean_ci([abs(e["td_along_m"]) for e in w])
    return out


def paired(a: list[dict], b: list[dict]) -> dict:
    from training.benchmark import mcnemar_exact, paired_table
    pa = [{"seed": e["seed"], "outcome": e["outcome"]} for e in a]
    pb = [{"seed": e["seed"], "outcome": e["outcome"]} for e in b]
    t = paired_table(pa, pb)
    return {"table": t, "mcnemar_exact_p": mcnemar_exact(t),
            "layout": "[[both_ok, first_only], [second_only, both_fail]]"}


def _seeds(n: int) -> list[int]:
    from training.benchmark import episode_seeds
    return episode_seeds(n, 0)


def _write(name: str, model: str | None, payload: dict) -> str:
    from training.benchmark import DEFAULT_CONFIG, provenance
    m = MODELS.get(model) if model else None
    payload = {"provenance": provenance(DEFAULT_CONFIG,
                                        os.path.join(ROOT, m[0]) if m else None,
                                        os.path.join(ROOT, m[1]) if m else None),
               **payload}
    path = os.path.join(ROOT, "results", f"wind_{name}{'_' + model if model else ''}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"  wrote {os.path.relpath(path, ROOT)}", flush=True)
    return path


def _strip(eps):
    return [{k: v for k, v in e.items() if k != "stacked_obs"} for e in eps]


# ---------------------------------------------------------------------------
# D1. Observation swap
# ---------------------------------------------------------------------------

D1_CONDITIONS = [
    # label, config, intervention
    ("a_control", "steady_wind", "none"),
    ("b_yaw_from_course", "steady_wind", "yaw_from_course"),
    ("c_course_from_yaw", "steady_wind", "course_from_yaw"),
    ("d_calm_control", "calm", "none"),
    ("d_calm_yaw_bias_15", "calm", "yaw_bias_15"),
]


def d1(model: str, n: int, workers: int) -> dict:
    seeds = _seeds(n)
    tasks = [{"model": model, "cfg": c, "seed": s, "intervention": i}
             for _, c, i in D1_CONDITIONS for s in seeds]
    tasks += [{"model": None, "cfg": c, "seed": s} for c in ("steady_wind", "calm") for s in seeds]
    res = run_tasks(tasks, workers)
    by = {}
    for label, c, i in D1_CONDITIONS:
        by[label] = [e for e in res if e["model"] == model and e["cfg"] == c and e["intervention"] == i]
    base = {c: [e for e in res if e["model"] == "baseline" and e["cfg"] == c]
            for c in ("steady_wind", "calm")}

    out = {"conditions": {}, "baseline": {c: summarise(v) for c, v in base.items()}}
    for label, c, _ in D1_CONDITIONS:
        s = summarise(by[label])
        ref = "a_control" if c == "steady_wind" else "d_calm_control"
        if label != ref:
            s["vs_control"] = paired(by[label], by[ref])
        s["vs_baseline"] = paired(by[label], base[c])
        out["conditions"][label] = s

    bx = out["baseline"]["steady_wind"]["mean_abs_cross_m"]["mean"]
    ax = out["conditions"]["a_control"]["mean_abs_cross_m"]["mean"]
    bbx = out["conditions"]["b_yaw_from_course"]["mean_abs_cross_m"]["mean"]
    excess = ax - bx
    out["cross_wind_excess_m"] = {"control": excess, "yaw_from_course": bbx - bx,
                                  "fraction_recovered": (excess - (bbx - bx)) / excess
                                  if excess > 0 else float("nan")}
    dc = out["conditions"]["d_calm_control"]["mean_track_err_deg"]["mean"]
    db = out["conditions"]["d_calm_yaw_bias_15"]["mean_track_err_deg"]["mean"]
    shift = [b["mean_track_err_deg"] - a["mean_track_err_deg"]
             for a, b in zip(by["d_calm_control"], by["d_calm_yaw_bias_15"])]
    out["yaw_bias_track_rotation_deg"] = {
        "shift": db - dc,
        "paired_shift": _mean_ci(shift),
        "reading": "about +15 if the policy steers its heading at home; about 0 if it steers its ground track",
    }
    out["episodes"] = {k: _strip(v) for k, v in by.items()}
    out["episodes"]["baseline"] = {c: _strip(v) for c, v in base.items()}
    _write("d1", model, {"description": "D1 observation swap", "n_per_condition": n, **out})
    return out


# ---------------------------------------------------------------------------
# D2. Input sensitivity
# ---------------------------------------------------------------------------

def d2(model: str, n_obs: int, workers: int) -> dict:
    import torch

    seeds = _seeds(40)
    out = {"description": "D2 input sensitivity: mean |d bank_mean / d raw input| x input std, "
                          "summed over the 8 stacked frames, as a share of all channels",
           "target_obs": n_obs, "by_config": {}}
    actor = _actor(model)
    pol = actor.model.policy
    vn = actor.vecnorm
    mean = torch.as_tensor(vn.obs_rms.mean, dtype=torch.float32)
    std_run = torch.as_tensor(np.sqrt(vn.obs_rms.var + vn.epsilon), dtype=torch.float32)
    clip = float(vn.clip_obs)
    for cfg in ("calm", "steady_wind"):
        res = run_tasks([{"model": model, "cfg": cfg, "seed": s, "mode": "log_obs"} for s in seeds],
                        workers)
        X = np.concatenate([np.asarray(e["stacked_obs"], dtype=np.float32) for e in res])
        rng = np.random.default_rng(0)
        if len(X) > n_obs:
            X = X[rng.choice(len(X), n_obs, replace=False)]
        x = torch.tensor(X, requires_grad=True)
        xn = torch.clamp((x - mean) / std_run, -clip, clip)
        bank = pol.get_distribution(xn).distribution.mean[:, 0]
        bank.sum().backward()
        g = x.grad.numpy()
        contrib = np.abs(g).mean(axis=0) * X.std(axis=0)
        n_stack = X.shape[1] // OBS_DIM
        per = contrib.reshape(n_stack, OBS_DIM)        # [frame (oldest first), channel]
        ch = per.sum(axis=0)
        share = ch / ch.sum()
        out["by_config"][cfg] = {
            "n_obs": int(len(X)),
            "share": {CHANNELS[i]: float(share[i]) for i in range(OBS_DIM)},
            "share_by_frame_newest_last": {c: (per[:, CHANNELS.index(c)] / ch.sum()).tolist()
                                           for c in ("yaw", "course", "dx_home", "dy_home")},
            "yaw_over_course": float(ch[YAW] / ch[COURSE]),
            "clipped_input_fraction": float((xn.detach().abs() >= clip).float().mean()),
        }
    _write("d2", model, out)
    return out


# ---------------------------------------------------------------------------
# D3. Seam and rotation dependence (existing benchmark logs)
# ---------------------------------------------------------------------------

D3_LOGS = {
    "v22": "results/benchmark_v22_best_15M.json",
    "v21": "results/benchmark_v21_best_3.5M.json",
}


def _sector_table(eps: list[dict]) -> dict:
    from scipy.stats import chi2_contingency
    rows = []
    succ = np.zeros((2, 8), dtype=int)
    for k in range(8):
        lo = -180 + 45 * k
        sel = [e for e in eps if lo <= e["b"] < lo + 45 or (k == 7 and e["b"] == 180)]
        ks = sum(e["outcome"] == "success" for e in sel)
        succ[:, k] = [ks, len(sel) - ks]
        rows.append({"sector_deg": [lo, lo + 45], "n": len(sel),
                     "success": ks / len(sel) if sel else float("nan"),
                     "success_wilson95": _wilson(ks, len(sel)) if sel else None,
                     "median_miss_m": float(np.median([e["miss"] for e in sel])) if sel else float("nan")})
    keep = succ.sum(axis=0) > 0
    chi = chi2_contingency(succ[:, keep]) if succ[0].sum() and succ[1].sum() else None
    seam = [e for e in eps if abs(e["b"]) >= 135]
    rest = [e for e in eps if abs(e["b"]) < 135]
    return {"sectors": rows,
            "chi2_homogeneity_p": float(chi[1]) if chi else float("nan"),
            "seam_sectors_success": sum(e["outcome"] == "success" for e in seam) / max(len(seam), 1),
            "seam_n": len(seam),
            "other_sectors_success": sum(e["outcome"] == "success" for e in rest) / max(len(rest), 1),
            "seam_median_miss_m": float(np.median([e["miss"] for e in seam])) if seam else float("nan"),
            "other_median_miss_m": float(np.median([e["miss"] for e in rest])) if rest else float("nan")}


def d3(model: str) -> dict:
    with open(os.path.join(ROOT, D3_LOGS[model]), encoding="utf-8") as f:
        log = json.load(f)
    stage0 = log["stages"]["0"]["episodes"]
    out = {"description": "D3 calm-air (stage 0) success and miss by launch bearing-to-home, "
                          "eight 45 deg sectors; sectors 1 and 8 border the +/-180 deg seam",
           "source": D3_LOGS[model]}
    for who in ("policy", "baseline"):
        eps = [{"b": math.degrees(math.atan2(-e["launch_state"][1], -e["launch_state"][0])),
                "outcome": e["outcome"], "miss": e["miss_distance_m"]} for e in stage0[who]]
        out[who] = _sector_table(eps)
    _write("d3", model, out)
    return out


# ---------------------------------------------------------------------------
# D4. Drift and r_path accounting
# ---------------------------------------------------------------------------

COMPONENTS = ["r_terminal", "r_path", "penalty_unreach", "penalty_stall", "penalty_bank",
              "penalty_smooth", "penalty_truncate"]


def _budget(eps: list[dict]) -> dict:
    tot = {c: float(np.mean([abs(e["components"].get(c, 0.0)) for e in eps])) for c in COMPONENTS}
    s = sum(tot.values())
    return {c: v / s for c, v in tot.items()}


def _training_budget(run_log: str) -> dict:
    by: dict[int, list] = {}
    with open(os.path.join(ROOT, run_log), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("reward_components"):
                by.setdefault(int(r["stage"]), []).append(r["reward_components"])
    out = {}
    for st, rows in sorted(by.items()):
        tot = {c: float(np.mean([abs(x.get(c, 0.0)) for x in rows])) for c in COMPONENTS}
        s = sum(tot.values())
        out[str(st)] = {c: v / s for c, v in tot.items()}
    return out


TRAIN_LOGS = {"v22": "data/runs/20261112T013358Z_seed0.jsonl"}


def d4(model: str, n: int, workers: int, d1_out: dict | None = None) -> dict:
    from scipy import stats
    seeds = _seeds(n)
    cfgs = ["steady_wind", "stage1", "stage2", "stage3"]
    tasks = [{"model": m, "cfg": c, "seed": s} for m in (model, None) for c in cfgs for s in seeds]
    res = run_tasks(tasks, workers)
    out = {"description": "D4 drift and r_path accounting. along: touchdown position along the "
                          "mean wind, downwind positive", "by_config": {}}
    for c in cfgs:
        pol = [e for e in res if e["model"] == model and e["cfg"] == c]
        base = [e for e in res if e["model"] == "baseline" and e["cfg"] == c]
        diff = [p["td_along_m"] - b["td_along_m"] for p, b in zip(pol, base)
                if "td_along_m" in p and "td_along_m" in b]
        rec = {}
        for who, eps in (("policy", pol), ("baseline", base)):
            w = [e for e in eps if "td_along_m" in e]
            ws = np.array([e["wind_speed"] for e in eps])
            rp = np.array([e["components"]["r_path"] for e in eps])
            rps = rp / (np.array([e["steps"] for e in eps]) * DT_RL)
            lr = stats.linregress(ws, rp)
            lrs = stats.linregress(ws, rps)
            rec[who] = {
                "success": sum(e["outcome"] == "success" for e in eps) / len(eps),
                "signed_along_m": _mean_ci([e["td_along_m"] for e in w]),
                "abs_cross_m": _mean_ci([abs(e["td_cross_m"]) for e in w]),
                "r_path_per_episode": _mean_ci(rp),
                "r_path_vs_wind": {"slope_per_ms": float(lr.slope), "intercept": float(lr.intercept),
                                   "r": float(lr.rvalue), "p": float(lr.pvalue)},
                "r_path_per_second_vs_wind": {"slope_per_ms": float(lrs.slope),
                                              "intercept": float(lrs.intercept),
                                              "r": float(lrs.rvalue), "p": float(lrs.pvalue)},
                "reward_budget_share": _budget(eps),
            }
        rec["paired_along_diff_policy_minus_baseline_m"] = _mean_ci(diff)
        out["by_config"][c] = rec
    if model in TRAIN_LOGS:
        out["training_reward_budget_share_by_stage"] = _training_budget(TRAIN_LOGS[model])
        out["training_log"] = TRAIN_LOGS[model]
    out["episodes"] = _strip(res)
    _write("d4", model, out)
    return out


# ---------------------------------------------------------------------------
# D5. Supervised representability
# ---------------------------------------------------------------------------

def _stack(frames: np.ndarray, n_stack: int) -> np.ndarray:
    """Per-step stacked inputs as PolicyActor builds them: zeros before the
    first frame, oldest first."""
    pad = np.zeros((n_stack - 1, frames.shape[1]), dtype=np.float32)
    f = np.concatenate([pad, frames])
    return np.stack([f[t:t + n_stack].ravel() for t in range(len(frames))])


def _fit_mlp(Xtr, Ytr, Xte, Yte, epochs: int, seed: int = 0) -> np.ndarray:
    """The policy's architecture ([128, 128], tanh) fitted by MSE on
    z-scored inputs and targets; returns held-out predictions."""
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.as_tensor((Ytr - ymu) / ysd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 128), torch.nn.Tanh(),
                              torch.nn.Linear(128, 128), torch.nn.Tanh(),
                              torch.nn.Linear(128, yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=g).split(256):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        p = net(torch.as_tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()
    return p * ysd + ymu


def _r2(y, p) -> list[float]:
    return [float(1 - ((y[:, k] - p[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum())
            for k in range(y.shape[1])]


def d5(n: int, workers: int, epochs: int = 30, n_stack: int = 8) -> dict:
    """Baseline flights in steady wind, observation v1 and v2 logged side by
    side with the baseline's bank command; the same MLP fitted to each."""
    from env.nav_features import V1, obs_v2_from_v1
    seeds = _seeds(n)
    res = run_tasks([{"model": None, "cfg": "steady_wind", "seed": s, "mode": "d5"} for s in seeds],
                    workers)
    data = {k: [] for k in ("v1", "v2", "v2_no_prev", "bank", "wind_ne", "wind_track", "ep")}
    for i, e in enumerate(res):
        v1 = np.asarray(e["stacked_obs"]["v1"], dtype=np.float32)
        bank = np.asarray(e["stacked_obs"]["bank_deg"], dtype=np.float32)
        prev = np.zeros((len(v1), 2), dtype=np.float32)
        prev[1:, 0] = bank[:-1] / 45.0
        # previous speed action is not logged by the baseline actor; the
        # baseline's bank is what is predicted, so only prev bank matters here
        v2 = np.stack([obs_v2_from_v1(v1[t], prev[t]) for t in range(len(v1))])
        w = np.asarray(e["wind_ne"], dtype=np.float32)
        c = v1[:, V1["course"]]
        tn, te = np.cos(c), np.sin(c)
        data["v1"].append(_stack(v1, n_stack))
        data["v2"].append(_stack(v2, n_stack))
        data["v2_no_prev"].append(_stack(v2[:, :-2], n_stack))
        data["bank"].append(bank[:, None])
        data["wind_ne"].append(np.tile(w, (len(v1), 1)))
        data["wind_track"].append(np.stack([w[0] * tn + w[1] * te, w[1] * tn - w[0] * te], 1))
        data["ep"].append(np.full(len(v1), i))
    D = {k: np.concatenate(v) for k, v in data.items()}
    test_eps = set(range(0, n, 5))                     # every fifth episode held out
    te = np.isin(D["ep"], list(test_eps))
    tr = ~te
    out = {"description": "D5 supervised representability: the policy's MLP shape fitted to "
                          "the baseline's bank command (degrees) and to the true wind, from "
                          "8 stacked frames of each observation version; held-out episodes",
           "n_episodes": n, "held_out_episodes": len(test_eps), "epochs": epochs,
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "bank_std_deg": float(D["bank"][te].std()),
           "bank_mae_deg": {}, "bank_r2": {}, "wind": {}}
    for name in ("v1", "v2", "v2_no_prev"):
        p = _fit_mlp(D[name][tr], D["bank"][tr], D[name][te], D["bank"][te], epochs)
        out["bank_mae_deg"][name] = float(np.abs(p - D["bank"][te]).mean())
        out["bank_r2"][name] = _r2(D["bank"][te], p)[0]
        print(f"  bank from {name}: MAE {out['bank_mae_deg'][name]:.2f} deg", flush=True)
    for name, target in (("v1", "wind_ne"), ("v1", "wind_track"), ("v2_no_prev", "wind_track")):
        p = _fit_mlp(D[name][tr], D[target][tr], D[name][te], D[target][te], epochs)
        out["wind"][f"{target}_from_{name}"] = {
            "r2": _r2(D[target][te], p),
            "mae_ms": np.abs(p - D[target][te]).mean(0).tolist(),
            "components": ["north", "east"] if target == "wind_ne" else ["tailwind", "crosswind_toward_right"]}
        print(f"  {target} from {name}: R2 {out['wind'][f'{target}_from_{name}']['r2']}", flush=True)
    _write("d5", None, out)
    return out


# ---------------------------------------------------------------------------
# D6. Critic calibration
# ---------------------------------------------------------------------------

def d6(model: str, n: int, workers: int) -> dict:
    seeds = _seeds(n)
    cfgs = ["calm", "steady_wind", "stage1"]
    res = run_tasks([{"model": model, "cfg": c, "seed": s, "mode": "critic"}
                     for c in cfgs for s in seeds], workers)
    out = {"description": "D6 critic calibration: V(s_t) against the realised discounted return "
                          "(learner reward units, the model's gamma), stochastic policy as trained, "
                          "at 0/25/50/75% of each episode",
           "by_config": {}}
    for c in cfgs:
        eps = [e for e in res if e["cfg"] == c]
        rec = {"n": len(eps), "truncated": sum(e["truncated"] for e in eps),
               "success": sum(e["outcome"] == "success" for e in eps) / len(eps)}
        allv, allg = [], []
        for f in ("0.0", "0.25", "0.5", "0.75"):
            v = np.array([e["critic"][f]["value"] for e in eps])
            g = np.array([e["critic"][f]["return"] for e in eps])
            allv += list(v)
            allg += list(g)
            ss = float(((g - g.mean()) ** 2).sum())
            rec[f] = {"r2": 1.0 - float(((g - v) ** 2).sum()) / ss if ss > 0 else float("nan"),
                      "corr": float(np.corrcoef(v, g)[0, 1]),
                      "mean_value": float(v.mean()), "mean_return": float(g.mean()),
                      "return_std": float(g.std())}
        v, g = np.array(allv), np.array(allg)
        rec["pooled_r2"] = 1.0 - float(((g - v) ** 2).sum()) / float(((g - g.mean()) ** 2).sum())
        out["by_config"][c] = rec
    _write("d6", model, out)
    return out


# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("test", choices=("d1", "d2", "d3", "d4", "d5", "d6", "all"))
    ap.add_argument("--model", action="append", choices=sorted(MODELS), default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--d2-obs", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    if a.test == "d5":                  # baseline flights only: no model
        d5(200, a.workers)
        return
    for model in a.model or ["v22"]:
        print(f"== {model}", flush=True)
        tests = ["d3", "d1", "d2", "d4", "d6"] if a.test == "all" else [a.test]
        for t in tests:
            print(f"-- {t}", flush=True)
            if t == "d1":
                d1(model, a.episodes, a.workers)
            elif t == "d2":
                d2(model, a.d2_obs, a.workers)
            elif t == "d3":
                d3(model)
            elif t == "d4":
                d4(model, a.episodes, a.workers)
            elif t == "d6":
                d6(model, a.episodes, a.workers)


if __name__ == "__main__":
    main()

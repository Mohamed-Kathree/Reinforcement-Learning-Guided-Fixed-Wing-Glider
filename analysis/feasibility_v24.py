"""
feasibility_v24.py
==================
Per-seed energy feasibility of the return-to-launch task on the 100 paired
benchmark seeds (episode_seeds(100, 0)), stages 0-3. Nothing trains; single
process, light on memory.

For every (stage, seed) the launch state, the mean wind and the randomised
airframe are read from the environment exactly as the benchmark builds it
(training.benchmark.make_env + env.reset(seed)). For that airframe the trimmed
glide polar is built from JSBSim's own aerodynamics (elevator trimmed to zero
pitching moment at each alpha, alpha below the stall, airspeed inside the
commandable 9.5-16 m/s band). Then:

  best ground glide ratio toward home, G = max over the polar of
      (sqrt(V^2 - w_cross^2) + w_along) / (sink(V) + w_down)
  (optimal speed-to-fly with a crab to cancel the crosswind)

  height available = launch height
                   + (V_launch^2 - V_best^2) / 2g        kinetic-energy credit
                   - height lost turning from the launch heading onto the
                     course home, coordinated turn at 30 deg bank
                     (penalty-free limit) at the best-glide speed

  reach margin = G * height available / (distance home - R_success)

margin < 1: home is out of reach even for a perfect pilot (straight line,
no gusts, no wasted energy). The model is optimistic on purpose (gusts, the
flare and the turn's ground track are ignored), so "infeasible" is a firm
claim and "feasible" only a necessary condition. Validation: no seed the
model calls infeasible may have been landed successfully by any controller
in results/ (checked and reported).

    python analysis/feasibility_v24.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

G0 = 9.80665
BANK_DEG = 30.0
R_SUCCESS = 20.0   # fallback only; the stage's own R_home_m (25/22/20/20) is read from the env


def _polar(params: dict) -> list[tuple[float, float, float]]:
    """[(alpha_deg, CL, CD)] trimmed, pre-stall, for one airframe."""
    from validate_glide import (DT, MS2FPS, Airframe, _airframe_fdm, stall_angle_deg)
    af = Airframe(params=params)
    alphas = np.arange(0.0, stall_angle_deg(params) - 0.25, 0.25)

    def sweep(e):
        f = _airframe_fdm(af, e)
        f["ic/vt-fps"] = 12.0 * MS2FPS
        f["ic/beta-deg"] = 0.0
        for k in ("p", "q", "r"):
            f[f"ic/{k}-rad_sec"] = 0.0
        out = []
        for a in alphas:
            f["ic/alpha-deg"] = float(a)
            f.run_ic()
            qs = f["aero/qbar-psf"] * f["metrics/Sw-sqft"]
            out.append((f["moments/m-aero-lbsft"], f["forces/fwz-aero-lbs"] / qs,
                        f["forces/fwx-aero-lbs"] / qs))
        return out
    e0, e1 = 0.0, 0.2
    s0, s1 = sweep(e0), sweep(e1)
    pol = []
    for a, (m0, cl0, cd0), (m1, cl1, cd1) in zip(alphas, s0, s1):
        if m1 == m0:
            continue
        t = -m0 / (m1 - m0)            # moment, CL, CD affine in elevator
        e = e0 + t * (e1 - e0)
        if -1.0 <= e <= 1.0:
            pol.append((float(a), cl0 + t * (cl1 - cl0), cd0 + t * (cd1 - cd0)))
    return pol


def _glide_points(pol, mass, s_m2, rho):
    """[(V, sink)] steady glide, airspeed inside the commandable band."""
    from env.glider_env import SPEED_CMD_MAX_MS, SPEED_CMD_MIN_MS
    pts = []
    for _, cl, cd in pol:
        if cl <= 0 or cd <= 0:
            continue
        gam = math.atan(cd / cl)
        v = math.sqrt(2 * mass * G0 * math.cos(gam) / (rho * s_m2 * cl))
        if SPEED_CMD_MIN_MS <= v <= SPEED_CMD_MAX_MS:
            pts.append((v, v * math.sin(gam)))
    return pts


def _case(cfg, stage, seed):
    from training.benchmark import make_env
    env = make_env(cfg, stage, 1)
    try:
        env.reset(seed=seed)
        st = env._fdm.state.copy()
        dr = dict(env._dr)
        w = env._wind.mean_ned
        gust = float(env._wind._intensity)
        r_ok = float(env.cfg.get("R_home_m", R_SUCCESS))
        f = env._fdm
        fd = f._fdm if hasattr(f, "_fdm") else None
        rho = float(fd["atmosphere/rho-slugs_ft3"]) * 515.378818 if fd is not None else 1.225
        s_m2 = float(fd["metrics/Sw-sqft"]) * 0.09290304 if fd is not None else None
        mass = float(fd["inertia/weight-lbs"]) * 0.45359237 if fd is not None else dr["mass_kg"]
        v0 = float(fd["velocities/vt-fps"]) * 0.3048 if fd is not None else float(st[3])
    finally:
        env.close()
    return st, dr, w, gust, rho, s_m2, mass, v0, r_ok


def analyse(cfg, stage, seed):
    st, dr, w, gust, rho, s_m2, mass, v0, r_ok = _case(cfg, stage, seed)
    pts = _glide_points(_polar(dr), mass, s_m2, rho)
    n, e, d = st[0], st[1], st[2]
    h = -d
    dist = math.hypot(n, e)
    u = np.array([-n, -e]) / dist                    # unit vector to home
    wh = np.array(w[:2])
    w_al = float(wh @ u)                             # + = tailwind toward home
    w_cr = float(abs(wh[0] * u[1] - wh[1] * u[0]))
    w_dn = float(w[2])
    best = None
    for v, s in pts:
        if v <= w_cr:
            continue
        gs = math.sqrt(v * v - w_cr * w_cr) + w_al
        sink = s + w_dn
        if gs <= 0 or sink <= 0:
            continue
        g = gs / sink
        if best is None or g > best[0]:
            best = (g, v, s, gs)
    still_ld = max(v / s for v, s in pts)
    # launch heading (body x) from the quaternion, course home with crab
    q0, q1, q2, q3 = st[6:10]
    psi = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    if best is None:
        return {"stage": stage, "seed": seed, "reachable": False, "margin": 0.0,
                "dist_m": dist, "h_m": h, "wind_ms": float(np.hypot(*wh)), "w_along": w_al,
                "w_cross": w_cr, "still_air_LD": still_ld, "note": "cannot make ground speed home"}
    g, v, s, gs = best
    crab = math.asin(min(1.0, w_cr / v))
    course = math.atan2(u[1], u[0])
    side = math.copysign(1.0, wh[0] * u[1] - wh[1] * u[0])
    heading_req = course + side * crab
    dpsi = abs((heading_req - psi + math.pi) % (2 * math.pi) - math.pi)
    phi = math.radians(BANK_DEG)
    nload = 1 / math.cos(phi)
    h_turn = (s * nload ** 1.5 + w_dn) * dpsi / (G0 * math.tan(phi) / v)
    h_ke = (v0 * v0 - v * v) / (2 * G0)
    h_av = h + h_ke - h_turn
    need = max(dist - r_ok, 1e-6)
    return {"stage": stage, "seed": seed, "dist_m": dist, "h_m": h, "v_launch": v0,
            "wind_ms": float(np.hypot(*wh)), "w_along": w_al, "w_cross": w_cr, "w_down": w_dn,
            "gust_sigma": gust, "R_success_m": r_ok, "still_air_LD": still_ld, "best_ground_LD": g, "v_best": v,
            "turn_deg": math.degrees(dpsi), "h_turn_m": h_turn, "h_ke_m": h_ke,
            "h_available_m": h_av, "reach_m": g * h_av, "margin": g * h_av / need,
            "reachable": g * h_av >= need}


def controller_successes():
    succ: dict = {}
    for f in glob.glob(os.path.join(ROOT, "results", "benchmark_v2*.json")):
        d = json.load(open(f))
        for st, s in d.get("stages", {}).items():
            for who in ("policy", "baseline"):
                for ep in s["episodes"].get(who, []):
                    if ep["outcome"] == "success":
                        succ.setdefault((int(st), ep["seed"]), set()).add(os.path.basename(f) + ":" + who)
    raw = os.path.join(ROOT, "results", "gated_v24_benchmark_raw.json")
    if os.path.exists(raw):
        for r in json.load(open(raw)):
            if r["outcome"] == "success":
                succ.setdefault((r["stage"], r["seed"]), set()).add(f"gated:{r['kind']}:{r['arg']}")
    return succ


def main():
    from training.benchmark import episode_seeds, load_config
    cfg = load_config(os.path.join(ROOT, "training", "configs", "v22_restored.yaml"))
    seeds = episode_seeds(100, 0)
    succ = controller_successes()
    rows = []
    for stage in range(4):
        for s in seeds:
            r = analyse(cfg, stage, s)
            r["any_controller_success"] = (stage, s) in succ
            r["n_controllers_success"] = len(succ.get((stage, s), ()))
            rows.append(r)
        print(f"stage {stage} done", flush=True)
    summary = {}
    for stage in range(4):
        rs = [r for r in rows if r["stage"] == stage]
        m = np.array([r["margin"] for r in rs])
        reach = [r for r in rs if r["reachable"]]
        summary[stage] = {
            "n": len(rs),
            "reachable": len(reach),
            "unreachable": len(rs) - len(reach),
            "marginal_1_to_1.25": int(np.sum((m >= 1) & (m < 1.25))),
            "margin_median": float(np.median(m)),
            "margin_p10": float(np.percentile(m, 10)),
            "still_air_LD_range": [float(min(r["still_air_LD"] for r in rs)),
                                   float(max(r["still_air_LD"] for r in rs))],
            "any_controller_success": sum(r["any_controller_success"] for r in rs),
            "success_on_unreachable_seed": sum(r["any_controller_success"] and not r["reachable"]
                                               for r in rs),
            "reachable_but_never_landed": sum(r["reachable"] and not r["any_controller_success"]
                                              for r in rs),
        }
    out = {"seed_set": "episode_seeds(100, 0)", "bank_deg": BANK_DEG, "R_success_m": R_SUCCESS,
           "model": __doc__.strip().splitlines()[0], "summary": summary, "rows": rows}
    path = os.path.join(ROOT, "results", "feasibility_v24.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    for st, v in summary.items():
        print(f"stage {st}: {v}")
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

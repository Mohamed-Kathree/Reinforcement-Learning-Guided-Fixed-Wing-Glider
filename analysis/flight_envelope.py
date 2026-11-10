"""
flight_envelope.py
==================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Steady-glide envelope of the domain-randomised airframes, computed from the
aircraft model itself (lift and drag read back from JSBSim with the sampled
parameters applied through JSBSimFDM.apply_domain_rand(), the same path the
environment uses) rather than from hand-copied coefficients.

Used for:
    * the airspeed command floor: it must exceed the stall speed of the
      worst airframe in DR_RANGES (heaviest, lowest lift peak, smallest wing)
    * the episode truncation limit: sized from the LONGEST possible flight
      (lightest, cleanest airframe at its minimum sink within the command
      range), so the best airframes are never cut off and the timeout
      bucket stays honest
    * the expected flight time used as the reward-shaping horizon

Steady glide: tan(gamma) = CD/CL, V = sqrt(2 W cos(gamma) / (rho S CL)),
sink = V sin(gamma). Elevator trim lift (CLde, about 1% of CL) and ground
effect are neglected.

Run:   python -m analysis.flight_envelope      (prints and writes analysis/flight_envelope.json)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from env.curriculum import DR_ONESIDED, DR_RANGES, STAGES  # noqa: E402
from sim.jsbsim_fdm import JSBSimFDM  # noqa: E402

RHO = 1.225          # kg/m^3, sea-level ISA (JSBSim's at these altitudes to <0.5%)
G = 9.81
FT2M = 0.3048
SLUG_FT3_TO_KG_M3 = 515.378818

REFERENCE_HEIGHT_M = 30.0
OUT_JSON = os.path.join(PROJ_ROOT, "analysis", "flight_envelope.json")


@dataclass
class Polar:
    clmax: float
    alpha_stall_deg: float
    v_stall_ms: float
    v: np.ndarray       # pre-stall steady-glide speeds, m/s (descending with alpha)
    sink: np.ndarray    # m/s
    ld: np.ndarray


def nominal_params() -> dict:
    """Midpoint of every two-sided range, lower bound of every one-sided one."""
    p = {k: 0.5 * (lo + hi) for k, (lo, hi) in DR_RANGES.items()}
    p.update({k: lo for k, (lo, hi) in DR_ONESIDED.items()})
    return p


def polar(params: dict, fdm: JSBSimFDM | None = None) -> Polar:
    """Steady-glide polar of one airframe (keys as in DR_RANGES)."""
    fdm = fdm or JSBSimFDM()
    state = np.zeros(13)
    state[2] = -300.0          # 300 m AGL: out of ground effect
    state[3] = 12.0
    state[6] = 1.0
    fdm.reset(state)
    fdm.apply_domain_rand(params)
    f = fdm._fdm

    weight_n = f["inertia/weight-lbs"] * 4.4482216
    s_m2 = f["metrics/Sw-sqft"] * FT2M**2
    rho = f["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3

    rows = []
    for alpha in np.arange(0.5, 24.01, 0.05):
        f["ic/alpha-deg"] = alpha
        f["ic/beta-deg"] = 0.0
        f.run_ic()
        f.run()
        qs = f["aero/qbar-psf"] * f["metrics/Sw-sqft"]
        cl = f["aero/coefficient/CLtotal"]
        cd = (f["aero/coefficient/CD0"] + f["aero/coefficient/CDi"]) / qs
        rows.append((f["aero/alpha-deg"], cl, cd))
    a, cl, cd = np.array(rows).T
    i_st = int(np.argmax(cl))
    clmax = float(cl[i_st])
    pre = slice(0, i_st + 1)
    gam = np.arctan(cd[pre] / cl[pre])
    v = np.sqrt(2 * weight_n * np.cos(gam) / (rho * s_m2 * cl[pre]))
    return Polar(
        clmax=clmax,
        alpha_stall_deg=float(a[i_st]),
        v_stall_ms=float(np.sqrt(2 * weight_n / (rho * s_m2 * clmax))),
        v=v, sink=v * np.sin(gam), ld=cl[pre] / cd[pre],
    )


def _corner(**overrides) -> dict:
    p = nominal_params()
    p.update(overrides)
    return p


def worst_stall_airframe() -> dict:
    """Heaviest, lowest lift peak, smallest wing. cl_mult is included at
    both ends to show it no longer moves the peak."""
    return _corner(mass_kg=DR_RANGES["mass_kg"][1],
                   clmax_scale=DR_RANGES["clmax_scale"][0],
                   sw_mult=DR_RANGES["sw_mult"][0])


def longest_flight_airframe() -> dict:
    """Lightest and cleanest: low mass, no added drag, lowest induced drag,
    highest lift peak, largest wing."""
    return _corner(mass_kg=DR_RANGES["mass_kg"][0], cd0_add=DR_ONESIDED["cd0_add"][0],
                   k_induced=DR_RANGES["k_induced"][0],
                   clmax_scale=DR_RANGES["clmax_scale"][1], sw_mult=DR_RANGES["sw_mult"][1])


def shortest_flight_airframe() -> dict:
    """Heaviest and draggiest: high mass, full added drag, highest induced
    drag, lowest lift peak, smallest wing."""
    return _corner(mass_kg=DR_RANGES["mass_kg"][1], cd0_add=DR_ONESIDED["cd0_add"][1],
                   k_induced=DR_RANGES["k_induced"][1],
                   clmax_scale=DR_RANGES["clmax_scale"][0], sw_mult=DR_RANGES["sw_mult"][0])


def max_start_energy_height_m(v_glide_ms: float) -> float:
    """Highest start in any stage: alt0 plus launch kinetic energy above the
    glide speed (the env's launch-speed ceiling is a fixed 15 m/s)."""
    from env.glider_env import LAUNCH_SPEED_MAX_MS
    alt0 = max(float(s["alt0_m"]) for s in STAGES)
    return alt0 + max(LAUNCH_SPEED_MAX_MS**2 - v_glide_ms**2, 0.0) / (2 * G)


def within(p: Polar, v_min: float, v_max: float) -> np.ndarray:
    return (p.v >= v_min) & (p.v <= v_max)


@dataclass
class Envelope:
    v_cmd_min: float
    v_cmd_max: float
    worst_stall_ms: float
    worst_stall_cl_mult_spread_ms: float
    longest_minsink_ms: float
    longest_v_ms: float
    longest_t30_s: float
    longest_t_start_s: float
    start_energy_height_m: float
    shortest_bestld: float
    shortest_v_ms: float
    shortest_t30_s: float
    nominal_bestld: float
    nominal_v_bestld_ms: float
    nominal_t30_s: float


def compute(v_cmd_min: float, v_cmd_max: float) -> Envelope:
    fdm = JSBSimFDM()
    lo_cl, hi_cl = DR_RANGES["cl_mult"]
    stall = [polar({**worst_stall_airframe(), "cl_mult": m}, fdm).v_stall_ms for m in (lo_cl, 1.0, hi_cl)]

    lp = polar(longest_flight_airframe(), fdm)
    ok = within(lp, v_cmd_min, v_cmd_max)
    j = int(np.argmin(np.where(ok, lp.sink, np.inf)))
    h_start = max_start_energy_height_m(float(lp.v[j]))

    sp = polar(shortest_flight_airframe(), fdm)
    ok_s = within(sp, max(v_cmd_min, sp.v_stall_ms), v_cmd_max)
    k = int(np.argmax(np.where(ok_s, sp.ld, -np.inf)))

    npol = polar(nominal_params(), fdm)
    ok_n = within(npol, v_cmd_min, v_cmd_max)
    n = int(np.argmax(np.where(ok_n, npol.ld, -np.inf)))

    return Envelope(
        v_cmd_min=v_cmd_min, v_cmd_max=v_cmd_max,
        worst_stall_ms=max(stall), worst_stall_cl_mult_spread_ms=max(stall) - min(stall),
        longest_minsink_ms=float(lp.sink[j]), longest_v_ms=float(lp.v[j]),
        longest_t30_s=REFERENCE_HEIGHT_M / float(lp.sink[j]),
        longest_t_start_s=h_start / float(lp.sink[j]), start_energy_height_m=h_start,
        shortest_bestld=float(sp.ld[k]), shortest_v_ms=float(sp.v[k]),
        shortest_t30_s=REFERENCE_HEIGHT_M / float(sp.sink[k]),
        nominal_bestld=float(npol.ld[n]), nominal_v_bestld_ms=float(npol.v[n]),
        nominal_t30_s=REFERENCE_HEIGHT_M / float(npol.sink[n]),
    )


def main() -> None:
    from env.glider_env import SPEED_CMD_MAX_MS, SPEED_CMD_MIN_MS
    env = compute(SPEED_CMD_MIN_MS, SPEED_CMD_MAX_MS)
    rec = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(env).items()}
    rec["reference_height_m"] = REFERENCE_HEIGHT_M
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    for k, v in rec.items():
        print(f"{k:32s} {v}")
    print(f"wrote {os.path.relpath(OUT_JSON, PROJ_ROOT)}")


if __name__ == "__main__":
    main()

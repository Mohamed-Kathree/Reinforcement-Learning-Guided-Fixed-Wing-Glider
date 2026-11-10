"""
validate_glide.py  (JSBSim version — SPEC V20 Phase 5)
======================================================
Five physics gates that must all PASS before RL training begins, for the
aircraft as generated and (--sweep) at the corners of the domain-
randomisation table.

Gates:
    1. energy_conservation  -- zero drag: full mech. energy (KE_trans +
                               KE_rot + PE) constant within 0.1% over 10 s
    2. glide_ratio          -- steady glide ratio from NED velocity: 10 s
                               settling, then the median over 10 s
    3. stall                -- CL drops past the airframe's own stall angle
    4. trimmed_stability    -- trimmed IC: pitch rate < 0.1 rad/s, 15 s
    5. attitude_integrity   -- Euler angles finite + bounded over 30 s

Every gate starts from the airframe's OWN trim, found by find_trim(),
from JSBSim's aero pitching moment about the CG (so CG offset, pitch trim
bias and every stability multiplier are included), with the glide angle
from CD/CL there and the speed that balances lift and weight. Two modes:

    fixed     elevator command zero; trim is wherever the moment balances
              (the original gate definition)
    elevator  elevator command trimmed so the airframe balances at the
              nominal airframe's trim LIFT COEFFICIENT, i.e. trimmed for
              speed the way a glider is trimmed on the field (trimming every
              airframe to one alpha instead would fly a low-lift-slope
              airframe far from its best glide); rejected if that needs more
              than ELEVATOR_TRIM_MAX of the throw. Gates 2, 4 and 5 then test
              the stick-fixed glide and stability at a flyable trim,
              separating mis-trimmed airframes from uncontrollable ones.

A roll trim bias is cancelled by the opposite aileron command in both modes.

Run:   python validate_glide.py              (the aircraft file as generated; elevator mode)
       python validate_glide.py --sweep [--trim fixed|elevator]
                                             (all sweep configs -> analysis/gate_sweep_<trim>.json)
       python validate_glide.py --random 300 (random full-range airframes
                                             -> analysis/gate_sweep_random.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import jsbsim
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from sim.jsbsim_fdm import DR_PROPERTY_MAP, MASS_KEYS, apply_airframe  # noqa: E402
from sim.mass_properties import fdm_baseline, nominal  # noqa: E402
from sim.mass_solver import (  # noqa: E402
    KG_PER_SLUG, KGM2_PER_SLUGFT2, MassPropertySolver, UnreachableTarget,
)

M2FT   = 3.280839895
FT2M   = 1.0 / M2FT
FPS2MS = FT2M               # 1 fps = FT2M m/s
MS2FPS = M2FT
LBF2N  = 4.4482216
SLUG_FT3_TO_KG_M3 = 515.378818

# Mass and inertia of the aircraft file as generated (aircraft/mass_properties.yaml
# via aircraft/generate_fdm.py): the bottom of each range, no point masses.
# Used by the legacy helpers below; the gates themselves read JSBSim's own
# mass properties, which equal these for the bare file
# (tests/test_mass_properties.py::test_generated_xml_matches_yaml).
_BASE = fdm_baseline()
MASS = _BASE["mass_kg"]   # kg
G    = 9.81               # m/s²
IXX  = _BASE["ixx"]       # kg·m²
IYY  = _BASE["iyy"]
IZZ  = _BASE["izz"]

# Trim angle of attack: Cm0 + Cm_alpha * alpha = 0 → alpha = 0.05/0.60 = 4.77°.
# Independent of mass while the CG stays on the aero reference point.
ALPHA_TRIM_DEG = 4.78
GAMMA_TRIM_DEG = -4.0

DT   = 1.0 / 200.0   # physics step, 200 Hz

# Stall angle of the unscaled CL table (its peak); a sampled airframe stalls
# at STALL_TABLE_DEG * clmax_scale / cl_mult (see the aircraft file).
STALL_TABLE_DEG = 12.0
STALL_CHECK_DEG = (4.0, 10.0, 12.0, 14.0, 22.0)   # at the unscaled stall angle

GLIDE_LO, GLIDE_HI = 10.0, 20.0
ELEVATOR_TRIM_MAX = 0.5    # fraction of full elevator throw a field trim may use
ENERGY_TOL = 1e-3
PITCH_RATE_LIMIT = 0.10

SWEEP_JSON = os.path.join(PROJ_ROOT, "analysis", "gate_sweep_{trim}.json")


# ---------------------------------------------------------------------------
# Legacy helpers (analysis/capture_baseline.py and the baseline JSONs use these)
# ---------------------------------------------------------------------------

def _make_fdm(root: str = PROJ_ROOT) -> jsbsim.FGFDMExec:
    """Create a 200 Hz FGFDMExec.  set_dt MUST precede load_model so the
    FCS rate-limiter caches the correct dt (JSBSim default is 1/120 s)."""
    fdm = jsbsim.FGFDMExec(root_dir=root)
    fdm.set_debug_level(0)
    fdm.set_dt(DT)
    ok = fdm.load_model("rlglider")
    if not ok:
        raise RuntimeError(f"load_model('rlglider') failed under {root}")
    return fdm


def trim_speed_ms(
    mass_kg: float = MASS,
    *,
    alt_m: float = 300.0,
    alpha_deg: float = ALPHA_TRIM_DEG,
    gamma_deg: float = GAMMA_TRIM_DEG,
) -> float:
    """Airspeed at which lift balances weight at the trim alpha.

    L = qbar * S * CL = m * g * cos(gamma). CL, wing area and air density are
    read from the loaded model at the trim alpha, so this follows the aircraft
    file rather than a copied constant. Replaces the fixed 10 m/s start speed,
    which was trim only for the 1.1 kg V19 aircraft.
    """
    fdm = _make_fdm()
    fdm["ic/h-agl-ft"]   = alt_m * M2FT
    fdm["ic/vt-fps"]     = 10.0 * MS2FPS
    fdm["ic/alpha-deg"]  = alpha_deg
    fdm["ic/gamma-deg"]  = gamma_deg
    fdm.run_ic()
    cl   = fdm["aero/coefficient/CLtotal"]
    s_m2 = fdm["metrics/Sw-sqft"] * FT2M * FT2M
    rho  = fdm["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3
    return float(np.sqrt(2.0 * mass_kg * G * np.cos(np.radians(gamma_deg)) / (rho * s_m2 * cl)))


def _set_near_trim_ic(
    fdm,
    *,
    alt_m: float = 300.0,
    vt_ms: float | None = None,
    alpha_deg: float = ALPHA_TRIM_DEG,
    gamma_deg: float = GAMMA_TRIM_DEG,
    heading_deg: float = 0.0,
) -> None:
    """Apply analytically computed near-trim IC and prime the sim.

    Trim point: Cm0 + Cm_alpha * alpha = 0 → alpha = 0.05/0.60 = 4.77°.
    vt_ms defaults to trim_speed_ms() for the model's mass, so that
    L = qbar*S*CL = mg cos(gamma) at the start.
    Elevator command zero (trim elevator is ~0 rad).
    """
    if vt_ms is None:
        vt_ms = trim_speed_ms(alt_m=alt_m, alpha_deg=alpha_deg, gamma_deg=gamma_deg)
    fdm["aero/cl-mult"]       = 1.0
    fdm["aero/cd0-add"]       = 0.0
    fdm["aero/cd-mult"]       = 1.0
    fdm["ic/h-agl-ft"]        = alt_m * M2FT
    fdm["ic/vt-fps"]          = vt_ms * MS2FPS
    fdm["ic/alpha-deg"]       = alpha_deg
    fdm["ic/gamma-deg"]       = gamma_deg
    fdm["ic/psi-true-deg"]    = heading_deg
    fdm.run_ic()
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/aileron-cmd-norm"]  = 0.0
    fdm["fcs/rudder-cmd-norm"]   = 0.0


def _total_energy(fdm, mass=None, ixx=None, iyy=None, izz=None, ixz=0.0) -> float:
    """Full mechanical energy: KE_translational + KE_rotational + PE  (J).

    Mass properties default to the bare generated file's (legacy callers).
    KE_rot includes the ixz cross term: 0.5 (Ixx p² + Iyy q² + Izz r² − 2 Ixz p r).
    """
    mass = MASS if mass is None else mass
    ixx = IXX if ixx is None else ixx
    iyy = IYY if iyy is None else iyy
    izz = IZZ if izz is None else izz
    u = fdm["velocities/u-fps"] * FPS2MS
    v = fdm["velocities/v-fps"] * FPS2MS
    w = fdm["velocities/w-fps"] * FPS2MS
    p = fdm["velocities/p-rad_sec"]
    q = fdm["velocities/q-rad_sec"]
    r = fdm["velocities/r-rad_sec"]
    h = fdm["position/h-agl-ft"] * FT2M

    ke_t = 0.5 * mass * (u*u + v*v + w*w)
    ke_r = 0.5 * (ixx*p*p + iyy*q*q + izz*r*r - 2.0*ixz*p*r)
    pe   = mass * G * h
    return ke_t + ke_r + pe


# ---------------------------------------------------------------------------
# Airframes and trim
# ---------------------------------------------------------------------------

@dataclass
class Airframe:
    """One configuration to gate. params uses env/curriculum.py's DR-table
    keys ({} = the aircraft file exactly as generated); root is the JSBSim
    root holding aircraft/rlglider/rlglider.xml (the ixz variants live in
    their own roots)."""
    name: str = "as_generated"
    params: dict = field(default_factory=dict)
    root: str = PROJ_ROOT


@dataclass
class Trim:
    alpha_deg: float
    gamma_deg: float
    vt_ms: float
    cl: float
    cd: float
    dm_dalpha: float          # aero pitching moment slope about the CG at trim (ft·lbf/deg)
    statically_stable: bool
    elevator_cmd: float = 0.0 # normalised elevator command held through the gates
    mode: str = "fixed"


def _airframe_fdm(af: Airframe, elevator_cmd: float = 0.0) -> jsbsim.FGFDMExec:
    """Loaded FDM with the airframe applied and actuators settled on their
    trim commands (lateral bias cancelled). Ends at a placeholder IC."""
    fdm = _make_fdm(af.root)
    fdm["ic/h-agl-ft"] = 300.0 * M2FT
    fdm["ic/vt-fps"] = 12.0 * MS2FPS
    fdm["ic/alpha-deg"] = 5.0
    fdm.run_ic()
    apply_airframe(fdm, af.params, MassPropertySolver(), float(fdm["metrics/Sw-sqft"]))
    _trim_commands(fdm, elevator_cmd)
    for _ in range(int(1.0 / DT)):     # 20 actuator time constants; run_ic keeps this state
        fdm.run()
    return fdm


def _trim_commands(fdm, elevator_cmd: float = 0.0) -> None:
    fdm["fcs/aileron-cmd-norm"]  = -fdm["fcs/trim-bias-roll"]
    fdm["fcs/elevator-cmd-norm"] = float(elevator_cmd)
    fdm["fcs/rudder-cmd-norm"]   = 0.0


def _moment_at(fdm, alpha: float) -> float:
    fdm["ic/alpha-deg"] = alpha
    fdm.run_ic()
    return fdm["moments/m-aero-lbsft"]


def _elevator_for_trim(af: Airframe, alpha_ref: float) -> float | None:
    """Elevator command balancing the pitching moment at alpha_ref. The
    moment is affine in elevator deflection, so a secant on settled
    actuators solves it; the result is verified. None if no command in
    [-1, 1] trims it."""
    def m(e):
        f = _airframe_fdm(af, e)
        f["ic/vt-fps"] = 12.0 * MS2FPS
        f["ic/beta-deg"] = 0.0
        for k in ("p", "q", "r"):
            f[f"ic/{k}-rad_sec"] = 0.0
        return _moment_at(f, alpha_ref)
    m0, m1 = m(0.0), m(0.2)
    if m1 == m0:
        return None
    e = -m0 * 0.2 / (m1 - m0)
    if not -1.0 <= e <= 1.0:
        return None
    return float(e) if abs(m(e)) <= 1e-6 + 1e-3 * abs(m0) else None


def _alpha_for_cl(af: Airframe, elevator_cmd: float, cl_ref: float) -> float | None:
    """Pre-stall alpha at which CLtotal (elevator lift included) equals cl_ref."""
    f = _airframe_fdm(af, elevator_cmd)
    f["ic/vt-fps"] = 12.0 * MS2FPS
    f["ic/beta-deg"] = 0.0
    for k in ("p", "q", "r"):
        f[f"ic/{k}-rad_sec"] = 0.0

    def cl(a):
        f["ic/alpha-deg"] = a
        f.run_ic()
        return f["aero/coefficient/CLtotal"]
    lo, hi = -6.0, stall_angle_deg(af.params)
    if not cl(lo) <= cl_ref <= cl(hi):
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cl(mid) < cl_ref else (lo, mid)
    return 0.5 * (lo + hi)


def _elevator_trim_alpha(af: Airframe, cl_ref: float) -> tuple[float, float] | None:
    """(alpha, elevator command) trimming `af` at lift coefficient cl_ref.
    Elevator lift moves CL slightly, so alpha and elevator are iterated."""
    e = 0.0
    for _ in range(4):
        alpha = _alpha_for_cl(af, e, cl_ref)
        if alpha is None:
            return None
        e = _elevator_for_trim(af, alpha)
        if e is None:
            return None
    return alpha, e


def find_trim(af: Airframe, mode: str = "fixed", cl_ref: float | None = None) -> Trim | None:
    """Trimmed glide of `af`, or None if no pitch trim exists between -6 and
    25 deg alpha (fixed) or within elevator authority at cl_ref (elevator).

    run_ic() evaluates the aerodynamics at the exact IC alpha without
    integrating, and keeps the settled actuator state, so the pitching
    moment about the CG (moments/m-aero-lbsft) can be scanned directly.
    """
    elevator_cmd = 0.0
    if mode == "elevator":
        trimmed = _elevator_trim_alpha(af, cl_ref)
        if trimmed is None:
            return None
        elevator_cmd = trimmed[1]
    fdm = _airframe_fdm(af, elevator_cmd)
    fdm["ic/vt-fps"] = 12.0 * MS2FPS
    fdm["ic/beta-deg"] = 0.0
    for k in ("p", "q", "r"):
        fdm[f"ic/{k}-rad_sec"] = 0.0

    def at(alpha):
        fdm["ic/alpha-deg"] = alpha
        fdm.run_ic()
        qs = fdm["aero/qbar-psf"] * fdm["metrics/Sw-sqft"]
        cd = (fdm["aero/coefficient/CD0"] + fdm["aero/coefficient/CDi"]) / qs
        return fdm["moments/m-aero-lbsft"], fdm["aero/coefficient/CLtotal"], cd

    grid = np.arange(-6.0, 25.0001, 0.25)
    m = np.array([at(a)[0] for a in grid])
    crossings = [i for i in range(len(grid) - 1) if m[i] == 0 or m[i] * m[i + 1] < 0]
    if not crossings:
        return None
    # Prefer a statically stable crossing (moment falling through zero).
    stable = [i for i in crossings if m[i + 1] < m[i]]
    i = (stable or crossings)[0]
    lo, hi = grid[i], grid[i + 1]
    m_lo = m[i]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m_mid = at(mid)[0]
        if np.sign(m_mid) == np.sign(m_lo):
            lo, m_lo = mid, m_mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    _, cl, cd = at(alpha)
    slope = (m[i + 1] - m[i]) / (grid[i + 1] - grid[i])
    if cl <= 0:
        return None
    gamma = -np.degrees(np.arctan(cd / cl))
    weight_n = fdm["inertia/weight-lbs"] * LBF2N
    s_m2 = fdm["metrics/Sw-sqft"] * FT2M**2
    rho = fdm["atmosphere/rho-slugs_ft3"] * SLUG_FT3_TO_KG_M3
    vt = np.sqrt(2 * weight_n * np.cos(np.radians(gamma)) / (rho * s_m2 * cl))
    return Trim(float(alpha), float(gamma), float(vt), float(cl), float(cd),
                float(slope), bool(slope < 0), float(elevator_cmd), mode)


def _trimmed(af: Airframe, trim: Trim, alt_m: float) -> jsbsim.FGFDMExec:
    fdm = _airframe_fdm(af, trim.elevator_cmd)
    fdm["ic/h-agl-ft"]     = alt_m * M2FT
    fdm["ic/vt-fps"]       = trim.vt_ms * MS2FPS
    fdm["ic/alpha-deg"]    = trim.alpha_deg
    fdm["ic/gamma-deg"]    = trim.gamma_deg
    fdm["ic/beta-deg"]     = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    for k in ("p", "q", "r"):
        fdm[f"ic/{k}-rad_sec"] = 0.0
    fdm.run_ic()
    _trim_commands(fdm, trim.elevator_cmd)
    return fdm


def _readback(fdm) -> dict:
    return dict(
        mass=fdm["inertia/mass-slugs"] * KG_PER_SLUG,
        ixx=fdm["inertia/ixx-slugs_ft2"] * KGM2_PER_SLUGFT2,
        iyy=fdm["inertia/iyy-slugs_ft2"] * KGM2_PER_SLUGFT2,
        izz=fdm["inertia/izz-slugs_ft2"] * KGM2_PER_SLUGFT2,
        ixz=fdm["inertia/ixz-slugs_ft2"] * KGM2_PER_SLUGFT2,
    )


# ---------------------------------------------------------------------------
# Gates. Each returns {"value": ..., "passed": bool, "detail": str}.
# `scale` shortens every duration (tests only; the gates are defined at 1.0).
# ---------------------------------------------------------------------------

def _result(value, passed, detail):
    return {"value": value, "passed": bool(passed), "detail": detail}


def gate_energy_conservation(af: Airframe, trim: Trim, scale: float = 1.0) -> dict:
    """Zero drag (aero/cd-mult=0): full mechanical energy within 0.1% / 10 s.

    Lift is perpendicular to the velocity vector and does no translational
    work; with drag zeroed, the remaining aerodynamic moments only exchange
    translational and rotational KE, so the total is conserved up to the
    integrator. Mass properties are JSBSim's own (point masses included).
    """
    fdm = _trimmed(af, trim, 300.0)
    fdm["aero/cd-mult"] = 0.0
    fdm.run()
    mp = _readback(fdm)
    e0 = _total_energy(fdm, **mp)
    drift = 0.0
    for _ in range(int(10.0 * scale / DT)):
        if not fdm.run():
            return _result(None, False, "fdm.run() returned False")
        drift = max(drift, abs(_total_energy(fdm, **mp) - e0) / abs(e0))
    return _result(drift, np.isfinite(drift) and drift < ENERGY_TOL,
                   f"drift={drift * 100:.3f}% (threshold < {ENERGY_TOL * 100:.1f}%)")


def gate_glide_ratio(af: Airframe, trim: Trim, scale: float = 1.0) -> dict:
    """Steady glide ratio (horizontal/vertical NED speed): 10 s settling from
    the trimmed IC, then the median over 10 s.

    The settling period is what makes this a property of the airframe: the
    old 5 s window from the start read 12.87 or 21.6 on the same airframe
    depending on start speed (docs/SPEC_V20_decisions.md).
    """
    fdm = _trimmed(af, trim, 300.0)
    settle, window = int(10.0 * scale / DT), int(10.0 * scale / DT)
    samples = []
    for i in range(settle + window):
        if not fdm.run():
            return _result(None, False, "fdm.run() returned False")
        if i < settle:
            continue
        vn = fdm["velocities/v-north-fps"]
        ve = fdm["velocities/v-east-fps"]
        vd = fdm["velocities/v-down-fps"]
        if vd * FPS2MS > 0.1:
            samples.append(np.hypot(vn, ve) / vd)
    if not samples:
        return _result(None, False, "never descended")
    glide = float(np.median(samples))
    return _result(glide, GLIDE_LO < glide < GLIDE_HI,
                   f"L/D={glide:.1f} (threshold {GLIDE_LO:.0f}-{GLIDE_HI:.0f})")


def stall_angle_deg(params: dict) -> float:
    return STALL_TABLE_DEG * float(params.get("clmax_scale", 1.0)) / float(params.get("cl_mult", 1.0))


def gate_stall(af: Airframe, trim: Trim | None = None, scale: float = 1.0) -> dict:
    """Lift force drops past the airframe's own stall angle.

    Checked at the same fractions of the stall angle as the original gate
    (4, 10, 12, 14, 22 deg for the unscaled 12 deg stall), all at 10 m/s so
    force ratios are CL ratios:
      pre-stall slope positive, CL rising to the peak, a drop just past it,
      and a sustained drop well past it.
    """
    k = stall_angle_deg(af.params) / STALL_TABLE_DEG
    angles = [a * k for a in STALL_CHECK_DEG]

    def lift(alpha_deg):
        fdm = _airframe_fdm(af)
        fdm["ic/vt-fps"]    = 10.0 * MS2FPS
        fdm["ic/alpha-deg"] = alpha_deg
        fdm["ic/gamma-deg"] = 0.0
        fdm["ic/h-agl-ft"]  = 200.0 * M2FT
        fdm.run_ic()
        fdm["fcs/elevator-cmd-norm"] = 0.0
        fdm.run()
        return fdm["aero/coefficient/CLalpha"]

    c4, c10, c12, c14, c22 = (lift(a) for a in angles)
    ok = c4 < c10 < c12 and c14 < c12 and c22 < c12
    return _result({f"{a:.2f}": c for a, c in zip(angles, (c4, c10, c12, c14, c22))}, ok,
                   f"CL drops past alpha={angles[2]:.1f} deg" if ok else
                   f"CL at {[round(a, 1) for a in angles]} deg: {[round(c, 3) for c in (c4, c10, c12, c14, c22)]}")


def gate_trimmed_stability(af: Airframe, trim: Trim, scale: float = 1.0) -> dict:
    """Trimmed IC, fixed trim controls: pitch rate < 0.1 rad/s for 15 s.

    Cm_alpha < 0 static stability and Cm_q < 0 damping keep the phugoid
    stable; an airframe whose CG offset or stability multipliers remove
    either diverges here.
    """
    fdm = _trimmed(af, trim, 300.0)
    max_q = 0.0
    for step in range(int(15.0 * scale / DT)):
        if not fdm.run():
            return _result(max_q, False, f"fdm.run() returned False at step {step}")
        max_q = max(max_q, abs(fdm["velocities/q-rad_sec"]))
        if not np.isfinite(max_q) or max_q >= PITCH_RATE_LIMIT:
            return _result(float(max_q), False,
                           f"|q|={max_q:.3f} rad/s at t={step * DT:.2f} s (threshold < {PITCH_RATE_LIMIT})")
    return _result(float(max_q), True, f"max|q|={max_q:.3f} rad/s (threshold < {PITCH_RATE_LIMIT})")


def gate_attitude_integrity(af: Airframe, trim: Trim, scale: float = 1.0) -> dict:
    """Euler angles finite and physically bounded over 30 s from the trimmed
    IC at 500 m: pitch in (-90, 90) deg, roll in (-180, 180) deg."""
    fdm = _trimmed(af, trim, 500.0)
    max_phi = max_theta = 0.0
    for step in range(int(30.0 * scale / DT)):
        if not fdm.run():
            return _result(None, False, f"fdm.run() returned False at step {step}")
        phi, theta, psi = fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]
        if not (np.isfinite(phi) and np.isfinite(theta) and np.isfinite(psi)):
            return _result(None, False, f"non-finite Euler angle at step {step}")
        if abs(theta) >= np.radians(90.0) or abs(phi) >= np.radians(180.0):
            return _result(None, False, f"attitude out of bounds at t={step * DT:.2f} s")
        max_phi, max_theta = max(max_phi, abs(phi)), max(max_theta, abs(theta))
    return _result({"max_abs_phi_rad": max_phi, "max_abs_theta_rad": max_theta}, True,
                   "Euler angles finite over 30s")


GATES = (
    ("energy_conservation", gate_energy_conservation),
    ("glide_ratio",         gate_glide_ratio),
    ("stall",               gate_stall),
    ("trimmed_stability",   gate_trimmed_stability),
    ("attitude_integrity",  gate_attitude_integrity),
)


_CL_REF: dict[str, float] = {}


def nominal_trim_cl() -> float:
    """Fixed-control trim lift coefficient of the nominal airframe: what the
    elevator-trimmed mode trims every airframe to."""
    if "nominal" not in _CL_REF:
        _CL_REF["nominal"] = find_trim(Airframe("nominal", nominal_params())).cl
    return _CL_REF["nominal"]


def run_gates(af: Airframe, scale: float = 1.0, mode: str = "fixed") -> dict:
    """All five gates for one airframe. A missing trim fails every gate that
    starts from trim; an unreachable mass target fails all of them. In
    elevator mode a trim needing more than ELEVATOR_TRIM_MAX of the throw
    also counts as no usable trim."""
    out = {"name": af.name, "params": dict(af.params),
           "root": os.path.relpath(af.root, PROJ_ROOT), "trim_mode": mode}
    try:
        trim = find_trim(af, mode, nominal_trim_cl() if mode == "elevator" else None)
        if trim is not None and mode == "elevator" and abs(trim.elevator_cmd) > ELEVATOR_TRIM_MAX:
            out["rejected_trim"] = asdict(trim)
            trim = None
    except UnreachableTarget as exc:
        out.update(trim=None, gates={n: _result(None, False, f"unreachable: {exc}") for n, _ in GATES},
                   passed=False)
        return out
    out["trim"] = asdict(trim) if trim else None
    gates = {}
    for name, fn in GATES:
        if trim is None and name != "stall":
            gates[name] = _result(None, False, "no usable pitch trim" + (
                f" within {ELEVATOR_TRIM_MAX:.0%} elevator at CL {nominal_trim_cl():.3f}"
                if mode == "elevator" else " between -6 and 25 deg"))
        else:
            gates[name] = fn(af, trim, scale)
    out["gates"] = gates
    out["passed"] = all(g["passed"] for g in gates.values())
    return out


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

def nominal_params() -> dict:
    """Every DR-table key at the aircraft as modelled: aero hooks at the
    defaults the aircraft file declares, wing area unscaled, mass/CG/inertia
    at the mass_properties.yaml nominal."""
    from env.curriculum import DR_ONESIDED, DR_RANGES
    fdm = _make_fdm()
    out = {}
    nom = nominal()
    for key in list(DR_RANGES) + list(DR_ONESIDED):
        if key in MASS_KEYS:
            out[key] = float(nom[key])
        elif key == "sw_mult":
            out[key] = 1.0
        else:
            out[key] = float(fdm[DR_PROPERTY_MAP[key]])
    return out


RANDOM_SEED = 20261110


def random_configs(n: int, seed: int = RANDOM_SEED, scale: float = 1.0) -> list[Airframe]:
    """n airframes sampled uniformly from the DR table at `scale`, the way the
    environment samples them: a check on JOINT combinations, which the
    single-extreme sweep and its two corners do not cover."""
    from env.curriculum import stage_ranges
    rng = np.random.default_rng(seed)
    ranges = stage_ranges(scale)
    return [Airframe(f"random_{i:03d}", {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()})
            for i in range(n)]


def sweep_configs() -> list[Airframe]:
    """nominal; every single-parameter extreme (full range, others nominal);
    the two specified corners; and the ixz validation variants."""
    from aircraft.generate_fdm import write_ixz_variants
    from env.curriculum import DR_ONESIDED, DR_RANGES

    nom = nominal_params()
    configs = [Airframe("nominal", dict(nom))]
    for key, (lo, hi) in {**DR_RANGES, **DR_ONESIDED}.items():
        configs.append(Airframe(f"{key}_lo", {**nom, key: lo}))
        configs.append(Airframe(f"{key}_hi", {**nom, key: hi}))

    R = {**DR_RANGES, **DR_ONESIDED}
    configs.append(Airframe("corner_heaviest_draggiest", {
        **nom, "mass_kg": R["mass_kg"][1], "cd0_add": R["cd0_add"][1],
        "k_induced": R["k_induced"][1], "clmax_scale": R["clmax_scale"][0],
        "sw_mult": R["sw_mult"][0]}))
    # Least stable: CG aft (cg_x_m is +forward), weakest static and damping
    # derivatives, steepest lift slope (an aft CG destabilises in proportion
    # to it), lightest mass and inertia.
    configs.append(Airframe("corner_lightest_least_stable", {
        **nom, "mass_kg": R["mass_kg"][0], "ixx": R["ixx"][0], "iyy": R["iyy"][0],
        "cg_x_m": R["cg_x_m"][0], "cl_mult": R["cl_mult"][1],
        "cmalpha_mult": R["cmalpha_mult"][0], "cmq_mult": R["cmq_mult"][0],
        "clp_mult": R["clp_mult"][0], "clbeta_mult": R["clbeta_mult"][0],
        "cnbeta_mult": R["cnbeta_mult"][0], "cnr_mult": R["cnr_mult"][0]}))

    roots = write_ixz_variants()
    configs.append(Airframe("ixz_lo", dict(nom), roots["lo"]))
    configs.append(Airframe("ixz_hi", dict(nom), roots["hi"]))
    return configs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_result(res: dict) -> None:
    for i, (name, _) in enumerate(GATES, start=1):
        g = res["gates"][name]
        print(f"[{'PASS' if g['passed'] else 'FAIL'}] Gate {i} — {name:20s} {g['detail']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Physics validation gates")
    ap.add_argument("--sweep", action="store_true",
                    help=f"gate every sweep config and write {os.path.relpath(SWEEP_JSON, PROJ_ROOT)}")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="gate N random full-range airframes instead of the sweep")
    ap.add_argument("--trim", choices=("fixed", "elevator"), default="elevator",
                    help="trim mode (see module docstring); elevator is the Phase 5 gate definition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or SWEEP_JSON.format(trim="random" if args.random else args.trim)

    if not args.sweep and not args.random:
        print("\nRunning physics validation gates...\n")
        res = run_gates(Airframe(), mode=args.trim)
        _print_result(res)
        print()
        n_failed = sum(not g["passed"] for g in res["gates"].values())
        if n_failed == 0:
            print("All 5 gates PASSED. Simulator ready for training.")
        else:
            print(f"{n_failed} of {len(GATES)} gates FAILED.")
            sys.exit(1)
        return

    import subprocess
    from datetime import datetime, timezone
    t0 = time.time()
    results = []
    configs = random_configs(args.random) if args.random else sweep_configs()
    for af in configs:
        res = run_gates(af, mode=args.trim)
        results.append(res)
        fails = [n for n, g in res["gates"].items() if not g["passed"]]
        print(f"{'PASS' if res['passed'] else 'FAIL'}  {af.name:32s} "
              + ("" if res["passed"] else "failed: " + ", ".join(fails)), flush=True)
    record = {
        "spec": "V20 Phase 5 — physics gates at the domain-randomisation corners",
        "trim_mode": args.trim,
        "random": {"n": args.random, "seed": RANDOM_SEED, "scale": 1.0} if args.random else None,
        "cl_ref": nominal_trim_cl() if args.trim == "elevator" else None,
        "provenance": {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT,
                                      capture_output=True, text=True).stdout.strip(),
            "jsbsim": jsbsim.__version__,
            "runtime_s": round(time.time() - t0, 1),
        },
        "n_configs": len(results),
        "n_passed": sum(r["passed"] for r in results),
        "configs": results,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1, default=float)
        f.write("\n")
    print(f"\n{record['n_passed']}/{record['n_configs']} configurations pass all five gates; "
          f"wrote {os.path.relpath(args.out, PROJ_ROOT)}")
    sys.exit(0 if record["n_passed"] == record["n_configs"] else 1)


if __name__ == "__main__":
    main()

"""
validate_glide.py  (JSBSim version — Phase 2)
==============================================
Five physics gates that must all PASS before RL training begins.

Gates:
    1. test_energy_conservation  -- zero drag: full mech. energy (KE_trans +
                                    KE_rot + PE) constant within 0.1% over 10 s
    2. test_glide_ratio          -- steady glide ratio from NED velocity, 5 s
    3. test_stall                -- CL drops past alpha_stall (table check)
    4. test_trimmed_stability    -- near-trim IC: pitch rate < 0.1 rad/s, 15 s
    5. test_attitude_integrity   -- Euler angles finite + bounded over 30 s

Run:   python validate_glide.py
"""

from __future__ import annotations

import os
import sys

import jsbsim
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
M2FT   = 3.280839895
FT2M   = 1.0 / M2FT
FPS2MS = FT2M               # 1 fps = FT2M m/s
MS2FPS = M2FT

MASS = 1.1    # kg  (GliderParams.m)
G    = 9.81   # m/s²
IXX  = 0.18   # kg·m²  (GliderParams.Ixx)
IYY  = 0.10
IZZ  = 0.26

DT   = 1.0 / 200.0   # physics step, 200 Hz


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fdm() -> jsbsim.FGFDMExec:
    """Create a 200 Hz FGFDMExec.  set_dt MUST precede load_model so the
    FCS rate-limiter caches the correct dt (JSBSim default is 1/120 s)."""
    fdm = jsbsim.FGFDMExec(root_dir=PROJ_ROOT)
    fdm.set_debug_level(0)
    fdm.set_dt(DT)
    ok = fdm.load_model("rlglider")
    if not ok:
        raise RuntimeError("load_model('rlglider') failed")
    return fdm


def _set_near_trim_ic(
    fdm,
    *,
    alt_m: float = 300.0,
    vt_ms: float = 10.0,
    alpha_deg: float = 4.78,
    gamma_deg: float = -4.0,
    heading_deg: float = 0.0,
) -> None:
    """Apply analytically computed near-trim IC and prime the sim.

    Trim point: Cm0 + Cm_alpha * alpha = 0 → alpha = 0.05/0.60 = 4.77°.
    At V=10 m/s: L = qbar*S*CL ≈ 10.6 N ≈ mg — within 2%.
    Elevator command zero (trim elevator is ~0 rad).
    """
    fdm["aero/cl-mult"]       = 1.0
    fdm["aero/cd0-add"]       = 0.0
    fdm["aero/cd-mult"]       = 1.0
    fdm["aero/ctrl-eff-mult"] = 1.0
    fdm["ic/h-agl-ft"]        = alt_m * M2FT
    fdm["ic/vt-fps"]          = vt_ms * MS2FPS
    fdm["ic/alpha-deg"]       = alpha_deg
    fdm["ic/gamma-deg"]       = gamma_deg
    fdm["ic/psi-true-deg"]    = heading_deg
    fdm.run_ic()
    fdm["fcs/elevator-cmd-norm"] = 0.0
    fdm["fcs/aileron-cmd-norm"]  = 0.0
    fdm["fcs/rudder-cmd-norm"]   = 0.0


def _total_energy(fdm) -> float:
    """Full mechanical energy: KE_translational + KE_rotational + PE  (J)."""
    u = fdm["velocities/u-fps"] * FPS2MS
    v = fdm["velocities/v-fps"] * FPS2MS
    w = fdm["velocities/w-fps"] * FPS2MS
    p = fdm["velocities/p-rad_sec"]
    q = fdm["velocities/q-rad_sec"]
    r = fdm["velocities/r-rad_sec"]
    h = fdm["position/h-agl-ft"] * FT2M

    ke_t = 0.5 * MASS * (u*u + v*v + w*w)
    ke_r = 0.5 * (IXX*p*p + IYY*q*q + IZZ*r*r)
    pe   = MASS * G * h
    return ke_t + ke_r + pe


# ---------------------------------------------------------------------------
# Gate 1 — Energy conservation (zero drag)
# ---------------------------------------------------------------------------

def test_energy_conservation() -> None:
    """Zero drag (aero/cd-mult=0): full mechanical energy within 0.1% / 10 s.

    Lift is perpendicular to the velocity vector → does no translational work.
    With drag zeroed, the only power inputs are gravitational (conservative)
    and aerodynamic moments coupling translational ↔ rotational KE.
    Including rotational KE in the energy tally makes the sum conserved even
    with pitch/roll/yaw damping terms active — those terms only transfer
    between translational and rotational DoF, not dissipate.
    The 0.1% tolerance covers JSBSim's Adams-Bashforth numerical integration.
    """
    fdm = _make_fdm()
    _set_near_trim_ic(fdm)
    fdm["aero/cd-mult"] = 0.0      # disable all drag forces

    # One warm-up step so aero state is primed from the IC
    fdm.run()
    E0        = _total_energy(fdm)
    max_drift = 0.0

    for _ in range(int(10.0 / DT)):
        if not fdm.run():
            raise AssertionError("fdm.run() returned False during energy test")
        E = _total_energy(fdm)
        drift = abs(E - E0) / abs(E0)
        max_drift = max(max_drift, drift)

    TOLERANCE = 1e-3   # 0.1 %
    if max_drift >= TOLERANCE:
        raise AssertionError(
            f"Energy drift {max_drift * 100:.3f}% >= 0.1% tolerance"
        )
    print(f"PASS  test_energy_conservation  max drift = {max_drift * 100:.3e}%")


# ---------------------------------------------------------------------------
# Gate 2 — Glide ratio
# ---------------------------------------------------------------------------

def test_glide_ratio() -> None:
    """Steady glide ratio (horizontal/vertical NED speed) over 5 s.

    Median of instantaneous samples is insensitive to phugoid drift.
    Expected range 10–20 for this prototype model (current L/D ≈ 14);
    limits will be tightened once real aircraft parameters are measured.
    """
    fdm = _make_fdm()
    _set_near_trim_ic(fdm)

    samples: list[float] = []
    for _ in range(int(5.0 / DT)):
        fdm.run()
        vn = fdm["velocities/v-north-fps"] * FPS2MS
        ve = fdm["velocities/v-east-fps"]  * FPS2MS
        vd = fdm["velocities/v-down-fps"]  * FPS2MS
        vh = np.sqrt(vn*vn + ve*ve)
        if vd > 0.1:
            samples.append(vh / vd)

    if not samples:
        raise AssertionError("Glider never descended — check flight dynamics")
    glide = float(np.median(samples))

    LO, HI = 10.0, 20.0
    if not (LO < glide < HI):
        raise AssertionError(
            f"Glide ratio {glide:.2f} outside ({LO}, {HI})"
        )
    print(
        f"PASS  test_glide_ratio          "
        f"glide ratio = {glide:.2f}  (in {LO}–{HI})"
    )


# ---------------------------------------------------------------------------
# Gate 3 — Stall behaviour
# ---------------------------------------------------------------------------

def test_stall() -> None:
    """Lift force drops past alpha_stall=12°.

    All runs share the same airspeed (V=10 m/s → identical qbar and S), so
    comparing raw CLalpha force values equals comparing CL coefficients.

    Checks:
      • Pre-stall slope positive:  CL(4°) < CL(10°)
      • CL peaks near stall alpha: CL(10°) < CL(12°)
      • Immediate post-stall drop: CL(14°) < CL(12°)
      • 10° past stall still lower: CL(22°) < CL(12°)
    """
    def _cl_lift_force(alpha_deg: float) -> float:
        fdm = _make_fdm()
        fdm["ic/vt-fps"]     = 10.0 * MS2FPS
        fdm["ic/alpha-deg"]  = alpha_deg
        fdm["ic/gamma-deg"]  = 0.0
        fdm["ic/h-agl-ft"]   = 200.0 * M2FT
        fdm.run_ic()
        fdm["fcs/elevator-cmd-norm"] = 0.0
        fdm.run()   # one step to compute aero coefficients
        return fdm["aero/coefficient/CLalpha"]   # lbf; same qbar → ratio = CL ratio

    cl_4  = _cl_lift_force(4.0)
    cl_10 = _cl_lift_force(10.0)
    cl_12 = _cl_lift_force(12.0)
    cl_14 = _cl_lift_force(14.0)
    cl_22 = _cl_lift_force(22.0)

    if not cl_4 < cl_10:
        raise AssertionError(
            f"Pre-stall slope not positive: CL(4°)={cl_4:.4f} >= CL(10°)={cl_10:.4f}"
        )
    if not cl_10 < cl_12:
        raise AssertionError(
            f"CL not rising to stall: CL(10°)={cl_10:.4f} >= CL(12°)={cl_12:.4f}"
        )
    if not cl_14 < cl_12:
        raise AssertionError(
            f"No CL drop at stall+2°: CL(14°)={cl_14:.4f} >= CL(12°)={cl_12:.4f}"
        )
    if not cl_22 < cl_12:
        raise AssertionError(
            f"CL not sustained drop at stall+10°: CL(22°)={cl_22:.4f} >= peak {cl_12:.4f}"
        )

    print(
        f"PASS  test_stall                "
        f"CL: 4°={cl_4:.3f}  10°={cl_10:.3f}  "
        f"12°(peak)={cl_12:.3f}  14°={cl_14:.3f}  22°={cl_22:.3f}"
    )


# ---------------------------------------------------------------------------
# Gate 4 — Trimmed stability
# ---------------------------------------------------------------------------

def test_trimmed_stability() -> None:
    """Near-trim IC: pitch rate < 0.1 rad/s for 15 s, zero control input.

    JSBSim's do_trim(0) fails for this model because the FCS lag filters
    make the trim Jacobian non-trivial for the built-in solver.  Instead we
    use the analytically computed trim IC (alpha≈4.78°, elevator=0) which
    was verified in Phase 1 to place the glider within 2% of true trim.
    The Cm_alpha < 0 static stability and Cm_q < 0 damping ensure the phugoid
    is stable — pitch rate remains negligible throughout the 15 s run.
    """
    fdm = _make_fdm()
    _set_near_trim_ic(fdm, alt_m=300.0, vt_ms=10.0)

    PITCH_LIMIT = 0.10   # rad/s
    max_q = 0.0

    for step in range(int(15.0 / DT)):
        if not fdm.run():
            raise AssertionError(f"fdm.run() returned False at step {step}")
        q = abs(fdm["velocities/q-rad_sec"])
        max_q = max(max_q, q)
        if q >= PITCH_LIMIT:
            raise AssertionError(
                f"Pitch rate {q:.4f} rad/s at step {step} >= {PITCH_LIMIT} rad/s "
                f"— model is longitudinally unstable"
            )

    print(
        f"PASS  test_trimmed_stability    "
        f"max pitch rate = {max_q:.4f} rad/s  (< 0.1)"
    )


# ---------------------------------------------------------------------------
# Gate 5 — Attitude integrity
# ---------------------------------------------------------------------------

def test_attitude_integrity() -> None:
    """Euler angles remain finite and physically bounded over 30 s.

    JSBSim normalises its quaternion internally; this gate verifies the
    result stays in the physical domain: pitch in (−90°, +90°), roll in
    (−180°, +180°), yaw finite, no NaN/Inf anywhere.  Starting at a higher
    altitude to avoid ground contact during the full 30 s glide.
    """
    fdm = _make_fdm()
    _set_near_trim_ic(fdm, alt_m=500.0, vt_ms=10.0)

    max_abs_phi   = 0.0
    max_abs_theta = 0.0

    for step in range(int(30.0 / DT)):
        if not fdm.run():
            raise AssertionError(f"fdm.run() returned False at step {step}")

        phi   = fdm["attitude/phi-rad"]
        theta = fdm["attitude/theta-rad"]
        psi   = fdm["attitude/psi-rad"]

        if not (np.isfinite(phi) and np.isfinite(theta) and np.isfinite(psi)):
            raise AssertionError(
                f"Non-finite Euler angle at step {step}: "
                f"phi={phi}  theta={theta}  psi={psi}"
            )
        if abs(theta) >= np.radians(90.0):
            raise AssertionError(
                f"Pitch {np.degrees(theta):.1f}° out of (−90°, +90°) at step {step}"
            )
        if abs(phi) >= np.radians(180.0):
            raise AssertionError(
                f"Roll {np.degrees(phi):.1f}° out of (−180°, +180°) at step {step}"
            )

        max_abs_phi   = max(max_abs_phi,   abs(phi))
        max_abs_theta = max(max_abs_theta, abs(theta))

    h_final = fdm["position/h-agl-ft"] * FT2M
    print(
        f"PASS  test_attitude_integrity   "
        f"max |roll|={np.degrees(max_abs_phi):.2f}°  "
        f"max |pitch|={np.degrees(max_abs_theta):.2f}°  "
        f"final alt={h_final:.1f} m"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== validate_glide.py  —  JSBSim physics gates ===\n")

    gates = [
        test_energy_conservation,
        test_glide_ratio,
        test_stall,
        test_trimmed_stability,
        test_attitude_integrity,
    ]

    results: list[tuple[str, bool, str]] = []
    for fn in gates:
        try:
            fn()
            results.append((fn.__name__, True, ""))
        except Exception as exc:
            print(f"FAIL  {fn.__name__:<36} {exc}")
            results.append((fn.__name__, False, str(exc)))

    print()
    all_passed = all(ok for _, ok, _ in results)
    for name, ok, _ in results:
        print(f"  {'PASSED' if ok else 'FAILED'}  {name}")
    print()
    if all_passed:
        print("All 5 gates PASSED — JSBSim model is flight-worthy.")
    else:
        n_failed = sum(1 for _, ok, _ in results if not ok)
        print(f"{n_failed} of {len(gates)} gates FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

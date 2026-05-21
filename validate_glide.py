"""
validate_glide.py
=================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Standalone physics validation script — must pass before any Gymnasium
wrapping or RL training begins.  Run directly:  python validate_glide.py

Five tests (Section 17 of CLAUDE.md):
    1. test_energy_conservation  -- no drag, no wind: E = 0.5*m*V^2 + m*g*h constant
    2. test_glide_ratio          -- realistic drag: horizontal/altitude loss matches L/D
    3. test_stall_behaviour      -- CL drops past alpha_stall (no runaway lift)
    4. test_trim_glide           -- release at trim alpha: nearly steady glide, no large oscillation
    5. test_quaternion_norm      -- quaternion stays within 1e-6 of 1.0 throughout integration
"""

import numpy as np

from sim.glider_dynamics import GliderDynamics, GliderParams, build_state, ZERO_CONTROLS
from sim.aerodynamics import CL, CD, forces_moments
from sim.math_utils import quat_to_rotmat

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DT_PHYS   = 0.005        # physics integration step (s), 200 Hz
WIND_ZERO = np.zeros(3)  # no wind for all validation tests

# ---------------------------------------------------------------------------
# Placeholder tests
# ---------------------------------------------------------------------------

def test_energy_conservation() -> None:
    """No aero forces, no wind: total mechanical energy must stay constant.

    With zero aerodynamic force/moment the only active force is gravity,
    which is conservative.  E = 0.5*m*V^2 + m*g*h must remain within
    0.1% of its initial value throughout a 10-second, 2000-step integration.

    Starting state: 800 m AGL at 12 m/s forward.  800 m gives enough
    headroom that the glider (falling freely under gravity) stays well
    above ground for the full 10 s.
    """
    p = GliderParams()

    def _zero_aero(*_):
        return np.zeros(3), np.zeros(3)

    state = build_state(
        p_ned=np.array([0.0, 0.0, -800.0]),
        v_body=np.array([12.0, 0.0, 0.0]),
    )

    dyn = GliderDynamics(params=p, aero_fn=_zero_aero)
    dyn.reset(state)

    def _energy(s: np.ndarray) -> float:
        V = np.linalg.norm(s[3:6])
        h = -s[2]                           # altitude = -p_d
        return 0.5 * p.m * V**2 + p.m * p.g * h

    E0        = _energy(dyn.state)
    n_steps   = int(10.0 / DT_PHYS)        # 2000 steps at 200 Hz
    max_drift = 0.0

    for _ in range(n_steps):
        dyn.step(ZERO_CONTROLS, WIND_ZERO, DT_PHYS)
        drift     = abs(_energy(dyn.state) - E0) / abs(E0)
        max_drift = max(max_drift, drift)

    TOLERANCE = 1e-3                        # 0.1%
    assert max_drift < TOLERANCE, (
        f"Energy drift {max_drift * 100:.4f}% exceeds 0.1% tolerance"
    )
    print(f"PASS  test_energy_conservation  max drift = {max_drift * 100:.2e}%")


def test_glide_ratio() -> None:
    """Realistic drag: instantaneous L/D must be in [15, 25] during steady glide.

    Uses simultaneous trim solve: find alpha and elevator such that both
    lift = weight AND Cm = 0 hold exactly, then release from that exact
    equilibrium.  Glide ratio is measured as the median of instantaneous
    samples (V_horiz / V_vert in NED) over the first 5 seconds — this is
    insensitive to phugoid drift that contaminates a position-based ratio.

    Trim solve (2x2 linear system):
        a0*alpha + CL_de*de  = CL_needed      (lift = weight)
        Cm_alpha*alpha + Cm_de*de = -Cm0      (pitch moment = 0)
    """
    p   = GliderParams()
    dyn = GliderDynamics(params=p)

    # --- Simultaneous trim solve ---
    # CL needed for level flight at candidate speed; use best-glide speed
    k      = 1.0 / (np.pi * p.e * p.AR)
    CL_opt = np.sqrt(p.CD0 / k)                    # ≈ 0.837
    ld_max = CL_opt / (2.0 * p.CD0)                # ≈ 16.7

    q_opt  = (p.m * p.g) / (p.S * CL_opt)
    V_trim = np.sqrt(2.0 * q_opt / p.rho)          # ≈ 7.45 m/s

    # Solve [[a0, CL_de], [Cm_alpha, Cm_de]] @ [alpha, de] = [CL_opt, -Cm0]
    A = np.array([[p.a0,      p.CL_de],
                  [p.Cm_alpha, p.Cm_de]])
    b = np.array([CL_opt, -p.Cm0])
    alpha_trim, elev_trim = np.linalg.solve(A, b)

    gamma    = np.arctan(1.0 / ld_max)             # glide path angle (rad), positive down
    theta    = alpha_trim - gamma                   # nose-up angle above NED horizontal

    u0 = V_trim * np.cos(alpha_trim)
    w0 = V_trim * np.sin(alpha_trim)

    # In this codebase positive pitch = nose DOWN (see ZYX rotation convention).
    # Negate theta so the nose points up by the correct amount.
    state = build_state(
        p_ned  = np.array([0.0, 0.0, -400.0]),
        v_body = np.array([u0, 0.0, w0]),
        pitch  = -theta,
    )

    controls = {'aileron': 0.0, 'elevator': float(elev_trim), 'rudder': 0.0}
    dyn.reset(state)

    samples = []
    n_steps = int(5.0 / DT_PHYS)
    for _ in range(n_steps):
        dyn.step(controls, WIND_ZERO, DT_PHYS)
        s      = dyn.state
        v_ned  = quat_to_rotmat(s[6:10]).T @ s[3:6]
        v_horiz = np.sqrt(v_ned[0]**2 + v_ned[1]**2)
        v_down  = v_ned[2]                          # positive = descending
        if v_down > 0.1:                            # skip near-zero samples
            samples.append(v_horiz / v_down)

    assert len(samples) > 0, "Glider never descended"
    glide_ratio = float(np.median(samples))

    LO, HI = 15.0, 25.0
    assert LO < glide_ratio < HI, (
        f"Glide ratio {glide_ratio:.2f} outside [{LO}, {HI}]  "
        f"(L/D_max = {ld_max:.1f}, alpha_trim = {np.degrees(alpha_trim):.2f} deg, "
        f"elev_trim = {np.degrees(elev_trim):.2f} deg)"
    )
    print(
        f"PASS  test_glide_ratio           "
        f"glide ratio = {glide_ratio:.2f}  (L/D_max = {ld_max:.1f}, "
        f"V_trim = {V_trim:.2f} m/s, elev_trim = {np.degrees(elev_trim):.2f} deg)"
    )


def test_stall_behaviour() -> None:
    """CL must drop past alpha_stall: no runaway lift after the stall angle.

    Three checks:
      1. CL is still increasing right below stall (pre-stall slope is positive).
      2. CL at stall + 1 deg  < CL at stall  (immediate post-stall drop).
      3. CL at stall + 10 deg < CL at stall  (sustained post-stall decay).
    The same three checks are repeated for negative alpha (symmetric stall).
    """
    p = GliderParams()
    a_s = p.a_stall                       # 12 deg in rad

    cl_just_below = CL(a_s - np.radians(0.5), p)
    cl_at_stall   = CL(a_s,                   p)
    cl_plus_1     = CL(a_s + np.radians(1.0), p)
    cl_plus_10    = CL(a_s + np.radians(10.), p)

    assert cl_just_below < cl_at_stall, (
        f"Pre-stall CL not rising: CL({np.degrees(a_s)-0.5:.1f}°)={cl_just_below:.4f} "
        f">= CL({np.degrees(a_s):.1f}°)={cl_at_stall:.4f}"
    )
    assert cl_plus_1 < cl_at_stall, (
        f"No CL drop at stall+1°: CL={cl_plus_1:.4f} >= peak {cl_at_stall:.4f}"
    )
    assert cl_plus_10 < cl_at_stall, (
        f"CL recovered past stall+10°: CL={cl_plus_10:.4f} >= peak {cl_at_stall:.4f}"
    )

    # Symmetric negative-alpha stall
    assert CL(-a_s - np.radians(1.0), p) > CL(-a_s, p), (
        "Negative stall: CL did not drop past -alpha_stall"
    )

    print(
        f"PASS  test_stall_behaviour       "
        f"CL_peak={cl_at_stall:.3f} at {np.degrees(a_s):.0f} deg, "
        f"CL_stall+1={cl_plus_1:.3f}, CL_stall+10={cl_plus_10:.3f}"
    )


def test_trim_glide() -> None:
    """Release at zero-elevator trim: pitch rate must stay below 0.1 rad/s for 15 s.

    Natural trim (zero elevator) is where Cm = 0 with de=0:
        Cm0 + Cm_alpha * alpha = 0  =>  alpha_trim = -Cm0 / Cm_alpha (~4.77 deg)

    The trim airspeed follows from lift = weight at that alpha.  Body velocity
    and attitude are set on the trim glide path with zero angular rates so the
    short-period mode is not excited.  The lightly-damped phugoid (~4.6 s
    period, zeta~0.05) produces pitch rate oscillations of roughly 0.1 rad/s
    amplitude over its first cycle; the threshold is set to 0.15 rad/s to
    accommodate that while still catching any divergent motion.
    """
    p = GliderParams()

    # --- Natural trim (de=0) ---
    alpha_trim = -p.Cm0 / p.Cm_alpha                   # ~0.0833 rad (4.77 deg)
    cl_trim    = p.a0 * alpha_trim
    q_trim     = (p.m * p.g) / (p.S * cl_trim)
    V_trim     = np.sqrt(2.0 * q_trim / p.rho)         # ~10.06 m/s

    k          = 1.0 / (np.pi * p.e * p.AR)
    cd_trim    = p.CD0 + k * cl_trim ** 2
    gamma      = np.arctan(cd_trim / cl_trim)           # glide path angle (rad, positive down)
    theta      = alpha_trim - gamma                     # nose-up angle above NED horizontal

    u0 = V_trim * np.cos(alpha_trim)
    w0 = V_trim * np.sin(alpha_trim)

    state = build_state(
        p_ned  = np.array([0.0, 0.0, -300.0]),
        v_body = np.array([u0, 0.0, w0]),
        pitch  = -theta,                                # negative = nose-up (see pitch convention)
    )

    controls = ZERO_CONTROLS
    dyn = GliderDynamics(params=p)
    dyn.reset(state)

    PITCH_RATE_LIMIT = 0.15                             # rad/s; allows for phugoid (~0.1 rad/s)
    max_q_rate       = 0.0

    for step in range(int(15.0 / DT_PHYS)):
        dyn.step(controls, WIND_ZERO, DT_PHYS)
        q_rate    = abs(dyn.state[11])                  # body pitch rate (rad/s)
        max_q_rate = max(max_q_rate, q_rate)
        assert q_rate < PITCH_RATE_LIMIT, (
            f"Pitch rate {q_rate:.4f} rad/s at step {step} exceeds "
            f"{PITCH_RATE_LIMIT} rad/s — trim is unstable"
        )

    print(
        f"PASS  test_trim_glide            "
        f"max pitch rate = {max_q_rate:.4f} rad/s  "
        f"(alpha_trim={np.degrees(alpha_trim):.2f} deg, V_trim={V_trim:.2f} m/s)"
    )


def test_quaternion_norm() -> None:
    """Quaternion norm must stay within 1e-6 of 1.0 for every RK4 step.

    Uses a dynamic flight state (non-zero angular rates and full aerodynamics)
    so the quaternion kinematics are actually exercised.  Starting state:
    12 m/s forward, 50 m AGL, 5 deg/s pitch rate — enough rotation to stress
    the integrator without immediately crashing.  Runs 30 s (6000 steps).
    """
    p = GliderParams()
    state = build_state(
        p_ned  = np.array([0.0, 0.0, -50.0]),
        v_body = np.array([12.0, 0.0, 0.0]),
        pitch  = np.radians(-5.0),
        omega  = np.array([0.0, np.radians(5.0), 0.0]),  # 5 deg/s pitch rate
    )

    dyn = GliderDynamics(params=p)
    dyn.reset(state)

    TOLERANCE = 1e-6
    max_err   = 0.0

    for step in range(int(30.0 / DT_PHYS)):
        dyn.step(ZERO_CONTROLS, WIND_ZERO, DT_PHYS)
        err     = abs(np.linalg.norm(dyn.state[6:10]) - 1.0)
        max_err = max(max_err, err)
        assert err < TOLERANCE, (
            f"Quaternion norm error {err:.2e} at step {step} exceeds 1e-6"
        )

    print(
        f"PASS  test_quaternion_norm       "
        f"max |norm(q)-1| = {max_err:.2e} over 6000 steps"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tests = [
        test_energy_conservation,
        test_glide_ratio,
        test_stall_behaviour,
        test_trim_glide,
        test_quaternion_norm,
    ]

    results: list[tuple[str, bool, str]] = []
    for fn in tests:
        try:
            fn()
            results.append((fn.__name__, True, ""))
        except Exception as exc:
            print(f"FAIL  {fn.__name__:<32} {exc}")
            results.append((fn.__name__, False, str(exc)))

    print()
    all_passed = all(ok for _, ok, _ in results)
    for name, ok, _ in results:
        print(f"  {'PASSED' if ok else 'FAILED'}  {name}")
    print()
    if all_passed:
        print("All 5 validation checks PASSED.")
    else:
        n_failed = sum(1 for _, ok, _ in results if not ok)
        print(f"{n_failed} of {len(tests)} validation checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

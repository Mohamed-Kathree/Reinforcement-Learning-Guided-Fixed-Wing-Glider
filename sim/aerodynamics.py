"""
aerodynamics.py
===============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Aerodynamic force and moment model for the 1.1 kg fixed-wing glider.
Implements a nonlinear lift curve (with smooth post-stall), a parabolic
drag polar, and a linear stability-derivative moment model.

Public API (consumed by GliderDynamics.derivatives via lazy import):
    forces_moments(V, alpha, beta, omega, controls, params)
        -> (F_body: ndarray[3], M_body: ndarray[3])
    CL(alpha, params) -> float   (also used by validate_glide.py)
    CD(cl, params)    -> float   (also used by validate_glide.py)

All aerodynamic parameters live on GliderParams so that domain
randomisation in env/ can override them per episode via a single copy.

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.math_utils import wind_to_body_forces


def CL(alpha: float, p) -> float:
    """Lift coefficient with smooth post-stall model.

    Pre-stall  : linear        CL = a0 * alpha
    Post-stall : exponential decay blended with 0.9*sin(2*alpha) to
    avoid a hard discontinuity at the stall angle.

    Args:
        alpha : angle of attack (rad)
        p     : GliderParams; uses p.a0 and p.a_stall

    Returns:
        cl: non-dimensional lift coefficient
    """
    if abs(alpha) <= p.a_stall:
        return float(p.a0 * alpha)
    cl_peak = p.a0 * p.a_stall
    excess  = abs(alpha) - p.a_stall
    return float(np.sign(alpha) * max(
        cl_peak * np.exp(-3.0 * excess),
        0.9 * np.sin(2.0 * alpha),
    ))


def CD(cl: float, p) -> float:
    """Drag coefficient from parabolic (Oswald) polar.

    CD = CD0 + k * CL^2,   k = 1 / (pi * e * AR)

    Args:
        cl : lift coefficient (output of CL())
        p  : GliderParams; uses p.CD0, p.e, p.AR

    Returns:
        cd: non-dimensional drag coefficient (always >= p.CD0)
    """
    k = 1.0 / (np.pi * p.e * p.AR)
    return float(p.CD0 + k * cl ** 2)


def forces_moments(
    V: float,
    alpha: float,
    beta: float,
    omega: NDArray,
    controls: dict,
    p,
) -> tuple[NDArray, NDArray]:
    """Compute aerodynamic forces [N] and moments [N*m] in the FRD body frame.

    Force coefficients are computed in the wind/stability frame, then
    transformed to body using the exact wind-to-body rotation matrix from
    math_utils (not the small-beta approximation in CLAUDE.md Section 6.5).

    Moment model: linear stability derivatives (Section 6.6).  Non-
    dimensional rates use the standard b/(2V) and c/(2V) scaling.

    Args:
        V       : airspeed (m/s)
        alpha   : angle of attack (rad)
        beta    : sideslip angle (rad)
        omega   : body angular rates [p, q, r] (rad/s)
        controls: {'aileron': rad, 'elevator': rad, 'rudder': rad (opt)}
        p       : GliderParams carrying all aero parameters

    Returns:
        F_body  : aerodynamic force in FRD body frame, shape (3,) [N]
        M_body  : aerodynamic moment in FRD body frame, shape (3,) [N*m]
    """
    q_dyn  = 0.5 * p.rho * V ** 2
    rudder = float(controls.get('rudder', 0.0))

    # --- Force coefficients ---
    cl = CL(alpha, p) + p.CL_de * controls['elevator']
    cd = CD(cl, p)    + p.CD_beta * beta ** 2
    cy = -p.CY_beta * beta + p.CY_dr * rudder

    # --- Wind-frame forces -> FRD body frame (exact rotation) ---
    L = q_dyn * p.S * cl    # lift  [N], perpendicular to velocity, upward
    D = q_dyn * p.S * cd    # drag  [N], along velocity, rearward
    Y = q_dyn * p.S * cy    # side force [N], positive rightward
    F_body = wind_to_body_forces(L, D, Y, alpha, beta)

    # --- Non-dimensional rate denominators ---
    b_2V = p.b / (2.0 * V)
    c_2V = p.c / (2.0 * V)

    # --- Moment coefficients (stability-derivative model) ---
    Cl_coeff = (
          p.Cl_beta * beta
        + p.Cl_p   * omega[0] * b_2V    # roll damping
        + p.Cl_r   * omega[2] * b_2V    # yaw-to-roll coupling
        + p.Cl_da  * controls['aileron']
    )
    Cm_coeff = (
          p.Cm0
        + p.Cm_alpha * alpha
        + p.Cm_q     * omega[1] * c_2V  # pitch damping
        + p.Cm_de    * controls['elevator']
    )
    Cn_coeff = (
          p.Cn_beta * beta
        + p.Cn_p   * omega[0] * b_2V    # roll-to-yaw coupling
        + p.Cn_r   * omega[2] * b_2V    # yaw damping
        + p.Cn_dr  * rudder
    )

    # Moments: q_dyn * S * coeff * moment_arm
    M_body = q_dyn * p.S * np.array([
        Cl_coeff * p.b,   # roll moment  (N*m)
        Cm_coeff * p.c,   # pitch moment (N*m)
        Cn_coeff * p.b,   # yaw moment   (N*m)
    ])

    return F_body, M_body

"""
math_utils.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Shared mathematical primitives used across sim/ and env/:
quaternion arithmetic, frame conversions, airdata, angle utilities,
and the RK4 integrator.

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


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: NDArray) -> NDArray:
    """3x3 rotation matrix R such that v_body = R @ v_ned.

    q = [q0, q1, q2, q3], q0 is scalar (ZYX Euler, NED->body).
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1.0 - 2.0*(q2**2 + q3**2),       2.0*(q1*q2 - q0*q3),       2.0*(q1*q3 + q0*q2)],
        [      2.0*(q1*q2 + q0*q3), 1.0 - 2.0*(q1**2 + q3**2),       2.0*(q2*q3 - q0*q1)],
        [      2.0*(q1*q3 - q0*q2),       2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1**2 + q2**2)],
    ], dtype=np.float64)


def euler_from_quat(q: NDArray) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in radians from quaternion [q0,q1,q2,q3].

    Uses ZYX (yaw->pitch->roll) Euler sequence matching standard NED aerospace.
    """
    q0, q1, q2, q3 = q
    roll  = np.arctan2(2.0*(q0*q1 + q2*q3), 1.0 - 2.0*(q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2.0*(q0*q2 - q3*q1), -1.0, 1.0))
    yaw   = np.arctan2(2.0*(q0*q3 + q1*q2), 1.0 - 2.0*(q2**2 + q3**2))
    return float(roll), float(pitch), float(yaw)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> NDArray:
    """Return unit quaternion [q0,q1,q2,q3] from ZYX Euler angles (rad).

    Rotation order applied to NED: yaw first, then pitch, then roll.
    Derived as q = q_yaw ⊗ q_pitch ⊗ q_roll.
    """
    cr, sr = np.cos(roll  / 2.0), np.sin(roll  / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw   / 2.0), np.sin(yaw   / 2.0)

    return np.array([
        cr*cp*cy + sr*sp*sy,   # q0 (scalar)
        sr*cp*cy - cr*sp*sy,   # q1
        cr*sp*cy + sr*cp*sy,   # q2
        cr*cp*sy - sr*sp*cy,   # q3
    ], dtype=np.float64)


def quat_mult(p: NDArray, q: NDArray) -> NDArray:
    """Hamilton product p ⊗ q for quaternions [q0,q1,q2,q3]."""
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dtype=np.float64)


def quat_kinematics(q: NDArray, omega: NDArray) -> NDArray:
    """Return q_dot = 0.5 * q ⊗ [0, p, q, r].

    omega: body angular rates [p, q, r] (rad/s, roll/pitch/yaw rate).
    """
    return 0.5 * quat_mult(q, np.array([0.0, omega[0], omega[1], omega[2]]))


def quat_normalize(q: NDArray) -> NDArray:
    """Return a copy of q scaled to unit length."""
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Airdata
# ---------------------------------------------------------------------------

def airdata(v_air_body: NDArray, v_min: float = 0.5) -> tuple[float, float, float]:
    """Return (V, alpha, beta) from body-frame air-relative velocity.

    v_air_body = v_body - R @ wind_ned  (m/s in FRD body frame).

    V     : airspeed (m/s), clamped to v_min to avoid singularities
    alpha : angle of attack (rad) = arctan2(w, u); positive when nose above velocity
    beta  : sideslip angle (rad) = arcsin(v/V); positive when wind from the right
    """
    u, v, w = float(v_air_body[0]), float(v_air_body[1]), float(v_air_body[2])
    V = max(np.sqrt(u**2 + v**2 + w**2), v_min)
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / V, -1.0, 1.0))
    return V, float(alpha), float(beta)


# ---------------------------------------------------------------------------
# Wind-to-body force transformation
# ---------------------------------------------------------------------------

def wind_to_body_matrix(alpha: float, beta: float) -> NDArray:
    """Exact 3x3 rotation matrix R_wb such that F_body = R_wb @ [-D, Y, -L].

    Columns are body-frame representations of wind-frame basis vectors:
      col 0 = velocity direction (x_wind)
      col 1 = y_wind (perpendicular to velocity, rightward in stability frame)
      col 2 = z_wind (downward perpendicular, completing RH system)

    Derivation: cross-product construction from v_body = V[ca*cb, sb, sa*cb].
    This is exact for all alpha and beta; the CLAUDE.md per-component formulas
    are an approximation for small beta (differ only in the ca*sb*Y -> Fx term).
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    return np.array([
        [ ca*cb, -ca*sb, -sa],
        [ sb,     cb,    0.0],
        [ sa*cb, -sa*sb,  ca],
    ], dtype=np.float64)


def wind_to_body_forces(
    L: float, D: float, Y: float, alpha: float, beta: float
) -> NDArray:
    """Transform aerodynamic forces from wind frame to FRD body frame.

    L : lift magnitude (N), positive upward
    D : drag magnitude (N), positive rearward (opposing motion)
    Y : side force (N), positive rightward

    In FRD body frame, upward lift produces negative Fz (z points down).
    Uses the exact wind-to-body rotation: F_body = R_wb @ [-D, Y, -L].
    """
    return wind_to_body_matrix(alpha, beta) @ np.array([-D, Y, -L])


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def wrap_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_2pi(angle: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return float(float(angle) % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def rk4_step(
    state: NDArray,
    controls: dict,
    t: float,
    dt: float,
    deriv_fn,
) -> NDArray:
    """Fourth-order Runge-Kutta step with quaternion renormalisation.

    deriv_fn(state, controls, t) -> state_dot  (same shape as state)

    The quaternion slice state[6:10] is renormalised after every step.
    Mandatory for glider dynamics; Euler integration diverges at 200 Hz.
    """
    k1 = deriv_fn(state,                 controls, t)
    k2 = deriv_fn(state + 0.5*dt*k1,    controls, t + 0.5*dt)
    k3 = deriv_fn(state + 0.5*dt*k2,    controls, t + 0.5*dt)
    k4 = deriv_fn(state + dt*k3,         controls, t + dt)

    new_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    new_state[6:10] /= np.linalg.norm(new_state[6:10])   # always renormalise
    return new_state

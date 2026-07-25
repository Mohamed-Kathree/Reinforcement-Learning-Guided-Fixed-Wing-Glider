"""
math_utils.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Shared mathematical primitives used across sim/ and env/:
quaternion/Euler conversions, frame transforms, angle utilities,
and the 13-element state vector builder.

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


def quat_normalize(q: NDArray) -> NDArray:
    """Return a copy of q scaled to unit length."""
    return q / np.linalg.norm(q)


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
# State vector construction
# ---------------------------------------------------------------------------

def _zero_state() -> NDArray:
    """13-element zero state with a valid identity quaternion."""
    s = np.zeros(13, dtype=np.float64)
    s[6] = 1.0  # q0=1 -> body axes aligned with NED, at origin
    return s


def build_state(
    *,
    p_ned:  NDArray | None = None,
    v_body: NDArray | None = None,
    roll:   float = 0.0,
    pitch:  float = 0.0,
    yaw:    float = 0.0,
    omega:  NDArray | None = None,
) -> NDArray:
    """Build a 13-element state vector from human-readable components.

    State layout: [0:3] p_ned, [3:6] v_body, [6:10] quat, [10:13] omega.
    All keyword arguments are optional; omitted quantities default to zero
    (position at NED origin, body aligned with NED, zero velocities).

    Args:
        p_ned  : NED position [p_n, p_e, p_d] (m)
        v_body : body-frame velocity [u, v, w] (m/s)
        roll   : roll angle (rad, ZYX convention)
        pitch  : pitch angle (rad)
        yaw    : yaw angle (rad)
        omega  : body angular rates [p, q, r] (rad/s)

    Example -- launch state 100 m AGL, 15 m/s nose-up at 15 deg, heading north:
        state = build_state(
            p_ned  = np.array([0, 0, -100]),
            v_body = np.array([15*np.cos(np.radians(15)), 0, -15*np.sin(np.radians(15))]),
            pitch  = np.radians(15),
        )
    """
    s = _zero_state()
    if p_ned  is not None:
        s[0:3]  = np.asarray(p_ned,  dtype=np.float64)
    if v_body is not None:
        s[3:6]  = np.asarray(v_body, dtype=np.float64)
    s[6:10] = euler_to_quat(roll, pitch, yaw)
    if omega  is not None:
        s[10:13] = np.asarray(omega, dtype=np.float64)
    return s


# Zero-deflection controls — useful as a starting point for tests/trim runs.
ZERO_CONTROLS: dict = {'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}

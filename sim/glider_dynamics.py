"""
glider_dynamics.py
==================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

6-DOF rigid-body equations of motion for an unpowered fixed-wing glider,
integrated with a fourth-order Runge-Kutta scheme at 200 Hz.

State vector (13 elements, float64):
    [0:3]   p_ned  -- NED position (m).  altitude = -state[2]
    [3:6]   v_body -- FRD body velocity (m/s): [u, v, w]
    [6:10]  q      -- attitude quaternion [q0,q1,q2,q3], NED->body, q0 scalar
    [10:13] omega  -- body angular rates [p, q, r] (rad/s)

Controls dict (keys, values in radians):
    'aileron'   -- lateral servo deflection
    'elevator'  -- longitudinal servo deflection
    'rudder'    -- directional servo deflection (optional, default 0)

Aerodynamics interface:
    forces_moments(V, alpha, beta, omega, controls, params)
        -> (F_body: ndarray[3], M_body: ndarray[3])
    If not supplied at construction time, sim.aerodynamics.forces_moments
    is imported lazily on first call.

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from sim.math_utils import (
    airdata,
    euler_from_quat,
    euler_to_quat,
    quat_kinematics,
    quat_to_rotmat,
    rk4_step,
)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

@dataclass
class GliderParams:
    """Physical and aerodynamic constants for the 1.1 kg fixed-wing glider.

    All values are the nominal (non-randomised) defaults; domain
    randomisation in env/ modifies copies of this object per episode.

    Inertia: After changing any of Ixx / Iyy / Izz / Ixz, call
    _build_inertia() to keep I and I_inv consistent.  When scaling the
    full tensor uniformly, replace I and I_inv directly:
        params.I     = base_I * scale
        params.I_inv = np.linalg.inv(params.I)

    Aerodynamics: a0, CD0, and all stability derivatives are plain fields
    — modify them directly for domain randomisation:
        params.a0  *= CL_slope_scale
        params.CD0 *= CD0_scale
    """

    # --- Geometry & mass ------------------------------------------------
    m:   float = 1.1    # kg,  all-up mass
    S:   float = 0.38   # m^2, wing area
    b:   float = 2.0    # m,   wingspan
    c:   float = 0.19   # m,   mean aerodynamic chord  (S/b)
    AR:  float = 10.5   # --   aspect ratio             (b^2/S)

    # --- Atmosphere (sea-level ISA) -------------------------------------
    rho: float = 1.225  # kg/m^3
    g:   float = 9.81   # m/s^2

    # --- Inertia tensor components (kg*m^2, FRD body frame) -------------
    Ixx: float = 0.18   # roll
    Iyy: float = 0.10   # pitch
    Izz: float = 0.26   # yaw
    Ixz: float = 0.01   # roll-yaw cross term (small; symmetric glider)

    # --- Lift model (sim/aerodynamics.py) --------------------------------
    a0:      float = 5.5                       # lift-curve slope (rad^-1)
    a_stall: float = float(np.radians(12.0))   # stall angle (rad)

    # --- Drag polar -------------------------------------------------------
    CD0: float = 0.025   # parasite/zero-lift drag coefficient
    e:   float = 0.85    # Oswald span efficiency factor

    # --- Force stability derivatives (per rad) ----------------------------
    CL_de:   float =  0.30   # elevator lift effectiveness
    CD_beta: float =  0.10   # drag increment due to sideslip (beta^2 term)
    CY_beta: float = -0.30   # side-force slope  [cy = -CY_beta*beta]
    CY_dr:   float =  0.00   # rudder side-force effectiveness

    # --- Roll moment derivatives (per rad) --------------------------------
    Cl_beta: float = -0.08   # dihedral roll-due-to-sideslip
    Cl_p:    float = -0.45   # roll damping (negative = stable)
    Cl_r:    float =  0.10   # yaw-to-roll coupling
    Cl_da:   float =  0.15   # aileron roll authority

    # --- Pitch moment derivatives (per rad) ------------------------------
    Cm0:      float =  0.05   # trim pitching moment at alpha=0
    Cm_alpha: float = -0.60   # pitch stiffness (negative = statically stable)
    Cm_q:     float = -8.0    # pitch damping (negative = stable)
    Cm_de:    float = -1.2    # elevator pitch authority

    # --- Yaw moment derivatives (per rad) --------------------------------
    Cn_beta: float =  0.06   # weathercock stability (positive = stable)
    Cn_p:    float = -0.03   # roll-to-yaw coupling
    Cn_r:    float = -0.08   # yaw damping (negative = stable)
    Cn_dr:   float = -0.05   # rudder authority

    def __post_init__(self) -> None:
        self._build_inertia()

    def _build_inertia(self) -> None:
        """Rebuild I (3x3) and I_inv from the scalar inertia components."""
        self.I: NDArray = np.array(
            [
                [ self.Ixx,  0.0,       -self.Ixz],
                [ 0.0,       self.Iyy,   0.0     ],
                [-self.Ixz,  0.0,        self.Izz],
            ],
            dtype=np.float64,
        )
        self.I_inv: NDArray = np.linalg.inv(self.I)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class GliderDynamics:
    """6-DOF rigid-body dynamics for the unpowered glider.

    Typical usage (physics loop at 200 Hz):
        dyn = GliderDynamics()
        dyn.reset(initial_state)
        while flying:
            state = dyn.step(controls, wind_ned, dt=0.005)

    For the RL environment (10 physics substeps per 20 Hz policy step):
        state = dyn.multi_step(controls, wind_ned, dt_phys=0.005, n_substeps=10)
    """

    def __init__(
        self,
        params: GliderParams | None = None,
        aero_fn: Callable | None = None,
    ) -> None:
        """
        Args:
            params  : glider physical parameters (default: GliderParams())
            aero_fn : aerodynamics callable; lazy-imports sim.aerodynamics
                      if None.  Signature:
                        fn(V, alpha, beta, omega, controls, params)
                          -> (F_body ndarray[3], M_body ndarray[3])
        """
        self.p = params if params is not None else GliderParams()
        self._aero_fn = aero_fn
        self.state: NDArray = _zero_state()

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def reset(self, state: NDArray) -> None:
        """Copy a 13-element state vector into the dynamics object."""
        self.state = np.array(state, dtype=np.float64)

    @property
    def position_ned(self) -> NDArray:
        """NED position [p_n, p_e, p_d] (m)."""
        return self.state[0:3].copy()

    @property
    def velocity_body(self) -> NDArray:
        """Body-frame velocity [u, v, w] (m/s)."""
        return self.state[3:6].copy()

    @property
    def attitude_quat(self) -> NDArray:
        """Attitude quaternion [q0, q1, q2, q3]."""
        return self.state[6:10].copy()

    @property
    def angular_rates(self) -> NDArray:
        """Body angular rates [p, q, r] (rad/s)."""
        return self.state[10:13].copy()

    @property
    def altitude(self) -> float:
        """Altitude above the NED origin (m), positive up.  = -p_d."""
        return float(-self.state[2])

    def euler_angles(self) -> tuple[float, float, float]:
        """Return (roll, pitch, yaw) in radians from the current attitude."""
        return euler_from_quat(self.state[6:10])

    def air_data(self, wind_ned: NDArray) -> tuple[float, float, float]:
        """Return (V, alpha, beta) for the current state and given wind.

        Args:
            wind_ned: NED wind vector [wn, we, wd] (m/s)

        Returns:
            V     : airspeed (m/s)
            alpha : angle of attack (rad)
            beta  : sideslip angle (rad)
        """
        R = quat_to_rotmat(self.state[6:10])
        v_air = self.state[3:6] - R @ wind_ned
        return airdata(v_air)

    # ------------------------------------------------------------------
    # Aerodynamics resolver
    # ------------------------------------------------------------------

    @property
    def _aero(self) -> Callable:
        if self._aero_fn is None:
            from sim.aerodynamics import forces_moments  # noqa: PLC0415
            self._aero_fn = forces_moments
        return self._aero_fn

    # ------------------------------------------------------------------
    # Equations of motion
    # ------------------------------------------------------------------

    def derivatives(
        self,
        state: NDArray,
        controls: dict,
        wind_ned: NDArray,
    ) -> NDArray:
        """Compute the 13-element state derivative vector.

        Implements the equations of motion from the design spec:

        Position kinematics  : p_dot_ned  = R^T @ v_body
        Translational (body) : v_dot      = F/m - omega x v_body
        Rotational (body)    : omega_dot  = I^-1 (M - omega x (I omega))
        Quaternion kinematics: q_dot      = 0.5 * q ⊗ [0, omega]

        where:
            F = F_aero + F_grav   (both in body frame)
            F_grav = R @ [0, 0, m*g]   (gravity [0,0,+mg] NED -> body)

        Args:
            state    : 13-element state vector at time t
            controls : servo deflection dict (rad)
            wind_ned : NED wind [wn, we, wd] (m/s); held constant per RK4 step

        Returns:
            state_dot: 13-element derivative array (same layout as state)
        """
        v_body = state[3:6]
        q      = state[6:10]
        omega  = state[10:13]

        # NED -> body rotation matrix
        R = quat_to_rotmat(q)

        # Air-relative velocity in body frame
        # v_air = v_body - R @ wind_ned
        # (R maps NED wind into body frame so we can subtract)
        v_air_body = v_body - R @ wind_ned
        V, alpha, beta = airdata(v_air_body)

        # Aerodynamic forces [N] and moments [N*m] in body frame
        F_aero, M = self._aero(V, alpha, beta, omega, controls, self.p)

        # Gravity in body frame.
        # In NED frame gravity is [0, 0, +m*g] (down = positive z).
        # R maps NED -> body, so F_grav_body = R @ [0, 0, m*g].
        F_grav = R @ np.array([0.0, 0.0, self.p.m * self.p.g])

        # Total force in body frame
        F = F_aero + F_grav

        # --- Translational dynamics (body frame, no thrust) ---
        # m * v_dot = F - omega x (m * v_body)
        # => v_dot = F/m - omega x v_body
        v_dot = F / self.p.m - np.cross(omega, v_body)

        # --- Rotational dynamics ---
        # I * omega_dot = M - omega x (I * omega)
        # => omega_dot = I^-1 (M - omega x (I omega))
        omega_dot = self.p.I_inv @ (M - np.cross(omega, self.p.I @ omega))

        # --- Quaternion kinematics ---
        # q_dot = 0.5 * q ⊗ [0, p, q, r]
        q_dot = quat_kinematics(q, omega)

        # --- Position kinematics ---
        # p_dot_NED = R^T @ v_body   (R^T maps body -> NED)
        p_dot = R.T @ v_body

        return np.concatenate([p_dot, v_dot, q_dot, omega_dot])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def step(
        self,
        controls: dict,
        wind_ned: NDArray,
        dt: float,
    ) -> NDArray:
        """Advance state by one physics step using RK4.

        dt should be 0.005 s (200 Hz). Wind and controls are held constant
        within the step (zero-order hold between calls).

        Returns a copy of the updated state vector.
        """
        def _deriv(s: NDArray, c: dict, _t: float) -> NDArray:
            return self.derivatives(s, c, wind_ned)

        self.state = rk4_step(self.state, controls, 0.0, dt, _deriv)
        return self.state.copy()

    def multi_step(
        self,
        controls: dict,
        wind_ned: NDArray,
        dt_phys: float,
        n_substeps: int,
    ) -> NDArray:
        """Run n_substeps consecutive RK4 steps.

        Called by env.step() to bridge the 20 Hz policy rate and the
        200 Hz physics rate:
            dyn.multi_step(controls, wind_ned, dt_phys=0.005, n_substeps=10)

        Controls and wind are held constant across all substeps; the
        environment updates them between multi_step calls.

        Returns a copy of the final state vector.
        """
        for _ in range(n_substeps):
            self.step(controls, wind_ned, dt_phys)
        return self.state.copy()


# ---------------------------------------------------------------------------
# Module-level helpers
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
            pitch  = -np.radians(15),   # nose up in NED = negative pitch? No: see note.
            yaw    = 0.0,
        )
    Note on pitch sign: pitch > 0 means nose up (positive alpha convention).
    For a launch pitched 15 deg nose-up: pitch = +np.radians(15) and
    w = -V*sin(gamma) (negative w in FRD = nose above velocity).
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

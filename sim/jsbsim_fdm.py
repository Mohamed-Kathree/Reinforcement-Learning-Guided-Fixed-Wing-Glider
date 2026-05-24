"""
jsbsim_fdm.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

JSBSim Flight Dynamics Model wrapper.  Drop-in replacement for GliderDynamics
with the same external interface so env/glider_env.py changes stay minimal.

Architecture role:
    JSBSim owns the 6-DOF EOM, aerodynamics, atmosphere, and actuator dynamics.
    Python retains the AttitudeController (ESP32 firmware parity), safety shield,
    sensor models, wind injection, reward, and RL stack.

Key gotcha — set_dt() BEFORE load_model():
    JSBSim's FCS components (actuator lag/rate limiters) cache the simulation
    delta-t at load_model() time.  If set_dt() is called after load_model()
    the actuators use the default 1/120 s dt, making rate limits 5/3 too loose
    at 200 Hz.  Always call set_dt() first.

State vector layout (13 elements, float64) — identical to GliderDynamics:
    [0:3]   p_ned  -- NED position (m).  p_d = -h_agl; p_d > 0 → ground impact
    [3:6]   v_body -- FRD inertial body velocity [u, v, w] (m/s)
    [6:10]  q      -- attitude quaternion [q0, q1, q2, q3], NED→body, q0 scalar
    [10:13] omega  -- body angular rates [p, q, r] (rad/s)

Controls format accepted by step():
    dict with keys 'aileron', 'elevator', 'rudder' in RADIANS
    (same as old ActuatorSuite.step() output / GliderDynamics.step() input)
    step() normalises them to [-1, 1] before writing to fcs/*-cmd-norm.
    The JSBSim FCS then applies actuator lag + rate limit + saturation.

NED position tracking:
    JSBSim's position/distance-from-start-*-mt properties give displacement
    from the IC (launch) point.  JSBSimFDM stores the launch NED offset at
    reset() and adds it to every read so position_ned is always relative to
    the home/origin of the episode.

Domain randomisation:
    Call apply_domain_rand() after reset() to write per-episode scale factors
    to the custom aero properties defined in aircraft/rlglider/rlglider.xml.

Units: SI at every public boundary (m, m/s, rad, rad/s).
       All English-unit conversions are internal to this module.
"""

from __future__ import annotations

import os
from typing import Union

import jsbsim
import numpy as np
from numpy.typing import NDArray

from sim.math_utils import euler_from_quat, euler_to_quat

# ---------------------------------------------------------------------------
# Unit conversion constants (English ↔ SI, used only inside this module)
# ---------------------------------------------------------------------------

M2FT   = 3.280839895
FT2M   = 1.0 / M2FT
MS2FPS = M2FT            # 1 m/s = M2FT fps
FPS2MS = FT2M            # 1 fps = FT2M m/s

# ---------------------------------------------------------------------------
# Surface deflection limits (from actuator_models.py — shared source of truth)
# ---------------------------------------------------------------------------

AILERON_MAX_RAD  = np.radians(25.0)
ELEVATOR_MAX_RAD = np.radians(20.0)
RUDDER_MAX_RAD   = np.radians(25.0)


# ---------------------------------------------------------------------------
# JSBSimFDM
# ---------------------------------------------------------------------------

class JSBSimFDM:
    """JSBSim-backed 6-DOF FDM with the same external interface as GliderDynamics.

    One instance per environment.  The FGFDMExec is created once at construction
    and reused across episodes via reset() → run_ic().  Do NOT create a new
    JSBSimFDM per episode — loading JSBSim XML is expensive.

    Thread safety: one instance is safe to use in a single thread.
    For vectorised training use one instance per environment (DummyVecEnv or
    SubprocVecEnv are both fine since each env owns its own FGFDMExec).
    """

    def __init__(
        self,
        data_root: str | None = None,
        dt_phys:   float      = 0.005,
        aircraft:  str        = "rlglider",
    ) -> None:
        """
        Args:
            data_root : project root containing aircraft/rlglider/rlglider.xml.
                        Defaults to the directory two levels above this file
                        (i.e. the project root when installed at sim/jsbsim_fdm.py).
            dt_phys   : physics time step (s).  Must match DT_PHYS in glider_env.py.
            aircraft  : JSBSim model name (loads aircraft/<name>/<name>.xml).
        """
        if data_root is None:
            # sim/ is one level below the project root
            data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._root     = data_root
        self._dt       = dt_phys
        self._aircraft = aircraft

        # Launch NED offset (set at reset, used to compute position relative to home)
        self._p_ned_launch: NDArray = np.zeros(3, dtype=np.float64)

        # Fault flag: set True if fdm.run() returns False (solver failure)
        self._fault: bool = False

        # Create the FGFDMExec (expensive — done once)
        self._fdm: jsbsim.FGFDMExec = self._make_fdm()

    # ------------------------------------------------------------------
    # Private setup
    # ------------------------------------------------------------------

    def _make_fdm(self) -> jsbsim.FGFDMExec:
        fdm = jsbsim.FGFDMExec(root_dir=self._root)
        fdm.set_debug_level(0)
        # CRITICAL: set_dt BEFORE load_model so FCS rate limiters cache the
        # correct dt.  The JSBSim default is 1/120 s; at 200 Hz this would
        # make every rate limit 5/3 too loose.
        fdm.set_dt(self._dt)
        ok = fdm.load_model(self._aircraft)
        if not ok:
            raise RuntimeError(
                f"JSBSim: load_model('{self._aircraft}') failed — "
                f"check that {self._root}/aircraft/{self._aircraft}/{self._aircraft}.xml exists"
            )
        return fdm

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self, state: NDArray) -> None:
        """Reset JSBSim to the given 13-element launch state.

        Converts the state vector into JSBSim ic/* properties, then calls
        run_ic() to prime the integrator.  The FCS actuator states are zeroed
        so each episode starts with neutral surfaces.

        Args:
            state : 13-element float64 array [p_ned(3), v_body(3), quat(4), omega(3)]
        """
        state = np.asarray(state, dtype=np.float64)

        # Store launch NED offset so position_ned is always home-relative
        self._p_ned_launch = state[0:3].copy()

        v_body = state[3:6]     # [u, v, w] m/s, FRD inertial
        quat   = state[6:10]    # [q0, q1, q2, q3]
        omega  = state[10:13]   # [p, q, r] rad/s

        # Altitude AGL:  state[2] = p_d = -h_agl  =>  h_agl = -state[2]
        altitude_m = float(-state[2])

        # Airspeed and flow angles from body velocity
        # (no wind at reset → inertial and air-relative velocity are equal)
        u, v, w = float(v_body[0]), float(v_body[1]), float(v_body[2])
        vt    = max(np.sqrt(u*u + v*v + w*w), 0.5)   # clamp to avoid alpha singularity
        alpha = float(np.arctan2(w, u))               # AoA (rad)
        beta  = float(np.arcsin(np.clip(v / vt, -1.0, 1.0)))   # sideslip (rad)

        # Euler attitude from quaternion (ZYX convention, NED→body)
        roll, pitch, yaw = euler_from_quat(quat)

        fdm = self._fdm

        # Neutral domain-randomisation properties (apply_domain_rand overrides)
        fdm["aero/cl-mult"]       = 1.0
        fdm["aero/cd0-add"]       = 0.0
        fdm["aero/cd-mult"]       = 1.0
        fdm["aero/ctrl-eff-mult"] = 1.0

        # Geographic reference: fixed at equator/prime meridian.
        # Absolute position does not matter for the RTL task — only relative
        # displacement (distance-from-start-*-mt) is used.
        fdm["ic/lat-gc-deg"]   = 0.0
        fdm["ic/long-gc-deg"]  = 0.0

        # Translational IC
        fdm["ic/h-agl-ft"]     = altitude_m * M2FT
        fdm["ic/vt-fps"]       = vt          * MS2FPS
        fdm["ic/alpha-deg"]    = np.degrees(alpha)
        fdm["ic/beta-deg"]     = np.degrees(beta)

        # Attitude IC
        fdm["ic/phi-deg"]      = np.degrees(roll)
        fdm["ic/theta-deg"]    = np.degrees(pitch)
        fdm["ic/psi-true-deg"] = np.degrees(yaw)

        # Angular rate IC
        fdm["ic/p-rad_sec"]    = float(omega[0])
        fdm["ic/q-rad_sec"]    = float(omega[1])
        fdm["ic/r-rad_sec"]    = float(omega[2])

        fdm.run_ic()

        # Zero surface commands — actuators start from rest each episode
        fdm["fcs/aileron-cmd-norm"]  = 0.0
        fdm["fcs/elevator-cmd-norm"] = 0.0
        fdm["fcs/rudder-cmd-norm"]   = 0.0

        self._fault = False

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(
        self,
        controls: dict,
        wind_ned: NDArray,
        dt: float,           # accepted for interface compatibility; must equal self._dt
    ) -> NDArray:
        """Advance the simulation by one physics step.

        Args:
            controls : {'aileron', 'elevator', 'rudder'} in RADIANS.
                       Identical format to old ActuatorSuite.step() output.
                       Internally normalised to [-1, 1] before writing to
                       fcs/*-cmd-norm; the JSBSim FCS applies actuator lag +
                       rate limit + saturation.
            wind_ned : NED wind vector [wn, we, wd] (m/s)
            dt       : physics step size (s); must equal DT_PHYS (0.005 s).
                       Parameter accepted for API compatibility.

        Returns:
            Current 13-element state array (same as self.state).
        """
        fdm = self._fdm

        # Inject steady wind (JSBSim computes alpha/beta/qbar from wind-relative V)
        fdm["atmosphere/wind-north-fps"] = float(wind_ned[0]) * MS2FPS
        fdm["atmosphere/wind-east-fps"]  = float(wind_ned[1]) * MS2FPS
        fdm["atmosphere/wind-down-fps"]  = float(wind_ned[2]) * MS2FPS

        # Normalise radian commands → [-1, 1] and write to FCS inputs
        fdm["fcs/aileron-cmd-norm"] = float(
            np.clip(controls.get("aileron",  0.0) / AILERON_MAX_RAD,  -1.0, 1.0)
        )
        fdm["fcs/elevator-cmd-norm"] = float(
            np.clip(controls.get("elevator", 0.0) / ELEVATOR_MAX_RAD, -1.0, 1.0)
        )
        fdm["fcs/rudder-cmd-norm"] = float(
            np.clip(controls.get("rudder",   0.0) / RUDDER_MAX_RAD,   -1.0, 1.0)
        )

        ok = fdm.run()
        if not ok:
            self._fault = True   # caller checks self.fault and ends the episode

        return self.state

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def apply_domain_rand(
        self,
        *,
        cl_mult:       float = 1.0,
        cd0_add:       float = 0.0,
        ctrl_eff_mult: float = 1.0,
    ) -> None:
        """Write per-episode domain-randomisation scale factors to JSBSim.

        Call this after reset() and before the first step().  These properties
        are referenced inside aircraft/rlglider/rlglider.xml and scale the
        aerodynamic model without reloading the XML.

        Args:
            cl_mult       : CL table multiplier (nominal 1.0; range 0.9–1.1)
            cd0_add       : additive CD0 offset (nominal 0.0; range −0.005–0.010)
            ctrl_eff_mult : control-effectiveness scale (nominal 1.0; range 0.75–1.10)
        """
        fdm = self._fdm
        fdm["aero/cl-mult"]       = float(cl_mult)
        fdm["aero/cd0-add"]       = float(cd0_add)
        fdm["aero/ctrl-eff-mult"] = float(ctrl_eff_mult)

    # ------------------------------------------------------------------
    # State read-outs  (SI, NED/FRD frames)
    # ------------------------------------------------------------------

    @property
    def position_ned(self) -> NDArray:
        """[N, E, D] in metres, relative to the episode home/origin.

        JSBSim distance-from-start tracks displacement from the IC (launch)
        point.  We add the stored launch offset so the returned position is
        always relative to the home point (NED origin used by the reward fn).
        """
        fdm = self._fdm
        p_n = (self._p_ned_launch[0]
               + fdm["position/distance-from-start-lat-mt"])
        p_e = (self._p_ned_launch[1]
               + fdm["position/distance-from-start-lon-mt"])
        # D derived from current AGL altitude;  p_d > 0 → crashed (below NED origin)
        p_d = -(fdm["position/h-agl-ft"] * FT2M)
        return np.array([p_n, p_e, p_d], dtype=np.float64)

    @property
    def velocity_body(self) -> NDArray:
        """[u, v, w] inertial body velocity in m/s, FRD."""
        fdm = self._fdm
        return np.array([
            fdm["velocities/u-fps"] * FPS2MS,
            fdm["velocities/v-fps"] * FPS2MS,
            fdm["velocities/w-fps"] * FPS2MS,
        ], dtype=np.float64)

    @property
    def attitude_quat(self) -> NDArray:
        """[q0, q1, q2, q3] attitude quaternion (NED→body, q0 scalar).

        Built from JSBSim Euler angles via euler_to_quat() so the quaternion
        convention matches GliderDynamics exactly.
        """
        roll, pitch, yaw = self.euler_angles()
        return euler_to_quat(roll, pitch, yaw)

    @property
    def angular_rates(self) -> NDArray:
        """[p, q, r] body angular rates in rad/s, FRD."""
        fdm = self._fdm
        return np.array([
            fdm["velocities/p-rad_sec"],
            fdm["velocities/q-rad_sec"],
            fdm["velocities/r-rad_sec"],
        ], dtype=np.float64)

    @property
    def altitude(self) -> float:
        """AGL altitude in metres (positive above ground)."""
        return float(self._fdm["position/h-agl-ft"] * FT2M)

    def euler_angles(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in radians, ZYX convention."""
        fdm = self._fdm
        return (
            float(fdm["attitude/phi-rad"]),
            float(fdm["attitude/theta-rad"]),
            float(fdm["attitude/psi-rad"]),
        )

    @property
    def airspeed(self) -> float:
        """True airspeed in m/s (wind-relative, from JSBSim aerodynamics)."""
        return float(self._fdm["velocities/vt-fps"] * FPS2MS)

    @property
    def state(self) -> NDArray:
        """13-element state array [p_ned(3), v_body(3), quat(4), omega(3)].

        Layout is identical to GliderDynamics.state so SensorSuite,
        analysis/plot_trajectories.py, and the reward function work unchanged.
        """
        s = np.empty(13, dtype=np.float64)
        s[0:3]   = self.position_ned
        s[3:6]   = self.velocity_body
        s[6:10]  = self.attitude_quat
        s[10:13] = self.angular_rates
        return s

    @property
    def fault(self) -> bool:
        """True if fdm.run() returned False (JSBSim solver failure).

        The env should treat a fault as a crash and end the episode.
        """
        return self._fault

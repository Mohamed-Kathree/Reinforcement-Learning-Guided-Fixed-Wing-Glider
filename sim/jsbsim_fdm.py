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
    N/E come from a local tangent-plane projection of JSBSim's true
    geocentric lat/lon relative to the lat/lon captured at reset() -- NOT
    from position/distance-from-start-lat/lon-mt, which report unsigned
    magnitude-only displacement and silently give the wrong direction once
    net displacement crosses to the other side of the launch point (see
    position_ned's docstring for the full explanation and validation).
    JSBSimFDM stores the launch NED offset at reset() and adds it to every
    read so position_ned is always relative to the home/origin of the episode.

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

from sim.mass_solver import N_POINT_MASSES, MassPropertySolver
from sim.math_utils import euler_from_quat, euler_to_quat

# ---------------------------------------------------------------------------
# Unit conversion constants (English ↔ SI, used only inside this module)
# ---------------------------------------------------------------------------

M2FT   = 3.280839895
FT2M   = 1.0 / M2FT
MS2FPS = M2FT            # 1 m/s = M2FT fps
FPS2MS = FT2M            # 1 fps = FT2M m/s
KG2LBS = 2.20462
M2IN   = 39.3701

# WGS84 semi-major axis (m). Used for a local tangent-plane (equirectangular)
# projection of lat/lon onto NED North/East -- see position_ned's docstring
# for why this replaced JSBSim's distance-from-start-lat/lon-mt properties.
R_EARTH_M = 6378137.0

# ---------------------------------------------------------------------------
# Surface deflection limits (from actuator_models.py — shared source of truth)
# ---------------------------------------------------------------------------

AILERON_MAX_RAD  = np.radians(25.0)
ELEVATOR_MAX_RAD = np.radians(20.0)
RUDDER_MAX_RAD   = np.radians(25.0)

# Domain-randomisation hooks declared in aircraft/rlglider/rlglider.xml.
# reset() restores each to the default the loaded model declares (read back
# at construction, never copied here), so no episode inherits the last one's.
DR_PROPERTIES: tuple[str, ...] = (
    "aero/cl-mult", "aero/clmax-scale", "aero/cd0-add", "aero/cd-mult",
    "aero/k-induced", "aero/cm0-offset",
    "aero/cmalpha-mult", "aero/cmq-mult", "aero/elev-eff-mult",
    "aero/clp-mult", "aero/ail-eff-mult", "aero/clbeta-mult",
    "aero/cnbeta-mult", "aero/cnr-mult", "aero/rud-eff-mult",
    "fcs/trim-bias-roll", "fcs/trim-bias-pitch",
)

# env/curriculum.py DR_RANGES / DR_ONESIDED key -> aircraft-file property,
# for every key written directly. Keys handled specially in
# apply_domain_rand(): MASS_KEYS (point-mass solver) and 'sw_mult'
# (multiplies the model's own metrics/Sw-sqft).
DR_PROPERTY_MAP: dict[str, str] = {
    "cl_mult":         "aero/cl-mult",
    "clmax_scale":     "aero/clmax-scale",
    "cd0_add":         "aero/cd0-add",
    "k_induced":       "aero/k-induced",
    "cm0_offset":      "aero/cm0-offset",
    "cmalpha_mult":    "aero/cmalpha-mult",
    "cmq_mult":        "aero/cmq-mult",
    "elev_eff_mult":   "aero/elev-eff-mult",
    "clp_mult":        "aero/clp-mult",
    "ail_eff_mult":    "aero/ail-eff-mult",
    "clbeta_mult":     "aero/clbeta-mult",
    "cnbeta_mult":     "aero/cnbeta-mult",
    "cnr_mult":        "aero/cnr-mult",
    "rud_eff_mult":    "aero/rud-eff-mult",
    "trim_bias_roll":  "fcs/trim-bias-roll",
    "trim_bias_pitch": "fcs/trim-bias-pitch",
}
MASS_KEYS: tuple[str, ...] = ("mass_kg", "cg_x_m", "ixx", "iyy")


def apply_airframe(fdm: jsbsim.FGFDMExec, params: dict[str, float],
                   solver: MassPropertySolver, sw_declared_sqft: float) -> None:
    """Write one airframe (DR-table keys) to a raw FGFDMExec; ends with run_ic().

    Shared by JSBSimFDM.apply_domain_rand() and validate_glide.py's sweep so
    both apply parameters identically. See apply_domain_rand() for the key
    semantics. Raises KeyError on unknown or incomplete keys.
    """
    unknown = set(params) - set(DR_PROPERTY_MAP) - set(MASS_KEYS) - {"sw_mult"}
    if unknown:
        raise KeyError(f"unknown domain-randomisation keys: {sorted(unknown)}")

    for key, prop in DR_PROPERTY_MAP.items():
        if key in params:
            fdm[prop] = float(params[key])
    if "sw_mult" in params:
        fdm["metrics/Sw-sqft"] = sw_declared_sqft * float(params["sw_mult"])

    present = [k for k in MASS_KEYS if k in params]
    if present and len(present) != len(MASS_KEYS):
        raise KeyError(f"mass properties need all of {MASS_KEYS}, got {present}")
    if present:
        solver.solve(fdm, *(float(params[k]) for k in MASS_KEYS))
    else:
        fdm.run_ic()

# Number of <contact> elements in aircraft/rlglider/rlglider.xml
# (NOSE, TAIL, LEFT_WINGTIP, RIGHT_WINGTIP) -- belly-skid ground contact points.
N_CONTACT_POINTS = 4


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

        # Launch lat/lon reference (radians), captured at reset() -- position_ned
        # projects current lat/lon onto a local tangent plane relative to this.
        self._lat0_rad: float = 0.0
        self._lon0_rad: float = 0.0

        # Fault flag: set True if fdm.run() returns False (solver failure)
        self._fault: bool = False

        # Create the FGFDMExec (expensive — done once)
        self._fdm: jsbsim.FGFDMExec = self._make_fdm()

        # <emptywt> as loaded, read back from JSBSim rather than hard-coded:
        # the XML is generated from aircraft/mass_properties.yaml, so this
        # follows it automatically. apply_domain_rand()'s mass_kg is reached
        # via the BALLAST point mass relative to this reference.
        self.empty_weight_kg: float = float(self._fdm["inertia/empty-weight-lbs"]) / KG2LBS

        # Neutral values of the domain-randomisation hooks, as declared by
        # the aircraft file.
        self._dr_defaults: dict[str, float] = {p: float(self._fdm[p]) for p in DR_PROPERTIES}
        # Wing area as declared; sw_mult scales this (writable, survives run_ic).
        self._dr_defaults["metrics/Sw-sqft"] = float(self._fdm["metrics/Sw-sqft"])

        # Per-episode mass/CG/inertia via point masses (sim/mass_solver.py).
        # Its cache rarely hits in training (continuous targets) but nearly
        # always in evaluation, where the Phase 5 sweep corners and the
        # frozen held-out airframes repeat exactly.
        self.mass_solver = MassPropertySolver()

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

        # Neutral domain-randomisation properties and an empty point-mass set
        # (apply_domain_rand overrides), so no episode inherits the last one's.
        for prop, value in self._dr_defaults.items():
            fdm[prop] = value
        for i in range(N_POINT_MASSES):
            fdm[f"inertia/pointmass-weight-lbs[{i}]"] = 0.0
        fdm["inertia/pointmass-location-X-inches[0]"] = 0.0

        # Geographic reference: fixed at equator/prime meridian.
        # Absolute position does not matter for the RTL task -- only relative
        # displacement matters, computed via a local tangent-plane projection
        # in position_ned (see below).
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

        # Capture the launch lat/lon as the reference point for position_ned's
        # tangent-plane projection (read back post-run_ic() rather than reusing
        # the ic/* values above, so this stays correct even if the IC solver
        # ever nudges them).
        self._lat0_rad = float(np.radians(fdm["position/lat-gc-deg"]))
        self._lon0_rad = float(np.radians(fdm["position/long-gc-deg"]))

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

        state = self.state
        if not np.all(np.isfinite(state)):
            # fdm.run() can return True even when the ground-contact spring-
            # damper model has numerically diverged (observed: a sudden multi-
            # metre altitude spike on touchdown followed by NaN the very next
            # step). Fold this into the same fault flag so the caller ends
            # the episode immediately instead of running up to MAX_STEPS on a
            # corrupted state.
            self._fault = True

        return state

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def apply_domain_rand(self, params: dict[str, float]) -> None:
        """Write one episode's sampled airframe to JSBSim.

        Call this after reset() and before the first step(). `params` uses the
        keys of env/curriculum.py's DR_RANGES / DR_ONESIDED; any key left out
        keeps the aircraft file's default. Unknown keys raise, so a renamed
        range cannot be silently dropped.

            DR_PROPERTY_MAP keys -> written to their property directly
            sw_mult              -> metrics/Sw-sqft = declared area * sw_mult
            mass_kg, cg_x_m, ixx, iyy (all four, or none)
                                 -> point masses solved by MassPropertySolver
                                    against JSBSim's own readback; cg_x_m is
                                    body frame, positive forward. Raises
                                    UnreachableTarget if a guard trips.

        The solver ends with run_ic(), which JSBSim needs to recompute mass
        properties. reset() already ran it once with an empty point-mass set;
        the second call leaves position, velocity and attitude unchanged since
        no ic/* property has changed.
        """
        apply_airframe(self._fdm, params, self.mass_solver, self._dr_defaults["metrics/Sw-sqft"])

    # ------------------------------------------------------------------
    # State read-outs  (SI, NED/FRD frames)
    # ------------------------------------------------------------------

    @property
    def position_ned(self) -> NDArray:
        """[N, E, D] in metres, relative to the episode home/origin.

        N/E are a local tangent-plane (equirectangular) projection of JSBSim's
        true geocentric lat/lon onto North/East, relative to the lat/lon
        captured at reset(). We add the stored launch NED offset so the
        result is relative to the home point (NED origin used by the reward
        fn), matching position_ned's contract everywhere else in the codebase.

        This does NOT use JSBSim's position/distance-from-start-lat-mt and
        -lon-mt properties, despite their name suggesting exactly this.
        Direct verification (see scratch diagnostics run against this
        aircraft model) showed those two properties report the UNSIGNED
        magnitude of net displacement along each axis, not signed
        displacement: e.g. flying due south for 25 m reports
        distance-from-start-lat-mt = +25, not -25; flying due west reports
        distance-from-start-lon-mt = +25, not -25. Using them directly as
        signed North/East meant position_ned silently reported the aircraft
        moving in the wrong direction whenever its net displacement from
        launch crossed from one side of the launch point to the other --
        reproduced with baseline episodes whose recorded ground track
        reversed direction mid-flight while attitude and body velocity
        stayed perfectly smooth (i.e. a position-tracking bug, not a real
        aerodynamic event). The tangent-plane projection used here was
        checked against independent dead-reckoning (integrating v_body
        through the attitude DCM) across straight and hard-turning flight,
        matching to within ~0.2 m over 5 s of sustained turning -- far
        tighter than the ~2.5 m GPS CEP already assumed for this hardware.
        Because a real GPS receiver also just reports raw lat/lon, this is
        the same conversion the real onboard flight computer will need to
        do, so it's the more sim-to-real-faithful approach as well, not
        merely a workaround.
        """
        fdm = self._fdm
        lat_rad = np.radians(fdm["position/lat-gc-deg"])
        lon_rad = np.radians(fdm["position/long-gc-deg"])
        p_n = self._p_ned_launch[0] + (lat_rad - self._lat0_rad) * R_EARTH_M
        p_e = (self._p_ned_launch[1]
               + (lon_rad - self._lon0_rad) * R_EARTH_M * np.cos(self._lat0_rad))
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
    def alpha(self) -> float:
        """True aerodynamic angle of attack (rad), wind-relative.

        NOT the same as arctan2(state[5], state[3]) -- that quantity is
        derived from INERTIAL body velocity and is only correct in still
        air; in wind it can be off by >15 deg (see scratch/audit1.py TEST 2).
        Use this for anything that needs the real aerodynamic condition
        (reward stall penalty, analysis). The safety shield deliberately
        does NOT use this -- see env/glider_env.py::_safety_shield, since
        the physical ESP32 has no AoA vane and must not depend on a
        quantity it cannot measure.
        """
        return float(self._fdm["aero/alpha-rad"])

    @property
    def beta(self) -> float:
        """True aerodynamic sideslip angle (rad), wind-relative."""
        return float(self._fdm["aero/beta-rad"])

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

    @property
    def control_surfaces(self) -> NDArray:
        """[aileron, elevator, rudder] ACTUAL surface deflection in radians
        (post actuator-lag/rate-limit, i.e. fcs/*-pos-rad) -- not the raw
        fcs/*-cmd-norm command. Used by the showcase dashboard's flight
        recorder to log true control-surface state; not read by training.
        """
        fdm = self._fdm
        return np.array([
            fdm["fcs/aileron-pos-rad"],
            fdm["fcs/elevator-pos-rad"],
            fdm["fcs/rudder-pos-rad"],
        ], dtype=np.float64)

    @property
    def touched_down(self) -> bool:
        """True if any belly-skid contact point has ground contact (WOW).

        The 4 <contact> points in rlglider.xml sit ~0.1 m below the CG (see
        their z-offset), so JSBSim's spring-damper ground-reaction model
        engages measurably BEFORE position_ned[2] (CG altitude) crosses zero.
        Use this -- not "p_d > 0" -- to detect ground contact: letting the
        sim keep running physics steps through the contact/bounce transient
        while only checking CG altitude gives the (fairly stiff) contact
        model room to do multiple compression/release cycles, which can
        numerically diverge (observed as a sudden multi-metre altitude spike
        immediately followed by a NaN state -- see the finite-state guard in
        step()).
        """
        fdm = self._fdm
        return any(
            bool(fdm[f"contact/unit[{i}]/WOW"]) for i in range(N_CONTACT_POINTS)
        )

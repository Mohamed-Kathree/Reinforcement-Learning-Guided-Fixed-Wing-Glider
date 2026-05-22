"""
actuator_models.py
==================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Servo actuator models for aileron, elevator, and rudder.  Each servo is a
first-order lag (time constant tau) followed by a rate limiter and a hard
deflection saturation — matching the physical digital servos on the glider.

The inner controller (AttitudeController in env/) commands deflections in
radians; the actuator suite converts those commands to the actual deflections
seen by the aerodynamics model, introducing realistic lag and rate limits.

Call sequence (inside env.step(), once per physics substep):
    actual = actuators.step(cmd_rad_dict, dt_phys)
    state  = dyn.step(actual, wind_ned, dt_phys)

Domain randomisation parameters set per episode via reset():
    tau  : servo lag time constant (s)       range [0.03, 0.12]
    rate : maximum slew rate (rad/s)         scale ×[0.7, 1.3]
    max  : deflection saturation (rad)       scale ×[0.8, 1.0]

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

import numpy as np


# ---------------------------------------------------------------------------
# Nominal servo parameters (physical defaults, pre-domain-randomisation)
# ---------------------------------------------------------------------------

@dataclass
class ServoParams:
    """Nominal physical parameters for one servo channel."""
    max_rad:    float   # deflection saturation (rad), symmetric ±
    rate_rad_s: float   # maximum slew rate (rad/s)
    tau_s:      float   # first-order lag time constant (s)


# Nominal defaults from CLAUDE.md Section 9
AILERON_PARAMS  = ServoParams(max_rad=np.radians(25.0), rate_rad_s=np.radians(200.0), tau_s=0.05)
ELEVATOR_PARAMS = ServoParams(max_rad=np.radians(20.0), rate_rad_s=np.radians(200.0), tau_s=0.05)
RUDDER_PARAMS   = ServoParams(max_rad=np.radians(25.0), rate_rad_s=np.radians(200.0), tau_s=0.06)


# ---------------------------------------------------------------------------
# Single-channel servo model
# ---------------------------------------------------------------------------

class ServoModel:
    """First-order lag + rate limiter + deflection saturation for one servo.

    Processing order per step() call:
      1. Clip command to ±max_rad   (hard saturation on the command)
      2. First-order lag toward clipped command
      3. Rate-limit the step increment to ±rate_rad_s * dt
      4. Saturate position to ±max_rad

    This ordering matches typical digital servo hardware: the servo tries to
    reach the commanded angle but is limited by both its slew rate and its
    mechanical travel.
    """

    def __init__(self, params: ServoParams) -> None:
        self._max  = params.max_rad
        self._rate = params.rate_rad_s
        self._tau  = params.tau_s
        self.pos: float = 0.0   # current deflection (rad)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self, pos: float = 0.0) -> None:
        """Zero (or set) the servo position at the start of an episode."""
        self.pos = float(pos)

    # ------------------------------------------------------------------
    # Per-physics-step advance
    # ------------------------------------------------------------------

    def step(self, cmd_rad: float, dt: float) -> float:
        """Advance servo by one physics step (dt seconds).

        Args:
            cmd_rad : commanded deflection (rad); will be saturated to ±max
            dt      : physics time step (s); should be 0.005 (200 Hz)

        Returns:
            Current servo position (rad) after lag and rate limiting.
        """
        cmd_clipped = float(np.clip(cmd_rad, -self._max, self._max))

        # First-order lag: exponential approach to command
        # pos_target = pos + (cmd - pos) * (1 - exp(-dt/tau))
        # Approximated as min(dt/tau, 1.0) to avoid exp overhead and
        # guarantee finite-time convergence when dt >> tau.
        alpha  = min(dt / self._tau, 1.0)
        target = self.pos + (cmd_clipped - self.pos) * alpha

        # Rate limit: servo cannot slew faster than rate_rad_s
        delta    = float(np.clip(target - self.pos, -self._rate * dt, self._rate * dt))
        self.pos = float(np.clip(self.pos + delta, -self._max, self._max))
        return self.pos

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def max_deflection(self) -> float:
        """Current deflection saturation limit (rad)."""
        return self._max

    @property
    def rate_limit(self) -> float:
        """Current slew rate limit (rad/s)."""
        return self._rate

    @property
    def time_constant(self) -> float:
        """Current lag time constant (s)."""
        return self._tau


# ---------------------------------------------------------------------------
# Three-channel actuator suite (aileron + elevator + rudder)
# ---------------------------------------------------------------------------

class ActuatorSuite:
    """Manages the three servo channels for one episode.

    Nominal parameters are stored; reset() applies domain randomisation
    by scaling them before handing them to fresh ServoModel instances.

    Typical usage (inside env.step(), one call per physics substep):

        actual_controls = self.actuators.step(cmd_dict, dt_phys)
        self.dyn.step(actual_controls, wind_ned, dt_phys)

    where cmd_dict and actual_controls both have keys:
        'aileron', 'elevator', 'rudder'  (values in radians)
    """

    def __init__(
        self,
        aileron_params:  ServoParams = AILERON_PARAMS,
        elevator_params: ServoParams = ELEVATOR_PARAMS,
        rudder_params:   ServoParams = RUDDER_PARAMS,
    ) -> None:
        self._nominal = {
            'aileron':  aileron_params,
            'elevator': elevator_params,
            'rudder':   rudder_params,
        }
        # Build with nominal params; reset() will randomise them per episode
        self.aileron  = ServoModel(aileron_params)
        self.elevator = ServoModel(elevator_params)
        self.rudder   = ServoModel(rudder_params)

    # ------------------------------------------------------------------
    # Episode reset (with optional domain randomisation)
    # ------------------------------------------------------------------

    def reset(
        self,
        rng:        np.random.Generator | None = None,
        tau_range:  tuple[float, float] = (0.03, 0.12),
        rate_scale: tuple[float, float] = (0.70, 1.30),
        max_scale:  tuple[float, float] = (0.80, 1.00),
    ) -> None:
        """Reinitialise servos, optionally with randomised parameters.

        When rng is None, nominal parameters are used (no randomisation).
        When rng is supplied, each servo gets independently sampled params
        within the given ranges — matching CLAUDE.md Section 13.

        Args:
            rng        : numpy Generator; pass None for deterministic nominal
            tau_range  : uniform range for lag time constant (s)
            rate_scale : multiplicative scale range applied to nominal rate
            max_scale  : multiplicative scale range applied to nominal max
        """
        for name, servo, attr in (
            ('aileron',  self.aileron,  'aileron'),
            ('elevator', self.elevator, 'elevator'),
            ('rudder',   self.rudder,   'rudder'),
        ):
            nom = self._nominal[name]
            if rng is not None:
                tau  = float(rng.uniform(*tau_range))
                rate = nom.rate_rad_s * float(rng.uniform(*rate_scale))
                max_ = nom.max_rad    * float(rng.uniform(*max_scale))
            else:
                tau  = nom.tau_s
                rate = nom.rate_rad_s
                max_ = nom.max_rad

            new_params = ServoParams(max_rad=max_, rate_rad_s=rate, tau_s=tau)
            new_servo  = ServoModel(new_params)
            new_servo.reset(pos=0.0)
            setattr(self, attr, new_servo)

    # ------------------------------------------------------------------
    # Per-physics-step advance
    # ------------------------------------------------------------------

    def step(self, controls: dict, dt: float) -> dict:
        """Advance all servos by one physics step.

        Args:
            controls : command dict with keys 'aileron', 'elevator', 'rudder'
                       (values in rad).  'rudder' is optional; defaults to 0.
            dt       : physics time step (s)

        Returns:
            actual : dict with same keys, values = actual servo positions (rad)
        """
        return {
            'aileron':  self.aileron.step( float(controls.get('aileron',  0.0)), dt),
            'elevator': self.elevator.step(float(controls.get('elevator', 0.0)), dt),
            'rudder':   self.rudder.step(  float(controls.get('rudder',   0.0)), dt),
        }

    # ------------------------------------------------------------------
    # State accessor (for obs building / logging)
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict:
        """Current deflections {'aileron', 'elevator', 'rudder'} in radians."""
        return {
            'aileron':  self.aileron.pos,
            'elevator': self.elevator.pos,
            'rudder':   self.rudder.pos,
        }

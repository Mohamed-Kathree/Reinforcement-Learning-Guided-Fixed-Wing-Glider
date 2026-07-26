"""
wind.py
=======
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Atmospheric wind model: constant mean wind (set per episode) plus
Ornstein-Uhlenbeck (first-order low-pass filtered) turbulence gusts.

The NED wind vector returned by step() is consumed by GliderDynamics as:
    v_air_body = v_body - R @ wind_ned

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class WindModel:
    """Constant mean wind + Ornstein-Uhlenbeck gust model.

    Mean wind is a fixed NED vector set at episode reset; it represents the
    prevailing wind direction and speed for the episode.

    Gusts are modelled as an Ornstein-Uhlenbeck process — a first-order
    low-pass filter driven by white noise — giving physically realistic
    correlated turbulence.  The gust state is updated every physics step
    (dt = 0.005 s, 200 Hz) and added to the mean to give the total wind.

    Curriculum knobs (set via reset()):
        mean_speed    : 0 -> 9 m/s across curriculum stages
        gust_intensity: 0 -> 2 m/s sigma of the driving white noise

    Altitude scaling: the HORIZONTAL mean wind in mean_ned is defined at
    z_ref AGL (2 m -- roughly the height wind is quoted at for small-UAV
    operations) and log-profile-scaled by step()'s altitude_m argument.
    A 20-40% gradient between 20 m and 2 m AGL is normal in the atmospheric
    surface layer this glider spends its whole flight inside; a flat mean
    wind that doesn't vary with altitude was previously a meaningful gap
    for a task whose entire episode happens between 0-25 m AGL. The gust
    and any vertical mean component (mean_ned[2], typically small thermal/
    mechanical lift or sink sampled per episode) are NOT altitude-scaled.

    Usage (inside env.step() physics loop):
        wind_ned = wind_model.step(altitude_m)
        state    = dyn.step(controls, wind_ned, dt_phys)
    """

    def __init__(
        self,
        tau:   float = 2.0,
        dt:    float = 0.005,
        z_ref: float = 2.0,
        z0:    float = 0.03,
    ) -> None:
        """
        Args:
            tau   : gust correlation time (s).  Larger tau -> slower, smoother gusts.
            dt    : physics integration step (s); must match GliderDynamics.
            z_ref : reference height (m AGL) at which mean_ned's horizontal
                    component is defined (see altitude scaling above).
            z0    : surface roughness length (m) for the log wind profile;
                    0.03 m is a typical "open terrain, few obstacles" value.
        """
        self.tau   = tau
        self.dt    = dt
        self.z_ref = z_ref
        self.z0    = z0

        self._mean_ned:  NDArray = np.zeros(3, dtype=np.float64)
        self._gust:      NDArray = np.zeros(3, dtype=np.float64)
        self._intensity: float   = 0.0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(
        self,
        mean_ned:        NDArray,
        gust_intensity:  float,
        rng:             np.random.Generator,
    ) -> None:
        """Initialise wind for a new episode.

        Args:
            mean_ned       : constant NED wind vector [wn, we, wd] (m/s).
                             Typically set by curriculum/domain randomisation:
                               direction = rng.uniform(0, 2*pi)
                               speed     = rng.uniform(0, cfg_mean_speed)
                               mean_ned  = speed * [cos(dir), sin(dir), 0]
            gust_intensity : std-dev (m/s) of the OU driving noise per axis.
                             0 => laminar (no gusts).
            rng            : numpy Generator for reproducible stochastic steps.
        """
        self._mean_ned  = np.asarray(mean_ned, dtype=np.float64).copy()
        self._intensity = float(gust_intensity)
        self._gust      = np.zeros(3, dtype=np.float64)
        self._rng       = rng

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self, altitude_m: float | None = None) -> NDArray:
        """Advance the gust state by one physics step and return total wind.

        Ornstein-Uhlenbeck update (Euler-Maruyama):
            gust += -gust * (dt/tau) + sigma_d * N(0, 1)
            sigma_d = intensity * sqrt(2*dt/tau)

        This scales the stochastic driving term by sqrt(dt) (not dt), which
        is what makes the process's stationary standard deviation equal to
        `intensity` exactly (an OU process integrated with a driving term
        scaled by dt instead of sqrt(dt) underestimates its stationary
        variance by a factor of ~dt/tau -- at dt=0.005s, tau=2s that's a
        ~14x-too-weak gust for the same configured intensity).

        Args:
            altitude_m : current AGL altitude (m), used to log-profile-scale
                         the horizontal mean wind component (see class
                         docstring). None (default) skips scaling entirely
                         (ratio = 1.0), for callers that don't track altitude.

        Returns:
            wind_ned : NED wind vector [wn, we, wd] (m/s), shape (3,).
        """
        if self._intensity > 0.0:
            sigma_d      = self._intensity * np.sqrt(2.0 * self.dt / self.tau)
            self._gust  += -self._gust * (self.dt / self.tau) + sigma_d * self._rng.normal(0.0, 1.0, 3)

        if altitude_m is None:
            profile_ratio = 1.0
        else:
            # Clipped to [0.5, 1.5]x: keeps the profile from blowing up near
            # the ground (log singularity at z0) or over-amplifying well
            # above the surface layer this glider actually operates in.
            z = max(float(altitude_m), 0.5)
            profile_ratio = float(np.clip(
                np.log(z / self.z0) / np.log(self.z_ref / self.z0), 0.5, 1.5
            ))

        wind_ned = self._mean_ned.copy()
        wind_ned[0:2] *= profile_ratio
        return wind_ned + self._gust

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def make_from_speed_direction(
        cls,
        speed:     float,
        direction: float,
        gust_intensity: float = 0.0,
        rng:       np.random.Generator | None = None,
        tau:       float = 2.0,
        dt:        float = 0.005,
    ) -> "WindModel":
        """Build and reset a WindModel from a horizontal speed + direction.

        Args:
            speed     : horizontal wind speed (m/s)
            direction : wind-from direction in NED azimuth (rad), 0 = from North
            gust_intensity: OU noise std-dev (m/s), 0 for laminar
            rng       : numpy Generator; creates a fresh one if None
            tau, dt   : forwarded to __init__

        Returns:
            Initialised WindModel ready to call step().
        """
        if rng is None:
            rng = np.random.default_rng()
        mean_ned = np.array([
            speed * np.cos(direction),
            speed * np.sin(direction),
            0.0,
        ], dtype=np.float64)
        model = cls(tau=tau, dt=dt)
        model.reset(mean_ned, gust_intensity, rng)
        return model

    # ------------------------------------------------------------------
    # State accessors (useful for logging / obs building)
    # ------------------------------------------------------------------

    @property
    def mean_ned(self) -> NDArray:
        """Current episode mean NED wind vector (m/s)."""
        return self._mean_ned.copy()

    @property
    def gust_ned(self) -> NDArray:
        """Current gust perturbation NED vector (m/s)."""
        return self._gust.copy()

    @property
    def total_ned(self) -> NDArray:
        """Mean + gust NED wind vector (same as last step() return value)."""
        return self._mean_ned + self._gust

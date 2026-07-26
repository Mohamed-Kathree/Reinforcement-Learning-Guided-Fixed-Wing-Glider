"""
sensor_models.py
================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Sensor suite for the 1.1 kg fixed-wing glider simulation.  Converts ground-truth
physics state into corrupted, rate-limited sensor readings that mimic what the
on-board Raspberry Pi 5 actually receives at runtime.

Key principle: the RL policy NEVER sees ground truth — only the outputs of these
models.  All noise parameters are domain-randomisation targets; set via reset().

Sensors modelled:
    GPS        :  5 Hz, 1.5 m / 0.3 m/s position & velocity white noise, plus
                  a per-episode correlated position bias (multipath/
                  ionospheric-like; see GPS_POS_BIAS_STD_M)
    IMU        : 200 Hz, 1 deg attitude / 0.5 deg/s rate white noise, plus a
                  per-episode correlated attitude bias -- larger on yaw
                  (magnetometer-like; see IMU_ATT_BIAS_STD_RAD / IMU_YAW_BIAS_STD_RAD)
    Baro       :  25 Hz, 1.0 m altitude noise (BMP280)
    Ultrasonic :  20 Hz, ~0.08 m range noise, dropout + range gating [0.2, 4.5] m
                  (RCWL-1655; landing-flare-only ground-proximity sensor, no LiDAR
                  is fitted on the hardware -- see as-built electronics notes.
                  Python identifiers below are still named lidar_*/LIDAR_* for
                  historical reasons; they carry the ultrasonic reading.)

    White noise is the easy case -- a policy learns to average it out in a
    couple of steps. The per-episode bias is what actually stresses sim-to-
    real transfer: it cannot be averaged away, only made robust to.

Update-rate model (zero-order hold):
    SensorSuite.step() is called at the 200 Hz physics rate.  Each sensor
    refreshes its internal buffer at its own Hz; between refreshes the last
    valid reading is held unchanged.  This means a 20 Hz policy step sees:
        GPS        : same sample for 4 consecutive policy steps (refreshes every 40 physics steps)
        IMU        : fresh sample every step (200 Hz == physics rate)
        Baro       : same sample for ~1-2 policy steps (refreshes every 8 physics steps)
        Ultrasonic : fresh sample every policy step (refreshes every 10 physics steps)

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

from sim.math_utils import euler_from_quat, quat_to_rotmat


# ---------------------------------------------------------------------------
# Update-rate constants (Hz)
# ---------------------------------------------------------------------------

GPS_HZ   = 5
IMU_HZ   = 200
BARO_HZ  = 25
LIDAR_HZ = 20    # ultrasonic (RCWL-1655), ~50 ms cycle -- see module docstring
PHYS_HZ  = 200   # physics integration rate; must match JSBSimFDM's dt_phys

# ---------------------------------------------------------------------------
# Per-episode correlated bias std-devs (Hz-independent constants)
# ---------------------------------------------------------------------------
# Real GPS multipath/ionospheric error and IMU (accelerometer/gyro/mag) bias
# are correlated over tens of seconds, not white -- a policy can learn to
# average white noise out in a couple of steps, but it cannot average away
# a persistent per-episode offset; it must be robust to it instead. Sampled
# once at reset(), held constant for the whole episode (see SensorSuite.reset()).
GPS_POS_BIAS_STD_M    = 1.5              # multipath/ionospheric-like position offset
IMU_ATT_BIAS_STD_RAD  = np.radians(1.0)  # roll/pitch: accelerometer-tilt-like bias
IMU_YAW_BIAS_STD_RAD  = np.radians(7.0)  # yaw: magnetometer bias near carbon fibre/servo current is much larger (real hardware: ~5-15 deg)


# ---------------------------------------------------------------------------
# Pure noise functions (stateless; used by SensorSuite internally)
# ---------------------------------------------------------------------------

def _gps_noise(
    p_ned:       NDArray,
    v_ned:       NDArray,
    rng:         np.random.Generator,
    noise_scale: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Add GPS position and velocity noise.

    Args:
        p_ned       : true NED position [pn, pe, pd] (m)
        v_ned       : true NED velocity [vn, ve, vd] (m/s)
        rng         : numpy Generator
        noise_scale : multiplier on standard deviations (domain randomisation)

    Returns:
        (p_meas, v_meas): noisy NED position and velocity
    """
    p_meas = p_ned + rng.normal(0.0, 1.5 * noise_scale, 3)
    v_meas = v_ned + rng.normal(0.0, 0.3 * noise_scale, 3)
    return p_meas, v_meas


def _imu_noise(
    euler:       NDArray,
    omega:       NDArray,
    rng:         np.random.Generator,
    noise_scale: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Add IMU attitude and angular-rate noise.

    Args:
        euler       : true [roll, pitch, yaw] (rad)
        omega       : true body angular rates [p, q, r] (rad/s)
        rng         : numpy Generator
        noise_scale : multiplier on standard deviations

    Returns:
        (euler_meas, omega_meas): noisy Euler angles and rates
    """
    att_noise  = rng.normal(0.0, np.radians(1.0) * noise_scale, 3)
    rate_noise = rng.normal(0.0, np.radians(0.5) * noise_scale, 3)
    return euler + att_noise, omega + rate_noise


def _baro_noise(
    altitude_m:  float,
    rng:         np.random.Generator,
    noise_scale: float = 1.0,
) -> float:
    """Add barometric altimeter noise.

    Args:
        altitude_m  : true altitude (m), positive up = -p_d
        rng         : numpy Generator
        noise_scale : multiplier on standard deviation

    Returns:
        Noisy altitude measurement (m)
    """
    return float(altitude_m + rng.normal(0.0, 1.0 * noise_scale))


def _lidar_noise(
    true_agl:     float,
    roll:         float,
    pitch:        float,
    rng:          np.random.Generator,
    dropout_prob: float = 0.10,
    noise_scale:  float = 1.0,
) -> tuple[float, bool]:
    """Compute slant-range-corrected ultrasonic AGL measurement with dropout.

    The ultrasonic (RCWL-1655) measures slant range to the ground; this is
    corrected to vertical AGL using:
        h_agl ≈ r_slant · cos(roll) · cos(pitch)

    The measurement is invalid when:
      - corrected AGL is outside [0.2, 4.5] m  (sensor range limits -- 5 m max
        range, unusable this close to the noise floor near the top), OR
      - a random dropout event occurs (bank angle / wind sensitive in reality).

    Only ever valid on final approach / flare, not as a continuous in-flight
    AGL floor -- see as-built electronics notes.

    Args:
        true_agl     : true vertical AGL (m) = -state[2] when on flat terrain
        roll         : current roll angle (rad)
        pitch        : current pitch angle (rad)
        rng          : numpy Generator
        dropout_prob : probability of a dropout on each call (domain randomisation)
        noise_scale  : multiplier on range noise std-dev

    Returns:
        (h_meas, valid):
            h_meas : corrected AGL measurement (m); 0.0 when invalid
            valid  : bool; True when the reading can be trusted
    """
    h_corrected = true_agl * np.cos(roll) * np.cos(pitch)
    in_range    = 0.2 < h_corrected < 4.5
    not_dropout = rng.random() > dropout_prob
    valid       = bool(in_range and not_dropout)
    if not valid:
        return 0.0, False
    return float(h_corrected + rng.normal(0.0, 0.08 * noise_scale)), True


# ---------------------------------------------------------------------------
# Stateful sensor suite
# ---------------------------------------------------------------------------

class SensorSuite:
    """Rate-limited, noisy sensor suite for the glider simulation.

    Maintains internal buffers for each sensor; each buffer refreshes at the
    sensor's native Hz and holds its last value between refreshes.

    Typical usage inside env.step():

        # One call per physics substep:
        self.sensors.step(self.dyn.state, self.rng)

        # After all substeps, build observation from last readings:
        obs = self.sensors.gps_pos    # (3,) NED position
        obs = self.sensors.imu_euler  # (3,) roll/pitch/yaw
        ...

    Domain-randomisation parameters are set once per episode via reset().
    """

    # Steps between sensor refreshes at PHYS_HZ = 200 Hz
    _GPS_PERIOD   = PHYS_HZ // GPS_HZ    # 40
    _IMU_PERIOD   = PHYS_HZ // IMU_HZ    # 1
    _BARO_PERIOD  = PHYS_HZ // BARO_HZ   # 8
    _LIDAR_PERIOD = PHYS_HZ // LIDAR_HZ  # 10

    def __init__(self) -> None:
        # Initialise with safe zero defaults; always call reset() before use.
        self._noise_scale:  float = 1.0
        self._dropout_prob: float = 0.10

        # Step counters — count DOWN; sensor refreshes when counter reaches 0
        self._gps_ctr:   int = 0
        self._imu_ctr:   int = 0
        self._baro_ctr:  int = 0
        self._lidar_ctr: int = 0

        # Last-valid measurement buffers
        self._gps_pos:    NDArray = np.zeros(3)
        self._gps_vel:    NDArray = np.zeros(3)
        self._imu_euler:  NDArray = np.zeros(3)
        self._imu_omega:  NDArray = np.zeros(3)
        self._baro_alt:   float   = 0.0
        self._lidar_agl:  float   = 0.0
        self._lidar_valid: bool   = False

        # Per-episode correlated bias (sampled once in reset(), held constant
        # for the episode; see GPS_POS_BIAS_STD_M etc. above).
        self._gps_pos_bias: NDArray = np.zeros(3)
        self._imu_att_bias: NDArray = np.zeros(3)   # [roll, pitch, yaw] bias (rad)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(
        self,
        state:        NDArray,
        rng:          np.random.Generator,
        noise_scale:  float = 1.0,
        dropout_prob: float = 0.10,
    ) -> None:
        """Initialise sensor state for a new episode.

        Seeds all buffers with a single noisy measurement drawn from the
        initial physics state so that the first observation is valid.

        Args:
            state        : 13-element glider state at episode start
            rng          : numpy Generator (shared with env)
            noise_scale  : noise std multiplier (0 = perfect, 1 = nominal)
            dropout_prob : LiDAR dropout probability per step (0–1)
        """
        self._noise_scale  = float(noise_scale)
        self._dropout_prob = float(dropout_prob)

        # Sample this episode's correlated bias (held constant until the next
        # reset()) -- BEFORE _refresh_all() so the first observation already
        # reflects it, same as the real receiver/IMU would from power-on.
        self._gps_pos_bias = rng.normal(0.0, GPS_POS_BIAS_STD_M * self._noise_scale, 3)
        self._imu_att_bias = np.array([
            rng.normal(0.0, IMU_ATT_BIAS_STD_RAD * self._noise_scale),
            rng.normal(0.0, IMU_ATT_BIAS_STD_RAD * self._noise_scale),
            rng.normal(0.0, IMU_YAW_BIAS_STD_RAD * self._noise_scale),
        ])

        # Stagger initial counters so all sensors don't update simultaneously
        self._gps_ctr   = 0
        self._imu_ctr   = 0
        self._baro_ctr  = 0
        self._lidar_ctr = 0

        # Seed buffers from initial state
        self._refresh_all(state, rng)

    # ------------------------------------------------------------------
    # Per-physics-step advance
    # ------------------------------------------------------------------

    def step(
        self,
        state: NDArray,
        rng:   np.random.Generator,
    ) -> None:
        """Advance sensor suite by one physics step (5 ms / 200 Hz).

        Each sensor's counter counts down; the buffer is refreshed when
        the counter reaches zero.  Call this once per RK4 substep inside
        the physics loop — *before* building the RL observation.

        Args:
            state : current 13-element glider state
            rng   : numpy Generator shared with the environment
        """
        # GPS (5 Hz: refresh every 40 physics steps)
        if self._gps_ctr <= 0:
            self._refresh_gps(state, rng)
            self._gps_ctr = self._GPS_PERIOD
        self._gps_ctr -= 1

        # IMU (200 Hz: refresh every physics step)
        if self._imu_ctr <= 0:
            self._refresh_imu(state, rng)
            self._imu_ctr = self._IMU_PERIOD
        self._imu_ctr -= 1

        # Baro (25 Hz: refresh every 8 physics steps)
        if self._baro_ctr <= 0:
            self._refresh_baro(state, rng)
            self._baro_ctr = self._BARO_PERIOD
        self._baro_ctr -= 1

        # Ultrasonic (20 Hz: refresh every 10 physics steps)
        if self._lidar_ctr <= 0:
            self._refresh_lidar(state, rng)
            self._lidar_ctr = self._LIDAR_PERIOD
        self._lidar_ctr -= 1

    # ------------------------------------------------------------------
    # Measurement accessors (read-only; always hold the last valid sample)
    # ------------------------------------------------------------------

    @property
    def gps_pos(self) -> NDArray:
        """Noisy NED position [pn, pe, pd] (m), shape (3,)."""
        return self._gps_pos.copy()

    @property
    def gps_vel(self) -> NDArray:
        """Noisy NED velocity [vn, ve, vd] (m/s), shape (3,)."""
        return self._gps_vel.copy()

    @property
    def imu_euler(self) -> NDArray:
        """Noisy [roll, pitch, yaw] (rad), shape (3,)."""
        return self._imu_euler.copy()

    @property
    def imu_omega(self) -> NDArray:
        """Noisy body angular rates [p, q, r] (rad/s), shape (3,)."""
        return self._imu_omega.copy()

    @property
    def baro_alt(self) -> float:
        """Noisy barometric altitude (m), positive up."""
        return self._baro_alt

    @property
    def lidar_agl(self) -> float:
        """Slant-corrected LiDAR AGL (m); 0.0 when invalid."""
        return self._lidar_agl

    @property
    def lidar_valid(self) -> bool:
        """True when the last LiDAR measurement passed range/dropout checks."""
        return self._lidar_valid

    # ------------------------------------------------------------------
    # Private refresh helpers
    # ------------------------------------------------------------------

    def _refresh_all(self, state: NDArray, rng: np.random.Generator) -> None:
        """Refresh every sensor buffer immediately (used in reset())."""
        self._refresh_gps(state, rng)
        self._refresh_imu(state, rng)
        self._refresh_baro(state, rng)
        self._refresh_lidar(state, rng)

    def _refresh_gps(self, state: NDArray, rng: np.random.Generator) -> None:
        R     = quat_to_rotmat(state[6:10])   # body -> NED DCM
        v_ned = R @ state[3:6]                # body -> NED velocity (no transpose)
        gps_pos, self._gps_vel = _gps_noise(
            state[0:3], v_ned, rng, self._noise_scale
        )
        # Per-episode correlated position bias (multipath/ionospheric-like),
        # on top of the white noise _gps_noise already added. Velocity is
        # left unbiased -- Doppler-derived GPS velocity is not subject to the
        # same slow-drifting position bias.
        self._gps_pos = gps_pos + self._gps_pos_bias

    def _refresh_imu(self, state: NDArray, rng: np.random.Generator) -> None:
        euler = np.array(euler_from_quat(state[6:10]))
        imu_euler, self._imu_omega = _imu_noise(
            euler, state[10:13], rng, self._noise_scale
        )
        # Per-episode correlated attitude bias (accelerometer-tilt/magnetometer
        # -like), on top of the white noise _imu_noise already added.
        self._imu_euler = imu_euler + self._imu_att_bias

    def _refresh_baro(self, state: NDArray, rng: np.random.Generator) -> None:
        altitude = float(-state[2])       # altitude = -p_d
        self._baro_alt = _baro_noise(altitude, rng, self._noise_scale)

    def _refresh_lidar(self, state: NDArray, rng: np.random.Generator) -> None:
        true_agl    = float(-state[2])
        roll, pitch = float(self._imu_euler[0]), float(self._imu_euler[1])
        self._lidar_agl, self._lidar_valid = _lidar_noise(
            true_agl, roll, pitch, rng, self._dropout_prob, self._noise_scale
        )

    # ------------------------------------------------------------------
    # Convenience: pack all readings into a flat dict for obs building
    # ------------------------------------------------------------------

    def get_readings(self) -> dict:
        """Return all current sensor readings as a plain dict.

        Keys match the observation vector description in CLAUDE.md Section 10.2:
            gps_pos     : ndarray(3)  NED position (m)
            gps_vel     : ndarray(3)  NED velocity (m/s)
            imu_euler   : ndarray(3)  roll/pitch/yaw (rad)
            imu_omega   : ndarray(3)  angular rates (rad/s)
            baro_alt    : float       altitude (m)
            lidar_agl   : float       AGL (m), 0 when invalid
            lidar_valid : bool        LiDAR validity flag
        """
        return {
            'gps_pos':     self._gps_pos.copy(),
            'gps_vel':     self._gps_vel.copy(),
            'imu_euler':   self._imu_euler.copy(),
            'imu_omega':   self._imu_omega.copy(),
            'baro_alt':    self._baro_alt,
            'lidar_agl':   self._lidar_agl,
            'lidar_valid': self._lidar_valid,
        }

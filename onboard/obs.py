"""
obs.py -- PiTelemetry -> observation v1, the 12 values the policy was trained on.

Mirrors env/glider_env.py GliderEnv._build_obs() channel for channel:

  [0] dx_home  m    GPS north of home    (lat/lon -> local tangent plane at home)
  [1] dy_home  m    GPS east of home
  [2] ground_speed  m/s  hypot(vn, ve)
  [3] course_angle  rad  atan2(ve, vn)
  [4] roll  [5] pitch  [6] yaw   rad (IMU; yaw aligned to GPS course after the throw)
  [7] baro_alt      m    above the FLIGHT-arm reference (home ground level)
  [8] vertical_speed m/s  -vd (GPS, positive = climbing)
  [9] lidar_agl     m    ultrasonic, 0 when invalid (the simulator's convention)
  [10] lidar_valid  0/1
  [11] req_glide_ratio   dist_home / max(baro_alt, 1) / 15

Known differences from the simulator, all small and logged:
  - vertical speed: NaN in the NMEA GPS fallback (no UBX); then a low-passed
    barometric rate is used instead (flagged in the record).
  - lat/lon -> metres uses the equirectangular projection at home (error
    < 1 cm over the few hundred metres flown).
"""

from __future__ import annotations

import math

import numpy as np

EARTH_R = 6_371_000.0


def latlon_e7_to_ne(lat_e7: int, lon_e7: int, home_lat_e7: int, home_lon_e7: int) -> tuple[float, float]:
    lat0 = math.radians(home_lat_e7 * 1e-7)
    n = math.radians((lat_e7 - home_lat_e7) * 1e-7) * EARTH_R
    e = math.radians((lon_e7 - home_lon_e7) * 1e-7) * EARTH_R * math.cos(lat0)
    return n, e


class ObsBuilder:
    def __init__(self, baro_rate_tau_s: float = 0.5) -> None:
        self.tau = baro_rate_tau_s
        self.reset()

    def reset(self) -> None:
        self._prev_alt = None
        self._prev_t = None
        self._vz_baro = 0.0
        self.used_baro_vz = False

    def build(self, t) -> np.ndarray:
        """t: a PiTelemetry (decoded). Returns the float32 observation."""
        dn, de = latlon_e7_to_ne(t.lat, t.lon, t.home_lat, t.home_lon)
        vn = 0.0 if math.isnan(t.vn) else t.vn
        ve = 0.0 if math.isnan(t.ve) else t.ve
        gs = math.hypot(vn, ve)
        course = math.atan2(ve, vn)
        alt = 0.0 if math.isnan(t.baro_alt) else t.baro_alt

        # barometric climb rate, used only when GPS vertical velocity is missing
        ts = t.t_ms * 1e-3
        if self._prev_t is not None and ts > self._prev_t:
            dt = ts - self._prev_t
            raw = (alt - self._prev_alt) / dt
            self._vz_baro += (dt / (self.tau + dt)) * (raw - self._vz_baro)
        self._prev_t, self._prev_alt = ts, alt
        if math.isnan(t.vd):
            vz, self.used_baro_vz = self._vz_baro, True
        else:
            vz, self.used_baro_vz = -t.vd, False

        sonar_ok = t.sonar >= 0.0
        dist = math.hypot(dn, de)
        return np.array([dn, de, gs, course, t.roll, t.pitch, t.yaw, alt, vz,
                         t.sonar if sonar_ok else 0.0, 1.0 if sonar_ok else 0.0,
                         dist / max(alt, 1.0) / 15.0], dtype=np.float32)

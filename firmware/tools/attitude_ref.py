"""
attitude_ref.py
===============
Python reference for the flight firmware's attitude estimator
(firmware/flight/src/core/attitude.c). No dependencies beyond the standard
library, so tests and tools can import it without vpython or pyserial.

`Kalman` is glider_viz.py's 2-state filter, unchanged (Hardware/Code/glider_viz.py).

`Attitude` is the glider_viz.py update step with the IMU frame CORRECTED
(Nov 2026 frame decision, firmware/flight/src/core/imu_frame.h): the board's
raw axes are already body axes (x forward, y right, z down; all signs +1), and
the accelerometer reads specific force, so level reads az = -1 g. Hence

    roll_a  = atan2(-ay, -az)                 (glider_viz.py had atan2(-ay, az))
    pitch_a = atan2( ax, sqrt(ay^2 + az^2))   (glider_viz.py had atan2(-ax, ...))
    rates   = gx, gy, gz as read              (glider_viz.py negated gx and gz)

The old negations compensated for the bench sketch's former sign defines
(AX -1, AY -1, AZ +1), which were a 180-degree yaw rotation of the board
rather than the board-to-body transform.

euler=False integrates body rates directly, exactly as glider_viz.py does;
euler=True mirrors the firmware's flight default (Euler-angle rates).
"""

from __future__ import annotations

import math

Q_ANGLE = 0.001
Q_BIAS = 0.003
R_MEASURE = 0.03
GATE_LO_G = 0.85
GATE_HI_G = 1.15
DT_MAX_S = 0.2
ACC_LSB = 16384.0      # +/- 2 g
GYRO_LSB = 131.0       # +/- 250 dps


class Kalman:
    """2-state Kalman filter (angle, gyro bias), verbatim from glider_viz.py."""

    def __init__(self):
        self.angle = 0.0
        self.bias = 0.0
        self.P = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, new_angle, new_rate, dt):
        rate_unbiased = new_rate - self.bias
        self.angle += dt * rate_unbiased

        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + Q_ANGLE)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += Q_BIAS * dt

        S = self.P[0][0] + R_MEASURE
        K = [self.P[0][0] / S, self.P[1][0] / S]

        y = new_angle - self.angle
        while y > 180.0:
            y -= 360.0
        while y < -180.0:
            y += 360.0

        self.angle += K[0] * y
        self.bias += K[1] * y

        P00, P01 = self.P[0][0], self.P[0][1]
        self.P[0][0] -= K[0] * P00
        self.P[0][1] -= K[0] * P01
        self.P[1][0] -= K[1] * P00
        self.P[1][1] -= K[1] * P01
        return self.angle


def accel_roll_deg(ay: float, az: float) -> float:
    return math.degrees(math.atan2(-ay, -az))


def accel_pitch_deg(ax: float, ay: float, az: float) -> float:
    return math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az)))


def _wrap180(d: float) -> float:
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


class Attitude:
    """One filter per axis; same sequencing as attitude.c rlg_attitude_update()."""

    def __init__(self, euler: bool = False):
        self.roll = Kalman()
        self.pitch = Kalman()
        self.yaw = 0.0
        self.seeded = False
        self.euler = euler
        self.trusted = False
        self.g_mag = 0.0

    def update(self, ax, ay, az, gx, gy, gz, dt) -> bool:
        """Body-frame sample: accel in g, gyro in deg/s, dt in s. False if unused."""
        roll_a = accel_roll_deg(ay, az)
        pitch_a = accel_pitch_deg(ax, ay, az)
        self.g_mag = math.sqrt(ax * ax + ay * ay + az * az)
        if not self.seeded:
            self.roll.angle, self.pitch.angle = roll_a, pitch_a
            self.seeded = True
            return False
        if not (dt > 0.0) or dt > DT_MAX_S:
            return False
        rr, pr, yr = gx, gy, gz
        if self.euler:
            phi, th = math.radians(self.roll.angle), math.radians(self.pitch.angle)
            ct = math.cos(th)
            if abs(ct) < 0.05:
                ct = -0.05 if ct < 0 else 0.05
            tt = math.sin(th) / ct
            rr = gx + (gy * math.sin(phi) + gz * math.cos(phi)) * tt
            pr = gy * math.cos(phi) - gz * math.sin(phi)
            yr = (gy * math.sin(phi) + gz * math.cos(phi)) / ct
        self.trusted = GATE_LO_G < self.g_mag < GATE_HI_G
        if self.trusted:
            self.roll.update(roll_a, rr, dt)
            self.pitch.update(pitch_a, pr, dt)
        else:
            self.roll.angle += dt * (rr - self.roll.bias)
            self.pitch.angle += dt * (pr - self.pitch.bias)
        self.roll.angle = _wrap180(self.roll.angle)
        self.yaw = _wrap180(self.yaw + dt * yr)
        return True

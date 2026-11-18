"""
safety.py
=========
Python reference for the firmware's safety layer (firmware/protocol/rlg_safety.c).
Rules, the two separate link timeouts, and the hardware seams are documented in
firmware/protocol/rlg_safety.h. tests/test_flighttest_safety.py runs the same
scenarios against this model and the compiled C; any disagreement fails.

Used by the stub glider, so the ground app behaves as it will against the
firmware before hardware is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from flighttest.protocol import packets as P

ACCEPTED, ARMING, ARMED, DISARMED_RESULT, REJECTED_FRAME, REJECTED_STALE, REJECTED_AIRBORNE = range(7)
RESULT_NAMES = ["ACCEPTED", "ARMING", "ARMED", "DISARMED", "REJECTED_FRAME", "REJECTED_STALE", "REJECTED_AIRBORNE"]

_U32 = 0xFFFFFFFF
N = len(P.SERVOS)


@dataclass
class Sensors:
    baro_alt_m: float = math.nan     # above the ground reference; NaN = no reference (does not vote)
    gps_speed_ms: float = 0.0
    gps_fix: int = 0                 # 0 none, 1 2D, 2 3D; speed votes only with a fix


GROUND = Sensors()


def _ground(mode: int) -> bool:
    return mode in (P.BENCH, P.REPLAY)


def _abs(v: int) -> int:
    return max(P.ABS_MIN_US, min(P.ABS_MAX_US, int(v)))


@dataclass
class SafetyState:
    neutral_us: list = field(default_factory=lambda: [1500] * N)
    min_us: list = field(default_factory=lambda: [P.ABS_MIN_US] * N)
    max_us: list = field(default_factory=lambda: [P.ABS_MAX_US] * N)
    mode: int = P.DISARMED
    armed: bool = False
    pending_mode: int = 0
    consecutive: int = 0
    have_seq: bool = False
    last_seq: int = 0
    last_cmd_ms: int = 0
    last_activity_ms: int = 0
    last_cmd_us: list = field(default_factory=lambda: [1500] * N)
    servo_us: list = field(default_factory=lambda: [1500] * N)
    timeout_disarmed: bool = False
    rejected_airborne: bool = False
    airborne: bool = False
    have_flight: bool = False
    last_flight_ms: int = 0
    pi_failsafe: bool = False
    last_pi_ms: int = 0
    pi_consecutive: int = 0


class Safety:
    def __init__(self, neutral_us: list | None = None, now_ms: int = 0) -> None:
        s = SafetyState()
        s.neutral_us = list(neutral_us) if neutral_us else [1500] * N
        s.last_cmd_us = list(s.neutral_us)
        now_ms &= _U32
        s.last_cmd_ms = s.last_activity_ms = s.last_pi_ms = now_ms
        self.s = s
        self._disarm(now_ms)

    # ---- internals mirror rlg_safety.c ----
    def _clamp(self, i: int, v: int) -> int:
        v = max(self.s.min_us[i], min(self.s.max_us[i], int(v)))
        return _abs(v)

    def _centre(self) -> None:
        self.s.servo_us = [self._clamp(i, self.s.neutral_us[i]) for i in range(N)]

    def _disarm(self, now_ms: int) -> None:
        s = self.s
        if s.armed and s.mode == P.FLIGHT:
            s.have_flight = True
            s.last_flight_ms = now_ms
        s.mode = P.DISARMED
        s.armed = False
        s.consecutive = 0
        s.pi_failsafe = False
        self._centre()

    def _apply(self, servo_us: list, now_ms: int) -> None:
        if list(servo_us) != self.s.last_cmd_us:
            self.s.last_activity_ms = now_ms
        self.s.last_cmd_us = list(servo_us)
        self.s.servo_us = [self._clamp(i, servo_us[i]) for i in range(N)]

    # ---- API ----
    def set_limits(self, min_us: list, max_us: list) -> bool:
        if any(lo > hi for lo, hi in zip(min_us, max_us)):
            return False
        self.s.min_us = [_abs(v) for v in min_us]
        self.s.max_us = [_abs(v) for v in max_us]
        if not self.s.armed:
            self._centre()
        else:
            self.s.servo_us = [self._clamp(i, v) for i, v in enumerate(self.s.servo_us)]
        return True

    def is_airborne(self, now_ms: int, sens: Sensors) -> bool:
        s = self.s
        now_ms &= _U32
        baro = not math.isnan(sens.baro_alt_m) and sens.baro_alt_m > P.AIRBORNE_ALT_M
        gps = sens.gps_fix > 0 and sens.gps_speed_ms > P.AIRBORNE_GROUND_SPEED_MS
        latch = (s.armed and s.mode == P.FLIGHT) or \
            (s.have_flight and ((now_ms - s.last_flight_ms) & _U32) <= P.FLIGHT_LATCH_MS)
        s.airborne = bool(baro or gps or latch)
        return s.airborne

    def on_command(self, now_ms: int, sens: Sensors, frame: bytes) -> int:
        s = self.s
        now_ms &= _U32
        try:
            c = P.Command.decode(bytes(frame))
        except ValueError:
            return REJECTED_FRAME
        if c.mode > P.REPLAY:
            return REJECTED_FRAME

        gap = (now_ms - s.last_cmd_ms) & _U32
        diff = (c.seq - s.last_seq) & 0xFFFF
        diff = diff - 0x10000 if diff >= 0x8000 else diff
        if s.have_seq and gap <= P.COMMAND_TIMEOUT_MS and diff <= 0:
            return REJECTED_STALE
        s.have_seq = True
        s.last_seq = c.seq

        if c.mode == P.DISARMED:
            self._disarm(now_ms)
            s.last_cmd_ms = now_ms
            return DISARMED_RESULT

        airborne = self.is_airborne(now_ms, sens)
        if _ground(c.mode) and airborne:
            s.rejected_airborne = True
            if _ground(s.mode):
                self._disarm(now_ms)
            s.consecutive = 0
            return REJECTED_AIRBORNE

        if s.armed and c.mode == s.mode:
            s.last_cmd_ms = now_ms
            if _ground(c.mode):
                self._apply(c.servo_us, now_ms)
            return ACCEPTED

        if s.armed:
            self._disarm(now_ms)
        if c.mode == s.pending_mode and s.consecutive > 0 and gap <= P.COMMAND_TIMEOUT_MS:
            s.consecutive += 1
        else:
            s.pending_mode = c.mode
            s.consecutive = 1
        s.last_cmd_ms = now_ms
        if s.consecutive < P.REARM_CONSECUTIVE_FRAMES:
            return ARMING

        s.mode = c.mode
        s.armed = True
        s.consecutive = 0
        s.timeout_disarmed = False
        s.rejected_airborne = False
        s.last_activity_ms = now_ms
        if c.mode == P.FLIGHT:
            s.have_flight = True
            s.last_flight_ms = now_ms
            s.pi_failsafe = True
            s.pi_consecutive = 0
        if _ground(c.mode):
            s.last_cmd_us = list(c.servo_us)
            s.servo_us = [self._clamp(i, c.servo_us[i]) for i in range(N)]
        return ARMED

    def on_pi_setpoint(self, now_ms: int) -> None:
        s = self.s
        now_ms &= _U32
        gap = (now_ms - s.last_pi_ms) & _U32
        s.last_pi_ms = now_ms
        if not s.pi_failsafe:
            return
        s.pi_consecutive = s.pi_consecutive + 1 if (s.pi_consecutive > 0 and gap <= P.PI_TIMEOUT_MS) else 1
        if s.pi_consecutive >= P.PI_RESUME_FRAMES:
            s.pi_failsafe = False
            s.pi_consecutive = 0

    def tick(self, now_ms: int, sens: Sensors | None = None) -> None:
        s = self.s
        now_ms &= _U32
        if s.armed and s.mode == P.FLIGHT:
            s.last_flight_ms = now_ms
            if not s.pi_failsafe and ((now_ms - s.last_pi_ms) & _U32) > P.PI_TIMEOUT_MS:
                s.pi_failsafe = True
                s.pi_consecutive = 0
        if sens is not None:
            self.is_airborne(now_ms, sens)
        gap = (now_ms - s.last_cmd_ms) & _U32
        if not s.armed and s.consecutive > 0 and gap > P.COMMAND_TIMEOUT_MS:
            s.consecutive = 0
        if not s.armed or not _ground(s.mode):
            return
        if gap > P.COMMAND_TIMEOUT_MS:
            self._disarm(now_ms)
            s.timeout_disarmed = True
            return
        if ((now_ms - s.last_activity_ms) & _U32) > P.ARM_IDLE_EXPIRY_MS:
            self._disarm(now_ms)

    def flight_output(self, us: list) -> bool:
        if not self.s.armed or self.s.mode != P.FLIGHT:
            return False
        self.s.servo_us = [self._clamp(i, us[i]) for i in range(N)]
        return True

    def status_bits(self) -> int:
        s = self.s
        return P.pack_status(mode=s.mode, armed=int(s.armed), airborne=int(s.airborne),
                             timeout_disarmed=int(s.timeout_disarmed),
                             cmd_rejected_airborne=int(s.rejected_airborne), pi_failsafe=int(s.pi_failsafe))

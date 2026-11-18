"""
safety.py
=========
Python reference for the firmware's command safety (firmware/protocol/rlg_safety.c).
Rules, design decisions and hardware seams are documented in
firmware/protocol/rlg_safety.h. tests/test_flighttest_safety.py runs the same
scenarios against this model and the compiled C; any disagreement fails.

Used by the ground app's stub glider, so the UI behaves like the firmware
before hardware is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flighttest.protocol import packets as P

ACCEPTED, ARMING, ARMED, DISARMED_RESULT, REJECTED_FRAME, REJECTED_STALE, REJECTED_AIRBORNE = range(7)
RESULT_NAMES = ["ACCEPTED", "ARMING", "ARMED", "DISARMED", "REJECTED_FRAME", "REJECTED_STALE", "REJECTED_AIRBORNE"]

_U32 = 0xFFFFFFFF
MIN_US = [s["min_us"] for s in P.SERVOS]
MAX_US = [s["max_us"] for s in P.SERVOS]
N = len(P.SERVOS)


def _ground(mode: int) -> bool:
    return mode in (P.BENCH, P.REPLAY)


def _clamp(i: int, v: int) -> int:
    return max(MIN_US[i], min(MAX_US[i], int(v)))


@dataclass
class SafetyState:
    neutral_us: list = field(default_factory=lambda: [1500] * N)
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


class Safety:
    def __init__(self, neutral_us: list | None = None, now_ms: int = 0) -> None:
        s = SafetyState()
        s.neutral_us = list(neutral_us) if neutral_us else [1500] * N
        s.last_cmd_us = list(s.neutral_us)
        s.last_cmd_ms = now_ms & _U32
        s.last_activity_ms = now_ms & _U32
        self.s = s
        self._disarm()

    # ---- internals mirror rlg_safety.c ----
    def _centre(self) -> None:
        self.s.servo_us = [_clamp(i, self.s.neutral_us[i]) for i in range(N)]

    def _disarm(self) -> None:
        self.s.mode = P.DISARMED
        self.s.armed = False
        self.s.consecutive = 0
        self._centre()

    def _apply(self, servo_us: list, now_ms: int) -> None:
        if list(servo_us) != self.s.last_cmd_us:
            self.s.last_activity_ms = now_ms
        self.s.last_cmd_us = list(servo_us)
        self.s.servo_us = [_clamp(i, servo_us[i]) for i in range(N)]

    # ---- API ----
    def on_command(self, now_ms: int, airborne: bool, frame: bytes) -> int:
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
            self._disarm()
            s.last_cmd_ms = now_ms
            return DISARMED_RESULT

        if _ground(c.mode) and airborne:
            s.rejected_airborne = True
            if _ground(s.mode):
                self._disarm()
            s.consecutive = 0
            return REJECTED_AIRBORNE

        if s.armed and c.mode == s.mode:
            s.last_cmd_ms = now_ms
            if _ground(c.mode):
                self._apply(c.servo_us, now_ms)
            return ACCEPTED

        if s.armed:
            self._disarm()
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
        if _ground(c.mode):
            s.last_cmd_us = list(c.servo_us)
            s.servo_us = [_clamp(i, c.servo_us[i]) for i in range(N)]
        return ARMED

    def tick(self, now_ms: int) -> None:
        s = self.s
        now_ms &= _U32
        gap = (now_ms - s.last_cmd_ms) & _U32
        if not s.armed and s.consecutive > 0 and gap > P.COMMAND_TIMEOUT_MS:
            s.consecutive = 0
        if not s.armed or not _ground(s.mode):
            return
        if gap > P.COMMAND_TIMEOUT_MS:
            self._disarm()
            s.timeout_disarmed = True
            return
        if ((now_ms - s.last_activity_ms) & _U32) > P.ARM_IDLE_EXPIRY_MS:
            self._disarm()

    def flight_output(self, us: list) -> bool:
        if not self.s.armed or self.s.mode != P.FLIGHT:
            return False
        self.s.servo_us = [_clamp(i, us[i]) for i in range(N)]
        return True

    def status_bits(self) -> int:
        return P.pack_status(mode=self.s.mode, armed=int(self.s.armed),
                             timeout_disarmed=int(self.s.timeout_disarmed),
                             cmd_rejected_airborne=int(self.s.rejected_airborne))

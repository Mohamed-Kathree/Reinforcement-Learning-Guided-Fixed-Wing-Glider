"""
stub_glider.py
==============
A simulated glider plus ground ESP32, speaking the real byte protocol, so the
ground app runs end to end without hardware. STUB: nothing here is a
measurement of the real aircraft.

What it models (from the flight-test brief's measured characteristics):
  - the firmware's command safety, via flighttest.safety (the Python reference
    of firmware/protocol/rlg_safety.c), so arming, the 150 ms timeout and the
    airborne refusal behave as the firmware will;
  - 5 Hz telemetry over a lossy ESP-NOW link (configurable loss, both ways);
  - barometer: +/-0.05 m noise, ~0.3 m slow drift, ~2 m warm-up transient
    decaying over the first two minutes; ground reference captured on arm
    (FLIGHT) after warm-up, never at boot;
  - sonar valid below 4.5 m, -1 above;
  - GPS: no fix for the first 20 s, then 2D, then 3D with a rising satellite count;
  - rail voltage: NOT measured by default, because the current wiring has no
    divider to an ADC pin (rail_v = -1, rail_v_valid = 0). `rail_sensor=True`
    simulates a fitted divider with sag under servo motion, for UI work only.
"""

from __future__ import annotations

import math
import random

from flighttest.protocol import packets as P
from flighttest.safety import Safety

TELEMETRY_PERIOD_MS = 200          # 5 Hz
BARO_WARMUP_MS = 120_000
HOME = (-33.9300, 18.6300)          # placeholder field position for the stub only


class StubGlider:
    def __init__(self, neutral_us: list[int] | None = None, now_ms: int = 0, seed: int = 0,
                 loss: float = 0.03, rail_sensor: bool = False) -> None:
        self.rng = random.Random(seed)
        self.safety = Safety(neutral_us, now_ms)
        self.boot_ms = now_ms
        self.next_tel_ms = now_ms
        self.seq = 0
        self.loss = loss
        self.rail_sensor = rail_sensor
        self.airborne = False
        self.alt_true = 0.0
        self.baro_ref_offset: float | None = None
        self.parser = P.StreamParser()
        self.outbox = bytearray()
        self.commands_received = 0
        self.commands_lost = 0
        self.last_result: int | None = None
        self._prev_servo = list(self.safety.s.servo_us)

    # ------------------------------------------------------------------ link side
    def receive(self, data: bytes, now_ms: int) -> None:
        """Bytes written by the ground station (ground ESP32 -> ESP-NOW -> glider)."""
        for pkt in self.parser.feed(data):
            if not isinstance(pkt, P.Command):
                continue
            if self.rng.random() < self.loss:
                self.commands_lost += 1
                continue
            self.commands_received += 1
            self.last_result = self.safety.on_command(now_ms, self.airborne, pkt.encode())

    def read(self) -> bytes:
        out = bytes(self.outbox)
        self.outbox.clear()
        return out

    # ------------------------------------------------------------------ physics-ish
    def _baro_raw(self, now_ms: int) -> float:
        t = (now_ms - self.boot_ms) / 1000.0
        warmup = 2.0 * math.exp(-t / 40.0)           # ~2 m transient, mostly gone by 2 min
        drift = 0.3 * math.sin(t / 300.0)            # ~0.3 m slow drift
        return self.alt_true + warmup + drift + self.rng.gauss(0.0, 0.05)

    def _gps(self, now_ms: int) -> tuple[int, int]:
        t = (now_ms - self.boot_ms) / 1000.0
        if t < 20:
            return 0, 0
        if t < 35:
            return 1, 4
        return 2, min(12, 6 + int((t - 35) / 15))

    def step(self, now_ms: int) -> None:
        self.safety.tick(now_ms)
        s = self.safety.s
        # Arming FLIGHT on the ground after warm-up captures the baro reference.
        warm = now_ms - self.boot_ms >= BARO_WARMUP_MS
        if s.armed and s.mode == P.FLIGHT and self.baro_ref_offset is None and warm and not self.airborne:
            self.baro_ref_offset = self._baro_raw(now_ms) - self.alt_true
        if not s.armed:
            self.baro_ref_offset = None if s.mode == P.DISARMED else self.baro_ref_offset

        while now_ms >= self.next_tel_ms:
            self.next_tel_ms += TELEMETRY_PERIOD_MS
            self.seq = (self.seq + 1) & 0xFFFF
            if self.rng.random() < self.loss:
                continue                                       # lost over ESP-NOW
            self.outbox += self._telemetry(now_ms, warm).encode()

    def _telemetry(self, now_ms: int, warm: bool) -> P.Telemetry:
        t = (now_ms - self.boot_ms) / 1000.0
        s = self.safety.s
        fix, sats = self._gps(now_ms)
        ref = self.baro_ref_offset if self.baro_ref_offset is not None else 0.0
        alt = self._baro_raw(now_ms) - ref
        sonar = self.alt_true + 0.12 + self.rng.gauss(0, 0.01) if self.alt_true < 4.5 else -1.0
        moved = sum(abs(a - b) for a, b in zip(s.servo_us, self._prev_servo))
        self._prev_servo = list(s.servo_us)
        if self.rail_sensor:
            rail = 5.08 - min(0.45, moved / 2000.0) + self.rng.gauss(0, 0.01)
        else:
            rail = -1.0
        status = P.pack_status(
            sonar_valid=int(sonar >= 0), gps_fix=fix, gps_sats=sats, airborne=int(self.airborne),
            baro_ref_captured=int(self.baro_ref_offset is not None), baro_warm=int(warm),
            rail_v_valid=int(self.rail_sensor)) | self.safety.status_bits()
        wob = 0.02 * math.sin(t * 1.3)
        return P.Telemetry(
            seq=self.seq, roll=wob, pitch=0.05 + 0.5 * wob, yaw=1.2, p=0.02 * math.cos(t * 1.3), q=0.0, r=0.0,
            alt_baro=alt, sonar=sonar,
            lat=HOME[0] if fix else 0.0, lon=HOME[1] if fix else 0.0, vn=0.0, ve=0.0,
            rail_v=rail, status=status)

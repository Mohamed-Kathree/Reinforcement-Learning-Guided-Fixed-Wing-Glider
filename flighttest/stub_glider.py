"""
stub_glider.py
==============
A simulated glider plus ground ESP32, speaking the real byte protocol, so the
ground app runs end to end without hardware. STUB: nothing here is a
measurement of the real aircraft.

What it models (flight-test brief and the 18 Nov hardware answers):
  - the firmware safety layer via flighttest.safety (the Python reference of
    firmware/protocol/rlg_safety.c): arming, the ESP-NOW 150 ms timeout for
    BENCH/REPLAY, the separate Pi-setpoint 150 ms timeout in FLIGHT, and the
    airborne test (baro > 5 m above the ground reference, GPS speed > 2 m/s with
    a fix, or armed in FLIGHT within 60 s);
  - 5 Hz telemetry over a lossy ESP-NOW link (configurable loss, both ways);
  - barometer: +/-0.05 m noise, ~0.3 m slow drift, ~2 m warm-up transient over
    the first two minutes; the ground reference is captured when FLIGHT is
    armed after warm-up, never at boot, so a cold barometer cannot vote airborne;
  - sonar valid below 4.5 m;
  - GPS: no fix for 20 s, then 2D, then 3D with a rising satellite count;
  - rail voltage on GPIO34 through a ~2:1 divider: the ESP32 sends the ADC pin
    voltage. The stub's true divider is 1.98:1, so an uncalibrated scale of 2.0
    reads ~1% high, as an uncalibrated board would. rail_sensor=False models
    the divider not fitted.
"""

from __future__ import annotations

import math
import random

from flighttest.protocol import packets as P
from flighttest.safety import Safety, Sensors

TELEMETRY_PERIOD_MS = 200          # 5 Hz
PI_PERIOD_MS = 50                  # Pi setpoints at 20 Hz
BARO_WARMUP_MS = 120_000
TRUE_DIVIDER = 1.98
HOME = (-33.9300, 18.6300)          # placeholder field position for the stub only


class StubGlider:
    def __init__(self, neutral_us: list[int] | None = None, now_ms: int = 0, seed: int = 0,
                 loss: float = 0.03, rail_sensor: bool = True,
                 limits: tuple[list[int], list[int]] | None = None) -> None:
        self.rng = random.Random(seed)
        self.safety = Safety(neutral_us, now_ms)
        if limits is not None:
            self.safety.set_limits(*limits)
        self.boot_ms = now_ms
        self.next_tel_ms = now_ms
        self.next_pi_ms = now_ms
        self.seq = 0
        self.loss = loss
        self.rail_sensor = rail_sensor
        self.alt_true = 0.0
        self.ground_speed = 0.0
        self.pi_alive = True
        self.baro_ref: float | None = None
        self.parser = P.StreamParser()
        self.outbox = bytearray()
        self.commands_received = 0
        self.commands_lost = 0
        self.last_result: int | None = None
        self._prev_servo = list(self.safety.s.servo_us)
        self._baro_cache = (None, 0.0)

    # ------------------------------------------------------------------ controls (stub only)
    def set_flying(self, flying: bool) -> None:
        """Carry the stub into the air (10 m up, 9 m/s) or back to the bench."""
        self.alt_true = 10.0 if flying else 0.0
        self.ground_speed = 9.0 if flying else 0.0

    @property
    def airborne(self) -> bool:
        return self.safety.s.airborne

    def arm_flight_locally(self, now_ms: int) -> None:
        """Simulate the flight-arming path (not from the ground app) for testing the latch."""
        base = self.safety.s.last_seq
        for k in range(P.REARM_CONSECUTIVE_FRAMES):
            f = P.Command(seq=(base + 1 + k) & 0xFFFF, mode=P.FLIGHT, servo_us=[1500] * 4).encode()
            self.safety.on_command(now_ms, self.sensors(now_ms), f)
        self._capture_ref(now_ms)

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
            self.last_result = self.safety.on_command(now_ms, self.sensors(now_ms), pkt.encode())
            if self.safety.s.armed and self.safety.s.mode == P.FLIGHT:
                self._capture_ref(now_ms)

    def read(self) -> bytes:
        out = bytes(self.outbox)
        self.outbox.clear()
        return out

    # ------------------------------------------------------------------ sensors
    def _baro_raw(self, now_ms: int) -> float:
        if self._baro_cache[0] == now_ms:
            return self._baro_cache[1]
        t = (now_ms - self.boot_ms) / 1000.0
        warmup = 2.0 * math.exp(-t / 40.0)
        drift = 0.3 * math.sin(t / 300.0)
        v = self.alt_true + warmup + drift + self.rng.gauss(0.0, 0.05)
        self._baro_cache = (now_ms, v)
        return v

    def _capture_ref(self, now_ms: int) -> None:
        if self.baro_ref is None and now_ms - self.boot_ms >= BARO_WARMUP_MS:
            self.baro_ref = self._baro_raw(now_ms) - self.alt_true

    def _gps(self, now_ms: int) -> tuple[int, int]:
        t = (now_ms - self.boot_ms) / 1000.0
        if t < 20:
            return 0, 0
        if t < 35:
            return 1, 4
        return 2, min(12, 6 + int((t - 35) / 15))

    def sensors(self, now_ms: int) -> Sensors:
        fix, _ = self._gps(now_ms)
        alt = math.nan if self.baro_ref is None else self._baro_raw(now_ms) - self.baro_ref
        return Sensors(baro_alt_m=alt, gps_speed_ms=self.ground_speed + abs(self.rng.gauss(0, 0.3)) if fix else 0.0,
                       gps_fix=fix)

    # ------------------------------------------------------------------ main step
    def step(self, now_ms: int) -> None:
        s = self.safety.s
        if s.armed and s.mode == P.FLIGHT and self.pi_alive:
            while now_ms >= self.next_pi_ms:
                self.next_pi_ms += PI_PERIOD_MS
                self.safety.on_pi_setpoint(now_ms)
        else:
            self.next_pi_ms = now_ms
        self.safety.tick(now_ms, self.sensors(now_ms))
        while now_ms >= self.next_tel_ms:
            self.next_tel_ms += TELEMETRY_PERIOD_MS
            self.seq = (self.seq + 1) & 0xFFFF
            if self.rng.random() < self.loss:
                continue                                       # lost over ESP-NOW
            self.outbox += self._telemetry(now_ms).encode()

    def _telemetry(self, now_ms: int) -> P.Telemetry:
        t = (now_ms - self.boot_ms) / 1000.0
        s = self.safety.s
        fix, sats = self._gps(now_ms)
        warm = now_ms - self.boot_ms >= BARO_WARMUP_MS
        alt = self._baro_raw(now_ms) - (self.baro_ref or 0.0)
        sonar = self.alt_true + 0.12 + self.rng.gauss(0, 0.01) if self.alt_true < 4.5 else -1.0
        moved = sum(abs(a - b) for a, b in zip(s.servo_us, self._prev_servo))
        self._prev_servo = list(s.servo_us)
        rail = 5.08 - min(0.45, moved / 2000.0) + self.rng.gauss(0, 0.01)
        adc = rail / TRUE_DIVIDER if self.rail_sensor else -1.0
        status = P.pack_status(
            sonar_valid=int(sonar >= 0), gps_fix=fix, gps_sats=sats,
            baro_ref_captured=int(self.baro_ref is not None), baro_warm=int(warm),
            rail_v_valid=int(self.rail_sensor)) | self.safety.status_bits()
        wob = 0.02 * math.sin(t * 1.3)
        return P.Telemetry(
            seq=self.seq, roll=wob, pitch=0.05 + 0.5 * wob, yaw=1.2, p=0.02 * math.cos(t * 1.3), q=0.0, r=0.0,
            alt_baro=alt, sonar=sonar,
            lat=HOME[0] if fix else 0.0, lon=HOME[1] if fix else 0.0,
            vn=self.ground_speed if fix else 0.0, ve=0.0,
            rail_adc_v=adc, status=status)

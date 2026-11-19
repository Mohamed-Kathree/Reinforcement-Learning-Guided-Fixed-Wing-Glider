"""
test_flight_core_v3.py
======================
Host tests for the flight firmware's Phase 2 C core (firmware/flight/src/core):
  - pilink.c   Pi UART receive parser (PiSetpoint) against the Python encoder:
               resync after noise and corruption, sequence gaps, duplicates;
               PiTelemetry built by the firmware decodes in Python.
  - ubx.c      u-blox NAV-POSLLH / NAV-VELNED / NAV-SOL decoding, checksum
               rejection, frame builder against a known reference frame.
  - flight_phase.c  launch and apex detection.
  - inner_loop.c    rlg_pitch_target() equals the simulator's trim schedule
               (and the full elevator command equals the simulator's whenever
               true airspeed equals the command, the case the KV term vanishes).
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import glider_env as G  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402

CORE = os.path.join(ROOT, "firmware", "flight", "src", "core")
_LIB = None
u8p = ctypes.POINTER(ctypes.c_uint8)


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no host C compiler: the C core is NOT tested")
    out = os.path.join(tempfile.mkdtemp(prefix="rlg_core3_"), "core3" + (".dll" if sys.platform == "win32" else ".so"))
    src = [os.path.join(CORE, f) for f in ("pilink.c", "ubx.c", "flight_phase.c", "inner_loop.c")]
    src += [os.path.join(ROOT, "firmware", "flight", "src", "protocol", "rlg_crc8.c"),
            os.path.join(ROOT, "tests", "c", "flight_core_helpers.c")]
    cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-shared", *src, "-o", out, "-lm"]
    if sys.platform != "win32":
        cmd.insert(5, "-fPIC")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    L = ctypes.CDLL(out)
    for name in ("t_pilink_size", "t_pitelemetry_size", "t_pisetpoint_size", "t_ubx_size", "t_phase_size", "t_ubx_build"):
        getattr(L, name).restype = ctypes.c_size_t
    L.t_pilink_feed.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint16),
                                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), u8p]
    L.t_ubx_feed.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
    L.t_ubx_build.argtypes = [ctypes.c_uint8, ctypes.c_uint8, u8p, ctypes.c_uint16, u8p]
    L.t_phase_update.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    L.rlg_pitch_target.argtypes = [ctypes.c_float, ctypes.c_float]
    L.rlg_pitch_target.restype = ctypes.c_float
    _LIB = L
    return L


def buf(n):
    return ctypes.create_string_buffer(n)


# ------------------------------------------------------------------ Pi link

class PiRx:
    def __init__(self):
        self.L = lib()
        self.b = buf(self.L.t_pilink_size())
        self.L.t_pilink_init(self.b)

    def feed(self, data: bytes) -> list:
        out = []
        seq, roll, spd, ctl = ctypes.c_uint16(), ctypes.c_float(), ctypes.c_float(), ctypes.c_uint8()
        for x in data:
            if self.L.t_pilink_feed(self.b, x, ctypes.byref(seq), ctypes.byref(roll), ctypes.byref(spd), ctypes.byref(ctl)):
                out.append((seq.value, round(roll.value, 5), round(spd.value, 5), ctl.value))
        return out

    def counts(self):
        a = (ctypes.c_uint32 * 4)()
        self.L.t_pilink_counts(self.b, a)
        return dict(zip(("ok", "bad", "gaps", "dup"), list(a)))


def sp(seq, roll=0.1, speed=11.0, ctl=1):
    return P.PiSetpoint(seq=seq, controller=ctl, roll_sp=roll, speed_cmd=speed, tlm_gaps=0).encode()


def test_sizes_match_python():
    L = lib()
    assert L.t_pisetpoint_size() == P.PiSetpoint.SIZE == 16
    assert L.t_pitelemetry_size() == P.PiTelemetry.SIZE == 83


def test_pilink_clean_stream():
    rx = PiRx()
    got = rx.feed(b"".join(sp(i, roll=0.01 * i) for i in range(1, 21)))
    assert [g[0] for g in got] == list(range(1, 21))
    assert got[4][1] == pytest.approx(0.05, abs=1e-6)
    assert rx.counts() == {"ok": 20, "bad": 0, "gaps": 0, "dup": 0}


def test_pilink_noise_corruption_gaps_and_partial():
    rx = PiRx()
    good = sp(1)
    bad = bytearray(sp(2)); bad[6] ^= 0xFF                 # payload corrupted -> CRC fails
    stream = b"\x00\x13\xa5\x99" + good + bytes(bad) + sp(3) + sp(7) + sp(7)
    got = rx.feed(stream[:-5])                              # last frame partial
    assert [g[0] for g in got] == [1, 3, 7]
    assert rx.feed(stream[-5:]) == [(7, 0.1, 11.0, 1)]      # completes; a duplicate seq
    c = rx.counts()
    assert c["ok"] == 4 and c["gaps"] == 1 + 3 and c["dup"] == 1 and c["bad"] >= 1


def test_pilink_random_fuzz_never_accepts_garbage():
    rx = PiRx()
    rng = random.Random(5)
    junk = bytes(rng.randrange(256) for _ in range(20000))
    got = rx.feed(junk)
    # random bytes can only produce a valid frame if sync, type and CRC all match by chance
    assert len(got) <= 2
    got = rx.feed(sp(100) + sp(101))
    assert [g[0] for g in got][-2:] == [100, 101]


def test_pitelemetry_built_by_firmware_decodes_in_python():
    L = lib()
    raw = (ctypes.c_uint8 * P.PiTelemetry.SIZE)()
    L.t_pitelemetry_example(raw)
    t = P.PiTelemetry.decode(bytes(raw))
    assert (t.seq, t.t_ms, t.lat, t.home_lon, t.gps_fix, t.gps_sats, t.gps_count) == (513, 123456, -339339590, 184546000, 2, 7, 42)
    assert t.roll == pytest.approx(0.25) and t.vd == pytest.approx(0.75) and t.sonar == -1.0
    assert (t.status, t.pi_rx_gaps, t.pi_rx_bad) == (0x00A40019, 3, 1)


# ------------------------------------------------------------------ UBX

def ubx_frame(cls, mid, payload: bytes) -> bytes:
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for x in body:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def posllh(itow, lat_e7, lon_e7, hmsl_mm=12000):
    return ubx_frame(1, 2, struct.pack("<IiiiiII", itow, lon_e7, lat_e7, hmsl_mm + 30000, hmsl_mm, 2500, 4000))


def velned(itow, vn, ve, vd):
    g = math.hypot(vn, ve)
    hdg = (math.degrees(math.atan2(ve, vn)) % 360) * 1e5
    return ubx_frame(1, 0x12, struct.pack("<IiiiIIiII", itow, round(vn * 100), round(ve * 100), round(vd * 100),
                                          round(math.sqrt(g * g + vd * vd) * 100), round(g * 100), round(hdg), 50, 100000))


def navsol(gps_fix, ok, num_sv):
    p = bytearray(52)
    p[10] = gps_fix; p[11] = 0x01 if ok else 0; p[47] = num_sv
    return ubx_frame(1, 6, bytes(p))


class Ubx:
    def __init__(self):
        self.L = lib()
        self.b = buf(self.L.t_ubx_size())
        self.L.t_ubx_init(self.b)

    def feed(self, data: bytes) -> int:
        return sum(self.L.t_ubx_feed(self.b, x) for x in data)

    def get(self):
        i = (ctypes.c_int32 * 3)(); f = (ctypes.c_float * 5)(); c = (ctypes.c_uint32 * 4)()
        self.L.t_ubx_get(self.b, i, f, c)
        return list(i), list(f), list(c)


def test_ubx_builder_matches_known_cfg_rate_frame():
    """The widely published 5 Hz CFG-RATE frame: B5 62 06 08 06 00 C8 00 01 00 01 00 DE 6A."""
    L = lib()
    pl = (ctypes.c_uint8 * 6)(*bytes.fromhex("C80001000100"))
    out = (ctypes.c_uint8 * 16)()
    n = L.t_ubx_build(6, 8, pl, 6, out)
    assert bytes(out[:n]).hex(" ").upper() == "B5 62 06 08 06 00 C8 00 01 00 01 00 DE 6A"
    assert bytes(out[:n]) == ubx_frame(6, 8, bytes.fromhex("C80001000100"))


def test_ubx_decodes_a_solution_pair():
    u = Ubx()
    stream = navsol(3, True, 8) + b"$GPGGA,junk*00\r\n" + posllh(1000, -339339590, 184546280) + velned(1000, 3.21, -4.5, 0.62)
    assert u.feed(stream) == 1
    ints, f, c = u.get()
    assert ints[:2] == [-339339590, 184546280] and ints[2] == 12000
    assert f[0] == pytest.approx(3.21) and f[1] == pytest.approx(-4.5) and f[2] == pytest.approx(0.62)
    assert f[3] == pytest.approx(math.hypot(3.21, 4.5), abs=0.01)
    assert c[:3] == [2, 8, 1]


def test_ubx_needs_matching_itow_and_valid_checksum():
    u = Ubx()
    assert u.feed(posllh(1000, 1, 2) + velned(1200, 1, 0, 0)) == 0      # different epochs
    bad = bytearray(posllh(1200, 1, 2)); bad[-1] ^= 1
    assert u.feed(bytes(bad)) == 0
    assert u.get()[2][3] == 1                                           # one bad checksum
    assert u.feed(posllh(1200, 5, 6)) == 1                              # pairs with the VELNED at 1200


@pytest.mark.parametrize("gps_fix,ok,expect", [(0, True, 0), (2, True, 1), (3, True, 2), (4, True, 2), (3, False, 0), (5, True, 0)])
def test_ubx_fix_mapping(gps_fix, ok, expect):
    u = Ubx()
    u.feed(navsol(gps_fix, ok, 6))
    assert u.get()[2][0] == expect


# ------------------------------------------------------------------ flight phase

class Phase:
    def __init__(self, apex=True):
        self.L = lib()
        self.b = buf(self.L.t_phase_size())
        self.L.t_phase_init(self.b, 1 if apex else 0)

    def step(self, t, armed=True, g=1.0, baro=0.0):
        return self.L.t_phase_update(self.b, t, 1 if armed else 0, g, baro)

    def causes(self):
        a = (ctypes.c_uint32 * 4)()
        self.L.t_phase_causes(self.b, a)
        return list(a)


DIS, ARMED, LAUNCH, AUTO = P.PHASE_DISARMED, P.PHASE_ARMED, P.PHASE_LAUNCH, P.PHASE_AUTONOMOUS


def run(ph, t0, t1, **kw):
    out = None
    for t in range(t0, t1, 4):
        out = ph.step(t, **kw)
    return out


def test_phase_arming_and_no_false_launch():
    ph = Phase()
    assert ph.step(0, armed=False) == DIS
    assert ph.step(4) == ARMED
    assert run(ph, 8, 2000, g=2.0) == ARMED          # carried around, 2 g: no launch
    assert run(ph, 2000, 2012, g=4.0) == ARMED       # 12 ms spike (a knock): no launch
    assert run(ph, 2012, 2100, g=1.0) == ARMED


def test_phase_throw_climb_apex_by_baro_descent():
    ph = Phase()
    ph.step(0)
    assert run(ph, 4, 40, g=4.0) == LAUNCH           # >= 20 ms above 2.5 g
    t = 40
    alt = 0.0
    while t < 2000:                                   # zoom climb 1.5 s up to ~20 m, then descent
        alt = 20.0 * math.sin(min(t - 40, 1500) / 1500 * math.pi / 2) - max(0, t - 1540) * 0.004
        p = ph.step(t, g=1.0, baro=alt)
        if p == AUTO:
            break
        t += 4
    assert p == AUTO
    launch_cause, apex_cause, t_launch, t_apex = ph.causes()
    assert (launch_cause, apex_cause) == (1, 1)
    assert 1540 < t_apex < 1900, t_apex               # shortly after the real peak at ~1540 ms
    assert run(ph, t, t + 1000, baro=alt) == AUTO     # stays autonomous
    assert ph.step(t + 1000, armed=False) == DIS      # disarm resets


def test_phase_apex_timeout_without_baro():
    ph = Phase()
    ph.step(0)
    run(ph, 4, 40, g=4.0)
    assert run(ph, 40, 3000, baro=float("nan")) == LAUNCH
    assert run(ph, 3000, 3100, baro=float("nan")) == AUTO
    assert ph.causes()[1] == 2


def test_phase_noisy_climb_does_not_trigger_early_apex():
    ph = Phase()
    ph.step(0)
    run(ph, 4, 40, g=4.0)
    rng = random.Random(3)
    t, apex_t = 40, None
    while t < 4000:
        alt = min(t - 40, 1500) * 0.012 + rng.gauss(0, 0.15)    # climbing to 18 m, then level
        if ph.step(t, baro=alt) == AUTO:
            apex_t = t
            break
        t += 4
    assert apex_t is None or apex_t > 1500, apex_t


def test_phase_handoff_at_launch_when_configured():
    ph = Phase(apex=False)
    ph.step(0)
    assert run(ph, 4, 40, g=4.0) == AUTO
    assert ph.causes()[:2] == [1, 3]


# ------------------------------------------------------------------ pitch target vs simulator

def test_pitch_target_constants_match_simulator():
    src = open(os.path.join(CORE, "inner_loop.h")).read()
    import re
    a = float(re.search(r"#define IL_ALPHA_TRIM\s+([0-9.]+)f", src).group(1))
    v = float(re.search(r"#define IL_V_TRIM\s+([0-9.]+)f", src).group(1))
    assert a == pytest.approx(G.ALPHA_TRIM, abs=1e-8) and v == G.V_TRIM


def test_pitch_target_equals_simulator_when_airspeed_equals_command():
    """With V_true == V_cmd the simulator's KV term is zero, so the firmware's
    elevator command must equal the simulator's exactly."""
    L = lib()
    import ctypes as C
    L.rlg_inner_loop.argtypes = [C.c_float] * 7

    class Surf(C.Structure):
        _fields_ = [("aileron", C.c_float), ("elevator", C.c_float), ("rudder", C.c_float),
                    ("shield_bank", C.c_uint8), ("shield_pitch", C.c_uint8)]
    L.rlg_inner_loop.restype = Surf
    ctrl = G.AttitudeController()
    rng = np.random.default_rng(2)
    for _ in range(3000):
        roll, pitch = rng.uniform(-0.75, 0.75), rng.uniform(-0.3, 0.3)
        p, q, r = rng.uniform(-1, 1, 3)
        bank, v = rng.uniform(-math.pi / 4, math.pi / 4), rng.uniform(9.5, 16.0)
        sim = ctrl.update(np.array([roll, pitch, 0.0]), np.array([p, q, r]), v, bank, v)
        s = L.rlg_inner_loop(roll, pitch, p, q, r, bank, L.rlg_pitch_target(v, roll))
        assert s.elevator == pytest.approx(sim["elevator"], abs=3e-5)
        assert s.aileron == pytest.approx(sim["aileron"], abs=3e-5)

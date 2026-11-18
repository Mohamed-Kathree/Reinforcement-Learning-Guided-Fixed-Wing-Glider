"""
test_flighttest_safety.py
=========================
The firmware safety layer (firmware/protocol/rlg_safety.h). Every scenario
runs against BOTH the Python reference (flighttest/safety.py) and the C code
(firmware/protocol/rlg_safety.c, compiled for the host). A randomised
differential test then drives both through thousands of identical event
sequences and requires identical observable state after every step.

Covers the 18 Nov 2026 rules:
  - two links, two timeouts: ESP-NOW 150 ms for BENCH/REPLAY only (never in
    FLIGHT); Pi setpoints 150 ms in FLIGHT -> failsafe glide, 5 fresh setpoints
    to resume;
  - airborne = baro > 5 m above the ground reference (NaN does not vote) OR
    GPS fix and speed > 2 m/s OR armed in FLIGHT within 60 s;
  - per-servo limits are runtime values (from the calibration file), inside
    the compiled absolute envelope;
  - arming by 5 consecutive frames.

Host-only: proves the logic, not the ESP32 integration (timers, PWM, ESP-NOW
callbacks, UART), which stays untested until hardware is available.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "c"))

from flighttest import safety as S  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402
from flighttest.safety import Sensors  # noqa: E402
import host_lib  # noqa: E402

NEUTRAL = [1480, 1510, 1500, 1495]
TIMEOUT = P.COMMAND_TIMEOUT_MS
REARM = P.REARM_CONSECUTIVE_FRAMES
GROUND = Sensors()                                   # no baro reference, no fix, stationary


class PyImpl:
    name = "python"

    def __init__(self, neutral=NEUTRAL, now=0):
        self.s = S.Safety(neutral, now)

    def cmd(self, now, sens, frame):
        return self.s.on_command(now, sens, frame)

    def tick(self, now, sens=GROUND):
        self.s.tick(now, sens)

    def pi(self, now):
        self.s.on_pi_setpoint(now)

    def airborne(self, now, sens):
        return self.s.is_airborne(now, sens)

    def limits(self, lo, hi):
        return self.s.set_limits(lo, hi)

    def flight(self, us):
        return self.s.flight_output(us)

    def failsafe(self):
        return self.s.s.pi_failsafe

    def state(self):
        return (self.s.s.mode, self.s.s.armed, tuple(self.s.s.servo_us), self.s.status_bits())


class CImpl:
    name = "c"

    def __init__(self, neutral=NEUTRAL, now=0):
        self.lib = host_lib.load()
        self.buf = ctypes.create_string_buffer(self.lib.rlg_safety_sizeof())
        self.lib.rlg_safety_init(self.buf, (ctypes.c_uint16 * 4)(*neutral), now & 0xFFFFFFFF)

    @staticmethod
    def _s(sens):
        return ctypes.byref(host_lib.CSensors(sens.baro_alt_m, sens.gps_speed_ms, sens.gps_fix))

    def cmd(self, now, sens, frame):
        return self.lib.rlg_safety_on_command(self.buf, now & 0xFFFFFFFF, self._s(sens),
                                              host_lib.buf(frame), len(frame))

    def tick(self, now, sens=GROUND):
        self.lib.rlg_safety_tick(self.buf, now & 0xFFFFFFFF, self._s(sens))

    def pi(self, now):
        self.lib.rlg_safety_on_pi_setpoint(self.buf, now & 0xFFFFFFFF)

    def airborne(self, now, sens):
        return self.lib.rlg_safety_is_airborne(self.buf, now & 0xFFFFFFFF, self._s(sens))

    def limits(self, lo, hi):
        return self.lib.rlg_safety_set_limits(self.buf, (ctypes.c_uint16 * 4)(*lo), (ctypes.c_uint16 * 4)(*hi))

    def flight(self, us):
        return self.lib.rlg_safety_flight_output(self.buf, (ctypes.c_uint16 * 4)(*us))

    def failsafe(self):
        return self.lib.rlg_safety_pi_failsafe(self.buf)

    def state(self):
        return (self.lib.rlg_safety_mode(self.buf), self.lib.rlg_safety_armed(self.buf),
                tuple(self.lib.rlg_safety_servo(self.buf, i) for i in range(4)),
                self.lib.rlg_safety_status_bits(self.buf))


IMPLS = [PyImpl] + ([CImpl] if host_lib.compiler() is not None else [])


@pytest.fixture(params=IMPLS, ids=lambda c: c.name)
def impl(request):
    return request.param


def frame(seq, mode, us=(1500, 1500, 1500, 1500)):
    return P.Command(seq=seq & 0xFFFF, mode=mode, servo_us=list(us)).encode()


def arm(x, mode, start_seq=1, t0=0, us=(1600, 1400, 1550, 1450), sens=GROUND):
    r = None
    for k in range(REARM):
        r = x.cmd(t0 + 50 * k, sens, frame(start_seq + k, mode, us))
    return r, start_seq + REARM, t0 + 50 * (REARM - 1)


# ---------------------------------------------------------------- arming, limits, ground timeout

def test_boots_disarmed_and_centred(impl):
    x = impl()
    mode, armed, servo, _ = x.state()
    assert mode == P.DISARMED and not armed and list(servo) == NEUTRAL


def test_one_frame_never_arms(impl):
    x = impl()
    assert x.cmd(0, GROUND, frame(1, P.BENCH, (2000,) * 4)) == S.ARMING
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_arms_after_consecutive_frames_and_clamps_to_envelope(impl):
    x = impl()
    r, seq, t = arm(x, P.BENCH, us=(3000, 100, 2500, 1500))
    assert r == S.ARMED
    assert x.state()[:3] == (P.BENCH, True, (2300, 600, 2300, 1500))


def test_runtime_limits_from_calibration(impl):
    x = impl()
    assert x.limits([700, 700, 900, 700], [2200, 2200, 2100, 2200])
    _, seq, t = arm(x, P.BENCH, us=(2300, 600, 2300, 1500))
    assert x.state()[2] == (2200, 700, 2100, 1500)
    assert not x.limits([1600, 600, 600, 600], [1500, 2300, 2300, 2300])    # min > max refused
    assert x.state()[2] == (2200, 700, 2100, 1500)                          # unchanged


def test_limits_cannot_widen_past_absolute_envelope(impl):
    x = impl()
    assert x.limits([100, 100, 100, 100], [3000, 3000, 3000, 3000])
    arm(x, P.BENCH, us=(100, 3000, 1500, 1500))
    assert x.state()[2] == (P.ABS_MIN_US, P.ABS_MAX_US, 1500, 1500)


def test_esp_now_timeout_centres_and_disarms_bench(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    x.tick(t + TIMEOUT)
    assert x.state()[1]
    x.tick(t + TIMEOUT + 1)
    mode, armed, servo, status = x.state()
    assert (mode, armed, list(servo)) == (P.DISARMED, False, NEUTRAL)
    assert P.unpack_status(status)["timeout_disarmed"] == 1


def test_stale_replay_packet_cannot_move_a_surface(impl):
    x = impl()
    _, seq, t = arm(x, P.REPLAY)
    x.tick(t + 500)
    assert x.cmd(t + 520, GROUND, frame(seq + 1, P.REPLAY, (2300, 600, 2000, 2300))) == S.ARMING
    assert list(x.state()[2]) == NEUTRAL


def test_rearm_after_timeout_needs_consecutive_frames(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    x.tick(t + 400)
    for k in range(REARM - 1):
        assert x.cmd(t + 500 + 50 * k, GROUND, frame(seq + k, P.BENCH)) == S.ARMING
    assert x.cmd(t + 500 + 50 * (REARM - 1), GROUND, frame(seq + REARM - 1, P.BENCH)) == S.ARMED


def test_mode_switch_centres_and_requires_rearm(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(2000,) * 4)
    assert x.cmd(t + 50, GROUND, frame(seq, P.REPLAY, (2000,) * 4)) == S.ARMING
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_disarm_command_is_immediate(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(2000,) * 4)
    assert x.cmd(t + 50, GROUND, frame(seq, P.DISARMED)) == S.DISARMED_RESULT
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_idle_expiry(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(1500,) * 4)
    now = t
    end = t + P.ARM_IDLE_EXPIRY_MS + 100
    while now < end:
        now += 50
        seq += 1
        x.cmd(now, GROUND, frame(seq, P.BENCH, (1500,) * 4))
        x.tick(now)
    assert x.state()[:2] == (P.DISARMED, False)


def test_bad_frames_change_nothing(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(1700,) * 4)
    before = x.state()
    good = bytearray(frame(seq, P.BENCH, (2000,) * 4))
    assert x.cmd(t + 10, GROUND, bytes(good[:-1]) + bytes([good[-1] ^ 0xFF])) == S.REJECTED_FRAME
    assert x.cmd(t + 20, GROUND, bytes(good[:-1])) == S.REJECTED_FRAME
    assert x.cmd(t + 30, GROUND, P.Command(seq=seq, mode=7, servo_us=[2000] * 4).encode()) == S.REJECTED_FRAME
    assert x.state() == before


def test_sequence_numbers(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, start_seq=65533)
    assert x.cmd(t + 50, GROUND, frame(seq - 1, P.BENCH, (2000,) * 4)) == S.REJECTED_STALE
    assert x.cmd(t + 60, GROUND, frame(seq, P.BENCH, (2000,) * 4)) == S.ACCEPTED


def test_clock_wrap(impl):
    t0 = 0xFFFFFFFF - 120
    x = impl(now=t0)
    _, seq, t = arm(x, P.BENCH, t0=t0)
    x.tick(t + TIMEOUT)
    assert x.state()[1]
    x.tick(t + TIMEOUT + 1)
    assert not x.state()[1]


# ---------------------------------------------------------------- airborne rules

def test_cold_barometer_cannot_vote_airborne(impl):
    """Before the ground reference exists (captured on arming after warm-up)
    the baro altitude is NaN and does not vote, however large the transient."""
    x = impl()
    assert not x.airborne(0, Sensors(baro_alt_m=math.nan))
    r, _, _ = arm(x, P.BENCH, sens=Sensors(baro_alt_m=math.nan))
    assert r == S.ARMED


def test_baro_threshold_is_5m(impl):
    x = impl()
    assert not x.airborne(0, Sensors(baro_alt_m=4.9))
    assert x.airborne(0, Sensors(baro_alt_m=5.1))
    assert x.cmd(0, Sensors(baro_alt_m=5.1), frame(1, P.BENCH)) == S.REJECTED_AIRBORNE


def test_gps_speed_votes_only_with_a_fix(impl):
    x = impl()
    assert not x.airborne(0, Sensors(gps_speed_ms=9.0, gps_fix=0))
    assert not x.airborne(0, Sensors(gps_speed_ms=1.9, gps_fix=2))
    assert x.airborne(0, Sensors(gps_speed_ms=2.1, gps_fix=1))


def test_flight_latch_refuses_bench_for_60s_after_flight(impl):
    x = impl()
    _, seq, t = arm(x, P.FLIGHT)
    for k in range(5):
        x.pi(t + 10 * k)
    t_off = t + 1000
    assert x.cmd(t_off, GROUND, frame(seq, P.DISARMED)) == S.DISARMED_RESULT
    # Sitting still on the ground with no sensor votes: still "airborne" for 60 s.
    for k in range(REARM):
        assert x.cmd(t_off + 50 * (k + 1), GROUND, frame(seq + 1 + k, P.BENCH)) == S.REJECTED_AIRBORNE
    assert x.airborne(t_off + P.FLIGHT_LATCH_MS, GROUND)
    assert not x.airborne(t_off + P.FLIGHT_LATCH_MS + 1, GROUND)
    r, _, _ = arm(x, P.BENCH, start_seq=seq + 100, t0=t_off + P.FLIGHT_LATCH_MS + 1000)
    assert r == S.ARMED


def test_armed_flight_counts_as_airborne(impl):
    x = impl()
    _, seq, t = arm(x, P.FLIGHT)
    assert x.airborne(t + 1, GROUND)
    assert x.cmd(t + 50, GROUND, frame(seq, P.REPLAY)) == S.REJECTED_AIRBORNE


def test_becoming_airborne_disarms_bench(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    assert x.cmd(t + 50, Sensors(baro_alt_m=6.0), frame(seq, P.BENCH)) == S.REJECTED_AIRBORNE
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


# ---------------------------------------------------------------- FLIGHT: the two links

def test_esp_now_silence_never_affects_flight(impl):
    x = impl()
    r, seq, t = arm(x, P.FLIGHT, us=(2300, 600, 2000, 2300))
    assert r == S.ARMED
    assert list(x.state()[2]) == NEUTRAL               # ground servo values ignored in FLIGHT
    for k in range(0, 10_000, 50):                     # ESP-NOW silent 10 s; Pi alive throughout
        x.pi(t + k)
        x.tick(t + k)
    assert x.state()[:2] == (P.FLIGHT, True) and not x.failsafe()


def test_pi_timeout_enters_failsafe_glide_and_resumes(impl):
    x = impl()
    _, seq, t = arm(x, P.FLIGHT)
    assert x.failsafe()                                # no Pi setpoints yet
    for k in range(P.PI_RESUME_FRAMES):
        x.pi(t + 50 * k)
    assert not x.failsafe()
    last = t + 50 * (P.PI_RESUME_FRAMES - 1)
    x.tick(last + P.PI_TIMEOUT_MS)
    assert not x.failsafe()
    x.tick(last + P.PI_TIMEOUT_MS + 1)                 # Pi silent > 150 ms
    assert x.failsafe() and x.state()[:2] == (P.FLIGHT, True)   # still flying, in failsafe glide
    assert P.unpack_status(x.state()[3])["pi_failsafe"] == 1
    t2 = last + 1000
    x.pi(t2)
    assert x.failsafe()                                # one setpoint is not enough
    for k in range(1, P.PI_RESUME_FRAMES):
        x.pi(t2 + 50 * k)
    assert not x.failsafe()


def test_pi_link_does_not_matter_outside_flight(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    x.tick(t + 100)                                    # no Pi setpoints at all
    assert not x.failsafe() and x.state()[:2] == (P.BENCH, True)


def test_flight_output_only_in_flight(impl):
    x = impl()
    arm(x, P.BENCH, us=(1500,) * 4)
    assert not x.flight([2000] * 4)
    x2 = impl()
    arm(x2, P.FLIGHT)
    assert x2.flight([1600, 1400, 2600, 1500])
    assert x2.state()[2] == (1600, 1400, 2300, 1500)


# ---------------------------------------------------------------- differential

@pytest.mark.skipif(host_lib.compiler() is None, reason="no C compiler: differential test needs the C side")
@pytest.mark.parametrize("seed", range(8))
def test_python_and_c_agree_on_random_sequences(seed):
    rng = random.Random(seed)
    neutral = [rng.randrange(1300, 1700) for _ in range(4)]
    t0 = rng.choice([0, 0xFFFFFFFF - 5000])
    a, b = PyImpl(neutral, t0), CImpl(neutral, t0)
    now, seq = t0, rng.randrange(65536)

    def sens():
        alt = rng.choice([math.nan, math.nan, rng.uniform(0, 4.9), rng.uniform(4.9, 5.1), rng.uniform(5.1, 20)])
        return Sensors(baro_alt_m=float(alt), gps_speed_ms=rng.choice([0.0, 1.0, 2.0, 2.5, 9.0]),
                       gps_fix=rng.choice([0, 0, 1, 2]))

    def f32(s):     # the C side stores floats; compare on identical inputs
        return Sensors(baro_alt_m=float(ctypes.c_float(s.baro_alt_m).value),
                       gps_speed_ms=float(ctypes.c_float(s.gps_speed_ms).value), gps_fix=s.gps_fix)

    for step in range(4000):
        now += rng.choice([0, 10, 50, 50, 50, 140, 150, 151, 400]) if rng.random() < 0.97 \
            else rng.choice([rng.randrange(59_000, 61_500), rng.randrange(100_000, 130_000)])
        k = rng.random()
        if k < 0.55:
            seq += rng.choice([1, 1, 1, 1, 0, -1, 2, 300])
            mode = rng.choice([P.BENCH, P.BENCH, P.REPLAY, P.FLIGHT, P.FLIGHT, P.DISARMED])
            f = bytearray(frame(seq, mode, [rng.choice([1500, rng.randrange(400, 2600)]) for _ in range(4)]))
            if rng.random() < 0.05:
                f[rng.randrange(len(f))] ^= 1 << rng.randrange(8)
            s = f32(sens())
            ra, rb = a.cmd(now, s, bytes(f)), b.cmd(now, s, bytes(f))
            assert ra == rb, (seed, step, S.RESULT_NAMES[ra], S.RESULT_NAMES[rb])
        elif k < 0.75:
            s = f32(sens())
            a.tick(now, s)
            b.tick(now, s)
        elif k < 0.90:
            a.pi(now)
            b.pi(now)
        elif k < 0.97:
            us = [rng.randrange(400, 2600) for _ in range(4)]
            assert a.flight(us) == b.flight(us), (seed, step)
        else:
            lo = [rng.randrange(400, 1500) for _ in range(4)]
            hi = [rng.randrange(1400, 2600) for _ in range(4)]
            assert a.limits(lo, hi) == b.limits(lo, hi), (seed, step)
        assert a.state() == b.state(), (seed, step, a.state(), b.state())
        assert a.failsafe() == b.failsafe(), (seed, step)

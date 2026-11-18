"""
test_flighttest_safety.py
=========================
Command safety for the control surfaces (flight-test brief, SAFETY section).
Every scenario runs against BOTH the Python reference (flighttest/safety.py)
and the C firmware code (firmware/protocol/rlg_safety.c, compiled for the host).
A randomised differential test then drives both through thousands of identical
event sequences and requires identical observable state after every step.

Host-only: this proves the logic, not the ESP32 integration (timers, PWM,
ESP-NOW callbacks), which remains untested until hardware is available.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "c"))

from flighttest import safety as S  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402
import host_lib  # noqa: E402

NEUTRAL = [1480, 1510, 1500, 1495]
TIMEOUT = P.COMMAND_TIMEOUT_MS
REARM = P.REARM_CONSECUTIVE_FRAMES


class PyImpl:
    name = "python"

    def __init__(self, neutral=NEUTRAL, now=0):
        self.s = S.Safety(neutral, now)

    def cmd(self, now, airborne, frame):
        return self.s.on_command(now, airborne, frame)

    def tick(self, now):
        self.s.tick(now)

    def flight(self, us):
        return self.s.flight_output(us)

    def state(self):
        return (self.s.s.mode, self.s.s.armed, tuple(self.s.s.servo_us), self.s.status_bits())


class CImpl:
    name = "c"

    def __init__(self, neutral=NEUTRAL, now=0):
        self.lib = host_lib.load()
        self.buf = ctypes.create_string_buffer(self.lib.rlg_safety_sizeof())
        n = (ctypes.c_uint16 * 4)(*neutral)
        self.lib.rlg_safety_init(self.buf, n, now & 0xFFFFFFFF)

    def cmd(self, now, airborne, frame):
        return self.lib.rlg_safety_on_command(self.buf, now & 0xFFFFFFFF, airborne, host_lib.buf(frame), len(frame))

    def tick(self, now):
        self.lib.rlg_safety_tick(self.buf, now & 0xFFFFFFFF)

    def flight(self, us):
        return self.lib.rlg_safety_flight_output(self.buf, (ctypes.c_uint16 * 4)(*us))

    def state(self):
        return (self.lib.rlg_safety_mode(self.buf), self.lib.rlg_safety_armed(self.buf),
                tuple(self.lib.rlg_safety_servo(self.buf, i) for i in range(4)),
                self.lib.rlg_safety_status_bits(self.buf))


IMPLS = [PyImpl]
if host_lib.compiler() is not None:
    IMPLS.append(CImpl)


@pytest.fixture(params=IMPLS, ids=lambda c: c.name)
def impl(request):
    return request.param


def frame(seq, mode, us=(1500, 1500, 1500, 1500)):
    return P.Command(seq=seq & 0xFFFF, mode=mode, servo_us=list(us)).encode()


def arm(x, mode, start_seq=1, t0=0, us=(1600, 1400, 1550, 1450), airborne=False):
    """Send REARM frames at 20 Hz; returns (next_seq, time of last frame)."""
    r = None
    for k in range(REARM):
        r = x.cmd(t0 + 50 * k, airborne, frame(start_seq + k, mode, us))
    return r, start_seq + REARM, t0 + 50 * (REARM - 1)


def test_boots_disarmed_and_centred(impl):
    x = impl()
    mode, armed, servo, _ = x.state()
    assert mode == P.DISARMED and not armed and list(servo) == NEUTRAL


def test_one_frame_never_arms(impl):
    x = impl()
    assert x.cmd(0, False, frame(1, P.BENCH, (2000, 2000, 2000, 2000))) == S.ARMING
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_arms_after_consecutive_frames_and_clamps(impl):
    x = impl()
    r, seq, t = arm(x, P.BENCH, us=(3000, 100, 2500, 1500))
    assert r == S.ARMED
    mode, armed, servo, status = x.state()
    assert mode == P.BENCH and armed
    assert servo == (2300, 600, 2000, 1500)          # hard limits, elevator 1000-2000
    assert P.unpack_status(status)["armed"] == 1


def test_timeout_centres_and_disarms(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    x.tick(t + TIMEOUT)                              # exactly 150 ms: still armed
    assert x.state()[1]
    x.tick(t + TIMEOUT + 1)
    mode, armed, servo, status = x.state()
    assert (mode, armed, list(servo)) == (P.DISARMED, False, NEUTRAL)
    assert P.unpack_status(status)["timeout_disarmed"] == 1


def test_three_missed_frames_at_20hz_disarm(impl):
    x = impl()
    _, seq, t = arm(x, P.REPLAY)
    for now in range(t + 10, t + 200, 10):          # frames at t+50, t+100, t+150 are missed
        x.tick(now)
    assert x.state()[0] == P.DISARMED


def test_stale_replay_packet_cannot_move_a_surface(impl):
    x = impl()
    _, seq, t = arm(x, P.REPLAY)
    x.tick(t + 500)                                  # link lost -> disarmed
    late = x.cmd(t + 520, False, frame(seq + 1, P.REPLAY, (2300, 600, 2000, 2300)))
    assert late == S.ARMING
    assert list(x.state()[2]) == NEUTRAL             # surfaces did not move


def test_rearm_after_timeout_needs_consecutive_frames(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    x.tick(t + 400)
    for k in range(REARM - 1):
        assert x.cmd(t + 500 + 50 * k, False, frame(seq + k, P.BENCH)) == S.ARMING
    assert x.cmd(t + 500 + 50 * (REARM - 1), False, frame(seq + REARM - 1, P.BENCH)) == S.ARMED


def test_arming_interrupted_by_gap_restarts_count(impl):
    x = impl()
    for k in range(REARM - 1):
        x.cmd(50 * k, False, frame(1 + k, P.BENCH))
    t = 50 * (REARM - 2) + TIMEOUT + 1
    x.tick(t)
    assert x.cmd(t + 1, False, frame(100, P.BENCH)) == S.ARMING     # count restarted at 1
    assert not x.state()[1]


def test_bench_and_replay_refused_when_airborne(impl):
    for mode in (P.BENCH, P.REPLAY):
        x = impl()
        for k in range(REARM + 2):
            assert x.cmd(50 * k, True, frame(1 + k, mode, (2000,) * 4)) == S.REJECTED_AIRBORNE
        mode_now, armed, servo, status = x.state()
        assert not armed and list(servo) == NEUTRAL
        assert P.unpack_status(status)["cmd_rejected_airborne"] == 1


def test_becoming_airborne_disarms_bench(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH)
    assert x.cmd(t + 50, True, frame(seq, P.BENCH)) == S.REJECTED_AIRBORNE
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_flight_mode_ignores_ground_servos_and_has_no_link_timeout(impl):
    x = impl()
    r, seq, t = arm(x, P.FLIGHT, us=(2300, 600, 2000, 2300))
    assert r == S.ARMED
    assert list(x.state()[2]) == NEUTRAL             # ground values ignored in FLIGHT
    x.tick(t + 10_000)                               # ESP-NOW silent for 10 s in flight
    assert x.state()[:2] == (P.FLIGHT, True)
    assert x.flight([1600, 1400, 2600, 1500])
    assert x.state()[2] == (1600, 1400, 2000, 1500)  # clamped


def test_flight_output_ignored_outside_flight(impl):
    x = impl()
    arm(x, P.BENCH, us=(1500,) * 4)
    assert not x.flight([2000] * 4)
    assert x.state()[2] == (1500,) * 4


def test_mode_switch_centres_and_requires_rearm(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(2000,) * 4)
    assert x.cmd(t + 50, False, frame(seq, P.REPLAY, (2000,) * 4)) == S.ARMING
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_disarm_command_is_immediate(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(2000,) * 4)
    assert x.cmd(t + 50, False, frame(seq, P.DISARMED)) == S.DISARMED_RESULT
    assert x.state()[:3] == (P.DISARMED, False, tuple(NEUTRAL))


def test_idle_expiry_disarms_but_activity_keeps_it(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(1500,) * 4)
    now = t
    while now < t + P.ARM_IDLE_EXPIRY_MS - 1000:     # unchanged values for ~119 s
        now += 50
        seq += 1
        x.cmd(now, False, frame(seq, P.BENCH, (1500,) * 4))
        x.tick(now)
    assert x.state()[1]
    seq += 1
    now += 50
    x.cmd(now, False, frame(seq, P.BENCH, (1510, 1500, 1500, 1500)))   # activity resets the clock
    end = now + P.ARM_IDLE_EXPIRY_MS + 100
    while now < end:
        now += 50
        seq += 1
        x.cmd(now, False, frame(seq, P.BENCH, (1510, 1500, 1500, 1500)))
        x.tick(now)
    assert x.state()[:2] == (P.DISARMED, False)


def test_bad_frames_change_nothing(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, us=(1700,) * 4)
    before = x.state()
    good = bytearray(frame(seq, P.BENCH, (2000,) * 4))
    corrupt = bytes(good[:-1]) + bytes([good[-1] ^ 0xFF])
    assert x.cmd(t + 10, False, corrupt) == S.REJECTED_FRAME
    assert x.cmd(t + 20, False, bytes(good[:-1])) == S.REJECTED_FRAME
    bad_mode = P.Command(seq=seq, mode=7, servo_us=[2000] * 4).encode()
    assert x.cmd(t + 30, False, bad_mode) == S.REJECTED_FRAME
    assert x.state() == before


def test_sequence_numbers(impl):
    x = impl()
    _, seq, t = arm(x, P.BENCH, start_seq=65533)     # crosses the u16 wrap while arming
    assert x.state()[1]
    assert x.cmd(t + 50, False, frame(seq - 1, P.BENCH, (2000,) * 4)) == S.REJECTED_STALE   # duplicate
    assert x.cmd(t + 60, False, frame(seq, P.BENCH, (2000,) * 4)) == S.ACCEPTED


def test_neutral_outside_limits_is_clamped(impl):
    x = impl(neutral=[500, 2500, 900, 1500])
    assert x.state()[2] == (600, 2300, 1000, 1500)


def test_clock_wrap(impl):
    t0 = 0xFFFFFFFF - 120
    x = impl(now=t0)
    _, seq, t = arm(x, P.BENCH, t0=t0)               # arming crosses the u32 millisecond wrap
    assert x.state()[1]
    x.tick(t + TIMEOUT)
    assert x.state()[1]
    x.tick(t + TIMEOUT + 1)
    assert not x.state()[1]


@pytest.mark.skipif(host_lib.compiler() is None, reason="no C compiler: differential test needs the C side")
@pytest.mark.parametrize("seed", range(6))
def test_python_and_c_agree_on_random_sequences(seed):
    rng = random.Random(seed)
    neutral = [rng.randrange(1300, 1700) for _ in range(4)]
    t0 = rng.choice([0, 0xFFFFFFFF - 5000])
    a, b = PyImpl(neutral, t0), CImpl(neutral, t0)
    now, seq, airborne = t0, rng.randrange(65536), False
    for step in range(3000):
        r = rng.random()
        now += rng.choice([0, 10, 50, 50, 50, 140, 150, 151, 400]) if r < 0.97 else rng.randrange(100_000, 130_000)
        if rng.random() < 0.02:
            airborne = not airborne
        kind = rng.random()
        if kind < 0.70:
            seq += rng.choice([1, 1, 1, 1, 0, -1, 2, 300])
            mode = rng.choice([P.BENCH, P.BENCH, P.REPLAY, P.FLIGHT, P.DISARMED])
            us = [rng.choice([1500, rng.randrange(400, 2600)]) for _ in range(4)]
            f = bytearray(frame(seq, mode, us))
            if rng.random() < 0.05:
                f[rng.randrange(len(f))] ^= 1 << rng.randrange(8)
            ra, rb = a.cmd(now, airborne, bytes(f)), b.cmd(now, airborne, bytes(f))
            assert ra == rb, (seed, step, S.RESULT_NAMES[ra], S.RESULT_NAMES[rb])
        elif kind < 0.95:
            a.tick(now)
            b.tick(now)
        else:
            us = [rng.randrange(400, 2600) for _ in range(4)]
            assert a.flight(us) == b.flight(us), (seed, step)
        assert a.state() == b.state(), (seed, step, a.state(), b.state())

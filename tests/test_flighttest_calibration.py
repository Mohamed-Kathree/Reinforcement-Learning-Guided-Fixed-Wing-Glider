"""
test_flighttest_calibration.py
==============================
The stored calibration file and its validator (flighttest/calibration.py):
neutrals inside the hard travel limits, directions of +/-1, an IMU transform
that is a signed permutation with determinant +1, and saves that refuse
invalid data.
"""

from __future__ import annotations

import itertools
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flighttest import calibration as C  # noqa: E402


def test_committed_file_is_valid_but_marked_unverified():
    cal = C.load()
    assert C.validate(cal) == []
    assert cal["imu"]["verified"] is False
    assert all(not s["measured"] for s in cal["servos"].values())


def test_every_signed_permutation_is_classified_by_handedness():
    """All 48 signed permutations: exactly the 24 right-handed ones pass."""
    ok = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0] * 3 for _ in range(3)]
            for r, c in enumerate(perm):
                m[r][c] = signs[r]
            cal = C.with_updates(C.load(), **{"imu.board_to_body": m})
            if C.validate(cal) == []:
                ok += 1
                assert C.det3(m) == 1
            else:
                assert C.det3(m) == -1
    assert ok == 24


def test_single_sign_flip_is_rejected_as_left_handed():
    cal = C.with_updates(C.load(), **{"imu.board_to_body": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]})
    problems = C.validate(cal)
    assert problems and "left-handed" in problems[0]


def test_non_permutation_rejected():
    for m in ([[1, 1, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [[2, 0, 0], [0, 1, 0], [0, 0, 1]]):
        assert C.validate(C.with_updates(C.load(), **{"imu.board_to_body": m}))


@pytest.mark.parametrize("path,value", [
    ("servos.aileron_l.neutral_us", 500),
    ("servos.elevator.neutral_us", 2100),      # elevator hard limit is 2000 us
    ("servos.rudder.direction", 0),
    ("baro.warmup_s", -1),
])
def test_bad_values_rejected(path, value):
    assert C.validate(C.with_updates(C.load(), **{path: value}))


def test_save_refuses_invalid_and_round_trips_valid(tmp_path):
    p = str(tmp_path / "cal.json")
    good = C.with_updates(C.load(), **{"servos.rudder.neutral_us": 1490, "servos.rudder.measured": True})
    C.save(good, p)
    assert C.load(p)["servos"]["rudder"]["neutral_us"] == 1490
    bad = C.with_updates(good, **{"imu.board_to_body": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    with pytest.raises(ValueError):
        C.save(bad, p)
    assert C.load(p)["servos"]["rudder"]["neutral_us"] == 1490   # file untouched


def test_neutral_order_matches_command_servo_order():
    cal = C.with_updates(C.load(), **{"servos.aileron_l.neutral_us": 1401, "servos.rudder.neutral_us": 1404})
    n = C.neutral_us(cal)
    assert n[0] == 1401 and n[3] == 1404


def test_imu_bench_checks_record_values():
    r = C.check_imu_readings(level_az_g=-0.98, nose_up_pitch=0.2, right_wing_down_roll=0.3, nose_right_gz=0.5)
    assert r["all_pass"] and r["checks"]["level_az"]["value"] == -0.98
    r = C.check_imu_readings(level_az_g=0.99, nose_up_pitch=0.2, right_wing_down_roll=-0.3, nose_right_gz=0.5)
    assert not r["all_pass"]
    assert not r["checks"]["level_az"]["pass"] and not r["checks"]["right_wing_down_roll"]["pass"]

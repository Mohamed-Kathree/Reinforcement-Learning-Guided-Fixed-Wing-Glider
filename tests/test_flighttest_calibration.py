"""
test_flighttest_calibration.py
==============================
The stored calibration file and its validator (flighttest/calibration.py):
per-servo type and limits (inside the type envelope, ordered min <= sweep_min
<= neutral <= sweep_max <= max), directions of +/-1, an IMU transform that is a
signed permutation with determinant +1, rail-voltage divider scaling on a free
ADC1 pin, and saves that refuse invalid data.
"""

from __future__ import annotations

import itertools
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flighttest import calibration as C  # noqa: E402


def test_committed_file_matches_the_hardware_answers():
    cal = C.load()
    assert C.validate(cal) == []
    assert cal["imu"]["verified"] is False and cal["rail"]["calibrated"] is False
    for name, sv in cal["servos"].items():           # 18 Nov: all four servos are MG90S
        assert sv["type"] == "MG90S" and (sv["min_us"], sv["max_us"]) == (600, 2300)
        assert (sv["sweep_min_us"], sv["sweep_max_us"]) == (700, 2200) and not sv["measured"]
    assert cal["rail"]["adc_gpio"] == 34 and cal["rail"]["brownout_v"] == 4.75


def test_every_signed_permutation_is_classified_by_handedness():
    ok = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0] * 3 for _ in range(3)]
            for r, c in enumerate(perm):
                m[r][c] = signs[r]
            valid = C.validate(C.with_updates(C.load(), **{"imu.board_to_body": m})) == []
            assert valid == (C.det3(m) == 1)
            ok += valid
    assert ok == 24


def test_single_sign_flip_is_rejected_as_left_handed():
    problems = C.validate(C.with_updates(C.load(), **{"imu.board_to_body": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}))
    assert problems and "left-handed" in problems[0]


@pytest.mark.parametrize("path,value", [
    ("servos.aileron_l.min_us", 500),            # below the MG90S envelope
    ("servos.rudder.max_us", 2400),              # above it
    ("servos.elevator.neutral_us", 2250),        # outside the sweep range
    ("servos.elevator.sweep_min_us", 2250),      # sweep_min above sweep_max
    ("servos.rudder.type", "SG90"),              # unknown type
    ("servos.rudder.direction", 0),
    ("servos.rudder.us_full_deflection", 20),
    ("servos.rudder.us_full_deflection", 900),     # 1500 + 900 leaves the 2300 limit
    ("servos.rudder.neutral_us", 1500.5),
    ("baro.warmup_s", -1),
    ("rail.adc_gpio", 35),                       # sonar echo pin
    ("rail.adc_gpio", 25),                       # ADC2: unusable with the radio on
    ("rail.scale", 0.0),
    ("rail.offset_v", 2.0),
    ("rail.brownout_v", 3.3),
])
def test_bad_values_rejected(path, value):
    assert C.validate(C.with_updates(C.load(), **{path: value}))


def test_swapping_a_servo_type_is_a_config_change():
    """Fitting an MG996R again would need its own envelope (1000-2000 us)."""
    cal = C.with_updates(C.load(), **{"servos.elevator.type": "MG996R"})
    assert C.validate(cal)                                       # 600-2300 no longer fits
    cal = C.with_updates(cal, **{"servos.elevator.min_us": 1000, "servos.elevator.max_us": 2000,
                                 "servos.elevator.sweep_min_us": 1050, "servos.elevator.sweep_max_us": 1950})
    assert C.validate(cal) == []


def test_save_refuses_invalid_and_round_trips_valid(tmp_path):
    p = str(tmp_path / "cal.json")
    good = C.with_updates(C.load(), **{"servos.rudder.neutral_us": 1490, "servos.rudder.measured": True})
    C.save(good, p)
    assert C.load(p)["servos"]["rudder"]["neutral_us"] == 1490
    with pytest.raises(ValueError):
        C.save(C.with_updates(good, **{"imu.board_to_body": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]}), p)
    assert C.load(p)["servos"]["rudder"]["neutral_us"] == 1490


def test_accessors_follow_command_servo_order():
    cal = C.with_updates(C.load(), **{"servos.aileron_l.neutral_us": 1401, "servos.rudder.neutral_us": 1404,
                                       "servos.rudder.max_us": 2250})
    assert C.neutral_us(cal)[0] == 1401 and C.neutral_us(cal)[3] == 1404
    lo, hi = C.limits(cal)
    assert lo == [600] * 4 and hi == [2300, 2300, 2300, 2250]
    assert C.sweep(cal) == ([700] * 4, [2200] * 4)


def test_rail_volts_scaling():
    cal = C.load()
    assert C.rail_volts(cal, 2.5) == pytest.approx(5.0)
    assert C.rail_volts(cal, -1.0) is None and C.rail_volts(cal, None) is None
    cal = C.with_updates(cal, **{"rail.scale": 1.98, "rail.offset_v": 0.02})
    assert C.rail_volts(cal, 2.5) == pytest.approx(4.97)


def test_imu_bench_checks_record_values():
    r = C.check_imu_readings(level_az_g=-0.98, nose_up_pitch=0.2, right_wing_down_roll=0.3, nose_right_gz=0.5)
    assert r["all_pass"] and r["checks"]["level_az"]["value"] == -0.98
    r = C.check_imu_readings(level_az_g=0.99, nose_up_pitch=0.2, right_wing_down_roll=-0.3, nose_right_gz=0.5)
    assert not r["all_pass"]

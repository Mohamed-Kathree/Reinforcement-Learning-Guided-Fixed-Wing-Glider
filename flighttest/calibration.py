"""
calibration.py
==============
The flight-test calibration file (flighttest/config/calibration.json): per-servo
neutral and direction, the IMU board-to-body transform, barometer warm-up.
Stored, never hard-coded. Every load and save is validated here.

IMU convention (flight-test brief): body frame x forward, y right, z down.
The board-to-body transform must be a signed permutation with determinant +1.
An odd number of sign flips gives a left-handed frame: plausible Euler angles,
silently wrong cross products. validate() refuses it.

Bench checks for a verified transform (check_imu_readings):
  level board        -> az near -1.00 g
  nose up            -> positive pitch
  right wing down    -> positive roll
  nose right         -> positive gz
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

from flighttest.protocol import packets as P

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "calibration.json")
SERVO_LIMITS = {s["name"]: (s["min_us"], s["max_us"]) for s in P.SERVOS}


def det3(m: list[list[int]]) -> int:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def validate(cal: dict) -> list[str]:
    """Every problem with a calibration dict; empty list = valid."""
    problems: list[str] = []
    servos = cal.get("servos", {})
    if set(servos) != set(SERVO_LIMITS):
        problems.append(f"servos must be exactly {sorted(SERVO_LIMITS)}, got {sorted(servos)}")
    for name, sv in servos.items():
        if name not in SERVO_LIMITS:
            continue
        lo, hi = SERVO_LIMITS[name]
        n = sv.get("neutral_us")
        if not isinstance(n, int) or not lo <= n <= hi:
            problems.append(f"{name}.neutral_us {n!r} outside the {lo}-{hi} us travel limit")
        if sv.get("direction") not in (1, -1):
            problems.append(f"{name}.direction must be +1 or -1, got {sv.get('direction')!r}")
    m = cal.get("imu", {}).get("board_to_body")
    if (not isinstance(m, list) or len(m) != 3 or any(not isinstance(r, list) or len(r) != 3 for r in m)
            or any(v not in (-1, 0, 1) for r in m for v in r)):
        problems.append("imu.board_to_body must be a 3x3 matrix of -1, 0, 1")
    else:
        rows_ok = all(sum(abs(v) for v in r) == 1 for r in m)
        cols_ok = all(sum(abs(m[r][c]) for r in range(3)) == 1 for c in range(3))
        if not (rows_ok and cols_ok):
            problems.append("imu.board_to_body must map each board axis to exactly one body axis")
        elif det3(m) != 1:
            problems.append(f"imu.board_to_body has determinant {det3(m)}: a left-handed frame "
                            "(odd number of sign flips). Flip one more axis or undo one.")
    w = cal.get("baro", {}).get("warmup_s")
    if not isinstance(w, (int, float)) or w < 0:
        problems.append(f"baro.warmup_s must be a non-negative number, got {w!r}")
    return problems


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    problems = validate(cal)
    if problems:
        raise ValueError("invalid calibration: " + "; ".join(problems))
    return cal


def save(cal: dict, path: str = DEFAULT_PATH) -> None:
    problems = validate(cal)
    if problems:
        raise ValueError("refusing to save invalid calibration: " + "; ".join(problems))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def neutral_us(cal: dict) -> list[int]:
    """Neutrals in Command.servo_us order."""
    return [cal["servos"][s["name"]]["neutral_us"] for s in P.SERVOS]


def board_to_body(cal: dict, v: list[float]) -> list[float]:
    m = cal["imu"]["board_to_body"]
    return [sum(m[r][c] * v[c] for c in range(3)) for r in range(3)]


def check_imu_readings(level_az_g: float, nose_up_pitch: float, right_wing_down_roll: float,
                       nose_right_gz: float, az_tol_g: float = 0.1) -> dict:
    """The four bench checks, on body-frame readings. Every value is returned
    alongside pass/fail so the measurement is recorded, not just the verdict."""
    checks = {
        "level_az": {"value": level_az_g, "expect": "near -1.00 g", "pass": abs(level_az_g + 1.0) <= az_tol_g},
        "nose_up_pitch": {"value": nose_up_pitch, "expect": "> 0", "pass": nose_up_pitch > 0},
        "right_wing_down_roll": {"value": right_wing_down_roll, "expect": "> 0", "pass": right_wing_down_roll > 0},
        "nose_right_gz": {"value": nose_right_gz, "expect": "> 0", "pass": nose_right_gz > 0},
    }
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values())}


def with_updates(cal: dict, **changes) -> dict:
    """Copy with dotted-path updates, e.g. with_updates(cal, **{'servos.rudder.neutral_us': 1490})."""
    out = deepcopy(cal)
    for path, value in changes.items():
        node = out
        *head, last = path.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    return out

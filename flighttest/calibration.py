"""
calibration.py
==============
The flight-test calibration file (flighttest/config/calibration.json): per-servo
type, neutral, direction, travel limits and sweep range; the IMU board-to-body
transform; barometer warm-up; rail-voltage divider scaling. Stored, never
hard-coded. Every load and save is validated here.

Servos: per-servo limits come from this file (the elevator servo has already
been swapped once). Each servo's limits must sit inside its type's envelope
(protocol schema servo_types), and
    min_us <= sweep_min_us <= neutral_us <= sweep_max_us <= max_us.

IMU convention: body frame x forward, y right, z down. The transform must be a
signed permutation with determinant +1; an odd number of sign flips gives a
left-handed frame (plausible Euler angles, silently wrong cross products).

Rail voltage: the ESP32 reports the voltage at the GPIO34 ADC pin; the rail is
scale * adc_v + offset_v. Indicative until calibrated against a multimeter.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy

from flighttest.protocol import packets as P

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "calibration.json")
SERVO_NAMES = [s["name"] for s in P.SERVOS]
RAIL_ADC_PINS = (34, 36, 39)      # free ADC1 input-only pins (32 = Pi UART, 35 = sonar echo)


def det3(m: list[list[int]]) -> int:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _servo_problems(name: str, sv: dict) -> list[str]:
    out = []
    t = sv.get("type")
    if t not in P.SERVO_TYPES:
        return [f"{name}.type {t!r} is not a known servo type {sorted(P.SERVO_TYPES)}"]
    lo_t, hi_t = P.SERVO_TYPES[t]
    keys = ("min_us", "sweep_min_us", "neutral_us", "sweep_max_us", "max_us")
    vals = [sv.get(k) for k in keys]
    if any(not isinstance(v, int) or isinstance(v, bool) for v in vals):
        return [f"{name}: {', '.join(keys)} must all be integers"]
    if vals[0] < lo_t or vals[-1] > hi_t:
        out.append(f"{name}: limits {vals[0]}-{vals[-1]} us exceed the {t} envelope {lo_t}-{hi_t} us")
    if vals != sorted(vals):
        out.append(f"{name}: need min_us <= sweep_min_us <= neutral_us <= sweep_max_us <= max_us, got {vals}")
    full = sv.get("us_full_deflection")
    if not isinstance(full, int) or isinstance(full, bool) or not 50 <= full <= 900:
        out.append(f"{name}.us_full_deflection must be an integer 50-900 us, got {full!r}")
    elif vals == sorted(vals) and not (vals[0] <= vals[2] - full and vals[2] + full <= vals[-1]):
        out.append(f"{name}: neutral +/- us_full_deflection ({vals[2]} +/- {full}) leaves the travel limits")
    if sv.get("direction") not in (1, -1):
        out.append(f"{name}.direction must be +1 or -1, got {sv.get('direction')!r}")
    return out


def validate(cal: dict) -> list[str]:
    """Every problem with a calibration dict; empty list = valid."""
    problems: list[str] = []
    servos = cal.get("servos", {})
    if set(servos) != set(SERVO_NAMES):
        problems.append(f"servos must be exactly {sorted(SERVO_NAMES)}, got {sorted(servos)}")
    for name, sv in servos.items():
        if name in SERVO_NAMES:
            problems += _servo_problems(name, sv)

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

    rail = cal.get("rail", {})
    if rail.get("adc_gpio") not in RAIL_ADC_PINS:
        problems.append(f"rail.adc_gpio must be a free ADC1 pin {RAIL_ADC_PINS} (ADC2 is unusable with the radio on)")
    sc, off, bo = rail.get("scale"), rail.get("offset_v"), rail.get("brownout_v")
    if not isinstance(sc, (int, float)) or not math.isfinite(sc) or not 1.0 <= sc <= 4.0:
        problems.append(f"rail.scale must be between 1 and 4 (divider ratio), got {sc!r}")
    if not isinstance(off, (int, float)) or not math.isfinite(off) or abs(off) > 0.5:
        problems.append(f"rail.offset_v must be within +/-0.5 V, got {off!r}")
    if not isinstance(bo, (int, float)) or not 4.0 <= bo <= 5.5:
        problems.append(f"rail.brownout_v must be 4.0-5.5 V, got {bo!r}")
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


def _per_servo(cal: dict, key: str) -> list[int]:
    return [cal["servos"][n][key] for n in SERVO_NAMES]


def neutral_us(cal: dict) -> list[int]:
    """In Command.servo_us order."""
    return _per_servo(cal, "neutral_us")


def limits(cal: dict) -> tuple[list[int], list[int]]:
    """(min_us, max_us) travel limits in Command.servo_us order."""
    return _per_servo(cal, "min_us"), _per_servo(cal, "max_us")


def full_deflection_us(cal: dict) -> list[int]:
    """Microseconds from neutral for the simulator's full deflection, per servo."""
    return _per_servo(cal, "us_full_deflection")


def directions(cal: dict) -> list[int]:
    return _per_servo(cal, "direction")


def sweep(cal: dict) -> tuple[list[int], list[int]]:
    """(sweep_min_us, sweep_max_us): the normal operating range, for UI sliders."""
    return _per_servo(cal, "sweep_min_us"), _per_servo(cal, "sweep_max_us")


def rail_volts(cal: dict, adc_v: float | None) -> float | None:
    """Rail voltage from the ADC pin voltage; None if not measured."""
    if adc_v is None or adc_v < 0:
        return None
    r = cal["rail"]
    return r["scale"] * adc_v + r["offset_v"]


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


# Simulator surface limits (rad), as firmware/flight/src/core/inner_loop.h IL_*_MAX.
SURFACE_MAX_RAD = {"aileron": math.radians(25.0), "elevator": math.radians(20.0), "rudder": math.radians(25.0)}
# Which simulator surface command drives each servo (both ailerons take the aileron command).
SERVO_SURFACE = {"aileron_l": "aileron", "aileron_r": "aileron", "elevator": "elevator", "rudder": "rudder"}


def deflection_to_us(cal: dict, aileron: float, elevator: float, rudder: float) -> list[int]:
    """Surface deflections (rad, simulator sign convention) -> servo pulses, exactly as
    the flight firmware does it (inner_loop.c rlg_deflection_to_us): neutral +
    direction * clip(defl / max, -1, 1) * us_full_deflection, clamped to the travel
    limits, rounded. Order = SERVO_NAMES. Checked against the C code in tests."""
    defl = {"aileron": aileron, "elevator": elevator, "rudder": rudder}
    out = []
    for name in SERVO_NAMES:
        sv = cal["servos"][name]
        surf = SERVO_SURFACE[name]
        frac = max(-1.0, min(1.0, defl[surf] / SURFACE_MAX_RAD[surf]))
        us = sv["neutral_us"] + sv["direction"] * frac * sv["us_full_deflection"]
        us = max(sv["min_us"], min(sv["max_us"], us))
        out.append(int(math.floor(us + 0.5)) if us >= 0 else int(math.ceil(us - 0.5)))
    return out


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

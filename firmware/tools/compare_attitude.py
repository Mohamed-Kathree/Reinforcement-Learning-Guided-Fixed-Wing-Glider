"""
compare_attitude.py -- Phase 1 check: firmware attitude vs the Python visualiser's filter.

The flight firmware's USB stream (console 'stream', or a bare '8' as
glider_viz.py sends) prints, per sample pair:
    D,t_ms,ax,ay,az,gx,gy,gz      raw body-frame counts (the visualiser's format)
    E,t_ms,roll_deg,pitch_deg,gate   the firmware's own Kalman output

This script runs the Python reference (attitude_ref.py, the glider_viz.py
filter with the corrected frame) on the D lines and compares it with the E
lines. Records a MEASUREMENT, not pass/fail: RMS and max difference per axis.

Type 'kin' on the console first so the firmware integrates body rates like the
visualiser (euler=False); otherwise pass --euler.

The stream is decimated (every 2nd sample, ~112 Hz), so the Python filter sees
twice the dt the firmware did: expect agreement to a fraction of a degree, not
bit-exactness. Bit-exactness of the C port is tests/test_flight_firmware.py.

    python firmware/tools/compare_attitude.py --port COM3 --seconds 30 --log run.csv
    python firmware/tools/compare_attitude.py --file run.csv
Confirm the COM port first (it changes between USB ports on this PC). USB
with the battery OFF.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_ref import ACC_LSB, GYRO_LSB, Attitude  # noqa: E402


def lines_from_port(port: str, seconds: float, log_path: str | None):
    import serial  # pyserial, only needed for live capture

    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)                       # the ESP32 resets when the port opens
    ser.write(b"8")
    log = open(log_path, "w") if log_path else None
    end = time.time() + seconds
    try:
        while time.time() < end:
            s = ser.readline().decode(errors="ignore").strip()
            if log and s[:2] in ("D,", "E,"):
                log.write(s + "\n")
            yield s
    finally:
        ser.write(b"stream\n")          # toggles it off again
        ser.close()
        if log:
            log.close()


def compare(lines, euler: bool = False, settle_s: float = 3.0) -> dict:
    att = Attitude(euler=euler)
    t_prev = None
    t0 = None
    py_at = {}
    diffs_r, diffs_p = [], []
    for s in lines:
        p = s.split(",")
        try:
            if p[0] == "D" and len(p) == 8:
                t = int(p[1])
                ax, ay, az = (int(v) / ACC_LSB for v in p[2:5])
                gx, gy, gz = (int(v) / GYRO_LSB for v in p[5:8])
                dt = 0.0 if t_prev is None else (t - t_prev) / 1000.0
                t_prev = t
                t0 = t if t0 is None else t0
                if att.update(ax, ay, az, gx, gy, gz, dt) or not py_at:
                    py_at[t] = (att.roll.angle, att.pitch.angle)
            elif p[0] == "E" and len(p) == 5:
                t = int(p[1])
                if t in py_at and t0 is not None and (t - t0) / 1000.0 >= settle_s:
                    pr, pp = py_at.pop(t)
                    dr = (float(p[2]) - pr + 180.0) % 360.0 - 180.0
                    diffs_r.append(dr)
                    diffs_p.append(float(p[3]) - pp)
        except ValueError:
            continue

    def stats(d):
        if not d:
            return {"n": 0}
        return {"n": len(d), "rms_deg": math.sqrt(sum(x * x for x in d) / len(d)),
                "max_abs_deg": max(abs(x) for x in d), "mean_deg": sum(d) / len(d)}

    return {"roll": stats(diffs_r), "pitch": stats(diffs_p)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--file")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--log")
    ap.add_argument("--euler", action="store_true", help="firmware left in Euler kinematics (default)")
    a = ap.parse_args(argv)
    if not (a.port or a.file):
        ap.error("--port or --file")
    lines = open(a.file).read().splitlines() if a.file else lines_from_port(a.port, a.seconds, a.log)
    r = compare(lines, euler=a.euler)
    for axis in ("roll", "pitch"):
        s = r[axis]
        if not s["n"]:
            print(f"{axis}: no matched samples")
        else:
            print(f"{axis}: n={s['n']} rms={s['rms_deg']:.3f} deg max={s['max_abs_deg']:.3f} deg "
                  f"mean={s['mean_deg']:+.3f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())

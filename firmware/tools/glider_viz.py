"""
glider_viz.py -- live 3-D attitude view, for the corrected IMU frame.

A copy of Hardware/Code/glider_viz.py brought into the repo. Changes:
  - the filter is attitude_ref.Attitude: glider_viz.py's Kalman, with the
    frame corrections described in attitude_ref.py (all signs +1; level reads
    az = -1 g). The original's negations matched the bench sketch's old,
    wrong sign defines and must not be used with the flight firmware.
  - the firmware's own roll/pitch (E lines from the flight firmware's stream)
    are shown next to the Python estimate, with the difference.
  - --port argument (confirm the COM port first: it changes between USB ports
    on this PC). USB with the battery OFF.

Works with the flight firmware (streams on a bare '8') and with the bench
sketch's '8' stream (D lines only).

    pip install vpython pyserial
    python firmware/tools/glider_viz.py --port COM3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_ref import ACC_LSB, GYRO_LSB, Attitude  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--euler", action="store_true", help="Python filter uses Euler kinematics")
    a = ap.parse_args(argv)

    import serial
    from vpython import arrow, box, canvas, color, compound, distant_light, label, rate, vector

    ser = serial.Serial(a.port, 115200, timeout=1)
    time.sleep(2)                     # ESP32 reboots when the port opens
    ser.write(b"8")
    time.sleep(0.5)
    ser.reset_input_buffer()
    print("stream started; hold the board still while the filter settles")

    canvas(title="ICM-20948 - Kalman attitude (body frame, corrected)", width=900, height=620,
           background=color.gray(0.12), forward=vector(-1, -0.35, -0.6))
    fuse = box(length=2.4, height=0.22, width=0.22, color=color.orange)
    wing = box(length=0.42, height=0.06, width=2.6, color=color.cyan)
    tail = box(length=0.4, height=0.06, width=1.0, pos=vector(-1.0, 0, 0), color=color.cyan)
    fin = box(length=0.4, height=0.5, width=0.06, pos=vector(-1.0, 0.25, 0), color=color.cyan)
    nose = box(length=0.3, height=0.18, width=0.18, pos=vector(1.3, 0, 0), color=color.red)
    glider = compound([fuse, wing, tail, fin, nose])
    arrow(pos=vector(0, 0, 0), axis=vector(2.2, 0, 0), shaftwidth=0.02, color=color.gray(0.4))
    arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 2.2), shaftwidth=0.02, color=color.gray(0.4))
    arrow(pos=vector(0, 0, 0), axis=vector(0, 1.6, 0), shaftwidth=0.02, color=color.gray(0.4))
    distant_light(direction=vector(0.3, 1, 0.5), color=color.gray(0.4))
    readout = label(pos=vector(0, -2.3, 0), height=15, box=False, color=color.white, text="initialising...")

    att = Attitude(euler=a.euler)
    t_prev = None
    fw = None                          # latest firmware (roll, pitch) from E lines
    dt = 0.0
    try:
        while True:
            rate(250)
            while ser.in_waiting > 400:          # drop backlog: never integrate stale samples
                ser.readline()
            p = ser.readline().decode(errors="ignore").strip().split(",")
            try:
                if p[0] == "E" and len(p) == 5:
                    fw = (float(p[2]), float(p[3]))
                    continue
                if p[0] != "D" or len(p) != 8:
                    continue
                t_ms = int(p[1])
                ax, ay, az = (int(v) / ACC_LSB for v in p[2:5])
                gx, gy, gz = (int(v) / GYRO_LSB for v in p[5:8])
            except ValueError:
                continue
            dt = 0.0 if t_prev is None else (t_ms - t_prev) / 1000.0   # ESP32 clock, not the host's
            t_prev = t_ms
            if not att.update(ax, ay, az, gx, gy, gz, dt):
                continue

            roll, pitch, yaw = att.roll.angle, att.pitch.angle, att.yaw
            r, pch, y = map(math.radians, (roll, pitch, yaw))
            cr, sr = math.cos(r), math.sin(r)
            cp, sp = math.cos(pch), math.sin(pch)
            cy, sy = math.cos(y), math.sin(y)
            fwd = (cp * cy, cp * sy, -sp)
            down = (cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp)
            glider.axis = vector(fwd[0], -fwd[2], fwd[1]) * 2.4
            glider.up = vector(-down[0], down[2], -down[1])

            fw_txt = "firmware: (no E lines)"
            if fw:
                fw_txt = (f"firmware roll {fw[0]:+7.2f}  pitch {fw[1]:+7.2f}   "
                          f"diff {fw[0] - roll:+5.2f} / {fw[1] - pitch:+5.2f}")
            readout.text = (
                f"roll {roll:+7.2f}   pitch {pitch:+7.2f}   yaw {yaw:+7.1f} (drifts)\n"
                f"{fw_txt}\n"
                f"bias est {att.roll.bias:+6.2f} / {att.pitch.bias:+6.2f} deg/s"
                f"    |a| {att.g_mag:4.2f} g   accel {'ON' if att.trusted else 'REJECTED'}"
                f"    dt {dt * 1000:4.1f} ms"
            )
    except KeyboardInterrupt:
        print(f"\nfinal bias estimates: roll {att.roll.bias:+.3f}, pitch {att.pitch.bias:+.3f} deg/s")
    finally:
        try:
            ser.write(b"stream\n")
            ser.close()
        except Exception:
            pass
        print("port closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

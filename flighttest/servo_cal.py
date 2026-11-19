"""
servo_cal.py -- hold one servo at a chosen pulse width, for the joint
calibration session. Talks to the flight computer through the ground station
(ESP-NOW bridge on USB) using real BENCH command frames, so every rlg_safety
rule applies: 5 frames to arm, 150 ms link timeout, refusal while airborne,
per-servo limits.

    python -m flighttest.servo_cal --port COM3 --servo elevator --us 1600 --hold 15

Arms BENCH, sends the frame at 20 Hz for --hold seconds (the other three servos
stay at their calibrated neutral), prints the flight computer's status lines
(the "us=" field is what it is actually outputting), then disarms: all servos
return to neutral. Ctrl-C disarms immediately.

Move in small steps near the ends of travel: a servo driven into a mechanical
stop stalls and draws current, and the power budget has no proven headroom.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flighttest import calibration as C  # noqa: E402
from flighttest.protocol import packets as P  # noqa: E402

MODE_DISARMED, MODE_BENCH = 0, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--servo", required=True, help="aileron_l | aileron_r | elevator | rudder | 0-3")
    ap.add_argument("--us", type=int, required=True)
    ap.add_argument("--hold", type=float, default=15.0)
    a = ap.parse_args(argv)

    import serial

    names = list(C.SERVO_NAMES)
    idx = int(a.servo) if a.servo.isdigit() else names.index(a.servo)
    cal = C.load()
    neutral = C.neutral_us(cal)
    if not 600 <= a.us <= 2300:
        print("refusing: outside the absolute envelope 600-2300 us")
        return 2
    us = list(neutral)
    us[idx] = a.us

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, 115200, 0.1
    ser.dtr = ser.rts = False                 # do not reset the ground station
    ser.open()
    stop = threading.Event()

    def reader():
        last = 0.0
        buf = b""
        while not stop.is_set():
            buf += ser.read(512)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                s = ln.decode("ascii", "ignore").strip()   # radio text has garbled bytes; never crash on them
                if "us=" in s and time.time() - last > 0.9:
                    last = time.time()
                    mode = "BENCH" if "BENCH" in s else ("DISARMED" if "DISARMED" in s else "?")
                    print(f"  flight: {mode:8s} {s[s.find('us='):]}")

    threading.Thread(target=reader, daemon=True).start()
    seq = 1

    def send(mode, vals):
        nonlocal seq
        ser.write(P.Command(seq=seq, mode=mode, servo_us=vals).encode())
        seq = (seq + 1) & 0xFFFF

    print(f"{names[idx]} -> {a.us} us (neutral {neutral[idx]}), holding {a.hold:.0f} s")
    try:
        t_end = time.time() + a.hold
        while time.time() < t_end:
            send(MODE_BENCH, us)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        for _ in range(4):
            send(MODE_DISARMED, list(neutral))
            time.sleep(0.05)
        time.sleep(1.2)
        stop.set()
        ser.close()
        print("disarmed: all servos back to neutral")
    return 0


if __name__ == "__main__":
    sys.exit(main())

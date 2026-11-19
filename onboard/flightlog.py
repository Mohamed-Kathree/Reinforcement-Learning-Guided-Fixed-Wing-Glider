"""
flightlog.py -- writes .rlglog files (format defined in flighttest/logs.py,
version 2) on the Pi's SD card, one file per FLIGHT arming.

Every PiTelemetry frame (50 Hz) becomes a record with the fields the dashboard
replays, plus the Pi's own decisions whenever the controller acted:
  "phase", "vd", "obs" (the 12 values fed to the controller), "action",
  "controller" (1 policy, 2 baseline), "handed_over", "wind_est".
NaN is not valid JSON: optional fields are written as null, and records
start once the barometric reference exists (captured at arming).

Lines are flushed as they are written so a power cut loses at most the last
partial line; the file is fsync'd every second.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import time

FORMAT, VERSION = "rlglog", 2


def _num(x, nd=5):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


class FlightLog:
    def __init__(self, log_dir: str, header: dict) -> None:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(log_dir, f"flight_{stamp}.rlglog")
        self.f = open(self.path, "w", encoding="utf-8", newline="\n", buffering=1)
        self.f.write(json.dumps({"format": FORMAT, "version": VERSION,
                                 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                 **header}) + "\n")
        self.t0_ms = None
        self.n = 0
        self._last_sync = time.monotonic()

    def record(self, t, phase: int, extra: dict | None = None) -> None:
        if math.isnan(t.baro_alt):
            return
        if self.t0_ms is None:
            self.t0_ms = t.t_ms
        r = {
            "t": round(((t.t_ms - self.t0_ms) & 0xFFFFFFFF) / 1000.0, 3),
            "roll": _num(t.roll), "pitch": _num(t.pitch), "yaw": _num(t.yaw),
            "p": _num(t.p), "q": _num(t.q), "r": _num(t.r),
            "alt_baro": round(t.baro_alt, 3), "sonar": _num(t.sonar, 3),
            "lat": t.lat * 1e-7, "lon": t.lon * 1e-7,
            "vn": _num(t.vn, 3) or 0.0, "ve": _num(t.ve, 3) or 0.0, "vd": _num(t.vd, 3),
            "rail_adc_v": _num(t.rail_adc_v, 3),
            "status": int(t.status), "servo_us": [int(x) for x in t.servo_us],
            "phase": phase, "esp_seq": t.seq,
        }
        if extra:
            r.update(extra)
        self.f.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.n += 1
        if time.monotonic() - self._last_sync > 1.0:
            self.f.flush()
            os.fsync(self.f.fileno())
            self._last_sync = time.monotonic()

    def close(self) -> None:
        if not self.f.closed:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()

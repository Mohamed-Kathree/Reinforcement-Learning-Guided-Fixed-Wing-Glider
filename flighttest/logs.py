"""
logs.py
=======
The full-rate flight log the Pi writes to its SD card, and its parser.

Format (.rlglog, JSON Lines, UTF-8). DEFINED HERE; the Pi logger does not
write it yet (hardware seam: the Pi-side logger must produce exactly this).

  line 1  header  {"format": "rlglog", "version": 1, "created_utc": ..., "source": ...,
                   "home": {"lat": ..., "lon": ...} | null, "calibration": {...}, "note": ...}
  line 2+ record  {"t": s since log start, "roll","pitch","yaw" (rad), "p","q","r" (rad/s),
                   "alt_baro" (m above launch reference), "sonar" (m, -1 invalid),
                   "lat","lon" (deg), "vn","ve" (m/s), "rail_v" (V, -1 not measured),
                   "status" (u32, protocol status bits), "servo_us": [4],
                   "action": [bank_cmd, speed_cmd] (optional, policy output)}

The record carries the same fields as the Telemetry packet, at full rate, so
the stream and the log can be compared directly. parse() turns a log into
replay frames in the same NED convention as the simulator recordings
(x north, y east, z down; home at the origin), so one viewer shows both.
"""

from __future__ import annotations

import datetime
import json
import math
import random

from flighttest.protocol import packets as P

FORMAT = "rlglog"
VERSION = 1
REQUIRED = ("t", "roll", "pitch", "yaw", "alt_baro", "lat", "lon", "status", "servo_us")
EARTH_R = 6_371_000.0


def latlon_to_ne(lat: float, lon: float, home_lat: float, home_lon: float) -> tuple[float, float]:
    """Equirectangular, adequate over a few hundred metres."""
    n = math.radians(lat - home_lat) * EARTH_R
    e = math.radians(lon - home_lon) * EARTH_R * math.cos(math.radians(home_lat))
    return n, e


def ne_to_latlon(n: float, e: float, home_lat: float, home_lon: float) -> tuple[float, float]:
    lat = home_lat + math.degrees(n / EARTH_R)
    lon = home_lon + math.degrees(e / (EARTH_R * math.cos(math.radians(home_lat))))
    return lat, lon


def parse(text: str) -> dict:
    """Parse a log. Bad lines are skipped and reported, never silently dropped."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty log")
    header = json.loads(lines[0])
    if header.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} file")
    if header.get("version") != VERSION:
        raise ValueError(f"log version {header.get('version')} not supported (expected {VERSION})")

    records, skipped = [], []
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                raise ValueError(f"missing {missing}")
            if len(r["servo_us"]) != len(P.SERVOS):
                raise ValueError("servo_us length")
            records.append(r)
        except (ValueError, TypeError) as exc:
            skipped.append({"line": i, "reason": str(exc)[:120]})

    home = header.get("home")
    if home is None:   # first record with a 3D fix
        for r in records:
            if P.unpack_status(int(r["status"]))["gps_fix"] == 2:
                home = {"lat": r["lat"], "lon": r["lon"], "derived": "first 3D fix"}
                break

    frames = []
    for r in records:
        st = P.unpack_status(int(r["status"]))
        if home is not None and st["gps_fix"] > 0:
            n, e = latlon_to_ne(r["lat"], r["lon"], home["lat"], home["lon"])
        else:
            n = e = None
        sonar = r.get("sonar", -1.0)
        rail = r.get("rail_v", -1.0)
        frames.append({
            "t": r["t"],
            "pos_ned": None if n is None else [n, e, -r["alt_baro"]],
            "euler": [r["roll"], r["pitch"], r["yaw"]],
            "rates": [r.get("p", 0.0), r.get("q", 0.0), r.get("r", 0.0)],
            "alt_baro": r["alt_baro"],
            "sonar": sonar if st["sonar_valid"] and sonar >= 0 else None,
            "rail_v": rail if st["rail_v_valid"] and rail >= 0 else None,
            "vel_ne": [r.get("vn", 0.0), r.get("ve", 0.0)],
            "servo_us": r["servo_us"],
            "action": r.get("action"),
            "mode": P.MODE_NAMES.get(st["mode"], str(st["mode"])),
            "armed": bool(st["armed"]),
            "gps_fix": st["gps_fix"], "gps_sats": st["gps_sats"],
        })
    times = [f["t"] for f in frames]
    return {
        "header": header,
        "home": home,
        "frames": frames,
        "skipped": skipped,
        "summary": {
            "n_frames": len(frames),
            "duration_s": (times[-1] - times[0]) if len(times) > 1 else 0.0,
            "rate_hz": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None,
            "frames_without_position": sum(f["pos_ned"] is None for f in frames),
            "max_alt_m": max((f["alt_baro"] for f in frames), default=None),
            "min_rail_v": min((f["rail_v"] for f in frames if f["rail_v"] is not None), default=None),
            "skipped_lines": len(skipped),
        },
    }


def write(path: str, header: dict, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"format": FORMAT, "version": VERSION, **header}) + "\n")
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def synthesize_from_trajectory(traj: dict, home: tuple[float, float], seed: int = 0,
                               calibration: dict | None = None) -> tuple[dict, list[dict]]:
    """STUB: a log in the Pi's format, made from a simulator recording, so the
    real-flight replay can be exercised before any real log exists. Labelled as
    synthesized in the header. Servo values are a nominal mapping of surface
    deflection (1000 us per rad about 1500), not a calibration."""
    rng = random.Random(seed)
    frames = traj["frames"]
    records = []
    for f in frames:
        n, e, d = f["pos_ned"]
        lat, lon = ne_to_latlon(n, e, *home)
        agl = f.get("agl", -d)
        ctrl = f.get("ctrl", {"ail": 0, "elev": 0, "rud": 0})
        servo = [int(1500 + 1000 * ctrl["ail"]), int(1500 - 1000 * ctrl["ail"]),
                 int(1500 + 1000 * ctrl["elev"]), int(1500 + 1000 * ctrl["rud"])]
        servo = [max(s["min_us"], min(s["max_us"], v)) for v, s in zip(servo, P.SERVOS)]
        status = P.pack_status(mode=P.FLIGHT, armed=1, airborne=int(agl > 3), sonar_valid=int(agl < 4.5),
                               gps_fix=2, gps_sats=9, baro_ref_captured=1, baro_warm=1)
        records.append({
            "t": round(f["t"], 3), "roll": f["euler"][0], "pitch": f["euler"][1], "yaw": f["euler"][2],
            "p": 0.0, "q": 0.0, "r": 0.0,
            "alt_baro": round(-d + rng.gauss(0, 0.05), 3),
            "sonar": round(agl, 3) if agl < 4.5 else -1.0,
            "lat": lat, "lon": lon, "vn": 0.0, "ve": 0.0, "rail_v": -1.0,
            "status": status, "servo_us": servo,
        })
    header = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": f"STUB: synthesized from simulator recording {traj['episode_id']}",
        "home": {"lat": home[0], "lon": home[1]},
        "calibration": calibration,
        "note": "Not a real flight. Generated so the replay can be tested without hardware.",
    }
    return header, records

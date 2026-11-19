"""
ground.py
=========
GroundStation: everything the ground app does with the ESP-NOW link.

  - reads bytes from the link, parses telemetry (CRC-checked, silent drops);
  - link statistics: packet rate (Hz), loss % from sequence gaps, CRC drops,
    last-packet age, RSSI when the link can report it;
  - a telemetry history for the live strip charts;
  - the command sender for BENCH / REPLAY sessions:
      * a session starts only on an explicit arm() call (UI action);
      * set_servos() may be called at any rate; commands go out on a fixed
        20 Hz schedule, so scrubbing a replay can never flood the link;
      * the session expires after UI_IDLE_EXPIRY_MS without a UI action
        (set_servos / keepalive); the firmware has its own longer expiry;
      * ending a session sends DISARMED several times (lossy link);
      * the glider's own report (status bits) is shown separately from what
        the ground requested: the firmware decides, the UI only asks.

Real hardware (19 Nov 2026, schema v3):
  - the bridge carries the flight computer's console text as well as binary
    frames; text is ASCII (never 0xA5), so the stream is split into frames
    and text lines (shown in the dashboard's console, never counted as drops);
  - Info packets (1 Hz) report the calibration the aircraft is using;
  - push_calibration() sends the calibration file as a CalibrationSet; the
    flight computer applies it only DISARMED and on the ground, stores it in
    NVS, and the push counts as done only when Info reports the same cal_id
    loaded from NVS (resent up to 3 times);
  - console() sends whitelisted console commands (FLIGHT is armed with the
    flight computer's own "arm" command; the ground app's arm() stays
    BENCH/REPLAY only).

Deterministic: step(now_ms) does all the work, so tests drive it with a fake
clock. backend/routers/flighttest.py calls step() from one background thread.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque

from flighttest import calibration as C
from flighttest.links import Link
from flighttest.protocol import packets as P

COMMAND_PERIOD_MS = 50            # 20 Hz maximum
UI_IDLE_EXPIRY_MS = 60_000        # UI arming expires after a minute without a UI action
STATS_WINDOW_MS = 5_000
HISTORY_LEN = 300                 # 60 s at 5 Hz
DISARM_BURST = 4

RAIL_HEADROOM_V = 0.15            # "low" warning band above brownout
CONSOLE_LEN = 300
CAL_RESEND_MS = 1500
CAL_MAX_SENDS = 3
CAL_GIVE_UP_MS = 6000
# Console commands the dashboard may send (firmware/flight/flight.ino runCommand).
CONSOLE_WHITELIST = {"arm", "disarm", "report", "help", "tlm", "uncal", "launch", "calclear", "kin", "srv on", "srv off"}


def calibration_id(cal: dict) -> int:
    """First 8 hex digits of the SHA-256 of the file as calibration.save() writes it
    (the same bytes flighttest/firmware_sync.py hashes for CAL_HASH)."""
    text = json.dumps(cal, indent=1) + "\n"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def calibration_packet(cal: dict, seq: int) -> bytes:
    names = C.SERVO_NAMES
    sv = [cal["servos"][n] for n in names]
    dir_neg = sum(1 << i for i, s in enumerate(sv) if int(s["direction"]) < 0)
    flags = (1 if all(s["measured"] for s in sv) else 0) | (2 if cal["imu"]["verified"] else 0)
    return P.CalibrationSet(seq=seq & 0xFFFF, cal_id=calibration_id(cal), flags=flags, dir_neg=dir_neg,
                            neutral_us=[int(s["neutral_us"]) for s in sv],
                            full_us=[int(s["us_full_deflection"]) for s in sv],
                            min_us=[int(s["min_us"]) for s in sv], max_us=[int(s["max_us"]) for s in sv]).encode()


class GroundStation:
    def __init__(self, link: Link | None = None, cal: dict | None = None) -> None:
        """cal: the calibration file (flighttest/calibration.py). Servo limits,
        neutrals and rail scaling all come from it; nothing is hard-coded."""
        self.link = link
        self.set_calibration(cal if cal is not None else C.load())
        self.parser = P.StreamParser()
        self.reset_stats()
        self.session: dict | None = None
        self.last_end: dict | None = None
        self.cmd_seq = 0
        self.next_cmd_ms = 0
        self.commands_sent = 0
        self.history: deque = deque(maxlen=HISTORY_LEN)
        self.latest: dict | None = None
        self.info: dict | None = None
        self.console_lines: deque = deque(maxlen=CONSOLE_LEN)
        self.console_seq = 0
        self.cal_push: dict | None = None
        self._rx = bytearray()
        self._text = bytearray()
        self.crc_drops = 0
        self._pending_disarm = 0

    def set_calibration(self, cal: dict) -> None:
        problems = C.validate(cal)
        if problems:
            raise ValueError("invalid calibration: " + "; ".join(problems))
        self.cal = cal
        self.neutral_us = C.neutral_us(cal)
        self.min_us, self.max_us = C.limits(cal)
        if getattr(self, "session", None):
            self.session["servo_us"] = self._clamp(self.session["servo_us"])

    def _clamp(self, us: list[int]) -> list[int]:
        return [max(lo, min(hi, int(v))) for v, lo, hi in zip(us, self.min_us, self.max_us)]

    # ------------------------------------------------------------------ link
    def connect(self, link: Link) -> None:
        self.disconnect()
        self.link = link
        self.parser = P.StreamParser()
        self._rx.clear()
        self._text.clear()
        self.crc_drops = 0
        self.info = None
        self.cal_push = None
        self.reset_stats()

    def disconnect(self) -> None:
        if self.link is not None:
            self.link.close()
        self.link = None
        self.session = None

    def reset_stats(self) -> None:
        self.events: deque = deque()          # (t_ms, received, lost)
        self.last_seq: int | None = None
        self.last_packet_ms: int | None = None
        self.received_total = 0
        self.lost_total = 0

    # ------------------------------------------------------------------ commands
    def arm(self, mode: int, now_ms: int) -> dict:
        if self.link is None:
            raise RuntimeError("no link: connect the ground station first")
        if mode not in (P.BENCH, P.REPLAY):
            raise ValueError("the ground app can only arm BENCH or REPLAY")
        self.session = {"mode": mode, "started_ms": now_ms, "last_ui_ms": now_ms,
                        "servo_us": list(self.neutral_us)}
        self.next_cmd_ms = now_ms
        self._pending_disarm = 0
        return self.session_info(now_ms)

    def set_servos(self, us: list[int], now_ms: int) -> list[int]:
        if self.session is None:
            raise RuntimeError("not armed: arm a BENCH or REPLAY session first")
        if len(us) != len(P.SERVOS):
            raise ValueError(f"need {len(P.SERVOS)} servo values")
        self.session["servo_us"] = self._clamp(us)
        self.session["last_ui_ms"] = now_ms
        return self.session["servo_us"]

    def keepalive(self, now_ms: int) -> None:
        if self.session is not None:
            self.session["last_ui_ms"] = now_ms

    def disarm(self, now_ms: int, reason: str = "requested") -> None:
        if self.session is not None:
            self.last_end = {"reason": reason, "at_ms": now_ms}
        self.session = None
        self._pending_disarm = DISARM_BURST
        self.next_cmd_ms = now_ms

    def _send(self, mode: int, servo_us: list[int]) -> None:
        self.cmd_seq = (self.cmd_seq + 1) & 0xFFFF
        self.link.write(P.Command(seq=self.cmd_seq, mode=mode, servo_us=servo_us).encode())
        self.commands_sent += 1

    # ------------------------------------------------------------------ main step
    def step(self, now_ms: int) -> None:
        if self.link is None:
            return
        self.link.step(now_ms)
        for pkt in self._demux(self.link.read(), now_ms):
            if isinstance(pkt, P.Telemetry):
                self._on_telemetry(pkt, now_ms)
            elif isinstance(pkt, P.Info):
                self._on_info(pkt, now_ms)
        self._cal_push_step(now_ms)
        while self.events and now_ms - self.events[0][0] > STATS_WINDOW_MS:
            self.events.popleft()

        if self.session is not None and now_ms - self.session["last_ui_ms"] > UI_IDLE_EXPIRY_MS:
            self.disarm(now_ms, reason="expired: no UI action for 60 s")
        if now_ms >= self.next_cmd_ms:
            if self.session is not None:
                self._send(self.session["mode"], self.session["servo_us"])
            elif self._pending_disarm > 0:
                self._send(P.DISARMED, self.neutral_us)
                self._pending_disarm -= 1
            # Steady 20 Hz cadence; after a stall, restart from now rather than
            # sending a catch-up burst (never more than one command per period).
            self.next_cmd_ms += COMMAND_PERIOD_MS
            if self.next_cmd_ms <= now_ms:
                self.next_cmd_ms = now_ms + COMMAND_PERIOD_MS

    # ------------------------------------------------------------------ stream split
    def _demux(self, data: bytes, now_ms: int) -> list:
        """Binary frames (from SYNC, fixed length per type, CRC-checked) and
        console text (everything else, ASCII) from one byte stream."""
        self._rx += data
        out = []
        while self._rx:
            if self._rx[0] != P.SYNC:
                i = self._rx.find(bytes([P.SYNC]))
                n = len(self._rx) if i < 0 else i
                self._take_text(bytes(self._rx[:n]), now_ms)
                del self._rx[:n]
                continue
            if len(self._rx) < 2:
                break
            cls = P.PACKET_TYPES.get(self._rx[1])
            if cls is None:
                self.crc_drops += 1
                del self._rx[:1]
                continue
            if len(self._rx) < cls.SIZE:
                break
            try:
                out.append(cls.decode(bytes(self._rx[:cls.SIZE])))
                del self._rx[:cls.SIZE]
            except ValueError:
                self.crc_drops += 1
                del self._rx[:1]
        return out

    def _take_text(self, chunk: bytes, now_ms: int) -> None:
        self._text += chunk
        while True:
            i = self._text.find(b"\n")
            if i < 0:
                break
            # Console text is printable ASCII. The bridge's 0xFF wake-up bytes, and any
            # byte mangled on the ground board's USB-serial path, are dropped here.
            line = bytes(b for b in self._text[:i] if 32 <= b < 127 or b == 9).decode("ascii")
            del self._text[:i + 1]
            if line.strip():
                self.console_seq += 1
                self.console_lines.append({"n": self.console_seq, "t_ms": now_ms, "text": line[:400]})
        if len(self._text) > 2000:                       # no newline for too long: discard
            self._text.clear()

    def console(self, line: str, now_ms: int) -> str:
        if self.link is None:
            raise RuntimeError("no link: connect the ground station first")
        cmd = " ".join(line.strip().lower().split())
        if cmd not in CONSOLE_WHITELIST:
            raise ValueError(f"not an allowed console command: {cmd!r} (allowed: {sorted(CONSOLE_WHITELIST)})")
        self.link.write((cmd + "\n").encode("ascii"))
        self.console_seq += 1
        self.console_lines.append({"n": self.console_seq, "t_ms": now_ms, "text": f"> {cmd}", "sent": True})
        return cmd

    # ------------------------------------------------------------------ calibration push
    def push_calibration(self, now_ms: int) -> dict:
        if self.link is None:
            raise RuntimeError("no link: connect the ground station first")
        st = (self.latest or {}).get("status") or {}
        if st.get("armed") or st.get("airborne"):
            raise RuntimeError("the aircraft must be DISARMED and on the ground to accept a calibration")
        self.cal_push = {"cal_id": calibration_id(self.cal), "sends": 0, "first_ms": now_ms, "last_ms": None,
                         "state": "sending"}
        self._send_cal(now_ms)
        return self.cal_push_info()

    def _send_cal(self, now_ms: int) -> None:
        self.cmd_seq = (self.cmd_seq + 1) & 0xFFFF
        self.link.write(calibration_packet(self.cal, self.cmd_seq))
        self.cal_push["sends"] += 1
        self.cal_push["last_ms"] = now_ms

    def _cal_push_step(self, now_ms: int) -> None:
        cp = self.cal_push
        if cp is None or cp["state"] != "sending":
            return
        if self.info and self.info["cal_id"] == cp["cal_id"] and self.info["cal_from_nvs"]:
            cp["state"] = "confirmed"
            cp["confirmed_ms"] = now_ms
        elif cp["sends"] < CAL_MAX_SENDS and now_ms - cp["last_ms"] >= CAL_RESEND_MS:
            self._send_cal(now_ms)
        elif now_ms - cp["first_ms"] > CAL_GIVE_UP_MS:
            cp["state"] = "not confirmed"

    def cal_push_info(self) -> dict | None:
        if self.cal_push is None:
            return None
        return {**self.cal_push, "cal_id_hex": f"{self.cal_push['cal_id']:08X}"}

    def _on_info(self, i, now_ms: int) -> None:
        self.info = {"t_ms": now_ms, "cal_id": i.cal_id, "cal_id_hex": f"{i.cal_id:08X}",
                     "cal_servos_measured": bool(i.cal_flags & 1), "cal_imu_verified": bool(i.cal_flags & 2),
                     "cal_from_nvs": bool(i.cal_flags & 4), "schema": i.schema, "loop_hz": i.loop_hz,
                     "pi_rx_gaps": i.pi_rx_gaps, "pi_rx_bad": i.pi_rx_bad, "pi_tlm_gaps": i.pi_tlm_gaps,
                     "gps_rate_hz": i.gps_rate_dhz / 10.0, "launch_cause": i.launch_cause & 0x0F,
                     "apex_cause": i.launch_cause >> 4, "pitch_sp": i.pitch_sp, "roll_sp": i.roll_sp,
                     "file_cal_id_hex": f"{calibration_id(self.cal):08X}"}

    def _on_telemetry(self, t: P.Telemetry, now_ms: int) -> None:
        lost = 0
        if self.last_seq is not None:
            gap = (t.seq - self.last_seq) & 0xFFFF
            if 0 < gap < 1000:
                lost = gap - 1
            # gap == 0 (duplicate) or a huge jump (glider rebooted): resync, count nothing
        self.last_seq = t.seq
        self.last_packet_ms = now_ms
        self.received_total += 1
        self.lost_total += lost
        self.events.append((now_ms, 1, lost))
        st = P.unpack_status(t.status)
        self.latest = {
            "seq": t.seq, "t_ms": now_ms,
            "roll": t.roll, "pitch": t.pitch, "yaw": t.yaw, "p": t.p, "q": t.q, "r": t.r,
            "alt_baro": t.alt_baro, "sonar": t.sonar if st["sonar_valid"] and t.sonar >= 0 else None,
            "lat": t.lat, "lon": t.lon, "vn": t.vn, "ve": t.ve,
            "rail_adc_v": t.rail_adc_v if st["rail_v_valid"] and t.rail_adc_v >= 0 else None,
            "rail_v": C.rail_volts(self.cal, t.rail_adc_v) if st["rail_v_valid"] else None,
            "status": st, "mode": P.MODE_NAMES.get(st["mode"], str(st["mode"])),
            "phase": P.PHASE_NAMES.get(st["phase"], str(st["phase"])),
        }
        self.latest["rail"] = self.rail_state(self.latest["rail_v"])
        self.history.append({k: self.latest[k] for k in
                             ("t_ms", "roll", "pitch", "r", "alt_baro", "sonar", "rail_v")}
                            | {"gps_sats": st["gps_sats"], "gps_fix": st["gps_fix"], "loss_pct": self.loss_pct()})

    # ------------------------------------------------------------------ reporting
    def rail_state(self, rail_v: float | None) -> dict:
        """Rail voltage with its trustworthiness: brownout below 4.75 V corrupts the Pi's SD card."""
        r = self.cal["rail"]
        level = None if rail_v is None else (
            "brownout" if rail_v < r["brownout_v"] else
            "low" if rail_v < r["brownout_v"] + RAIL_HEADROOM_V else "ok")
        return {"volts": rail_v, "level": level, "brownout_v": r["brownout_v"],
                "calibrated": bool(r["calibrated"]), "measured": rail_v is not None,
                "note": None if rail_v is None else (None if r["calibrated"]
                                                     else "indicative: divider scale not yet calibrated")}

    def rate_hz(self) -> float:
        n = sum(r for _, r, _ in self.events)
        return n / (STATS_WINDOW_MS / 1000.0)

    def loss_pct(self) -> float | None:
        rec = sum(r for _, r, _ in self.events)
        lost = sum(lo for _, _, lo in self.events)
        return None if rec + lost == 0 else 100.0 * lost / (rec + lost)

    def session_info(self, now_ms: int) -> dict | None:
        if self.session is None:
            return None
        s = self.session
        return {"mode": P.MODE_NAMES[s["mode"]], "servo_us": s["servo_us"],
                "expires_in_ms": max(0, UI_IDLE_EXPIRY_MS - (now_ms - s["last_ui_ms"]))}

    def snapshot(self, now_ms: int) -> dict:
        link = self.link
        age = None if self.last_packet_ms is None else now_ms - self.last_packet_ms
        return {
            "ground": {
                "connected": link is not None,
                "kind": link.kind if link else None,
                "port": link.port if link else None,
                "rate_hz": round(self.rate_hz(), 2),
                "loss_pct": None if self.loss_pct() is None else round(self.loss_pct(), 1),
                "crc_drops": self.crc_drops,
                "rssi_dbm": link.rssi_dbm if link else None,
                "last_packet_age_ms": age,
                "receiving": age is not None and age < 1000,
                "received_total": self.received_total,
                "lost_total": self.lost_total,
                "commands_sent": self.commands_sent,
            },
            "telemetry": self.latest,
            "session": self.session_info(now_ms),
            "last_session_end": self.last_end,
            "info": self.info,
            "calibration_push": self.cal_push_info(),
            "file_cal_id_hex": f"{calibration_id(self.cal):08X}",
            "console": list(self.console_lines)[-40:],
        }

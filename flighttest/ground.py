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

Deterministic: step(now_ms) does all the work, so tests drive it with a fake
clock. backend/routers/flighttest.py calls step() from one background thread.
"""

from __future__ import annotations

from collections import deque

from flighttest.links import Link
from flighttest.protocol import packets as P

COMMAND_PERIOD_MS = 50            # 20 Hz maximum
UI_IDLE_EXPIRY_MS = 60_000        # UI arming expires after a minute without a UI action
STATS_WINDOW_MS = 5_000
HISTORY_LEN = 300                 # 60 s at 5 Hz
DISARM_BURST = 4

LIMITS = [(s["min_us"], s["max_us"]) for s in P.SERVOS]


def _clamp(us: list[int]) -> list[int]:
    return [max(lo, min(hi, int(v))) for v, (lo, hi) in zip(us, LIMITS)]


class GroundStation:
    def __init__(self, link: Link | None = None, neutral_us: list[int] | None = None) -> None:
        self.link = link
        self.neutral_us = list(neutral_us) if neutral_us else [1500] * len(P.SERVOS)
        self.parser = P.StreamParser()
        self.reset_stats()
        self.session: dict | None = None
        self.last_end: dict | None = None
        self.cmd_seq = 0
        self.next_cmd_ms = 0
        self.commands_sent = 0
        self.history: deque = deque(maxlen=HISTORY_LEN)
        self.latest: dict | None = None
        self._pending_disarm = 0

    # ------------------------------------------------------------------ link
    def connect(self, link: Link) -> None:
        self.disconnect()
        self.link = link
        self.parser = P.StreamParser()
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
        self.session["servo_us"] = _clamp(us)
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
        for pkt in self.parser.feed(self.link.read()):
            if isinstance(pkt, P.Telemetry):
                self._on_telemetry(pkt, now_ms)
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
            "rail_v": t.rail_v if st["rail_v_valid"] and t.rail_v >= 0 else None,
            "status": st, "mode": P.MODE_NAMES.get(st["mode"], str(st["mode"])),
        }
        self.history.append({k: self.latest[k] for k in
                             ("t_ms", "roll", "pitch", "r", "alt_baro", "sonar", "rail_v")}
                            | {"gps_sats": st["gps_sats"], "gps_fix": st["gps_fix"], "loss_pct": self.loss_pct()})

    # ------------------------------------------------------------------ reporting
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
                "crc_drops": self.parser.dropped,
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
        }

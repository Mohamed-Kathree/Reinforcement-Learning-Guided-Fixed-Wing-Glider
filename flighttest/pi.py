"""
pi.py
=====
The Pi over WiFi, on the ground only (log retrieval, status).

Pi-side HTTP contract (served by flighttest/pi_server.py on the Pi):
    GET /status            {"hostname", "ip", "disk_free_bytes", "disk_total_bytes", "log_dir", "time_utc"}
    GET /logs              [{"name", "size_bytes", "modified_utc"}], newest first
    GET /logs/<name>       the raw .rlglog file

PiClient talks to a real Pi. HARDWARE SEAM: untested against the Pi; it uses
only urllib, with short timeouts so an absent Pi reports "unreachable" quickly.
StubPi answers the same calls in process, with logs synthesized from simulator
recordings (clearly labelled as such inside each log).
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import socket
import urllib.error
import urllib.parse
import urllib.request

from flighttest import logs

SAFE_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def safe_log_name(name: str) -> str:
    if not name.endswith(".rlglog") or not set(name) <= SAFE_NAME or ".." in name or name.startswith("."):
        raise ValueError(f"not a log name: {name!r}")
    return name


class PiClient:
    kind = "wifi"

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout_s) as r:
            return r.read()

    def status(self) -> dict:
        try:
            st = json.loads(self._get("/status"))
            return {"reachable": True, "kind": self.kind, "address": self.base_url, **st}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"reachable": False, "kind": self.kind, "address": self.base_url, "error": str(exc)[:160]}

    def list_logs(self) -> list[dict]:
        return json.loads(self._get("/logs"))

    def fetch(self, name: str) -> bytes:
        return self._get("/logs/" + urllib.parse.quote(safe_log_name(name)))


class StubPi:
    """STUB Pi: a local directory of synthesized logs behind the same interface."""

    kind = "stub"

    def __init__(self, log_dir: str, episodes_dir: str | None = None, home=(-33.93, 18.63)) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if episodes_dir and not any(n.endswith(".rlglog") for n in os.listdir(log_dir)):
            self._seed_logs(episodes_dir, home)

    def _seed_logs(self, episodes_dir: str, home) -> None:
        names = sorted((n for n in os.listdir(episodes_dir)
                        if n.startswith("gated-") and n.endswith(".json") and not n.endswith(".summary.json")),
                       key=lambda n: os.path.getmtime(os.path.join(episodes_dir, n)), reverse=True)[:3]
        for i, n in enumerate(names):
            with open(os.path.join(episodes_dir, n), encoding="utf-8") as f:
                traj = json.load(f)
            header, records = logs.synthesize_from_trajectory(traj, home, seed=i)
            logs.write(os.path.join(self.log_dir, f"stub-{traj['episode_id']}.rlglog"), header, records)

    def status(self) -> dict:
        du = shutil.disk_usage(self.log_dir)
        return {"reachable": True, "kind": self.kind, "address": "stub (simulated Pi)",
                "hostname": "rlglider-pi (stub)", "ip": "127.0.0.1",
                "disk_free_bytes": du.free, "disk_total_bytes": du.total, "log_dir": self.log_dir,
                "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    def list_logs(self) -> list[dict]:
        out = []
        for n in os.listdir(self.log_dir):
            if n.endswith(".rlglog"):
                p = os.path.join(self.log_dir, n)
                out.append({"name": n, "size_bytes": os.path.getsize(p),
                            "modified_utc": datetime.datetime.fromtimestamp(
                                os.path.getmtime(p), datetime.timezone.utc).isoformat(timespec="seconds")})
        return sorted(out, key=lambda x: x["modified_utc"], reverse=True)

    def fetch(self, name: str) -> bytes:
        with open(os.path.join(self.log_dir, safe_log_name(name)), "rb") as f:
            return f.read()


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"

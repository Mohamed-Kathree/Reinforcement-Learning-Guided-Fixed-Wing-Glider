"""
pi_server.py -- runs ON THE RASPBERRY PI, ground use only (log retrieval over WiFi).

UNTESTED ON THE PI. Standard library only, so it needs nothing installed:

    python3 pi_server.py --log-dir /home/pi/logs --port 8090

Serves the contract in flighttest/pi.py:
    GET /status, GET /logs, GET /logs/<name>
Read-only: it never writes, deletes or executes anything. Only *.rlglog files
in --log-dir are served; names are checked against a strict whitelist.

Do not run it in flight: WiFi is for the ground. The flight logger (not yet
written) owns the SD card while flying.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def make_handler(log_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/status":
                du = shutil.disk_usage(log_dir)
                return self._json({"hostname": socket.gethostname(), "ip": _ip(),
                                   "disk_free_bytes": du.free, "disk_total_bytes": du.total, "log_dir": log_dir,
                                   "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
            if self.path == "/logs":
                out = []
                for n in os.listdir(log_dir):
                    if n.endswith(".rlglog"):
                        p = os.path.join(log_dir, n)
                        out.append({"name": n, "size_bytes": os.path.getsize(p),
                                    "modified_utc": datetime.datetime.fromtimestamp(
                                        os.path.getmtime(p), datetime.timezone.utc).isoformat(timespec="seconds")})
                return self._json(sorted(out, key=lambda x: x["modified_utc"], reverse=True))
            if self.path.startswith("/logs/"):
                name = unquote(self.path[len("/logs/"):])
                if not name.endswith(".rlglog") or not set(name) <= SAFE or ".." in name or name.startswith("."):
                    return self._json({"error": "bad name"}, 400)
                p = os.path.join(log_dir, name)
                if not os.path.isfile(p):
                    return self._json({"error": "not found"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(os.path.getsize(p)))
                self.end_headers()
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return None
            return self._json({"error": "not found"}, 404)

        def log_message(self, *args):  # quiet
            pass

    return Handler


def _ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.path.expanduser("~/logs"))
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(a.log_dir)).serve_forever()


if __name__ == "__main__":
    main()

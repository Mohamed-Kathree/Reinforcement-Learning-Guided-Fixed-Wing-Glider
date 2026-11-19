"""
flighttest.py
=============
Flight-test API (flight-test phase 2). Thin layer over the flighttest package:

    GET  /api/ft/status                 ground link + Pi, reported independently
    GET  /api/ft/ports                  serial ports (needs pyserial)
    POST /api/ft/connect?transport=stub|serial&port=COM5&baud=115200
    POST /api/ft/disconnect
    POST /api/ft/pi?url=http://<pi>:8090   (no url = stub Pi)
    POST /api/ft/arm?mode=bench|replay   explicit UI arming
    POST /api/ft/disarm
    POST /api/ft/servos   {"servo_us": [4]}   any rate; sent at <= 20 Hz
    POST /api/ft/keepalive
    GET/PUT /api/ft/calibration         validated on every save
    POST /api/ft/stub?flying=&loss=&rail_sensor=&pi_alive=&flight_arm=   stub glider only
    GET  /api/ft/pi/logs ; POST /api/ft/pi/logs/{name}/fetch
    GET  /api/ft/logs ; GET /api/ft/logs/{name}       fetched logs, parsed for replay
    WS   /ws/ft                         snapshot + new telemetry points, 5 Hz

The ground station runs in one background thread (10 ms step), started on first
use. Hardware seams: SerialLink and PiClient are untested against hardware;
pushing calibration neutrals to the ESP32 (NVS) is not implemented.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect

from .. import config

router = APIRouter(tags=["flighttest"])

FLIGHTLOG_DIR = config.DATA_DIR / "flightlogs"
STUB_PI_DIR = config.DATA_DIR / "pi_stub"


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


class Runtime:
    def __init__(self) -> None:
        from flighttest import calibration
        from flighttest.ground import GroundStation
        self.lock = threading.Lock()
        self.cal = calibration.load()
        self.ground = GroundStation(cal=self.cal)
        self.pi = None
        self.stub_glider = None
        self._pi_cache: tuple[float, dict] | None = None
        self._thread = threading.Thread(target=self._loop, name="ft-ground", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self.lock:
                try:
                    self.ground.step(_now_ms())
                except Exception as exc:  # a broken link must not kill the loop
                    self.ground.last_end = {"reason": f"link error: {exc}"[:200], "at_ms": _now_ms()}
                    self.ground.disconnect()
            time.sleep(0.01)

    def pi_client(self):
        if self.pi is None:
            from flighttest.pi import StubPi
            self.pi = StubPi(str(STUB_PI_DIR), str(config.EPISODES_DIR))
        return self.pi

    def pi_status(self) -> dict:
        now = time.monotonic()
        if self._pi_cache and now - self._pi_cache[0] < 3.0:
            return self._pi_cache[1]
        pi = self.pi_client()
        st = pi.status()
        if st.get("reachable"):
            try:
                st["logs"] = pi.list_logs()
            except Exception as exc:
                st["logs"] = []
                st["logs_error"] = str(exc)[:160]
        self._pi_cache = (now, st)
        return st


_rt: Runtime | None = None


def rt() -> Runtime:
    global _rt
    if _rt is None:
        _rt = Runtime()
    return _rt


def json_safe(x):
    """NaN and +/-inf -> None, recursively. The real aircraft sends NaN (e.g. baro
    altitude before the arm-time reference exists); NaN is not valid JSON and the
    browser's JSON.parse rejected every WebSocket message (found 20 Nov 2026)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


@router.get("/api/ft/status")
def status() -> dict:
    r = rt()
    with r.lock:
        snap = r.ground.snapshot(_now_ms())
    snap["pi"] = r.pi_status()
    snap["stub"] = None if r.stub_glider is None else {
        "flying": r.stub_glider.alt_true > 0, "airborne": r.stub_glider.airborne, "loss": r.stub_glider.loss,
        "rail_sensor": r.stub_glider.rail_sensor, "pi_alive": r.stub_glider.pi_alive}
    return json_safe(snap)


@router.get("/api/ft/ports")
def ports() -> dict:
    from flighttest.links import list_serial_ports
    return list_serial_ports()


@router.post("/api/ft/connect")
def connect(transport: str = "stub", port: Optional[str] = None, baud: int = 57600) -> dict:
    # 57600: the ground station's USB-serial path corrupted bursts at 115200 (20 Nov 2026)
    from flighttest.links import SerialLink, StubLink
    from flighttest.stub_glider import StubGlider
    from flighttest import calibration
    r = rt()
    with r.lock:
        if transport == "stub":
            r.stub_glider = StubGlider(calibration.neutral_us(r.cal), now_ms=_now_ms(), seed=int(time.time()),
                                       limits=calibration.limits(r.cal))
            r.ground.connect(StubLink(r.stub_glider))
        elif transport == "serial":
            if port == "auto":
                port = _find_ground_station(baud)
                if port is None:
                    raise HTTPException(404, "no ground station found: is it plugged in, and is the "
                                             "flight computer on? (a port counts once it carries valid telemetry)")
            if not port:
                raise HTTPException(400, "serial needs a port, e.g. COM5 or /dev/ttyUSB0")
            try:
                link = SerialLink(port, baud)
            except Exception as exc:
                raise HTTPException(400, f"could not open {port}: {exc}")
            r.stub_glider = None
            r.ground.connect(link)
        else:
            raise HTTPException(400, "transport must be 'stub' or 'serial'")
        return r.ground.snapshot(_now_ms())["ground"]


def _find_ground_station(baud: int) -> Optional[str]:
    """The first serial port that carries valid frames (Telemetry/Info) within 2.5 s.
    Ports busy in another program are skipped."""
    from flighttest.links import list_serial_ports
    from flighttest.protocol import packets as P
    try:
        import serial
    except ImportError:
        return None
    for p in list_serial_ports().get("ports", []):
        try:
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = p["device"], baud, 0.1
            s.dtr = s.rts = False                     # do not reset the board
            s.open()
        except Exception:
            continue
        try:
            parser, t0 = P.StreamParser(), time.time()
            while time.time() - t0 < 2.5:
                if any(isinstance(x, (P.Telemetry, P.Info)) for x in parser.feed(s.read(512))):
                    return p["device"]
        finally:
            s.close()
    return None


@router.post("/api/ft/disconnect")
def disconnect() -> dict:
    r = rt()
    with r.lock:
        r.ground.disarm(_now_ms(), "disconnect")
        r.ground.step(_now_ms())          # send the first DISARMED before closing
        r.ground.disconnect()
        r.stub_glider = None
    return {"connected": False}


@router.post("/api/ft/pi")
def set_pi(url: Optional[str] = None) -> dict:
    from flighttest.pi import PiClient
    r = rt()
    r.pi = PiClient(url) if url else None
    r._pi_cache = None
    return r.pi_status()


def _mode(name: str) -> int:
    from flighttest.protocol import packets as P
    m = {"bench": P.BENCH, "replay": P.REPLAY}.get(name.lower())
    if m is None:
        raise HTTPException(400, "mode must be 'bench' or 'replay'")
    return m


@router.post("/api/ft/arm")
def arm(mode: str) -> dict:
    r = rt()
    with r.lock:
        try:
            return r.ground.arm(_mode(mode), _now_ms())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))


@router.post("/api/ft/disarm")
def disarm() -> dict:
    r = rt()
    with r.lock:
        r.ground.disarm(_now_ms(), "requested")
    return {"session": None}


@router.post("/api/ft/servos")
def servos(servo_us: list[int] = Body(..., embed=True), envelope: bool = Body(False, embed=True)) -> dict:
    r = rt()
    with r.lock:
        try:
            return {"servo_us": r.ground.set_servos(servo_us, _now_ms(), envelope=envelope)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@router.post("/api/ft/surfaces")
def surfaces(aileron: float = Body(...), elevator: float = Body(...), rudder: float = Body(...)) -> dict:
    """Surface deflections (rad, simulator convention) mapped through the saved
    calibration exactly as the flight firmware maps them (replay on the aircraft)."""
    r = rt()
    with r.lock:
        try:
            return {"servo_us": r.ground.set_surfaces(aileron, elevator, rudder, _now_ms())}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@router.post("/api/ft/keepalive")
def keepalive() -> dict:
    r = rt()
    with r.lock:
        r.ground.keepalive(_now_ms())
        return {"session": r.ground.session_info(_now_ms())}


@router.get("/api/ft/calibration")
def get_calibration() -> dict:
    from flighttest.calibration import validate
    r = rt()
    return {"calibration": r.cal, "problems": validate(r.cal)}


@router.put("/api/ft/calibration")
def put_calibration(cal: dict = Body(...)) -> dict:
    from flighttest import calibration
    r = rt()
    try:
        calibration.save(cal)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    with r.lock:
        r.cal = cal
        r.ground.set_calibration(cal)
        if r.stub_glider is not None:     # the stub "firmware" picks up neutrals and limits immediately
            r.stub_glider.safety.s.neutral_us = calibration.neutral_us(cal)
            r.stub_glider.safety.set_limits(*calibration.limits(cal))
    # Saving does not change the aircraft: POST /api/ft/calibration/push sends it
    # (CalibrationSet over ESP-NOW; applied only DISARMED and on the ground, stored in NVS).
    return {"calibration": cal, "problems": [], "pushed_to_firmware": False}


@router.post("/api/ft/calibration/push")
def push_calibration() -> dict:
    """Send the saved calibration to the aircraft. Confirmed when the aircraft's
    Info packet reports the same calibration id loaded from NVS (see snapshot
    'calibration_push')."""
    r = rt()
    with r.lock:
        try:
            return {"calibration_push": r.ground.push_calibration(_now_ms())}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))


@router.post("/api/ft/console")
def console(line: str = Body(..., embed=True)) -> dict:
    """A whitelisted flight-computer console command (arm, disarm, report, ...)."""
    r = rt()
    with r.lock:
        try:
            return {"sent": r.ground.console(line, _now_ms())}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@router.post("/api/ft/stub")
def stub(flying: Optional[bool] = None, loss: Optional[float] = None, rail_sensor: Optional[bool] = None,
         pi_alive: Optional[bool] = None, flight_arm: Optional[bool] = None) -> dict:
    r = rt()
    with r.lock:
        g = r.stub_glider
        if g is None:
            raise HTTPException(409, "the stub glider is not connected")
        if flying is not None:            # whether the firmware then *detects* airborne is up to its rules
            g.set_flying(flying)
        if pi_alive is not None:
            g.pi_alive = pi_alive
        if flight_arm:                    # simulate arming FLIGHT (not a ground-app action)
            g.arm_flight_locally(_now_ms())
        if loss is not None:
            if not 0.0 <= loss <= 1.0:
                raise HTTPException(400, "loss must be 0..1")
            g.loss = loss
        if rail_sensor is not None:
            g.rail_sensor = rail_sensor
        return {"flying": g.alt_true > 0, "airborne": g.airborne, "loss": g.loss,
                "rail_sensor": g.rail_sensor, "pi_alive": g.pi_alive}


@router.get("/api/ft/pi/logs")
def pi_logs() -> list[dict]:
    try:
        return rt().pi_client().list_logs()
    except Exception as exc:
        raise HTTPException(502, f"Pi unreachable: {exc}")


@router.post("/api/ft/pi/logs/{name}/fetch")
def pi_fetch(name: str) -> dict:
    from flighttest.pi import safe_log_name
    try:
        name = safe_log_name(name)
        data = rt().pi_client().fetch(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"could not fetch from the Pi: {exc}")
    FLIGHTLOG_DIR.mkdir(parents=True, exist_ok=True)
    (FLIGHTLOG_DIR / name).write_bytes(data)
    return {"name": name, "size_bytes": len(data)}


@router.get("/api/ft/logs")
def local_logs() -> list[dict]:
    FLIGHTLOG_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(({"name": p.name, "size_bytes": p.stat().st_size} for p in FLIGHTLOG_DIR.glob("*.rlglog")),
                  key=lambda x: x["name"])


@router.get("/api/ft/logs/{name}")
def local_log(name: str) -> dict:
    from flighttest import logs
    from flighttest.pi import safe_log_name
    try:
        p = FLIGHTLOG_DIR / safe_log_name(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not p.is_file():
        raise HTTPException(404, "log not fetched yet")
    try:
        return {"name": name, **logs.parse(p.read_text(encoding="utf-8"))}
    except ValueError as exc:
        raise HTTPException(422, f"could not parse {name}: {exc}")


@router.websocket("/ws/ft")
async def ws_ft(websocket: WebSocket) -> None:
    await websocket.accept()
    r = rt()
    last_t = -1
    try:
        while True:
            with r.lock:
                now = _now_ms()
                snap = r.ground.snapshot(now)
                new = [h for h in r.ground.history if h["t_ms"] > last_t]
            if new:
                last_t = new[-1]["t_ms"]
            await websocket.send_json(json_safe({"now_ms": now, "snapshot": snap, "history": new}))
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, RuntimeError):
        return

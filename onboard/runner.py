"""
runner.py -- the Pi's flight program (systemd service rlglider-onboard).

    python3 -m onboard.runner --config onboard/config.json

Every 50 ms (20 Hz, the training control rate):
  1. read all PiTelemetry from the ESP32 (logged at 50 Hz);
  2. follow the ESP32's flight phase (status bits):
       DISARMED  -> nothing to fly; heartbeat setpoints keep the link fresh;
       ARMED     -> a new .rlglog is opened (home in the header); heartbeat;
       LAUNCH    -> heartbeat (the ESP32 holds wings level in the zoom climb);
       AUTONOMOUS-> the controller starts fresh (frame stack reset, gate reset:
                    the same state as a training episode start, which is the
                    apex) and flies: observation v1 -> action -> PiSetpoint;
  3. back to DISARMED -> log closed.
A heartbeat is a PiSetpoint with controller = 0: the ESP32 counts it for link
health but never flies it. If telemetry stops for stale_s the runner stops
acting (heartbeat only); the ESP32 flies its failsafe glide on its own once
setpoints stop or are heartbeats.

The ESP32 decides when the Pi's setpoints are flown and checks every limit;
the Pi cannot move a surface outside AUTONOMOUS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from onboard import controllers as C  # noqa: E402
from onboard.flightlog import FlightLog  # noqa: E402
from onboard.link import EspLink, open_serial  # noqa: E402
from onboard.obs import ObsBuilder  # noqa: E402
from onboard.vendor import packets as P  # noqa: E402
from onboard.vendor.constants import SPEED_CMD_CENTRE_MS  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "onboard", "config.json")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("model", "log_dir"):
        if cfg.get(key) and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(ROOT, cfg[key])
    return cfg


class Runner:
    def __init__(self, cfg: dict, link: EspLink, clock=time.monotonic, logging: bool = True) -> None:
        self.cfg = cfg
        self.link = link
        self.clock = clock
        self.logging = logging
        self.actor = C.build(cfg)
        self.obsb = ObsBuilder()
        self.log: FlightLog | None = None
        self.phase = P.PHASE_DISARMED
        self.active = False
        self.steps = 0
        self.last_rx = None
        self.last_action = None
        self.events: list[str] = []
        self.model_meta = getattr(getattr(self.actor, "policy", self.actor), "net", None)

    # -- helpers ----------------------------------------------------------
    def _event(self, msg: str) -> None:
        self.events.append(msg)
        print(f"[onboard] {msg}", flush=True)

    def _open_log(self, t) -> None:
        if not self.logging:
            return
        st = P.unpack_status(t.status)
        meta = self.model_meta.meta if self.model_meta is not None else None
        header = {
            "source": "Raspberry Pi onboard runner",
            "home": ({"lat": t.home_lat * 1e-7, "lon": t.home_lon * 1e-7} if st.get("home_set") else None),
            "controller": self.cfg.get("controller", "gated"),
            "gate": self.cfg.get("gate"),
            "model": None if meta is None else {k: meta.get(k) for k in ("source_model", "model_sha256", "vecnorm_sha256")},
            "calibration": None,
            "note": "pitch target = trim schedule without the true-airspeed term (decision 19 Nov 2026)",
        }
        self.log = FlightLog(self.cfg.get("log_dir", os.path.join(ROOT, "logs")), header)
        self._event(f"log opened {self.log.path}")

    def _close_log(self) -> None:
        if self.log is not None:
            self.log.close()
            self._event(f"log closed ({self.log.n} records)")
            self.log = None

    # -- one 20 Hz step ---------------------------------------------------------
    def tick(self) -> None:
        now = self.clock()
        frames = self.link.poll()
        if frames:
            self.last_rx = now
        t = self.link.latest
        stale = t is None or self.last_rx is None or now - self.last_rx > self.cfg.get("stale_s", 0.25)

        if t is not None:
            phase = P.unpack_status(t.status)["phase"]
            if phase != self.phase:
                self._event(f"phase {P.PHASE_NAMES.get(self.phase)} -> {P.PHASE_NAMES.get(phase)}")
                if self.phase == P.PHASE_DISARMED and phase >= P.PHASE_ARMED:
                    self._open_log(t)
                if phase == P.PHASE_AUTONOMOUS:
                    self.actor.reset(0)
                    self.obsb.reset()
                    self.active = True
                    self.steps = 0
                if phase == P.PHASE_DISARMED:
                    self.active = False
                    self._close_log()
                self.phase = phase

        extra = None
        if self.active and not stale and self.phase == P.PHASE_AUTONOMOUS:
            obs = self.obsb.build(t)
            action = self.actor.act(obs)
            roll_sp, speed = C.to_setpoint(action)
            ctl = C.active_controller(self.actor)
            self.link.send(roll_sp, speed, ctl)
            self.steps += 1
            self.last_action = action
            gate = getattr(self.actor, "gate", None)
            extra = {"obs": [round(float(x), 4) for x in obs], "action": [round(float(x), 4) for x in action],
                     "roll_sp": round(roll_sp, 4), "speed_cmd": round(speed, 3), "controller": ctl,
                     "baro_vz": self.obsb.used_baro_vz}
            if gate is not None:
                w = gate.wind_estimate_ne
                extra["handed_over"] = bool(gate.handed_over)
                extra["wind_est_ne"] = [round(float(w[0]), 3), round(float(w[1]), 3)]
        else:
            self.link.send(0.0, SPEED_CMD_CENTRE_MS, 0)          # heartbeat, never flown

        if self.log is not None:
            for fr in frames:
                self.log.record(fr, self.phase, extra if fr is t else None)

    def close(self) -> None:
        self._close_log()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--port", help="override the config's serial device")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    port = open_serial(a.port or cfg["serial"], cfg.get("baud", 115200))
    link = EspLink(port)
    r = Runner(cfg, link)
    period = 1.0 / cfg.get("rate_hz", 20)
    print(f"[onboard] controller={cfg.get('controller')} serial={a.port or cfg['serial']} rate={1/period:.0f} Hz", flush=True)
    nxt = time.monotonic()
    last_report = nxt
    overruns = 0
    try:
        while True:
            r.tick()
            nxt += period
            now = time.monotonic()
            if now > nxt:
                overruns += 1
                nxt = now
            else:
                time.sleep(nxt - now)
            if now - last_report > 5.0:
                last_report = now
                age = None if r.last_rx is None else now - r.last_rx
                print(f"[onboard] phase={P.PHASE_NAMES.get(r.phase)} active={r.active} steps={r.steps} "
                      f"rx={link.rx_frames} gaps={link.tlm_gaps} dropped={link.dropped} "
                      f"tlm_age={'-' if age is None else f'{age:.2f}s'} overruns={overruns}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

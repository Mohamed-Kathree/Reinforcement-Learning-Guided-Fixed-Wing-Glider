"""
sim_check.py -- DEV PC ONLY. The Pi's flight program, closed-loop against the
simulator, through the real wire protocol.

SimInTheLoop is a benchmark actor (reset/act) that, every 50 ms step:
  1. turns the environment's observation into a PiTelemetry frame, exactly the
     fields the ESP32 sends (position as lat/lon 1e-7 deg around a home point,
     velocity NED, attitude, baro, sonar; phase AUTONOMOUS, home set);
  2. encodes it to bytes and feeds them to onboard.runner.Runner over an
     in-memory serial port;
  3. decodes the PiSetpoint bytes the runner writes back and converts bank +
     speed command into the normalised action the environment steps with.
So it exercises link parsing, the observation builder, the numpy policy, the
vendored gate/baseline and the setpoint mapping together.

    # equivalence with the benchmark actor (SB3 policy + repository gate)
    python -m onboard.sim_check compare --stages 0 1 2 3 --n 10
    # the pitch-target decision: Run 4 gated, simulator inner loop with and
    # without the true-airspeed term, DEVELOPMENT seeds only (never the
    # benchmark seeds), at most 3 worker processes
    python -m onboard.sim_check airspeed --stages 0 1 2 3 --n 200 --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from onboard.obs import EARTH_R  # noqa: E402
from onboard.vendor import packets as P  # noqa: E402
from onboard.vendor.constants import BANK_CMD_SCALE_RAD, SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS  # noqa: E402

HOME = (-339339000, 184546000)          # an arbitrary home point, 1e-7 deg
RUN4_CONFIG = os.path.join(ROOT, "training", "configs", "v22_restored.yaml")
RUN4_MODEL = ("checkpoints_v24_run4/best_model.zip", "checkpoints_v24_run4/vecnormalize_best.pkl")
STATUS_AUTO = P.pack_status(mode=P.FLIGHT, armed=1, airborne=1, phase=P.PHASE_AUTONOMOUS, home_set=1,
                            gps_fix=2, gps_sats=9, baro_ref_captured=1, baro_warm=1, gps_ubx=1, yaw_aligned=1)


class MemPort:
    def __init__(self) -> None:
        self.inbound = bytearray()
        self.outbound = bytearray()

    def read(self, n: int) -> bytes:
        out = bytes(self.inbound[:n]); del self.inbound[:n]
        return out

    def write(self, b: bytes) -> int:
        self.outbound += b
        return len(b)


def telemetry_from_obs(obs, seq: int, t_ms: int) -> P.PiTelemetry:
    dn, de, gs, course, roll, pitch, yaw, alt, vz, lidar, lidar_ok, _ = (float(x) for x in obs)
    lat0 = math.radians(HOME[0] * 1e-7)
    lat = HOME[0] + round(math.degrees(dn / EARTH_R) * 1e7)
    lon = HOME[1] + round(math.degrees(de / (EARTH_R * math.cos(lat0))) * 1e7)
    return P.PiTelemetry(seq=seq & 0xFFFF, t_ms=t_ms, roll=roll, pitch=pitch, yaw=yaw, p=0.0, q=0.0, r=0.0,
                         baro_alt=alt, sonar=lidar if lidar_ok > 0.5 else -1.0, lat=lat, lon=lon,
                         home_lat=HOME[0], home_lon=HOME[1], vn=gs * math.cos(course), ve=gs * math.sin(course),
                         vd=-vz, gps_fix=2, gps_sats=9, gps_age_ms=0, gps_count=seq & 0xFFFF,
                         status=STATUS_AUTO, pi_rx_gaps=0, pi_rx_bad=0, servo_us=[1500] * 4, rail_adc_v=-1.0)


class SimInTheLoop:
    """A benchmark actor that is the Pi runner behind the wire protocol."""

    def __init__(self, onboard_cfg: dict) -> None:
        self.cfg = onboard_cfg
        self.name = f"onboard_{onboard_cfg.get('controller', 'gated')}"
        self.obs_version = 1
        self.max_obs_err = 0.0
        self.actions: list = []

    def reset(self, seed: int) -> None:
        from onboard.link import EspLink
        from onboard.runner import Runner
        self.port = MemPort()
        self.t = 0.0
        self.seq = 0
        self.runner = Runner(self.cfg, EspLink(self.port), clock=lambda: self.t, logging=False)
        self.actions = []

    def act(self, obs) -> np.ndarray:
        self.seq += 1
        self.t += 0.05
        self.port.inbound += telemetry_from_obs(obs, self.seq, int(round(self.t * 1000))).encode()
        self.runner.tick()
        sp = [p for p in P.StreamParser().feed(bytes(self.port.outbound)) if isinstance(p, P.PiSetpoint)]
        self.port.outbound.clear()
        assert sp and sp[-1].controller != 0, "the runner did not fly this step"
        rebuilt = self.runner.obsb.build  # noqa: F841  (the runner already built it; compare below)
        a = np.array([sp[-1].roll_sp / BANK_CMD_SCALE_RAD,
                      (sp[-1].speed_cmd - SPEED_CMD_CENTRE_MS) / SPEED_CMD_SCALE_MS], dtype=np.float32)
        self.actions.append(a)
        return a


def onboard_cfg(controller: str = "gated") -> dict:
    from onboard.runner import load_config
    cfg = load_config()
    cfg["controller"] = controller
    return cfg


def reference_actor(controller: str = "gated"):
    from baseline.wind_gated import WindGatedActor
    from training.benchmark import BaselineActor, PolicyActor, load_config
    if controller == "baseline":
        return BaselineActor()
    pol = PolicyActor(*[os.path.join(ROOT, p) for p in RUN4_MODEL])
    if controller == "policy":
        return pol
    return WindGatedActor.from_config(pol, load_config()["gate"]["run4"], "window")


class Recording:
    def __init__(self, actor) -> None:
        self.actor, self.name, self.obs_version = actor, actor.name, 1
        self.actions = []

    def reset(self, seed):
        self.actor.reset(seed); self.actions = []

    def act(self, obs):
        a = np.asarray(self.actor.act(obs), dtype=np.float32)
        self.actions.append(a)
        return a

    def episode_record(self):
        return self.actor.episode_record() if hasattr(self.actor, "episode_record") else {}


def dev_seeds(n: int) -> list[int]:
    with open(os.path.join(ROOT, "analysis", "seeds", "dev_seeds.json")) as f:
        return json.load(f)["seeds"][:n]


def _episode(args):
    stage, seed, kind, airspeed_term, controller = args
    from training.benchmark import evaluate_seed, load_config
    cfg = load_config(RUN4_CONFIG)
    cfg["pitch_airspeed_term"] = airspeed_term
    actor = SimInTheLoop(onboard_cfg(controller)) if kind == "onboard" else Recording(reference_actor(controller))
    rec = evaluate_seed(cfg, stage, actor, seed, 1)
    return rec, [a.tolist() for a in actor.actions]


def compare(stages, n, controller="gated") -> dict:
    out = {}
    for stage in stages:
        rows = []
        for seed in dev_seeds(n):
            r_ref, a_ref = _episode((stage, seed, "reference", True, controller))
            r_onb, a_onb = _episode((stage, seed, "onboard", True, controller))
            k = min(len(a_ref), len(a_onb))
            diff = float(np.max(np.abs(np.array(a_ref[:k]) - np.array(a_onb[:k])))) if k else 0.0
            rows.append({"seed": seed, "outcome_ref": r_ref["outcome"], "outcome_onboard": r_onb["outcome"],
                         "steps_ref": len(a_ref), "steps_onboard": len(a_onb), "max_action_diff": diff})
        same = sum(r["outcome_ref"] == r["outcome_onboard"] for r in rows)
        out[str(stage)] = {"n": len(rows), "same_outcome": same,
                           "max_action_diff": max(r["max_action_diff"] for r in rows), "episodes": rows}
        print(f"stage {stage}: same outcome {same}/{len(rows)}, max |action diff| {out[str(stage)]['max_action_diff']:.2e}")
    return out


def airspeed(stages, n, workers, controller="gated") -> dict:
    import multiprocessing as mp
    workers = min(int(workers), 3)                        # standing rule: never more than 3
    out = {"seed_set": "analysis/seeds/dev_seeds.json", "n": n, "controller": controller,
           "model": RUN4_MODEL[0], "config": "training/configs/v22_restored.yaml", "stages": {}}
    for stage in stages:
        res = {}
        for term in (True, False):
            jobs = [(stage, s, "reference", term, controller) for s in dev_seeds(n)]
            t0 = time.time()
            with mp.get_context("spawn").Pool(workers) as pool:
                recs = [r for r, _ in pool.map(_episode, jobs)]
            k = sum(r["outcome"] == "success" for r in recs)
            crash = sum(r["outcome"] == "crash" for r in recs)
            res["with_term" if term else "without_term"] = {
                "success": k, "crash": crash, "n": len(recs), "outcomes": [r["outcome"] for r in recs],
                "mean_quality": float(np.mean([r["quality"] for r in recs])), "seconds": round(time.time() - t0)}
        a, b = res["with_term"]["outcomes"], res["without_term"]["outcomes"]
        tab = [[sum(x == "success" and y == "success" for x, y in zip(a, b)), sum(x == "success" and y != "success" for x, y in zip(a, b))],
               [sum(x != "success" and y == "success" for x, y in zip(a, b)), sum(x != "success" and y != "success" for x, y in zip(a, b))]]
        from training.benchmark import mcnemar_exact
        res["paired"] = {"table": tab, "mcnemar_p": mcnemar_exact(tab)}
        out["stages"][str(stage)] = res
        print(f"stage {stage}: success with term {res['with_term']['success']}/{n}, without {res['without_term']['success']}/{n}"
              f" (crash {res['with_term']['crash']} / {res['without_term']['crash']}), McNemar p={res['paired']['mcnemar_p']:.3f}",
              flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["compare", "airspeed"])
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--controller", default="gated")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = compare(a.stages, a.n, a.controller) if a.what == "compare" else airspeed(a.stages, a.n, a.workers, a.controller)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

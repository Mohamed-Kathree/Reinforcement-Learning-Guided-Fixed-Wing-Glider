"""
test_onboard.py
===============
The Raspberry Pi flight program (onboard/), on the dev PC.

  - vendored copies are current (onboard.vendor_sync --check) and the Pi-side
    import graph has no simulator / torch / gym / SB3;
  - the exported numpy policy equals SB3's deterministic action, and its
    observation pipeline equals training/benchmark.py PolicyActor;
  - through the real wire protocol (PiTelemetry bytes in, PiSetpoint bytes
    out), the runner's gated controller matches the repository's
    WindGatedActor on recorded simulator observations, handover included;
  - observation builder conventions (sonar invalid -> 0, vertical speed sign,
    baro-rate fallback, required glide ratio);
  - the runner's phase handling: heartbeats before the apex, controller from
    AUTONOMOUS, stale telemetry, log written and readable by flighttest.logs;
  - link sequence-gap counting;
  - config.json matches the frozen gate and the exported model's source files.

Development seeds only (analysis/seeds/dev_seeds.json); never the benchmark seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from onboard import controllers as C  # noqa: E402
from onboard import sim_check as S  # noqa: E402
from onboard.link import EspLink  # noqa: E402
from onboard.obs import ObsBuilder  # noqa: E402
from onboard.policy import NumpyPolicy, PolicyActor  # noqa: E402
from onboard.runner import Runner, load_config  # noqa: E402
from onboard.vendor import packets as P  # noqa: E402

MODEL_NPZ = os.path.join(ROOT, "onboard", "model", "run4.npz")


def test_vendor_copies_are_current():
    r = subprocess.run([sys.executable, "-m", "onboard.vendor_sync", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_pi_import_graph_is_light():
    code = ("import sys; import onboard.runner, onboard.controllers; "
            "heavy=[m for m in ('gymnasium','jsbsim','torch','stable_baselines3','env','baseline','training','sim') if m in sys.modules]; "
            "print(heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_config_matches_frozen_gate_and_model():
    from training.benchmark import load_config as bench_cfg
    cfg = load_config()
    assert cfg["controller"] == "gated"
    g = bench_cfg()["gate"]["run4"]["window"]
    assert cfg["gate"] == {k: g[k] for k in ("threshold_ms", "t_earliest_s", "t_latest_s")}
    meta = NumpyPolicy(MODEL_NPZ).meta
    for k_src, k_sha in (("source_model", "model_sha256"), ("source_vecnorm", "vecnorm_sha256")):
        path = os.path.join(ROOT, meta[k_src])
        if not os.path.exists(path):
            pytest.skip(f"{meta[k_src]} not present on this machine")
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == meta[k_sha], f"{meta[k_src]} changed since export"


def test_numpy_policy_equals_sb3():
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(ROOT, S.RUN4_MODEL[0]), device="cpu")
    net = NumpyPolicy(MODEL_NPZ)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, size=(500, 96)).astype(np.float32)
    ref, _ = model.predict(x, deterministic=True)
    got = np.stack([net.forward(v) for v in x])
    assert np.abs(ref - got).max() < 1e-5


def _recorded_obs(stage: int, seed: int):
    from training.benchmark import evaluate_seed, load_config as bench_cfg

    class Rec(S.Recording):
        def act(self, obs):
            self.obs = getattr(self, "obs", [])
            self.obs.append(np.array(obs))
            return super().act(obs)
    rec = Rec(S.reference_actor())
    evaluate_seed(bench_cfg(S.RUN4_CONFIG), stage, rec, seed, 1)
    return rec.obs


def test_policy_pipeline_equals_benchmark_actor():
    from training.benchmark import PolicyActor as BenchActor
    obs = _recorded_obs(0, S.dev_seeds(1)[0])
    ref = BenchActor(*[os.path.join(ROOT, p) for p in S.RUN4_MODEL])
    mine = PolicyActor(MODEL_NPZ)
    ref.reset(0); mine.reset(0)
    d = max(float(np.abs(ref.act(o) - mine.act(o)).max()) for o in obs)
    assert d < 1e-5, d


def test_wire_protocol_gated_controller_matches_reference_with_handover():
    """Open loop on recorded observations: the runner behind PiTelemetry/PiSetpoint
    bytes vs the repository's WindGatedActor. Differences come only from the wire
    format (positions as 1e-7 deg, about 1 cm) and float32 arithmetic."""
    found = False
    for seed in S.dev_seeds(8):
        obs = _recorded_obs(3, seed)
        ref = S.reference_actor(); ref.reset(seed)
        on = S.SimInTheLoop(S.onboard_cfg()); on.reset(seed)
        diffs = [float(np.abs(np.asarray(ref.act(o)) - on.act(o)).max()) for o in obs]
        gate_on = on.runner.actor.gate
        assert ref.gate.handover_step == gate_on.handover_step
        assert max(diffs) < 1e-3, max(diffs)
        if ref.handed_over:
            found = True
            break
    assert found, "no development seed among the first 8 handed over at stage 3"


def tlm(**kw):
    base = dict(seq=1, t_ms=1000, roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=0.0, baro_alt=20.0, sonar=-1.0,
                lat=S.HOME[0], lon=S.HOME[1], home_lat=S.HOME[0], home_lon=S.HOME[1], vn=0.0, ve=0.0, vd=0.0,
                gps_fix=2, gps_sats=8, gps_age_ms=0, gps_count=1, status=0, pi_rx_gaps=0, pi_rx_bad=0,
                servo_us=[1500] * 4, rail_adc_v=-1.0)
    base.update(kw)
    return P.PiTelemetry(**base)


def test_obs_builder_conventions():
    b = ObsBuilder()
    # 100 m north, 50 m east of home (1e-7 deg steps), flying east at 10 m/s, sinking 1 m/s
    lat = S.HOME[0] + round(math.degrees(100.0 / 6_371_000.0) * 1e7)
    lon = S.HOME[1] + round(math.degrees(50.0 / (6_371_000.0 * math.cos(math.radians(S.HOME[0] * 1e-7)))) * 1e7)
    o = b.build(tlm(lat=lat, lon=lon, vn=0.0, ve=10.0, vd=1.0, baro_alt=10.0, sonar=-1.0, yaw=1.0))
    assert o[0] == pytest.approx(100.0, abs=0.02) and o[1] == pytest.approx(50.0, abs=0.02)
    assert o[2] == pytest.approx(10.0) and o[3] == pytest.approx(math.pi / 2)
    assert o[6] == pytest.approx(1.0) and o[7] == pytest.approx(10.0) and o[8] == pytest.approx(-1.0)
    assert (o[9], o[10]) == (0.0, 0.0)                                  # sonar invalid -> 0, not -1
    assert o[11] == pytest.approx(math.hypot(100, 50) / 10.0 / 15.0, rel=1e-4)
    o = b.build(tlm(t_ms=1100, sonar=2.5, baro_alt=0.4))
    assert (o[9], o[10]) == (pytest.approx(2.5), 1.0)
    assert o[11] == pytest.approx(0.0, abs=1e-6)                        # at home; alt clamps to 1 m
    # NMEA fallback: no GPS vertical speed -> barometric climb rate, flagged
    b = ObsBuilder()
    for i in range(40):
        o = b.build(tlm(t_ms=1000 + 50 * i, baro_alt=20.0 - 0.1 * i, vd=float("nan")))
    assert b.used_baro_vz and o[8] == pytest.approx(-2.0, abs=0.1)


def test_link_counts_sequence_gaps_and_ignores_garbage():
    port = S.MemPort()
    link = EspLink(port)
    port.inbound += b"\x00\xa5\x07junk" + tlm(seq=1).encode() + tlm(seq=2).encode() + tlm(seq=5).encode()
    frames = link.poll()
    assert [f.seq for f in frames] == [1, 2, 5] and link.tlm_gaps == 2
    link.send(0.1, 12.0, 1)
    sp = P.PiSetpoint.decode(bytes(port.outbound))
    assert (sp.controller, sp.tlm_gaps) == (1, 2) and sp.roll_sp == pytest.approx(0.1)


def status(phase, **kw):
    return P.pack_status(mode=P.FLIGHT if phase else P.DISARMED, armed=1 if phase else 0, phase=phase,
                         home_set=1 if phase else 0, gps_fix=2, baro_ref_captured=1, **kw)


def test_runner_phases_heartbeats_log(tmp_path):
    from flighttest import logs as L
    cfg = load_config()
    cfg["log_dir"] = str(tmp_path)
    port, clock = S.MemPort(), [0.0]
    r = Runner(cfg, EspLink(port), clock=lambda: clock[0])
    seq = [0]

    def step(phase, n=1, **kw):
        out = []
        for _ in range(n):
            seq[0] += 1
            clock[0] += 0.05
            port.inbound += tlm(seq=seq[0], t_ms=int(clock[0] * 1000), status=status(phase), **kw).encode()
            r.tick()
            out += [p for p in P.StreamParser().feed(bytes(port.outbound)) if isinstance(p, P.PiSetpoint)]
            port.outbound.clear()
        return out

    assert all(s.controller == 0 for s in step(P.PHASE_DISARMED, 3))       # heartbeat only
    assert r.log is None
    assert all(s.controller == 0 for s in step(P.PHASE_ARMED, 5))
    assert r.log is not None                                                # log opened at arming
    assert all(s.controller == 0 for s in step(P.PHASE_LAUNCH, 5, vd=-5.0))
    sp = step(P.PHASE_AUTONOMOUS, 40, vn=8.0, ve=2.0, vd=1.0)
    assert all(s.controller == C.CTRL_POLICY for s in sp) and r.steps == 40
    assert all(abs(s.roll_sp) <= math.radians(45) + 1e-6 and 9.5 - 1e-4 <= s.speed_cmd <= 16.0 + 1e-4 for s in sp)
    # telemetry stops: heartbeat, the controller does not act on stale data
    clock[0] += 1.0
    r.tick()
    hb = [p for p in P.StreamParser().feed(bytes(port.outbound)) if isinstance(p, P.PiSetpoint)]
    port.outbound.clear()
    assert hb and hb[-1].controller == 0 and r.steps == 40
    step(P.PHASE_DISARMED, 2)
    assert r.log is None
    logs = [f for f in os.listdir(tmp_path) if f.endswith(".rlglog")]
    assert len(logs) == 1
    parsed = L.parse(open(os.path.join(tmp_path, logs[0]), encoding="utf-8").read())
    assert parsed["home"] is not None and parsed["summary"]["skipped_lines"] == 0
    acted = [f for f in parsed["frames"] if f["action"] is not None]
    assert len(acted) == 40 and parsed["summary"]["n_frames"] >= 50


def test_simulator_airspeed_term_default_unchanged():
    from env import glider_env as G
    assert G.AttitudeController.use_airspeed_term is True
    env = G.GliderEnv(cfg={})
    assert env._ctrl.use_airspeed_term is True
    env2 = G.GliderEnv(cfg={"pitch_airspeed_term": False})
    assert env2._ctrl.use_airspeed_term is False


def test_airspeed_term_switch_reaches_the_env_through_the_benchmark_path():
    """The measurement builds envs with training.benchmark.make_env (flat_env_cfg):
    the switch must survive that path, and actually change the flight."""
    from training.benchmark import evaluate_seed, load_config as bench_cfg, make_env
    for term in (True, False):
        cfg = bench_cfg(S.RUN4_CONFIG)
        cfg["reward"]["pitch_airspeed_term"] = term
        assert make_env(cfg, 0, 1)._ctrl.use_airspeed_term is term
    seed = S.dev_seeds(1)[0]
    traj = {}
    for term in (True, False):
        cfg = bench_cfg(S.RUN4_CONFIG)
        cfg["reward"]["pitch_airspeed_term"] = term
        rec = S.Recording(S.reference_actor())
        evaluate_seed(cfg, 0, rec, seed, 1)
        traj[term] = np.array(rec.actions)
    k = min(len(traj[True]), len(traj[False]))
    assert np.abs(traj[True][:k] - traj[False][:k]).max() > 1e-3, "the switch had no effect on the flight"

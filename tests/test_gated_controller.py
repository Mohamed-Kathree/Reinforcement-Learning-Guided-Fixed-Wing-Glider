"""
SPEC V24 Phase 3: the wind-gated controller decides from on-board signals
only, is tuned on development seeds disjoint from the benchmark, hands over
one way, and logs its gate state with every episode.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_PATH = os.path.join(ROOT, "baseline", "wind_gate.py")
DEV_SEEDS = os.path.join(ROOT, "analysis", "seeds", "dev_seeds.json")
# sha256 of dev_seeds.json with LF line endings (git may check it out as CRLF).
DEV_SEEDS_SHA256 = "9b323c694d9faed58d1ccbbd72ba9c9bb1ee3c8c455acd40603ac8f27318ea91"
SIM_MODULES = ("sim", "jsbsim", "env.glider_env", "env.reward", "env.curriculum", "gymnasium")


class _ZeroPolicy:
    """Stand-in for a trained policy: wings level, centre airspeed."""
    obs_version = 1

    def __init__(self):
        self.calls = 0

    def reset(self, seed):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return np.zeros(2, dtype=np.float32)


def _record_episode(stage: int, seed: int, wind_override: float | None = None):
    """Fly the zero policy behind a live gate; return the sensor stream, the
    actions flown and the gate's per-step decisions."""
    from baseline.wind_gated import WindGatedActor
    from training.benchmark import load_config, make_env

    env = make_env(load_config(), stage, 1)
    if wind_override is not None:
        env.cfg["wind_speed"] = wind_override
    actor = WindGatedActor(_ZeroPolicy(), threshold_ms=1.0, mode="window",
                           t_earliest_s=3.0, t_latest_s=20.0)
    actor.reset(seed)
    obs, _ = env.reset(seed=seed)
    stream, actions, decisions = [], [], []
    try:
        while True:
            stream.append(np.array(obs, dtype=float))
            a = actor.act(obs)
            decisions.append(actor.gate.handed_over)
            actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        true_wind = env._wind._mean_ned[:2].astype(float).copy()
    finally:
        env.close()
    return stream, actions, decisions, true_wind


def _replay(stream, actions):
    from baseline.wind_gate import WindGate
    gate = WindGate(threshold_ms=1.0, mode="window", t_earliest_s=3.0, t_latest_s=20.0)
    out = []
    for obs, a in zip(stream, actions):
        gate.update(obs)
        out.append(gate.handed_over)
        gate.record_action(a)
    return out, gate


def test_gate_ignores_true_wind():
    from training.benchmark import load_config, make_env

    stream, actions, live, wind = _record_episode(2, 4242)
    assert np.hypot(*wind) > 1.0
    # Replay the identical sensor stream while a different true wind is live
    # in the simulator (calm Stage 0 env), and with no env at all.
    calm = make_env(load_config(), 0, 1)
    calm.reset(seed=1)
    try:
        assert np.hypot(*calm._wind._mean_ned[:2]) < 1e-9
        during_calm, _ = _replay(stream, actions)
    finally:
        calm.close()
    standalone, gate = _replay(stream, actions)
    assert live == during_calm == standalone
    # And the gate does read the sensors: a stream from calm air does not hand over.
    stream0, actions0, live0, _ = _record_episode(0, 4242)
    assert not any(live0)
    assert any(live), "a >1 m/s wind episode should trip a 1.0 m/s gate"


def test_gate_imports_no_sim_state():
    tree = ast.parse(open(GATE_PATH, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"__future__", "math", "numpy", "env.nav_features", "env.wind_estimate"}
    # And at run time, nothing pulls the simulator in transitively.
    code = ("import sys; import baseline.wind_gate; "
            f"print([m for m in sys.modules if m.startswith({SIM_MODULES!r})])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out


def test_speed_command_mapping_matches_env():
    import baseline.wind_gate as g
    import env.glider_env as e
    assert (g.SPEED_CMD_MIN_MS, g.SPEED_CMD_MAX_MS, g.SPEED_CMD_CENTRE_MS, g.SPEED_CMD_SCALE_MS) == \
        (e.SPEED_CMD_MIN_MS, e.SPEED_CMD_MAX_MS, e.SPEED_CMD_CENTRE_MS, e.SPEED_CMD_SCALE_MS)


def test_dev_and_benchmark_seeds_disjoint():
    from training.benchmark import episode_seeds
    with open(DEV_SEEDS) as f:
        doc = json.load(f)
    dev = doc["seeds"]
    assert dev == episode_seeds(200, 7) and len(set(dev)) == 200
    bench = set(episode_seeds(100, 0))
    selection = set(episode_seeds(10, 1000))      # in-training model selection (eval.seed)
    assert not set(dev) & bench
    assert not set(dev) & selection
    assert not bench & selection
    # No frozen held-out airframe seed set exists in the repo (SPEC V24 names
    # one; sim/jsbsim_fdm.py only mentions it), so there is nothing else to check.


def test_dev_seed_file_hash_stable():
    raw = open(DEV_SEEDS, "rb").read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == DEV_SEEDS_SHA256


def _synthetic_obs(gs, course, yaw):
    from env.nav_features import V1
    o = np.zeros(12)
    o[V1["dx_home"]], o[V1["dy_home"]] = 50.0, 0.0
    o[V1["ground_speed"]], o[V1["course"]], o[V1["yaw"]] = gs, course, yaw
    o[V1["baro_alt"]] = 20.0
    return o


@pytest.mark.parametrize("mode", ["single", "window"])
def test_handover_is_one_way(mode):
    """Strong apparent wind early, then none: once handed over, the policy is
    never asked again."""
    from baseline.wind_gated import WindGatedActor
    pol = _ZeroPolicy()
    actor = WindGatedActor(pol, threshold_ms=1.0, t_decide_s=6.0, mode=mode,
                           t_earliest_s=3.0, t_latest_s=10.0)
    actor.reset(0)
    calls_at_handover, flags = None, []
    for k in range(600):
        t = k * 0.05
        # Flying north at 12.75 m/s air, 5 m/s east wind for 8 s, then calm.
        we = 5.0 if t < 8.0 else 0.0
        vn, ve = 12.75, we
        actor.act(_synthetic_obs(math.hypot(vn, ve), math.atan2(ve, vn), 0.0))
        flags.append(actor.handed_over)
        if actor.handed_over and calls_at_handover is None:
            calls_at_handover = pol.calls
    assert actor.handed_over
    first = flags.index(True)
    assert all(flags[first:])                 # no step after the handover returns to the policy
    assert pol.calls == calls_at_handover
    rec = actor.episode_record()
    assert rec["handover_time_s"] <= (6.0 if mode == "single" else 8.0)


def test_activation_rate_logged():
    from baseline.wind_gated import WindGatedActor
    from training.benchmark import evaluate_seed, load_config
    cfg = load_config()
    for stage, seed in ((0, 5), (2, 4242)):
        actor = WindGatedActor.from_config(_ZeroPolicy(), cfg["gate"], "window")
        r = evaluate_seed(cfg, stage, actor, seed)
        g = r["gate"]
        assert set(g) >= {"mode", "handed_over", "handover_time_s",
                          "baseline_share_of_flight", "wind_est_at_decision_ms", "steps"}
        assert g["steps"] == r["steps"]
        assert (g["handover_time_s"] is None) == (not g["handed_over"])
        assert 0.0 <= g["baseline_share_of_flight"] <= 1.0

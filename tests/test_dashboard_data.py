"""
test_dashboard_data.py
======================
The dashboard's selected controller and its figures.

  1. frontend/src/data/validation.json is exactly what
     analysis/export_dashboard_data.py produces from the committed result
     files (no hand edits, no stale copy);
  2. the figures the Validation page relies on match the benchmark files;
  3. backend/recorder.py's "gated" controller is run 4's best policy behind
     the frozen window gate, records the gate's decision, and leaves the
     other controllers' recordings unchanged (gate = None).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "frontend", "src", "data", "validation.json")
RUN4 = os.path.join(ROOT, "results", "gated_v24_run4_benchmark.json")
RAW = os.path.join(ROOT, "results", "gated_v24_run4_benchmark_raw.json")
CKPT = os.path.join(ROOT, "checkpoints_v24_run4", "best_model.zip")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw benchmark file not present")
def test_validation_json_is_regenerated_output(tmp_path, monkeypatch):
    import analysis.export_dashboard_data as ex
    out = tmp_path / "validation.json"
    monkeypatch.setattr(ex, "OUT", str(out))
    ex.main()
    assert _load(str(out)) == _load(DATA), "validation.json is stale: run analysis/export_dashboard_data.py"


def test_dashboard_figures_match_benchmark():
    data = _load(DATA)
    bench = _load(RUN4)
    assert data["controller"]["gate"]["threshold_ms"] == 1.5
    assert (data["controller"]["gate"]["t_earliest_s"], data["controller"]["gate"]["t_latest_s"]) == (8.0, 15.0)
    for s in data["stages"]:
        b = bench["stages"][str(s["stage"])]
        assert s["gated"]["success"] == round(b["window"]["buckets"]["success"]["rate"], 4)
        assert s["gated"]["crash"] == round(b["window"]["buckets"]["crash"]["rate"], 4)
        assert s["baseline"]["success"] == round(b["baseline"]["buckets"]["success"]["rate"], 4)
        assert s["run4_alone"]["success"] == round(b["v22_alone"]["buckets"]["success"]["rate"], 4)
        assert s["paired_success"]["p"] == pytest.approx(b["window"]["mcnemar_exact_p"], rel=1e-3)
        t = s["crash_timing"]
        assert t["after_handover"] + t["policy_only_after_window_opened"] + t["before_window_opened"] == t["total"]
    significant = [s["stage"] for s in data["stages"] if s["paired_success"]["p"] < 0.05
                   and s["gated"]["success"] > s["baseline"]["success"]]
    assert significant == [0]


@pytest.mark.skipif(not os.path.exists(CKPT), reason="run 4 checkpoint not present")
def test_gated_recording_uses_run4_and_frozen_gate(tmp_path, monkeypatch):
    from backend import config, recorder
    from training.benchmark import load_config
    monkeypatch.setattr(config, "EPISODES_DIR", tmp_path)
    monkeypatch.setattr(recorder, "_gated_actor", None)

    actor = recorder._load_gated_actor()
    g = load_config()["gate"]["run4"]["window"]
    assert actor.gate.mode == "window"
    assert actor.gate.threshold_ms == g["threshold_ms"]
    assert (actor.gate.t_earliest_s, actor.gate.t_latest_s) == (g["t_earliest_s"], g["t_latest_s"])

    traj = recorder.record_episode(controller="gated", stage=2, seed=31)
    assert traj.meta.controller == "gated"
    assert traj.meta.gate is not None and traj.meta.gate.mode == "window"
    if traj.meta.gate.handed_over:
        assert g["t_earliest_s"] <= traj.meta.gate.handover_time_s <= g["t_latest_s"]
        assert traj.meta.gate.wind_est_at_decision_ms > g["threshold_ms"]
    assert (tmp_path / f"{traj.episode_id}.json").exists()

    base = recorder.record_episode(controller="baseline", stage=0, seed=11)
    assert base.meta.gate is None


def test_unknown_controller_rejected():
    from backend import recorder
    with pytest.raises(ValueError):
        recorder.record_episode(controller="v22", stage=0, seed=1)

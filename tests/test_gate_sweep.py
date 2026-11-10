"""
test_gate_sweep.py
==================
SPEC V20 Phase 5 — physics gates at the domain-randomisation corners.

The full-duration sweeps are run by `python validate_glide.py --sweep` and
`--random 300` and committed as analysis/gate_sweep_*.json. These tests run
every sweep config at shortened durations (crashes, NaN), the nominal config
at full duration (against the Phase 2 reference), and check the committed
results are current and passing.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import validate_glide as vg  # noqa: E402

QUICK = 0.05   # duration scale for the crash/NaN pass; gates are defined at 1.0


@pytest.fixture(scope="module")
def configs():
    return vg.sweep_configs()


@pytest.fixture(scope="module")
def quick_results(configs):
    return [vg.run_gates(af, scale=QUICK, mode="elevator") for af in configs]


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def test_sweep_covers_spec(configs):
    """nominal, every single-parameter extreme, the two specified corners,
    and the ixz validation variants."""
    from env.curriculum import DR_ONESIDED, DR_RANGES
    names = {c.name for c in configs}
    assert "nominal" in names
    for key in list(DR_RANGES) + list(DR_ONESIDED):
        assert {f"{key}_lo", f"{key}_hi"} <= names, key
    assert {"corner_heaviest_draggiest", "corner_lightest_least_stable", "ixz_lo", "ixz_hi"} <= names
    assert len(names) == len(configs)


def test_sweep_runs_all_configs(configs, quick_results):
    assert len(quick_results) == len(configs)
    for res in quick_results:
        assert res["trim"] is not None, f"{res['name']}: no usable trim"
        assert set(res["gates"]) == {n for n, _ in vg.GATES}, res["name"]


def test_no_config_produces_nan(quick_results):
    for res in quick_results:
        for name, gate in res["gates"].items():
            assert gate["value"] is not None, f"{res['name']}.{name}: {gate['detail']}"
            vals = list(_numbers(gate["value"]))
            assert vals and all(math.isfinite(v) for v in vals), f"{res['name']}.{name}"
        assert all(math.isfinite(v) for v in _numbers(res["trim"])), res["name"]


def test_nominal_matches_baseline():
    """The nominal airframe (aircraft as modelled, YAML-nominal mass) at full
    duration reproduces the Phase 2 reference (the bare generated file)
    within 1% wherever the quantity is mass-independent: glide ratio and
    the lift forces at the stall check angles. Energy drift and pitch rate
    are near zero in both, where a relative comparison is meaningless, so
    both must sit far inside their thresholds instead."""
    with open(os.path.join(PROJ_ROOT, "analysis", "baseline_v20_defaults.json"), encoding="utf-8") as f:
        ref = json.load(f)["gate_values"]
    res = vg.run_gates(vg.Airframe("nominal", vg.nominal_params()), mode="elevator")
    g = res["gates"]
    assert res["passed"]
    assert g["glide_ratio"]["value"] == pytest.approx(ref["gate2_glide_ratio_median"], rel=0.01)
    for (alpha, cl), ref_cl in zip(g["stall"]["value"].items(), ref["gate3_clalpha_force_lbf"].values()):
        assert cl == pytest.approx(ref_cl, rel=0.01), alpha
    assert g["energy_conservation"]["value"] < 0.1 * vg.ENERGY_TOL
    assert ref["gate1_energy_drift_frac"] < 0.1 * vg.ENERGY_TOL
    assert g["trimmed_stability"]["value"] < 0.1 * vg.PITCH_RATE_LIMIT
    assert ref["gate4_max_abs_q_rad_s"] < 0.1 * vg.PITCH_RATE_LIMIT
    assert res["trim"]["elevator_cmd"] == pytest.approx(0.0, abs=1e-6)


def test_elevator_trim_separates_mistrim_from_instability(configs):
    """CG 13 mm aft alone fails the zero-elevator gates (its fixed-control
    trim sits at the stall) but passes trimmed for speed with a small
    elevator command: mis-trimmed, not uncontrollable."""
    cg_aft = next(c for c in configs if c.name == "cg_x_m_lo")
    fixed = vg.run_gates(cg_aft, mode="fixed")
    trimmed = vg.run_gates(cg_aft, mode="elevator")
    assert not fixed["passed"]
    assert trimmed["passed"]
    assert abs(trimmed["trim"]["elevator_cmd"]) < 0.15


def _committed(name):
    with open(os.path.join(PROJ_ROOT, "analysis", name), encoding="utf-8") as f:
        return json.load(f)


def test_committed_sweep_is_current_and_passes(configs):
    """THE TRAINING GATE: every sweep configuration passes all five gates,
    and the committed sweep was run on the current configurations."""
    rec = _committed("gate_sweep_elevator.json")
    assert rec["trim_mode"] == "elevator"
    assert rec["n_passed"] == rec["n_configs"] == len(configs)
    current = {c.name: c.params for c in configs}
    for res in rec["configs"]:
        assert res["passed"], res["name"]
        assert res["params"] == pytest.approx(current[res["name"]], rel=1e-12), res["name"]


def test_committed_random_sweep_passes():
    """Joint combinations: the committed random full-range sample is current
    (same seed reproduces the same airframes) and every airframe passes."""
    rec = _committed("gate_sweep_random.json")
    assert rec["random"]["n"] >= 300
    regenerated = vg.random_configs(rec["random"]["n"], rec["random"]["seed"])
    assert [r["name"] for r in rec["configs"]] == [a.name for a in regenerated]
    for res, af in zip(rec["configs"], regenerated):
        assert res["params"] == pytest.approx(af.params, rel=1e-12), res["name"]
        assert res["passed"], res["name"]

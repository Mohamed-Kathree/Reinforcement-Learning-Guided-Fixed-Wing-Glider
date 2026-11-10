"""
test_dr_hooks.py
================
SPEC V20 Phase 3 — domain-randomisation hooks in the aircraft file.

The refactor turns hard-coded literals into properties at their default
values. It must change no physics at all, so the invariance tests compare
trajectories to 1e-9, not 1%.

Reference points:
    analysis/baseline_v20_defaults.json
        near-trim snapshots captured at the end of Phase 2
    tests/data/historical/rlglider_v20_phase2.xml
        the generated aircraft file before this refactor (frozen copy of
        aircraft/rlglider/rlglider.xml at f49969d), flown side by side with
        the current one through control doublets and sideslip, so the
        control and lateral terms that a trimmed glide never exercises are
        compared as well.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import jsbsim
import numpy as np
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from analysis.capture_baseline import trim_snapshot  # noqa: E402
from sim.jsbsim_fdm import DR_PROPERTIES  # noqa: E402
from sim.mass_properties import nominal  # noqa: E402
from sim.mass_solver import MassPropertySolver  # noqa: E402

DT = 0.005
HISTORICAL_XML = os.path.join(PROJ_ROOT, "tests", "data", "historical", "rlglider_v20_phase2.xml")
BASELINE_JSON = os.path.join(PROJ_ROOT, "analysis", "baseline_v20_defaults.json")

# Default each property must load with (the literal it replaced, or neutral).
DEFAULTS = {
    "aero/k-induced":     0.03566,
    "aero/clmax-scale":   1.0,
    "aero/cm0-offset":    0.0,
    "aero/cmalpha-mult":  1.0,
    "aero/cmq-mult":      1.0,
    "aero/elev-eff-mult":     1.0,
    "aero/clp-mult":      1.0,
    "aero/ail-eff-mult":     1.0,
    "aero/clbeta-mult":   1.0,
    "aero/cnbeta-mult":   1.0,
    "aero/cnr-mult":      1.0,
    "aero/rud-eff-mult":     1.0,
    "fcs/trim-bias-roll":  0.0,
    "fcs/trim-bias-pitch": 0.0,
}

STATE_PROPS = (
    "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
    "position/h-agl-ft", "position/lat-gc-deg", "position/long-gc-deg",
    "fcs/aileron-pos-rad", "fcs/elevator-pos-rad", "fcs/rudder-pos-rad",
)


def _fdm(root: str = PROJ_ROOT):
    fdm = jsbsim.FGFDMExec(root_dir=root)
    fdm.set_debug_level(0)
    fdm.set_dt(DT)
    assert fdm.load_model("rlglider")
    return fdm


def _set_ic(fdm, *, vt_ms=12.4, alpha_deg=5.0, beta_deg=3.0, p=0.3, q=0.2, r=0.25):
    fdm["ic/h-agl-ft"] = 300.0 * 3.280839895
    fdm["ic/vt-fps"] = vt_ms * 3.280839895
    fdm["ic/alpha-deg"] = alpha_deg
    fdm["ic/beta-deg"] = beta_deg
    fdm["ic/p-rad_sec"] = p
    fdm["ic/q-rad_sec"] = q
    fdm["ic/r-rad_sec"] = r
    fdm.run_ic()


# ---------------------------------------------------------------------------

def test_all_properties_exist():
    fdm = _fdm()
    for prop, default in DEFAULTS.items():
        assert fdm[prop] == default, prop
    for prop in DR_PROPERTIES:
        fdm[prop]   # raises if missing


# ---------------------------------------------------------------------------
# Invariance (1e-9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["trim_snapshot", "trim_snapshot_nominal_mass"])
def test_defaults_reproduce_baseline(which):
    """Near-trim states at 0, 1, 2, 5, 10 s match baseline_v20_defaults.json."""
    with open(BASELINE_JSON, encoding="utf-8") as f:
        ref = json.load(f)[which]
    got = trim_snapshot(nominal_mass=(which == "trim_snapshot_nominal_mass"))
    for t, state in got.items():
        for k, v in state.items():
            assert v == pytest.approx(ref[t][k], rel=1e-9, abs=1e-12), f"{t}.{k}"


def _doublet(step: int) -> tuple[float, float, float]:
    """Aileron, elevator, rudder (norm) doublets, staggered over 6 s."""
    t = step * DT
    a = 0.4 if 0.5 <= t < 1.0 else (-0.4 if 1.0 <= t < 1.5 else 0.0)
    e = 0.3 if 2.0 <= t < 2.5 else (-0.3 if 2.5 <= t < 3.0 else 0.0)
    r = 0.5 if 3.5 <= t < 4.0 else (-0.5 if 4.0 <= t < 4.5 else 0.0)
    return a, e, r


def _fly(fdm, n_steps: int) -> np.ndarray:
    out = np.empty((n_steps, len(STATE_PROPS)))
    for i in range(n_steps):
        a, e, r = _doublet(i)
        fdm["fcs/aileron-cmd-norm"] = a
        fdm["fcs/elevator-cmd-norm"] = e
        fdm["fcs/rudder-cmd-norm"] = r
        assert fdm.run()
        out[i] = [fdm[p] for p in STATE_PROPS]
    return out


@pytest.mark.parametrize("control_eff", [1.0, 0.8])
@pytest.mark.parametrize("nominal_mass", [False, True])
def test_refactor_matches_pre_refactor_file(tmp_path, control_eff, nominal_mass):
    """Old and new aircraft files, same mass settings, same inputs: identical
    6 s trajectories with sideslip, body rates and all three surfaces moving.

    control_eff = 0.8 also checks the retired single control multiplier maps
    onto the per-surface ones: old file with it at 0.8, new file with
    clda/cmde/cndr-mult all at 0.8.
    """
    old_root = tmp_path / "old"
    (old_root / "aircraft" / "rlglider").mkdir(parents=True)
    shutil.copy(HISTORICAL_XML, old_root / "aircraft" / "rlglider" / "rlglider.xml")

    old, new = _fdm(str(old_root)), _fdm()
    nom = nominal()
    for fdm in (old, new):
        _set_ic(fdm)
        if nominal_mass:
            MassPropertySolver().solve(fdm, nom["mass_kg"], 0.004, nom["ixx"], nom["iyy"])
    old["aero/" + "ctrl" + "-eff-mult"] = control_eff   # retired name; old file only
    for prop in ("aero/ail-eff-mult", "aero/elev-eff-mult", "aero/rud-eff-mult"):
        new[prop] = control_eff

    a, b = _fly(old, 1200), _fly(new, 1200)
    assert np.all(np.isfinite(a))
    scale = np.maximum(1.0, np.abs(a))
    worst = np.max(np.abs(a - b) / scale)
    assert worst <= 1e-9, f"max scaled state difference {worst:.3e}"


# ---------------------------------------------------------------------------
# Each hook does what it says
# ---------------------------------------------------------------------------

AERO_FUNCS = (
    "aero/coefficient/CLalpha", "aero/coefficient/CLde", "aero/coefficient/CLtotal",
    "aero/coefficient/CD0", "aero/coefficient/CDi", "aero/coefficient/CDbeta",
    "aero/coefficient/CYbeta",
    "aero/moment/Clbeta", "aero/moment/Clp", "aero/moment/Clr", "aero/moment/Clda",
    "aero/moment/Cldr", "aero/moment/Cm0", "aero/moment/Cmalpha", "aero/moment/Cmq",
    "aero/moment/Cmde", "aero/moment/Cnbeta", "aero/moment/Cnp", "aero/moment/Cnr",
    "aero/moment/Cndr", "aero/moment/Cnda",
)

# property, value, alpha (deg), {function: expected ratio new/default or None}.
# None = must change, in the direction given by `increase` below. Every
# function NOT listed must be unchanged, which catches a hook wired to the
# wrong term.
CASES = [
    ("aero/k-induced",    0.03566 * 1.5, 5.0, {"aero/coefficient/CDi": 1.5}),
    ("aero/cm0-offset",   0.05,          5.0, {"aero/moment/Cm0": 2.0}),
    ("aero/cmalpha-mult", 1.5,           5.0, {"aero/moment/Cmalpha": 1.5}),
    ("aero/cmq-mult",     1.5,           5.0, {"aero/moment/Cmq": 1.5}),
    ("aero/elev-eff-mult",    1.5,           5.0, {"aero/moment/Cmde": 1.5, "aero/coefficient/CLde": 1.5,
                                               "aero/coefficient/CLtotal": None, "aero/coefficient/CDi": None}),
    ("aero/clp-mult",     1.5,           5.0, {"aero/moment/Clp": 1.5}),
    ("aero/ail-eff-mult",    1.5,           5.0, {"aero/moment/Clda": 1.5, "aero/moment/Cnda": 1.5}),
    ("aero/clbeta-mult",  1.5,           5.0, {"aero/moment/Clbeta": 1.5}),
    ("aero/cnbeta-mult",  1.5,           5.0, {"aero/moment/Cnbeta": 1.5}),
    ("aero/cnr-mult",     1.5,           5.0, {"aero/moment/Cnr": 1.5}),
    ("aero/rud-eff-mult",    1.5,           5.0, {"aero/moment/Cndr": 1.5, "aero/moment/Cldr": 1.5}),
    # Past the 12 deg stall: a higher CL peak means more lift at 14 deg.
    ("aero/clmax-scale",  1.5,          14.0, {"aero/coefficient/CLalpha": None,
                                               "aero/coefficient/CLtotal": None, "aero/coefficient/CDi": None}),
]


def _lockstep(prop, value, alpha_deg):
    """Two identical FDMs, surfaces deflected; then one step with `prop` changed
    on one of them. JSBSim evaluates aerodynamics from the pre-integration
    state, so after that step the function values differ only through `prop`."""
    fdms = [_fdm(), _fdm()]
    for fdm in fdms:
        _set_ic(fdm, alpha_deg=alpha_deg)
        for _ in range(10):
            fdm["fcs/aileron-cmd-norm"] = 0.5
            fdm["fcs/elevator-cmd-norm"] = 0.5
            fdm["fcs/rudder-cmd-norm"] = 0.5
            fdm.run()
    fdms[1][prop] = value
    for fdm in fdms:
        fdm.run()
    return ({f: fdms[0][f] for f in AERO_FUNCS}, {f: fdms[1][f] for f in AERO_FUNCS})


@pytest.mark.parametrize("prop, value, alpha_deg, expected", CASES, ids=[c[0] for c in CASES])
def test_each_multiplier_has_effect(prop, value, alpha_deg, expected):
    base, mod = _lockstep(prop, value, alpha_deg)
    for func in AERO_FUNCS:
        b, m = base[func], mod[func]
        if func not in expected:
            assert m == b, f"{prop} changed {func}: {b} -> {m}"
        elif expected[func] is None:
            assert m != b, f"{prop} did not change {func}"
        else:
            assert b != 0.0, f"{func} is zero at the test condition; ratio meaningless"
            assert m / b == pytest.approx(expected[func], rel=1e-9), func

    if prop == "aero/clmax-scale":
        assert mod["aero/coefficient/CLalpha"] > base["aero/coefficient/CLalpha"]


def test_clmax_scale_keeps_pre_stall_slope():
    """At 5 deg (pre-stall) clmax-scale must leave lift essentially unchanged.

    Not exact: the table is only piecewise-linear (slope 0.0960/deg on 0-4 deg,
    0.09598/deg on 4-8 deg), and alpha / s moves the lookup across the 4 deg
    breakpoint, a 3e-5 relative difference. A plain multiplier would be off by
    15% here."""
    base, mod = _lockstep("aero/clmax-scale", 1.15, 5.0)
    assert mod["aero/coefficient/CLalpha"] == pytest.approx(base["aero/coefficient/CLalpha"], rel=1e-4)


# ---------------------------------------------------------------------------
# Trim bias
# ---------------------------------------------------------------------------

def test_trim_bias_deflects_surface():
    """A roll bias with zero command deflects the aileron through the actuator
    lag and produces a steady positive (right-wing-down) roll rate."""
    fdm = _fdm()
    _set_ic(fdm, alpha_deg=4.78, beta_deg=0.0, p=0.0, q=0.0, r=0.0)
    bias = 0.04
    fdm["fcs/trim-bias-roll"] = bias
    target = bias * 0.4363

    p_hist, da_hist = [], []
    for _ in range(int(3.0 / DT)):
        fdm["fcs/aileron-cmd-norm"] = 0.0
        fdm.run()
        p_hist.append(fdm["velocities/p-rad_sec"])
        da_hist.append(fdm["fcs/aileron-pos-rad"])

    # Through the lag, not a step: first sample well short of the target.
    assert 0.0 < da_hist[0] < 0.5 * target
    assert da_hist[-1] == pytest.approx(target, rel=1e-3)

    # Steady roll rate: positive throughout the last 2 s and nearly constant.
    p_late = np.array(p_hist[int(1.0 / DT):])
    assert np.all(p_late > 0)
    assert p_late.std() < 0.15 * p_late.mean()

    # Pitch bias reaches the elevator the same way.
    fdm["fcs/trim-bias-pitch"] = -0.03
    for _ in range(int(1.0 / DT)):
        fdm.run()
    assert fdm["fcs/elevator-pos-rad"] == pytest.approx(-0.03 * 0.3491, rel=1e-3)


# ---------------------------------------------------------------------------
# Retired property
# ---------------------------------------------------------------------------

# Historical material that legitimately records the retired name, plus this
# file, whose test name is fixed by the spec.
HISTORICAL_PREFIXES = ("runs_archive_", "checkpoints_archive_", "tests/data/historical/",
                       "tests/test_dr_hooks.py")


def test_ctrl_eff_mult_removed():
    needles = ("ctrl" + "-eff-" + "mult", "ctrl" + "_eff_" + "mult")
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=PROJ_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8").split("\0")
    hits = []
    for rel in filter(None, files):
        if rel.startswith(HISTORICAL_PREFIXES):
            continue
        path = os.path.join(PROJ_ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore").lower()
        hits += [f"{rel}: {n}" for n in needles if n in text]
    assert not hits, "retired control-effectiveness multiplier still referenced:\n" + "\n".join(hits)


def test_cl_mult_scales_slope_not_peak():
    """cl_mult sets the lift-curve slope only (SPEC V20 Phase 4): pre-stall CL
    scales with it, CLmax does not. A whole-table multiplier would also move
    the peak and compound with clmax-scale at the stall corner."""
    def cl_curve(cl_mult):
        fdm = _fdm()
        _set_ic(fdm, beta_deg=0.0, p=0.0, q=0.0, r=0.0)
        fdm["aero/cl-mult"] = cl_mult
        out = []
        for a in np.arange(1.0, 24.0, 0.05):
            fdm["ic/alpha-deg"] = a
            fdm.run_ic()
            fdm.run()
            # CL and the alpha JSBSim actually evaluated it at (one step on
            # from the IC, so not exactly `a`, and not the same in both runs).
            out.append((fdm["aero/alpha-deg"], fdm["aero/coefficient/CLtotal"]))
        return np.array(out)

    base, steep = cl_curve(1.0), cl_curve(1.2)
    # ~3 deg: both on the table's 0-4 deg segment, so the slope ratio is the multiplier.
    i3 = int(round((3.0 - 1.0) / 0.05))
    slope = lambda row: row[1] / row[0]
    assert slope(steep[i3]) / slope(base[i3]) == pytest.approx(1.2, rel=1e-3)
    assert steep[:, 1].max() == pytest.approx(base[:, 1].max(), rel=2e-3)

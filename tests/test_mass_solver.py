"""
test_mass_solver.py
===================
SPEC V20 Phase 2 — point-mass solver for per-episode mass, CG and inertia.
"""

from __future__ import annotations

import os
import sys

import jsbsim
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from sim.mass_solver import (  # noqa: E402
    CACHE_QUANTUM, CG_TOL_M, MAX_POINT_MASS_KG, REL_TOL,
    MassPropertySolver, UnreachableTarget, check_point_mass_layout, read_mass_properties,
)

# (mass kg, CG mm forward, Ixx, Iyy) -> Izz = Ixx + Iyy
CORNERS = [
    (1.50,   0, 0.14, 0.10),
    (1.64, +15, 0.21, 0.16),
    (1.80, -15, 0.32, 0.26),
    (1.64,   0, 0.16, 0.12),
]


def _fdm():
    fdm = jsbsim.FGFDMExec(root_dir=PROJ_ROOT)
    fdm.set_debug_level(0)
    fdm.set_dt(0.005)
    assert fdm.load_model("rlglider")
    fdm["ic/h-agl-ft"] = 300.0
    fdm["ic/vt-fps"] = 39.0
    fdm.run_ic()
    return fdm


@pytest.fixture(scope="module")
def fdm():
    return _fdm()


def test_point_mass_layout(fdm):
    check_point_mass_layout(fdm)


@pytest.mark.parametrize("mass, cg_mm, ixx, iyy", CORNERS)
def test_solver_hits_targets(fdm, mass, cg_mm, ixx, iyy):
    sol = MassPropertySolver().solve(fdm, mass, cg_mm / 1e3, ixx, iyy)
    a = read_mass_properties(fdm)   # readback after solve, not the returned copy
    assert a["mass_kg"] == pytest.approx(mass, rel=REL_TOL)
    assert a["ixx"] == pytest.approx(ixx, rel=REL_TOL)
    assert a["iyy"] == pytest.approx(iyy, rel=REL_TOL)
    assert abs(a["cg_x_m"] - cg_mm / 1e3) <= CG_TOL_M
    assert sol["achieved"] == pytest.approx(a)


def test_cg_sign_convention(fdm):
    """cg_x_m is body frame, positive forward; JSBSim's cg-x-in is structural,
    positive aft. A forward target must move JSBSim's CG to NEGATIVE x."""
    MassPropertySolver().solve(fdm, 1.64, +0.015, 0.21, 0.16)
    assert fdm["inertia/cg-x-in"] < 0
    MassPropertySolver().solve(fdm, 1.64, -0.015, 0.21, 0.16)
    assert fdm["inertia/cg-x-in"] > 0


@pytest.mark.parametrize("mass, cg_mm, ixx, iyy", CORNERS)
def test_izz_follows_flat_plate(fdm, mass, cg_mm, ixx, iyy):
    a = MassPropertySolver().solve(fdm, mass, cg_mm / 1e3, ixx, iyy)["achieved"]
    assert a["izz"] == pytest.approx(a["ixx"] + a["iyy"], rel=0.03)
    assert a["izz"] == pytest.approx(ixx + iyy, rel=0.03)
    # Rigid-body triangle inequality: never exceeded.
    assert a["izz"] <= (a["ixx"] + a["iyy"]) * (1 + 1e-6)


def test_solver_deterministic(fdm):
    target = (1.71, 0.006, 0.27, 0.19)
    a = MassPropertySolver().solve(fdm, *target)
    b = MassPropertySolver().solve(fdm, *target)
    for k in ("ballast_kg", "ballast_x_m", "pair_y_kg", "pair_x_kg"):
        assert a[k] == b[k], k


def test_cache_returns_same_solution(fdm):
    solver = MassPropertySolver()
    target = (1.62, -0.004, 0.19, 0.15)
    first = solver.solve(fdm, *target)
    assert (solver.cache_hits, solver.cache_misses) == (0, 1)

    nudged = tuple(v + 0.2 * CACHE_QUANTUM for v in target)
    hit = solver.solve(fdm, *nudged)
    assert (solver.cache_hits, solver.cache_misses) == (1, 1)

    fresh = MassPropertySolver().solve(fdm, *nudged)
    for k in ("ballast_kg", "ballast_x_m", "pair_y_kg", "pair_x_kg"):
        assert hit[k] == first[k], k
        assert hit[k] == pytest.approx(fresh[k], abs=0.02), k

    # The cached weights, re-applied, still meet the nudged target.
    solver.apply(fdm, hit)
    a = read_mass_properties(fdm)
    for name, idx in (("mass_kg", 0), ("ixx", 2), ("iyy", 3)):
        assert a[name] == pytest.approx(nudged[idx], rel=REL_TOL), name
    assert abs(a["cg_x_m"] - nudged[1]) <= CG_TOL_M + CACHE_QUANTUM


@pytest.mark.parametrize("target, why", [
    ((1.64, 0.0, 0.21, 5.0),   "Iyy far beyond the 1.5 kg pair limit"),
    ((1.64, 0.0, 0.10, 0.16),  "Ixx below the baked-in base; pairs only add"),
    ((3.50, 0.0, 0.14, 0.10),  "needs +2.0 kg ballast, beyond the 1.5 kg guard"),
    ((1.64, 0.5, 0.21, 0.16),  "CG 500 mm forward, beyond ballast authority"),
    ((-0.5, 0.0, 0.14, 0.10),  "non-positive total mass"),
])
def test_solver_raises_on_unreachable(fdm, target, why):
    with pytest.raises(UnreachableTarget):
        MassPropertySolver().solve(fdm, *target)


def test_point_mass_guard_bounds(fdm):
    """No accepted solution uses a point mass beyond the conditioning guard."""
    for mass, cg_mm, ixx, iyy in CORNERS:
        sol = MassPropertySolver().solve(fdm, mass, cg_mm / 1e3, ixx, iyy)
        for k in ("ballast_kg", "pair_y_kg", "pair_x_kg"):
            assert abs(sol[k]) <= MAX_POINT_MASS_KG

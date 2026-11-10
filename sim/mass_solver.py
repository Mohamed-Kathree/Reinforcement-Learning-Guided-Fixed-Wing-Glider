"""
mass_solver.py
==============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Per-episode mass, CG and inertia targets, reached through JSBSim point masses.

JSBSim's inertia/i??-slugs_ft2 properties are read-only outputs (verified on
JSBSim 1.3.1: writes are silently ignored), so the only runtime handle on
inertia is point masses. aircraft/rlglider/rlglider.xml bakes in the BOTTOM of
each mass_properties.yaml inertia range and declares five point masses:

    index  name     location (structural frame, X aft)   freedom
    0      BALLAST  x variable, y = 0                    total mass, CG x
    1      IN_Y_R   y = +0.9 m                           \\ equal weights:
    2      IN_Y_L   y = -0.9 m                           /  ixx (+ izz)
    3      IN_X_F   x = -0.5 m (forward)                 \\ equal weights:
    4      IN_X_A   x = +0.5 m (aft)                     /  iyy (+ izz)

Izz is not a target: every freedom adds equally to izz and to ixx or iyy, so
izz = ixx + iyy follows (the flat-plate upper bound; see mass_properties.yaml).

Sign conventions:
    cg_x_m is BODY frame, positive FORWARD of the reference CG, matching
    GliderEnv's cg_offset_m. JSBSim point-mass locations and inertia/cg-x-in
    are STRUCTURAL frame, positive AFT, so every x crossing that boundary is
    negated here and nowhere else.

Ballast weight is signed. At heavy, high-inertia targets the inertia pairs
alone exceed the mass budget and the ballast goes negative (about -0.56 kg at
1.80 kg / Ixx 0.32 / Iyy 0.26). That is the deliberate mechanism documented
in the aircraft file; the guards in solve() reject any solution that is
unphysical (non-positive inertia or mass) or ill-conditioned (|w| > 1.5 kg).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

LBS_PER_KG       = 2.2046226218
IN_PER_M         = 39.37007874
KG_PER_SLUG      = 14.5939029
KGM2_PER_SLUGFT2 = 1.35581795

BALLAST, IN_Y_R, IN_Y_L, IN_X_F, IN_X_A = range(5)

MAX_POINT_MASS_KG = 1.5     # conditioning guard on every point mass
MAX_BALLAST_X_M   = 1.0     # ballast travel either side of the reference CG
CG_SCALE_M        = 0.015   # residual scale for CG: the YAML residual half-width

# Acceptance: 1% relative on mass and inertia; CG absolute, 1% of CG_SCALE_M,
# since a relative tolerance is meaningless at a CG target of zero.
REL_TOL   = 0.01
CG_TOL_M  = 0.01 * CG_SCALE_M

# Cache key quantum on each target (kg, m, kg m^2).
CACHE_QUANTUM = 1e-3


class UnreachableTarget(ValueError):
    """The requested mass properties cannot be reached within the guards."""


def _wprop(i: int) -> str:
    return f"inertia/pointmass-weight-lbs[{i}]"


def _xprop(i: int) -> str:
    return f"inertia/pointmass-location-X-inches[{i}]"


def _yprop(i: int) -> str:
    return f"inertia/pointmass-location-Y-inches[{i}]"


def read_mass_properties(fdm) -> dict:
    """JSBSim's own mass properties for the current point masses, SI units."""
    return {
        "mass_kg": fdm["inertia/mass-slugs"] * KG_PER_SLUG,
        "cg_x_m":  -fdm["inertia/cg-x-in"] / IN_PER_M,      # structural -> body
        "ixx":     fdm["inertia/ixx-slugs_ft2"] * KGM2_PER_SLUGFT2,
        "iyy":     fdm["inertia/iyy-slugs_ft2"] * KGM2_PER_SLUGFT2,
        "izz":     fdm["inertia/izz-slugs_ft2"] * KGM2_PER_SLUGFT2,
    }


def check_point_mass_layout(fdm) -> None:
    """Fail loudly if the aircraft file's point masses are not the expected
    order and geometry (JSBSim exposes them by index only)."""
    y_r, y_l = fdm[_yprop(IN_Y_R)], fdm[_yprop(IN_Y_L)]
    x_f, x_a = fdm[_xprop(IN_X_F)], fdm[_xprop(IN_X_A)]
    ok = (
        y_r > 0 and np.isclose(y_l, -y_r)
        and x_f < 0 and np.isclose(x_a, -x_f)
        and fdm[_yprop(BALLAST)] == 0.0
        and fdm[_xprop(IN_Y_R)] == 0.0 and fdm[_xprop(IN_Y_L)] == 0.0
        and fdm[_yprop(IN_X_F)] == 0.0 and fdm[_yprop(IN_X_A)] == 0.0
    )
    if not ok:
        raise RuntimeError(
            "rlglider.xml point masses are not BALLAST, IN_Y_R, IN_Y_L, IN_X_F, "
            "IN_X_A in that order with symmetric locations; see aircraft/rlglider.template.xml"
        )


class MassPropertySolver:
    """Solve the four point-mass freedoms for a target mass, CG and inertia.

    Unknowns: BALLAST weight, BALLAST x-position, y-pair weight, x-pair weight.
    Targets:  total mass, CG x, Ixx, Iyy.  Izz follows as Ixx + Iyy.

    Solves against JSBSim's own reported mass properties via
    scipy.optimize.least_squares, rather than a reimplemented parallel-axis
    calculation, so the solver cannot drift from what the simulator integrates.
    The parallel-axis estimate is used only as the starting point.

    Note: solve() and apply() call fdm.run_ic(), which re-initialises the
    state from the ic/* properties. Set the episode's initial conditions
    first; the extra run_ic() then leaves the state unchanged.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------

    @staticmethod
    def _key(mass_kg, cg_x_m, ixx, iyy) -> tuple:
        return tuple(int(round(v / CACHE_QUANTUM)) for v in (mass_kg, cg_x_m, ixx, iyy))

    @staticmethod
    def _write(fdm, u) -> None:
        w_b, x_b, w_y, w_x = (float(v) for v in u)
        fdm[_wprop(BALLAST)] = w_b * LBS_PER_KG
        fdm[_xprop(BALLAST)] = -x_b * IN_PER_M              # body -> structural
        for i, w in ((IN_Y_R, w_y), (IN_Y_L, w_y), (IN_X_F, w_x), (IN_X_A, w_x)):
            fdm[_wprop(i)] = w * LBS_PER_KG
        fdm.run_ic()

    # ------------------------------------------------------------------

    def solve(self, fdm, mass_kg: float, cg_x_m: float, ixx: float, iyy: float) -> dict:
        """Return point-mass weights reaching the targets, or raise UnreachableTarget.

        The returned dict holds the four unknowns plus `achieved` (JSBSim
        readback at the solution). Solutions are cached per target, quantised
        to CACHE_QUANTUM; a hit returns the solution of the first target that
        fell in that cell.
        """
        key = self._key(mass_kg, cg_x_m, ixx, iyy)
        if key in self._cache:
            self.cache_hits += 1
            return _copy(self._cache[key])
        self.cache_misses += 1

        check_point_mass_layout(fdm)
        target = np.array([mass_kg, cg_x_m, ixx, iyy], dtype=float)

        # Starting point from the parallel-axis relations, against the bare model.
        self._write(fdm, (0.0, 0.0, 0.0, 0.0))
        base = read_mass_properties(fdm)
        y_arm = fdm[_yprop(IN_Y_R)] / IN_PER_M
        x_arm = -fdm[_xprop(IN_X_F)] / IN_PER_M
        w_y0 = max((ixx - base["ixx"]) / (2 * y_arm**2), 0.0)
        w_x0 = max((iyy - base["iyy"]) / (2 * x_arm**2), 0.0)
        w_b0 = mass_kg - base["mass_kg"] - 2 * w_y0 - 2 * w_x0
        x_b0 = cg_x_m * mass_kg / w_b0 if abs(w_b0) > 1e-3 else 0.0

        lo = np.array([-MAX_POINT_MASS_KG, -MAX_BALLAST_X_M, 0.0, 0.0])
        hi = np.array([MAX_POINT_MASS_KG, MAX_BALLAST_X_M, MAX_POINT_MASS_KG, MAX_POINT_MASS_KG])
        span = hi - lo
        u0 = np.clip([w_b0, x_b0, w_y0, w_x0], lo + 1e-6 * span, hi - 1e-6 * span)
        scale = np.array([mass_kg, CG_SCALE_M, ixx, iyy])

        def residual(u):
            self._write(fdm, u)
            a = read_mass_properties(fdm)
            got = np.array([a["mass_kg"], a["cg_x_m"], a["ixx"], a["iyy"]])
            return (got - target) / scale

        res = least_squares(residual, u0, bounds=(lo, hi), method="trf",
                            x_scale=[0.1, 0.1, 0.1, 0.1],
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)

        sol = {
            "ballast_kg":  float(res.x[0]),
            "ballast_x_m": float(res.x[1]),
            "pair_y_kg":   float(res.x[2]),
            "pair_x_kg":   float(res.x[3]),
        }
        self.apply(fdm, sol)
        sol["achieved"] = read_mass_properties(fdm)
        self._check(sol, mass_kg, cg_x_m, ixx, iyy)

        self._cache[key] = _copy(sol)
        return sol

    def apply(self, fdm, weights: dict) -> None:
        """Write a solve() result to the point masses and re-run the IC."""
        self._write(fdm, (weights["ballast_kg"], weights["ballast_x_m"],
                          weights["pair_y_kg"], weights["pair_x_kg"]))

    # ------------------------------------------------------------------

    @staticmethod
    def _check(sol: dict, mass_kg, cg_x_m, ixx, iyy) -> None:
        a = sol["achieved"]
        problems = []
        for name in ("ballast_kg", "pair_y_kg", "pair_x_kg"):
            if abs(sol[name]) > MAX_POINT_MASS_KG:
                problems.append(f"|{name}| = {abs(sol[name]):.3f} kg > {MAX_POINT_MASS_KG} kg")
        if a["mass_kg"] <= 0:
            problems.append(f"achieved mass {a['mass_kg']:.4f} kg is not positive")
        for name in ("ixx", "iyy", "izz"):
            if a[name] <= 0:
                problems.append(f"achieved {name} {a[name]:.4f} kg m^2 is not positive")
        for name, tgt in (("mass_kg", mass_kg), ("ixx", ixx), ("iyy", iyy)):
            if abs(a[name] - tgt) > REL_TOL * abs(tgt):
                problems.append(f"{name} {a[name]:.4f} vs target {tgt:.4f} (> {REL_TOL:.0%})")
        if abs(a["cg_x_m"] - cg_x_m) > CG_TOL_M:
            problems.append(f"cg_x {a['cg_x_m'] * 1e3:.2f} mm vs target {cg_x_m * 1e3:.2f} mm")
        if problems:
            raise UnreachableTarget(
                f"mass properties (m={mass_kg}, cg={cg_x_m}, ixx={ixx}, iyy={iyy}) "
                "unreachable: " + "; ".join(problems)
            )


def _copy(sol: dict) -> dict:
    out = dict(sol)
    if "achieved" in out:
        out["achieved"] = dict(out["achieved"])
    return out

"""
mass_properties.py
==================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Loader for aircraft/mass_properties.yaml, the single source of truth for the
airframe's mass, balance, inertia and reference geometry. Every entry carries
a `value`, a `source` provenance tag and, where uncertain, a `range`.

Two derived views are used across the codebase:

    fdm_baseline()  -- what aircraft/generate_fdm.py bakes into rlglider.xml:
                       the BOTTOM of each mass/inertia range, because the
                       per-episode point masses can only add inertia.
    nominal()       -- the central estimate of the as-built aircraft.
"""

from __future__ import annotations

import os

import yaml

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASS_PROPERTIES_PATH = os.path.join(PROJ_ROOT, "aircraft", "mass_properties.yaml")

PROVENANCE_TAGS = ("measured", "derived_cad", "estimated", "design_target")


def load_mass_properties(path: str = MASS_PROPERTIES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_entries(mp: dict):
    """Yield (section, key, entry) for every leaf entry in the YAML."""
    for section, entries in mp.items():
        for key, entry in entries.items():
            yield section, key, entry


def _lo(entry: dict) -> float:
    return float(entry["range"][0]) if "range" in entry else float(entry["value"])


def fdm_baseline(mp: dict | None = None) -> dict:
    """Mass properties baked into the generated aircraft file (SI units).

    izz uses the factor's nominal value, not its low end: the point masses add
    equal amounts to izz and to ixx (y-pair) or iyy (x-pair), so izz stays on
    the flat-plate line izz = (ixx + iyy) * factor only if the baseline is on
    it too.
    """
    mp = mp or load_mass_properties()
    ixx = _lo(mp["inertia"]["ixx"])
    iyy = _lo(mp["inertia"]["iyy"])
    return {
        "mass_kg":      _lo(mp["mass"]["flying_kg"]),
        "ixx":          ixx,
        "iyy":          iyy,
        "izz":          (ixx + iyy) * float(mp["inertia"]["izz_factor"]["value"]),
        "ixz":          float(mp["inertia"]["ixz"]["value"]),
        "wing_area_m2": float(mp["geometry"]["wing_area_m2"]["value"]),
        "span_m":       float(mp["geometry"]["span_m"]["value"]),
        "chord_m":      float(mp["geometry"]["chord_mean_m"]["value"]),
    }


def nominal(mp: dict | None = None) -> dict:
    """Central estimate of the as-built aircraft (SI units)."""
    mp = mp or load_mass_properties()
    ixx = float(mp["inertia"]["ixx"]["value"])
    iyy = float(mp["inertia"]["iyy"]["value"])
    return {
        "mass_kg": float(mp["mass"]["flying_kg"]["value"]),
        "cg_x_m":  float(mp["cg"]["residual_m"]["value"]),
        "ixx":     ixx,
        "iyy":     iyy,
        "izz":     (ixx + iyy) * float(mp["inertia"]["izz_factor"]["value"]),
    }

"""
test_mass_properties.py
=======================
SPEC V20 Phase 1 — mass properties as data.

aircraft/mass_properties.yaml is the single source of truth; the aircraft
file is generated from it by aircraft/generate_fdm.py and must never drift.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import jsbsim
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from aircraft.generate_fdm import OUTPUT_PATH, render, strip_timestamp  # noqa: E402
from sim.mass_properties import (  # noqa: E402
    PROVENANCE_TAGS, fdm_baseline, iter_entries, load_mass_properties,
)

KG2SLUG    = 1.0 / 14.5939029
KGM2_2SLFT2 = 1.0 / 1.35581795


@pytest.fixture(scope="module")
def mp() -> dict:
    return load_mass_properties()


def test_yaml_schema_valid(mp):
    entries = list(iter_entries(mp))
    assert entries, "mass_properties.yaml has no entries"
    for section, key, entry in entries:
        name = f"{section}.{key}"
        assert "value" in entry, f"{name} has no value"
        assert "source" in entry, f"{name} has no source"
        assert entry["source"] in PROVENANCE_TAGS, f"{name} source {entry['source']!r} not in {PROVENANCE_TAGS}"
        if "range" in entry:
            lo, hi = entry["range"]
            assert lo <= entry["value"] <= hi, f"{name}: value {entry['value']} outside range [{lo}, {hi}]"
        if "dr_range" in entry:
            dlo, dhi = entry["dr_range"]
            assert entry.get("dr_reason"), f"{name}: dr_range without dr_reason"
            assert lo <= dlo <= entry["value"] <= dhi <= hi, f"{name}: dr_range {entry['dr_range']} not inside range"


def test_generated_xml_parses():
    ET.fromstring(render(timestamp="T").encode("utf-8"))
    fdm = jsbsim.FGFDMExec(root_dir=PROJ_ROOT)
    fdm.set_debug_level(0)
    assert fdm.load_model("rlglider"), "JSBSim failed to load the generated aircraft file"


def test_generated_xml_matches_yaml(mp):
    base = fdm_baseline(mp)
    assert base["mass_kg"] == mp["mass"]["flying_kg"]["range"][0]
    assert base["ixx"] == mp["inertia"]["ixx"]["range"][0]
    assert base["iyy"] == mp["inertia"]["iyy"]["range"][0]

    # In the file as written ...
    root = ET.parse(OUTPUT_PATH).getroot()
    mb = root.find("mass_balance")
    assert float(mb.find("emptywt").text) == pytest.approx(base["mass_kg"], rel=1e-12)
    for tag in ("ixx", "iyy", "izz", "ixz"):
        assert float(mb.find(tag).text) == pytest.approx(base[tag], rel=1e-9), tag

    # ... and as JSBSim actually loads it (no point-mass contribution yet).
    fdm = jsbsim.FGFDMExec(root_dir=PROJ_ROOT)
    fdm.set_debug_level(0)
    fdm.load_model("rlglider")
    fdm.run_ic()
    assert fdm["inertia/mass-slugs"] == pytest.approx(base["mass_kg"] * KG2SLUG, rel=1e-5)
    # JSBSim's own KG*M2 -> SLUG*FT2 constant differs from the exact one by
    # ~9e-5 relative, so this tolerance checks the value, not the conversion.
    for tag in ("ixx", "iyy", "izz"):
        got = fdm[f"inertia/{tag}-slugs_ft2"]
        assert got == pytest.approx(base[tag] * KGM2_2SLFT2, rel=1e-3), tag


def test_generation_is_deterministic():
    a = render(timestamp="2000-01-01T00:00:00+00:00")
    b = render(timestamp="2099-12-31T23:59:59+00:00")
    assert a != b, "timestamp line missing from the generated header"
    assert strip_timestamp(a) == strip_timestamp(b)
    assert strip_timestamp(render()) == strip_timestamp(render())


def test_no_handwritten_drift():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        committed = f.read()
    assert strip_timestamp(committed) == strip_timestamp(render()), (
        "aircraft/rlglider/rlglider.xml differs from what aircraft/generate_fdm.py "
        "produces. Edit mass_properties.yaml or rlglider.template.xml, never the XML, "
        "then run: python aircraft/generate_fdm.py"
    )


def test_ixz_variants_differ_only_in_ixz(mp):
    """The ixz validation variants (fixed at the YAML value in training; the
    sensitivity is measured on these instead) span the YAML range and change
    nothing but the ixz element and the header note."""
    from aircraft.generate_fdm import ixz_variant_values

    values = ixz_variant_values()
    assert [values["lo"], values["hi"]] == mp["inertia"]["ixz"]["range"]
    assert values["mid"] == mp["inertia"]["ixz"]["value"]

    base = strip_timestamp(render(timestamp="T")).splitlines()
    for label, value in values.items():
        var = strip_timestamp(render(timestamp="T", ixz=value)).splitlines()
        var = [ln for ln in var if "VALIDATION VARIANT" not in ln]
        diff = [(a, b) for a, b in zip(base, var) if a != b]
        assert len(var) == len(base), label
        if value == mp["inertia"]["ixz"]["value"]:
            assert diff == [], label
        else:
            assert len(diff) == 1 and "<ixz" in diff[0][0], (label, diff)
            root = ET.fromstring("\n".join(var).encode("utf-8"))
            assert float(root.find("mass_balance/ixz").text) == pytest.approx(value)

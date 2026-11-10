"""
generate_fdm.py
===============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Generates aircraft/rlglider/rlglider.xml from aircraft/rlglider.template.xml
and aircraft/mass_properties.yaml. The generated file is committed, but must
never be edited by hand: tests/test_mass_properties.py::test_no_handwritten_drift
regenerates it and fails on any difference.

Run:   python aircraft/generate_fdm.py                 (write if content changed)
       python aircraft/generate_fdm.py --check         (exit 1 if stale)
       python aircraft/generate_fdm.py --ixz-variants  (validation artefacts)

The output's header names the generator, the SHA-256 of the YAML and template
it was built from, and a generation timestamp. The timestamp line is the only
part allowed to differ between two generations from identical inputs; the file
is only rewritten when something other than that line changes.

ixz variants: ixz's two estimates disagree in sign, so it is fixed at its
YAML value (the range midpoint) and NOT randomised in training. Its
sensitivity is measured instead by flying aircraft files at the range low,
value and high: --ixz-variants writes those to VARIANTS_DIR (gitignored),
one JSBSim root per variant, for the Phase 5 gate sweep. They are never
loaded by the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from sim.mass_properties import MASS_PROPERTIES_PATH, fdm_baseline, load_mass_properties  # noqa: E402

TEMPLATE_PATH = os.path.join(PROJ_ROOT, "aircraft", "rlglider.template.xml")
OUTPUT_PATH   = os.path.join(PROJ_ROOT, "aircraft", "rlglider", "rlglider.xml")
VARIANTS_DIR  = os.path.join(PROJ_ROOT, "analysis", "fdm_variants")

TIMESTAMP_PREFIX = "     Generated : "


def _sha256(path: str) -> str:
    # Line endings normalised so a CRLF checkout (git autocrlf on Windows)
    # hashes the same as the LF content in the repository.
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def _fmt(x: float) -> str:
    # Rounded to 10 significant figures so float noise from the izz sum
    # (0.14 + 0.10 = 0.24000000000000002) never reaches the file.
    return format(float(x), ".10g")


def render(
    yaml_path: str = MASS_PROPERTIES_PATH,
    template_path: str = TEMPLATE_PATH,
    timestamp: str | None = None,
    ixz: float | None = None,
) -> str:
    """Return the generated aircraft file as a string.

    ixz overrides the YAML value; for validation variants only.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = fdm_baseline(load_mass_properties(yaml_path))
    variant_note = []
    if ixz is not None:
        base["ixz"] = float(ixz)
        variant_note = [f"     VALIDATION VARIANT: ixz = {_fmt(ixz)} kg m^2 (not for training)"]

    header = "\n".join([
        "<!-- GENERATED FILE: DO NOT EDIT BY HAND.",
        "     Generator : aircraft/generate_fdm.py",
        f"     Source    : aircraft/mass_properties.yaml  sha256={_sha256(yaml_path)}",
        f"     Template  : aircraft/rlglider.template.xml sha256={_sha256(template_path)}",
        f"{TIMESTAMP_PREFIX}{timestamp}",
        *variant_note,
        "     Edit the YAML or the template, then rerun the generator. -->",
    ])
    subs = {
        "GENERATED_HEADER": header,
        "EMPTYWT_KG":   _fmt(base["mass_kg"]),
        "IXX":          _fmt(base["ixx"]),
        "IYY":          _fmt(base["iyy"]),
        "IZZ":          _fmt(base["izz"]),
        "IXZ":          _fmt(base["ixz"]),
        "WING_AREA_M2": _fmt(base["wing_area_m2"]),
        "SPAN_M":       _fmt(base["span_m"]),
        "CHORD_M":      _fmt(base["chord_m"]),
    }

    with open(template_path, encoding="utf-8") as f:
        text = f.read()
    for key, val in subs.items():
        token = f"@@{key}@@"
        if text.count(token) != 1:
            raise ValueError(f"template must contain {token} exactly once, found {text.count(token)}")
        text = text.replace(token, val)
    if "@@" in text:
        raise ValueError("unsubstituted @@ token left in generated output")
    return text


def ixz_variant_values(yaml_path: str = MASS_PROPERTIES_PATH) -> dict[str, float]:
    """{label: ixz} at the YAML range low, value and high."""
    entry = load_mass_properties(yaml_path)["inertia"]["ixz"]
    lo, hi = entry["range"]
    return {"lo": float(lo), "mid": float(entry["value"]), "hi": float(hi)}


def write_ixz_variants(out_dir: str = VARIANTS_DIR) -> dict[str, str]:
    """Write one JSBSim root per ixz variant; return {label: root_dir}."""
    roots = {}
    for label, value in ixz_variant_values().items():
        root = os.path.join(out_dir, f"ixz_{label}")
        model_dir = os.path.join(root, "aircraft", "rlglider")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "rlglider.xml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(render(ixz=value))
        roots[label] = root
    return roots


def strip_timestamp(text: str) -> str:
    """Generated text with the timestamp line removed, for comparisons."""
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(TIMESTAMP_PREFIX))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file is stale")
    ap.add_argument("--ixz-variants", action="store_true",
                    help=f"write ixz validation variants under {os.path.relpath(VARIANTS_DIR, PROJ_ROOT)}")
    args = ap.parse_args()

    if args.ixz_variants:
        for label, root in write_ixz_variants().items():
            print(f"wrote {os.path.relpath(root, PROJ_ROOT)} (ixz {label})")
        return

    new = render()
    old = None
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            old = f.read()
    stale = old is None or strip_timestamp(old) != strip_timestamp(new)

    if args.check:
        print("stale" if stale else "up to date")
        sys.exit(1 if stale else 0)
    if stale:
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        print(f"wrote {os.path.relpath(OUTPUT_PATH, PROJ_ROOT)}")
    else:
        print("up to date; not rewritten")


if __name__ == "__main__":
    main()

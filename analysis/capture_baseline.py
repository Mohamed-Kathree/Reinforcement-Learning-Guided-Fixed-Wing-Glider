"""
capture_baseline.py
===================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

SPEC V20 Phase 0 — capture the pre-change reference that every later
regression claim is measured against.

Records:
    * pytest tests/test_dynamics.py pass/fail counts
    * validate_glide.py exit code and its printed gate lines
    * the numeric value behind each of the five gates, recomputed with
      validate_glide's own helpers (the script prints only some of them)
    * a short near-trim trajectory snapshot, so a later refactor of the
      aircraft file can be checked state-by-state, not just gate-by-gate

Run:   python -m analysis.capture_baseline [--out analysis/baseline_v19.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import validate_glide as vg  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_pytest() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_dynamics.py", "-v", "-p", "no:cacheprovider"],
        cwd=PROJ_ROOT, capture_output=True, text=True,
    )
    tests = dict(re.findall(r"::(\w+) (PASSED|FAILED|ERROR|SKIPPED)", proc.stdout))
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {
        "command": "pytest tests/test_dynamics.py -v",
        "exit_code": proc.returncode,
        "passed": sum(v == "PASSED" for v in tests.values()),
        "failed": sum(v in ("FAILED", "ERROR") for v in tests.values()),
        "skipped": sum(v == "SKIPPED" for v in tests.values()),
        "summary_line": summary,
        "tests": tests,
    }


def run_validate_glide() -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "validate_glide.py"],
        cwd=PROJ_ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("[PASS]", "[FAIL]"))]
    return {"command": "python validate_glide.py", "exit_code": proc.returncode, "gate_lines": lines}


# ---------------------------------------------------------------------------
# Numeric gate values — same IC and loop as validate_glide, returning numbers
# ---------------------------------------------------------------------------

def gate_values() -> dict:
    # Gate 1 — energy drift with zero drag
    fdm = vg._make_fdm()
    vg._set_near_trim_ic(fdm)
    fdm["aero/cd-mult"] = 0.0
    fdm.run()
    e0 = vg._total_energy(fdm)
    drift = 0.0
    for _ in range(int(10.0 / vg.DT)):
        fdm.run()
        drift = max(drift, abs(vg._total_energy(fdm) - e0) / abs(e0))

    # Gate 2 — median glide ratio over 5 s
    fdm = vg._make_fdm()
    vg._set_near_trim_ic(fdm)
    samples = []
    for _ in range(int(5.0 / vg.DT)):
        fdm.run()
        vn = fdm["velocities/v-north-fps"] * vg.FPS2MS
        ve = fdm["velocities/v-east-fps"] * vg.FPS2MS
        vd = fdm["velocities/v-down-fps"] * vg.FPS2MS
        if vd > 0.1:
            samples.append(np.hypot(vn, ve) / vd)
    glide = float(np.median(samples))

    # Gate 3 — CLalpha lift force (lbf) at the five check angles
    def _cl(alpha_deg: float) -> float:
        f = vg._make_fdm()
        f["ic/vt-fps"] = 10.0 * vg.MS2FPS
        f["ic/alpha-deg"] = alpha_deg
        f["ic/gamma-deg"] = 0.0
        f["ic/h-agl-ft"] = 200.0 * vg.M2FT
        f.run_ic()
        f["fcs/elevator-cmd-norm"] = 0.0
        f.run()
        return float(f["aero/coefficient/CLalpha"])

    cl = {f"{a:g}": _cl(a) for a in (4.0, 10.0, 12.0, 14.0, 22.0)}

    # Gate 4 — max |q| over 15 s from near-trim
    fdm = vg._make_fdm()
    vg._set_near_trim_ic(fdm, alt_m=300.0, vt_ms=10.0)
    max_q = 0.0
    for _ in range(int(15.0 / vg.DT)):
        fdm.run()
        max_q = max(max_q, abs(fdm["velocities/q-rad_sec"]))

    # Gate 5 — Euler bounds over 30 s
    fdm = vg._make_fdm()
    vg._set_near_trim_ic(fdm, alt_m=500.0, vt_ms=10.0)
    max_phi = max_theta = 0.0
    finite = True
    for _ in range(int(30.0 / vg.DT)):
        fdm.run()
        phi, theta, psi = fdm["attitude/phi-rad"], fdm["attitude/theta-rad"], fdm["attitude/psi-rad"]
        finite &= bool(np.isfinite(phi) and np.isfinite(theta) and np.isfinite(psi))
        max_phi, max_theta = max(max_phi, abs(phi)), max(max_theta, abs(theta))

    return {
        "gate1_energy_drift_frac": drift,
        "gate2_glide_ratio_median": glide,
        "gate3_clalpha_force_lbf": cl,
        "gate4_max_abs_q_rad_s": max_q,
        "gate5_max_abs_phi_rad": max_phi,
        "gate5_max_abs_theta_rad": max_theta,
        "gate5_all_finite": finite,
    }


def trim_snapshot() -> dict:
    """State at t = 0, 1, 2, 5, 10 s from validate_glide's near-trim IC."""
    keys = {
        "u_fps": "velocities/u-fps", "w_fps": "velocities/w-fps",
        "q_rad_s": "velocities/q-rad_sec", "theta_rad": "attitude/theta-rad",
        "alpha_rad": "aero/alpha-rad", "vt_fps": "velocities/vt-fps",
        "h_agl_ft": "position/h-agl-ft", "qbar_psf": "aero/qbar-psf",
        "mass_slugs": "inertia/mass-slugs",
        "ixx_slugs_ft2": "inertia/ixx-slugs_ft2", "iyy_slugs_ft2": "inertia/iyy-slugs_ft2",
        "izz_slugs_ft2": "inertia/izz-slugs_ft2", "cg_x_in": "inertia/cg-x-in",
    }
    fdm = vg._make_fdm()
    vg._set_near_trim_ic(fdm)
    marks = {0: "t0", int(1 / vg.DT): "t1", int(2 / vg.DT): "t2",
             int(5 / vg.DT): "t5", int(10 / vg.DT): "t10"}
    out = {}
    for step in range(int(10 / vg.DT) + 1):
        if step in marks:
            out[marks[step]] = {k: float(fdm[p]) for k, p in keys.items()}
        fdm.run()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("analysis", "baseline_v19.json"))
    args = ap.parse_args()

    import jsbsim

    record = {
        "spec": "V20 Phase 0 — baseline capture (V19 airframe state)",
        "provenance": {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "python": sys.version.split()[0],
            "jsbsim": jsbsim.__version__,
            "aircraft_xml_sha256": _sha256(os.path.join(PROJ_ROOT, "aircraft", "rlglider", "rlglider.xml")),
        },
        "pytest_dynamics": run_pytest(),
        "validate_glide": run_validate_glide(),
        "gate_values": gate_values(),
        "trim_snapshot": trim_snapshot(),
    }
    path = os.path.join(PROJ_ROOT, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    p, v = record["pytest_dynamics"], record["validate_glide"]
    print(f"pytest: {p['passed']} passed, {p['failed']} failed (exit {p['exit_code']})")
    print(f"validate_glide: exit {v['exit_code']}")
    print(json.dumps(record["gate_values"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""
test_runner.py
==============
Subprocess wrappers around the RL repo's real validation entry points --
validate_glide.py (5 physics gates) and `pytest tests/test_dynamics.py`
(8 unit tests) -- plus a lightweight in-process GliderEnv probe. This module
never reimplements the checks; it only runs the real ones and parses their
real output.

Regexes below are matched against validate_glide.py's actual print format
(read directly from the source, not guessed):
    [PASS] Gate 1 — Energy Conservation   drift=0.012%  (threshold < 0.1%)
    [FAIL] Gate 2  Glide ratio 8.10 outside (10.0, 20.0)
Note FAIL lines carry no gate name (see validate_glide.py main()), so the
name is filled in from GATE_NAMES for failures.
"""
from __future__ import annotations

import re
import subprocess

import numpy as np

from . import config
from .schemas import EnvProbeResult, GateResult, TestRunResult, UnitTestResult

GATE_NAMES: dict[int, str] = {
    1: "Energy Conservation",
    2: "Glide Ratio",
    3: "Stall Behaviour",
    4: "Trim Stability",
    5: "Attitude Integrity",
}

_PASS_RE = re.compile(r"^\[PASS\]\s+Gate\s+(\d+)\s+[—-]\s+(.+?)\s{2,}(.+)$")
_FAIL_RE = re.compile(r"^\[FAIL\]\s+Gate\s+(\d+)\s+(.+)$")
_PYTEST_RE = re.compile(r"^\S*::(\S+)\s+(PASSED|FAILED|ERROR)")


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(config.GLIDER_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_physics_gates() -> tuple[list[GateResult], str]:
    proc = _run([str(config.GLIDER_PYTHON), "validate_glide.py"], timeout=120)
    results: dict[int, GateResult] = {}

    for line in proc.stdout.splitlines():
        line = line.strip()
        m = _PASS_RE.match(line)
        if m:
            gate_num = int(m.group(1))
            results[gate_num] = GateResult(
                gate=gate_num, name=m.group(2).strip(), passed=True, detail=m.group(3).strip(),
            )
            continue
        m = _FAIL_RE.match(line)
        if m:
            gate_num = int(m.group(1))
            results[gate_num] = GateResult(
                gate=gate_num,
                name=GATE_NAMES.get(gate_num, f"Gate {gate_num}"),
                passed=False,
                detail=m.group(2).strip(),
            )

    return [results[n] for n in sorted(results)], proc.stdout + proc.stderr


def run_unit_tests() -> tuple[list[UnitTestResult], str]:
    proc = _run(
        [str(config.GLIDER_PYTHON), "-m", "pytest", "tests/test_dynamics.py", "-v", "--tb=short", "--no-header"],
        timeout=180,
    )
    results: list[UnitTestResult] = []
    for line in proc.stdout.splitlines():
        m = _PYTEST_RE.match(line.strip())
        if m:
            results.append(UnitTestResult(name=m.group(1), passed=m.group(2) == "PASSED"))
    return results, proc.stdout + proc.stderr


def run_env_probe() -> EnvProbeResult:
    """In-process smoke test of GliderEnv -- requires the backend process
    itself to be running under GLIDER_PYTHON (see config.py) so the import
    resolves."""
    try:
        from env.curriculum import STAGES
        from env.glider_env import GliderEnv

        env = GliderEnv(cfg=dict(STAGES[0]), seed=0)
        obs, _info = env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs2, _reward, _term, _trunc, _info2 = env.step(action)

        ok = tuple(obs.shape) == (11,) and tuple(env.action_space.shape) == (2,) and np.isfinite(obs2).all()
        return EnvProbeResult(
            passed=ok,
            obs_shape=list(obs.shape),
            action_shape=list(env.action_space.shape),
            detail="reset() + step() succeeded" if ok else "unexpected observation/action shape or non-finite obs",
        )
    except Exception as exc:  # surfaced to the UI as a failed card, not raised
        return EnvProbeResult(passed=False, obs_shape=[], action_shape=[], detail=f"{type(exc).__name__}: {exc}")


def run_all() -> TestRunResult:
    gates, gate_stdout = run_physics_gates()
    unit_tests, pytest_stdout = run_unit_tests()
    env_probe = run_env_probe()

    all_passed = (
        bool(gates) and all(g.passed for g in gates)
        and bool(unit_tests) and all(t.passed for t in unit_tests)
        and env_probe.passed
    )

    return TestRunResult(
        physics_gates=gates,
        unit_tests=unit_tests,
        env_probe=env_probe,
        all_passed=all_passed,
        raw_stdout=gate_stdout + "\n" + pytest_stdout,
    )

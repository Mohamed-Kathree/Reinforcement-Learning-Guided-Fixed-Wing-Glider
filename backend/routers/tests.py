"""
routers/tests.py
================
POST /api/tests/run -- runs the real physics gates, unit tests, and an env
probe, returning actual pass/fail results for the Validation panel.
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import test_runner
from ..schemas import TestRunResult

router = APIRouter(prefix="/api/tests", tags=["tests"])


@router.post("/run", response_model=TestRunResult)
def run_tests() -> TestRunResult:
    return test_runner.run_all()

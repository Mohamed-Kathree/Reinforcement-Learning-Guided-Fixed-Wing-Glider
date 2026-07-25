"""
routers/baseline.py
====================
POST /api/baseline/run -- benchmarks DeterministicRTL across all 4
curriculum stages and records every episode for the Replay tab.
"""
from __future__ import annotations

from fastapi import APIRouter

from .. import baseline_runner
from ..schemas import BaselineRunResult

router = APIRouter(prefix="/api/baseline", tags=["baseline"])


@router.post("/run", response_model=BaselineRunResult)
def run(n_episodes: int = 20, seed_base: int = 0) -> BaselineRunResult:
    return baseline_runner.run_baseline(n_episodes=n_episodes, seed_base=seed_base)

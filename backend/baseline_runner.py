"""
baseline_runner.py
===================
Benchmarks the deterministic RTL controller across all 4 curriculum stages,
recording every episode as a full Trajectory (so the Baseline panel's
episode picker has something real to hand off to the Replay tab).

Runs synchronously -- at the default 20 episodes/stage this is a few
seconds, not minutes (see FRONTEND_BUILD_CONTEXT.md Section 6 gotcha about
50-episode runs; the frontend shows a spinner while this call is in flight).
"""
from __future__ import annotations

from . import recorder
from .schemas import BaselineRunResult, EpisodeSummary, StageBaselineResult

N_STAGES = 4


def run_baseline(n_episodes: int = 20, seed_base: int = 0) -> BaselineRunResult:
    """Run `n_episodes` baseline episodes at each of the 4 curriculum stages.

    Args:
        n_episodes : episodes per stage.
        seed_base  : base RNG seed; stage/episode offsets keep every episode
                     across the whole run reproducible and distinct.
    """
    stage_results: list[StageBaselineResult] = []

    for stage in range(N_STAGES):
        n_success = n_crash = n_timeout = 0
        episodes: list[EpisodeSummary] = []

        for i in range(n_episodes):
            seed = seed_base + stage * 10_000 + i
            traj = recorder.record_episode(controller="baseline", stage=stage, seed=seed)
            episodes.append(EpisodeSummary(episode_id=traj.episode_id, outcome=traj.meta.outcome))
            if traj.meta.outcome == "success":
                n_success += 1
            elif traj.meta.outcome == "crash":
                n_crash += 1
            else:
                n_timeout += 1

        stage_results.append(StageBaselineResult(
            stage=stage,
            n_episodes=n_episodes,
            n_success=n_success,
            n_crash=n_crash,
            n_timeout=n_timeout,
            success_rate=n_success / n_episodes,
            episodes=episodes,
        ))

    return BaselineRunResult(stages=stage_results)

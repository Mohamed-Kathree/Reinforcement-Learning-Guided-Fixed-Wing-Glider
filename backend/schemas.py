"""
schemas.py
==========
Pydantic response models for the dashboard API. Mirrors frontend/src/types.ts
field-for-field -- the data contract is sacred (FRONTEND_BUILD_CONTEXT.md
Section 4): change a field here, change it there in the same commit.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# Episode outcome, shared by TrajectoryMeta and EpisodeSummary below.
#
# "soft_landing" was added when the sim/RL side moved to the Phase 4
# precision-landing task (touchdown-based termination + a continuous
# 0-1 landing "quality" grade, replacing the old cross-R_home_m-and-stop
# task -- see env/reward.py). Under that task most episodes that don't hit
# the strict "success" bar (quality > 0.5 and centred) still land cleanly,
# just off-centre or a bit rough -- they are NOT timeouts. Before this
# field existed, recorder.py bucketed every non-success/non-crash episode
# as "timeout", which was actively misleading (a landed-but-off-centre
# episode showed up identically to one that never touched down at all).
# "timeout" now means what it says: truncated without ever touching down.
Outcome = Literal["success", "soft_landing", "crash", "timeout"]


# ---------------------------------------------------------------------------
# Validation panel (Milestone 1)
# ---------------------------------------------------------------------------

class GateResult(BaseModel):
    gate: int
    name: str
    passed: bool
    detail: str


class UnitTestResult(BaseModel):
    name: str
    passed: bool


class EnvProbeResult(BaseModel):
    passed: bool
    obs_shape: list[int]
    action_shape: list[int]
    detail: str


class TestRunResult(BaseModel):
    physics_gates: list[GateResult]
    unit_tests: list[UnitTestResult]
    env_probe: EnvProbeResult
    all_passed: bool
    raw_stdout: str


# ---------------------------------------------------------------------------
# Flight trajectory -- the core data contract (Milestones 2-4)
# ---------------------------------------------------------------------------

class TrajectoryMeta(BaseModel):
    stage: int
    controller: Literal["baseline", "rl"]
    outcome: Outcome
    R_home: float
    alt0: float
    seed: int
    n_frames: int
    duration_s: float
    quality: float = 0.0            # landing-quality grade [0,1]; 0 for a real timeout
    final_dist_home: float = 0.0    # dist_home (m) at the final frame


class ControlSurfaces(BaseModel):
    ail: float
    elev: float
    rud: float


class Frame(BaseModel):
    t: float
    pos_ned: list[float]   # [n, e, d] metres
    euler: list[float]     # [roll, pitch, yaw] radians
    v_body: list[float]    # [u, v, w] m/s
    ctrl: ControlSurfaces
    dist_home: float
    agl: float


class Trajectory(BaseModel):
    episode_id: str
    meta: TrajectoryMeta
    home_ned: list[float]
    frames: list[Frame]


# ---------------------------------------------------------------------------
# Baseline panel (Milestone 4)
# ---------------------------------------------------------------------------

class EpisodeSummary(BaseModel):
    episode_id: str
    outcome: Outcome
    quality: float = 0.0


class StageBaselineResult(BaseModel):
    stage: int
    n_episodes: int
    n_success: int
    n_soft_landing: int
    n_crash: int
    n_timeout: int
    success_rate: float
    mean_quality: float
    episodes: list[EpisodeSummary]


class BaselineRunResult(BaseModel):
    stages: list[StageBaselineResult]


# ---------------------------------------------------------------------------
# Training monitor (Milestone 5)
# ---------------------------------------------------------------------------

class TrainingStatus(BaseModel):
    running: bool
    pid: Optional[int] = None
    total_timesteps: Optional[int] = None


class TrainingMetric(BaseModel):
    timesteps: int
    stage: int
    success_rate: float
    ep_rew_mean: Optional[float] = None

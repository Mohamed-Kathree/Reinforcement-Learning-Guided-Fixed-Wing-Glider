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
    outcome: Literal["success", "crash", "timeout"]
    R_home: float
    alt0: float
    seed: int
    n_frames: int
    duration_s: float


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
    outcome: Literal["success", "crash", "timeout"]


class StageBaselineResult(BaseModel):
    stage: int
    n_episodes: int
    n_success: int
    n_crash: int
    n_timeout: int
    success_rate: float
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

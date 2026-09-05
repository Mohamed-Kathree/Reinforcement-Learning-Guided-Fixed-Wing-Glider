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

class FlightConditions(BaseModel):
    """The env-config values actually applied for this episode -- either the
    chosen stage's preset, or a caller-supplied override of some subset of
    them (see /api/episode/record's wind_speed/gust_intensity/etc. query
    params). wind_speed/gust_intensity/sensor_noise/dropout_prob are each
    the UPPER BOUND GliderEnv.reset() draws this episode's actual value
    from (env/glider_env.py uniform-samples within [0, cfg value] each
    reset) -- not a fixed exact value, by the same design curriculum
    stages already use.
    """
    wind_speed: float
    gust_intensity: float
    sensor_noise: float
    dropout_prob: float
    launch_offset_min_m: float
    launch_offset_max_m: float


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
    # Default is the pre-this-field placeholder (all zero) so the 827
    # already-recorded episodes in data/episodes/ (none of which have a
    # "conditions" key) still load -- see Frame.wind_ned's default above
    # for the same pattern applied earlier in this file.
    conditions: FlightConditions = FlightConditions(
        wind_speed=0.0, gust_intensity=0.0, sensor_noise=0.0, dropout_prob=0.0,
        launch_offset_min_m=0.0, launch_offset_max_m=0.0,
    )


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
    wind_ned: list[float] = [0.0, 0.0, 0.0]   # [wn, we, wd] m/s, total (mean+gust);
                                               # default keeps old recordings loadable
    # V16 Phase C2 §C2.1/§C2.2 -- straight from env/reward.py's compute_reward()
    # info dict (ground-truth, ever step), not recomputed here. Defaults keep
    # the 837 pre-this-field recordings loadable (same pattern as wind_ned
    # above), though those will simply show "no violation, ever" until
    # re-recorded -- the field didn't exist when they were made.
    stall_violation: bool = False
    bank_violation: bool = False
    unreach_violation: bool = False
    alpha_deg: float = 0.0
    roll_deg: float = 0.0
    airspeed: float = 0.0


class Trajectory(BaseModel):
    episode_id: str
    meta: TrajectoryMeta
    home_ned: list[float]
    frames: list[Frame]


# V16 Phase D §D1/§D3 -- lightweight per-episode summary for the analysis
# views (touchdown scatter, failure gallery), so listing/filtering across
# the whole data/episodes/ archive doesn't require parsing every full
# Trajectory's frame array. Written as a data/episodes/<id>.summary.json
# sidecar by recorder.record_episode() (same "small sidecar next to the
# big file" pattern data/runs/<id>.meta.json already established in Phase A
# §A5), and backfilled lazily for pre-this-phase episodes the first time
# GET /api/episode/summaries reads them (see backend/routers/episodes.py).
class EpisodeRecordSummary(BaseModel):
    episode_id: str
    stage: int
    controller: str
    outcome: str
    quality: float
    seed: int
    R_home: float
    touchdown_ned: list[float]   # final frame's pos_ned [n, e, d]
    home_ned: list[float]


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
    # SB3 diagnostics (V16 Phase A §A1), read from the model's own logger --
    # None until the first rollout/episode completes, never fabricated.
    ep_len_mean: Optional[float] = None
    explained_variance: Optional[float] = None
    approx_kl: Optional[float] = None
    clip_fraction: Optional[float] = None
    entropy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    policy_gradient_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    # Curriculum detail (§A2)
    advance_threshold: Optional[float] = None
    episodes_at_stage: Optional[int] = None
    # 4-way rolling outcome breakdown (§A4) -- success/soft_landing/crash/
    # timeout, matching Outcome above (NOT env/reward.py's own 3-way
    # success/crash/timeout the V16 spec's literal text described --
    # collapsing soft-landings into "timeout" was a real bug, already fixed
    # elsewhere in this project; not reintroducing it here).
    outcome_counts: Optional[dict[str, int]] = None
    # Per-component reward breakdown for the most recently completed episode
    # (§A3), keyed by env/reward.py's component names plus 'penalty_truncate'
    # (env/glider_env.py's MAX_STEPS backstop, which compute_reward() itself
    # never sees).
    reward_components: Optional[dict[str, float]] = None
    # V16 Phase C2 §C2.1 -- constraint-violation flags for the live annunciator
    # lamp row, sampled from the single step at which this record was emitted
    # (self.locals['infos'][0] in WebStreamCallback, same info dict
    # compute_reward() returns) -- a live update roughly every emit_freq
    # steps, not every raw step, matching this stream's existing bounded-
    # frequency design. None before the first step's info is available.
    stall_violation: Optional[bool] = None
    bank_violation: Optional[bool] = None
    unreach_violation: Optional[bool] = None
    alpha_deg: Optional[float] = None
    roll_deg: Optional[float] = None


class TrainingRunMeta(BaseModel):
    """One entry in GET /api/training/runs -- mirrors a data/runs/<run_id>.meta.json
    file written once by WebStreamCallback at training start (V16 Phase A §A5)."""
    run_id: str
    start_time: str
    seed: int
    total_timesteps: int
    git_hash: Optional[str] = None
    n_envs: Optional[int] = None


class LiveFrame(BaseModel):
    t: float
    pos_ned: list[float]   # [n, e, d] metres -- position only, this is a 2D top-down view


class LiveEpisode(BaseModel):
    """The most recently completed episode's true-state ground track, for the
    training panel's live top-down view (V16 Phase A §A6 / Phase B §B7).
    Written by WebStreamCallback to data/live_episode.json -- NOT the full
    Trajectory/TrajectoryMeta shape used by recorded-episode replay: a live
    training rollout has no meaningful seed/controller/FlightConditions the
    way a manually recorded evaluation episode does, so this is a smaller,
    purpose-built shape rather than force-fitting that one.
    """
    frames: list[LiveFrame]
    outcome: Optional[str] = None
    quality: float = 0.0
    dist_home: float = 0.0
    R_home: float = 20.0
    stage: int = 0
    updated_at: str

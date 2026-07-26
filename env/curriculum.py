"""
curriculum.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Staged difficulty scheduler for RL training.  Starts at Stage 0 (perfect
conditions, wide home radius) and advances automatically when the rolling
success rate over the last N episodes crosses a threshold.

Four stages (CLAUDE.md Section 12):
    0 : calm air, no noise, R_home=25 m   -- learns basic RTL geometry
    1 : light wind, mild noise            -- adds disturbance rejection
    2 : moderate wind + gusts             -- adds robust aero uncertainty
    3 : full domain randomisation         -- final deployment difficulty

Launch altitude and horizontal offset (as-built hardware notes):
    The glider is hand/ground-launched, not tow-released at altitude.
    Realistic peak altitude after the launch zoom-climb is ~15-25 m, so
    alt0_m is held in that band across all stages (unlike wind/noise, this
    is a hardware constant, not something that should vary with difficulty).
    launch_offset_{min,max}_m are scaled to fit inside the glide range
    available from that altitude (L/D ~= 12-14 per validate_glide.py) while
    staying comfortably clear of R_home_m so the task remains well-posed.
    R_home_m is widened versus the original design to account for the
    NEO-6M GPS's ~2.5 m CEP (vs the ~1.5 m originally assumed).

Phase 4 precision-landing keys (sigma_centre_m, sink_bad, roll_bad_deg):
    R_home_m no longer TERMINATES the episode (see env/reward.py's module
    docstring -- only ground contact/fault does); it and these three new keys
    are purely scoring-strictness parameters that widen at easy stages and
    tighten to the final Stage 3 values, same spirit as wind/noise. sink_ok /
    roll_ok_deg / alpha_ok_deg / alpha_bad_deg stay FIXED across all stages
    (in env/reward.py's DEFAULT_REWARD_CFG) so the definition of "clean
    landing" never changes -- only how strictly distance-from-centre and the
    sink/roll BAD thresholds are graded.

Launch speed/pitch randomisation (deliberately NOT a per-stage key here):
    Launch speed and pitch are randomised per episode across a wide range
    (see LAUNCH_SPEED_MIN_MS etc. in env/glider_env.py, or launch.speed_*_ms
    / launch.pitch_*_deg in training/configs/base.yaml) to cover the range of
    energy states the not-yet-built ESP32 apex-detection firmware might hand
    control over at -- a true apex (near-level, ~trim speed) vs. something
    closer to the original release-state assumption (nose-up, near launch
    speed). Unlike wind/noise/R_home/altitude, this does NOT vary by stage:
    it represents a fixed hardware uncertainty to be robust to at every
    difficulty level, not something that gets easier/harder with curriculum
    progress. It replaces the old flat `launch_jitter` scalar (a small
    multiplicative jitter around one fixed release-state value), which
    covered a much narrower and less realistic range and has been removed
    from STAGES.

Advance rule (CLAUDE.md Section 15):
    advance_threshold = 0.80  (80 % success rate)
    rolling_window    = 100   episodes
    Rolling buffer is cleared after each advance to re-evaluate on the
    harder stage from scratch.

Usage (inside training/train.py CurriculumCallback):
    scheduler = CurriculumScheduler()
    ...
    advanced = scheduler.record_episode(success=info['success'])
    if advanced:
        logger.info(f'Advanced to stage {scheduler.current_stage}')
    env.set_stage(scheduler.current_cfg)

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

from collections import deque
from typing import Deque


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES: list[dict] = [
    # Stage 0 — perfect conditions: learn the geometry with no distractions
    dict(
        wind_speed          = 0.0,
        gust_intensity      = 0.0,
        sensor_noise        = 0.0,
        dropout_prob        = 0.0,
        R_home_m            = 25.0,
        alt0_m              = 25.0,
        launch_offset_min_m = 50.0,
        launch_offset_max_m = 90.0,
        aero_scale_range    = (1.0,  1.0),
        mass_range          = (1.1,  1.1),
        sigma_centre_m      = 12.0,
        sink_bad            = 3.5,
        roll_bad_deg        = 45.0,
    ),
    # Stage 1 — light wind, mild noise: add first disturbances
    dict(
        wind_speed          = 3.0,
        gust_intensity      = 0.5,
        sensor_noise        = 0.3,
        dropout_prob        = 0.05,
        R_home_m            = 22.0,
        alt0_m              = 23.0,
        launch_offset_min_m = 45.0,
        launch_offset_max_m = 80.0,
        aero_scale_range    = (0.9,  1.1),
        mass_range          = (1.0,  1.2),
        sigma_centre_m      = 10.0,
        sink_bad            = 3.0,
        roll_bad_deg        = 40.0,
    ),
    # Stage 2 — moderate wind + gusts, meaningful noise
    dict(
        wind_speed          = 6.0,
        gust_intensity      = 1.0,
        sensor_noise        = 0.7,
        dropout_prob        = 0.15,
        R_home_m            = 20.0,
        alt0_m              = 21.0,
        launch_offset_min_m = 40.0,
        launch_offset_max_m = 70.0,
        aero_scale_range    = (0.85, 1.15),
        mass_range          = (0.9,  1.3),
        sigma_centre_m      = 9.0,
        sink_bad            = 2.75,
        roll_bad_deg        = 37.0,
    ),
    # Stage 3 — full domain randomisation: deployment difficulty
    dict(
        wind_speed          = 9.0,
        gust_intensity      = 2.0,
        sensor_noise        = 1.0,
        dropout_prob        = 0.25,
        R_home_m            = 20.0,
        alt0_m              = 20.0,
        launch_offset_min_m = 40.0,
        launch_offset_max_m = 70.0,
        aero_scale_range    = (0.8,  1.2),
        mass_range          = (0.85, 1.35),
        sigma_centre_m      = 8.0,
        sink_bad            = 2.5,
        roll_bad_deg        = 35.0,
    ),
]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Tracks rolling episode success rate and advances the difficulty stage.

    Args:
        advance_threshold : success rate required to advance (default 0.80)
        rolling_window    : number of recent episodes used for the rate (default 100)

    Attributes:
        current_stage : int, index into STAGES (read-only via property)
        success_rate  : float, rolling success rate over the last window episodes
        current_cfg   : dict, stage config for the current difficulty level
        episodes_seen : int, total episodes recorded since last reset()
    """

    def __init__(
        self,
        advance_threshold: float = 0.80,
        rolling_window:    int   = 100,
    ) -> None:
        if not 0.0 < advance_threshold <= 1.0:
            raise ValueError(f"advance_threshold must be in (0, 1]; got {advance_threshold}")
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1; got {rolling_window}")

        self._threshold: float = advance_threshold
        self._window:    int   = rolling_window
        self._stage:     int   = 0
        self._buffer:    Deque[bool] = deque(maxlen=rolling_window)
        self.episodes_seen: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_episode(self, success: bool) -> bool:
        """Record the outcome of one episode and check for a stage advance.

        Args:
            success : True if the episode ended with the glider reaching home

        Returns:
            True if the stage was advanced this call, False otherwise.
            Callers can use this to log stage transitions.
        """
        self._buffer.append(bool(success))
        self.episodes_seen += 1
        return self._maybe_advance()

    def reset(self, stage: int = 0) -> None:
        """Reset the scheduler to a given stage and clear the rolling buffer.

        Call this at the start of a new training run or to restart from a
        specific stage during evaluation.

        Args:
            stage : stage index to reset to (default 0)
        """
        if not 0 <= stage < len(STAGES):
            raise ValueError(f"stage must be in [0, {len(STAGES)-1}]; got {stage}")
        self._stage = stage
        self._buffer.clear()
        self.episodes_seen = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> int:
        """Current stage index (0-indexed)."""
        return self._stage

    @property
    def stage_count(self) -> int:
        """Total number of curriculum stages."""
        return len(STAGES)

    @property
    def at_final_stage(self) -> bool:
        """True when the scheduler has reached the hardest stage."""
        return self._stage >= len(STAGES) - 1

    @property
    def current_cfg(self) -> dict:
        """Stage config dict for the current difficulty level.

        Returns a shallow copy so callers can freely merge additional keys
        without mutating the canonical STAGES definition:
            cfg = {**DEFAULT_REWARD_CFG, **scheduler.current_cfg}
        """
        return dict(STAGES[self._stage])

    @property
    def success_rate(self) -> float:
        """Rolling success rate over the last rolling_window episodes.

        Returns 0.0 when no episodes have been recorded yet.
        """
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    @property
    def episodes_in_window(self) -> int:
        """Number of episodes currently in the rolling window."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_advance(self) -> bool:
        """Advance stage if the rolling success rate meets the threshold.

        Only advances once the buffer is full (rolling_window episodes
        recorded at the current stage) to avoid premature promotion on
        small samples at the start of training.

        Returns True if the stage was advanced.
        """
        if self.at_final_stage:
            return False

        # Require a full window before evaluating
        if len(self._buffer) < self._window:
            return False

        if self.success_rate >= self._threshold:
            self._stage += 1
            self._buffer.clear()   # re-evaluate from scratch on the new stage
            return True

        return False

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler("
            f"stage={self._stage}/{len(STAGES)-1}, "
            f"success_rate={self.success_rate:.2%}, "
            f"window={len(self._buffer)}/{self._window}, "
            f"threshold={self._threshold:.0%})"
        )

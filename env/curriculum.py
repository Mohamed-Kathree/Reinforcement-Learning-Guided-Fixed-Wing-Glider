"""
curriculum.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Staged difficulty scheduler for RL training.  Starts at Stage 0 (perfect
conditions, wide home radius) and advances automatically when the rolling
success rate over the last N episodes crosses a threshold.

Four stages (CLAUDE.md Section 12):
    0 : calm air, no noise, R_home=30 m   -- learns basic RTL geometry
    1 : light wind, mild noise            -- adds disturbance rejection
    2 : moderate wind + gusts             -- adds robust aero uncertainty
    3 : full domain randomisation         -- final deployment difficulty

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
Rotation R maps NED -> BODY: v_body = R @ v_ned

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
        wind_speed      = 0.0,
        gust_intensity  = 0.0,
        sensor_noise    = 0.0,
        dropout_prob    = 0.0,
        R_home_m        = 30.0,
        alt0_m          = 120.0,
        launch_jitter   = 0.0,
        aero_scale_range= (1.0,  1.0),
        mass_range      = (1.1,  1.1),
    ),
    # Stage 1 — light wind, mild noise: add first disturbances
    dict(
        wind_speed      = 3.0,
        gust_intensity  = 0.5,
        sensor_noise    = 0.3,
        dropout_prob    = 0.05,
        R_home_m        = 20.0,
        alt0_m          = 100.0,
        launch_jitter   = 0.1,
        aero_scale_range= (0.9,  1.1),
        mass_range      = (1.0,  1.2),
    ),
    # Stage 2 — moderate wind + gusts, meaningful noise
    dict(
        wind_speed      = 6.0,
        gust_intensity  = 1.0,
        sensor_noise    = 0.7,
        dropout_prob    = 0.15,
        R_home_m        = 15.0,
        alt0_m          = 90.0,
        launch_jitter   = 0.2,
        aero_scale_range= (0.85, 1.15),
        mass_range      = (0.9,  1.3),
    ),
    # Stage 3 — full domain randomisation: deployment difficulty
    dict(
        wind_speed      = 9.0,
        gust_intensity  = 2.0,
        sensor_noise    = 1.0,
        dropout_prob    = 0.25,
        R_home_m        = 12.0,
        alt0_m          = 80.0,
        launch_jitter   = 0.3,
        aero_scale_range= (0.8,  1.2),
        mass_range      = (0.85, 1.35),
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

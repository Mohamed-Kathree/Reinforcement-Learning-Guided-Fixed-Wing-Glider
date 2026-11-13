"""
wind_gate.py
============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

The handover decision of the wind-gated controller (baseline/wind_gated.py),
kept apart so that it provably reads only what the aircraft has on board
(SPEC V24 Phase 3.1):

    - the sensor observation (observation v1): GPS ground speed and course,
      IMU heading;
    - the controller's own command history (the airspeed it commanded).

It imports nothing from the simulator: no flight model, no wind model, no
environment (tests/test_gated_controller.py checks the import graph). The
wind it decides on is the on-board estimate of env/wind_estimate.py.

Modes
-----
single  One check at t_decide_s: hand over if |w_est| > threshold_ms.
        The V23 controller (threshold 1.0 m/s at 15 s, tuned on development
        seeds).
window  The earlier-handover variant: hand over the first time |w_est| >
        threshold_ms anywhere in [t_earliest_s, t_latest_s]. In strong wind
        the estimate passes the threshold early, before it has converged.
at_step Hand over at a fixed step (analysis only: lets a tuning sweep replay
        a handover time found from a recorded estimate trace).

Every mode is one-way: once handed over, the baseline flies to touchdown (a
hand back would restart the baseline's pattern state machine mid-flight).
"""

from __future__ import annotations

import math

import numpy as np

from env.nav_features import V1, obs_version_of
from env.wind_estimate import WindEstimator

DT_RL = 0.05
# Airspeed command mapping of env/glider_env.py (action[1] in [-1, 1] ->
# 9.5..16 m/s), repeated here so the gate does not import the environment;
# tests/test_gated_controller.py asserts the two agree.
SPEED_CMD_MIN_MS = 9.5
SPEED_CMD_MAX_MS = 16.0
SPEED_CMD_CENTRE_MS = 0.5 * (SPEED_CMD_MIN_MS + SPEED_CMD_MAX_MS)
SPEED_CMD_SCALE_MS = 0.5 * (SPEED_CMD_MAX_MS - SPEED_CMD_MIN_MS)

MODES = ("single", "window", "at_step")


class WindGate:
    def __init__(self, threshold_ms: float = 1.0, t_decide_s: float = 15.0,
                 mode: str = "single", t_earliest_s: float = 5.0,
                 t_latest_s: float = 15.0, at_step: int | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if threshold_ms < 0 or t_decide_s < 0 or t_earliest_s < 0 or t_latest_s < t_earliest_s:
            raise ValueError("thresholds and times must be >= 0, t_latest_s >= t_earliest_s")
        if mode == "at_step" and at_step is None:
            raise ValueError("mode 'at_step' needs at_step")
        self.threshold_ms = float(threshold_ms)
        self.t_decide_s = float(t_decide_s)
        self.mode = mode
        self.t_earliest_s = float(t_earliest_s)
        self.t_latest_s = float(t_latest_s)
        self.at_step = at_step
        self._est = WindEstimator(dt=DT_RL)
        self.reset()

    def reset(self) -> None:
        self._est.reset()
        self.step = 0
        self._v_cmd = SPEED_CMD_CENTRE_MS
        self.handed_over = False
        self.handover_step: int | None = None
        self.wind_at_decision: float | None = None
        self.trace: list[float] = []          # |w_est| at every step (diagnostics)

    @property
    def t(self) -> float:
        return self.step * DT_RL

    @property
    def wind_estimate_ne(self) -> np.ndarray:
        return self._est.w.copy()

    def update(self, obs) -> bool:
        """Feed one sensor observation (before acting on it); returns whether
        the baseline flies this step."""
        if obs_version_of(obs) != 1:
            raise ValueError("WindGate reads observation v1, the raw sensor vector")
        gs, course, yaw = float(obs[V1["ground_speed"]]), float(obs[V1["course"]]), float(obs[V1["yaw"]])
        self._est.update(gs * math.cos(course), gs * math.sin(course), yaw, self._v_cmd)
        w = float(np.hypot(*self._est.w))
        self.trace.append(w)
        if not self.handed_over:
            t = self.t
            if self.mode == "single":
                if self.wind_at_decision is None and t >= self.t_decide_s:
                    self.wind_at_decision = w
                    self._hand_over(w > self.threshold_ms)
            elif self.mode == "window":
                if self.t_earliest_s <= t <= self.t_latest_s and w > self.threshold_ms:
                    self.wind_at_decision = w
                    self._hand_over(True)
            elif self.step >= self.at_step:
                self.wind_at_decision = w
                self._hand_over(True)
        return self.handed_over

    def _hand_over(self, yes: bool) -> None:
        if yes:
            self.handed_over = True
            self.handover_step = self.step

    def record_action(self, action) -> None:
        """The command actually flown this step (from whichever controller):
        the estimator's airspeed input for the next step."""
        self._v_cmd = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS
        self.step += 1


def first_crossing_step(trace: list[float], threshold_ms: float,
                        t_earliest_s: float, t_latest_s: float) -> int | None:
    """The step at which mode 'window' hands over, from a recorded |w_est|
    trace of the policy flying alone (the flight is identical up to then)."""
    for k, w in enumerate(trace):
        t = k * DT_RL
        if t > t_latest_s:
            return None
        if t >= t_earliest_s and w > threshold_ms:
            return k
    return None


def single_check_step(trace: list[float], threshold_ms: float, t_decide_s: float) -> int | None:
    """The step at which mode 'single' hands over, from the same trace."""
    for k, w in enumerate(trace):
        if k * DT_RL >= t_decide_s:
            return k if w > threshold_ms else None
    return None

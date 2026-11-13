"""
wind_gated.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

A wind-gated controller: the learned policy flies where it is strong (calm
air: V22 59% against the baseline's 27% at Stage 0) and the scripted
baseline (DeterministicRTL) takes over where it is not (steady wind).

The decision uses only what the aircraft can compute on board: the wind
estimate of env/wind_estimate.py, fed from the GPS velocity, the IMU heading
and the airspeed command being flown. The policy flies from launch; at
t_decide seconds, if the estimated wind speed exceeds threshold_ms, the
baseline takes over for the rest of the flight (one-way: a mid-flight hand
back would restart the baseline's pattern state machine). The policy's
frame stack keeps being fed until then, exactly as when it flies alone.

Both controllers read the same raw observation vector (observation v1 for
the V20-V22 policies), so the actor fits training/benchmark.py's loop.
"""

from __future__ import annotations

import math

import numpy as np

from baseline.deterministic_rtl import DeterministicRTL
from env.nav_features import V1, obs_version_of
from env.wind_estimate import WindEstimator

_DT_RL = 0.05


class WindGatedActor:
    def __init__(self, policy_actor, threshold_ms: float = 1.5, t_decide_s: float = 15.0) -> None:
        if threshold_ms < 0 or t_decide_s < 0:
            raise ValueError("threshold_ms and t_decide_s must be >= 0")
        self.policy = policy_actor
        self.threshold_ms = float(threshold_ms)
        self.t_decide_s = float(t_decide_s)
        self.obs_version = int(getattr(policy_actor, "obs_version", 1))
        self.name = f"gated_{self.threshold_ms:g}ms_{self.t_decide_s:g}s"
        self._baseline = DeterministicRTL()
        self._est = WindEstimator(dt=_DT_RL)

    def reset(self, seed: int) -> None:
        self.policy.reset(seed)
        self._baseline.reset()
        self._est.reset()
        self._t = 0.0
        self._v_cmd = None
        self.handed_over = False
        self.wind_at_decision = None
        self.estimates: dict[float, float] = {}     # |w_est| at 5, 10, 15, 20 s (diagnostics)

    def _update_estimate(self, obs) -> None:
        from env.glider_env import SPEED_CMD_CENTRE_MS
        if obs_version_of(obs) != 1:
            raise ValueError("WindGatedActor reads observation v1 (the V20-V22 policies)")
        gs, course, yaw = float(obs[V1["ground_speed"]]), float(obs[V1["course"]]), float(obs[V1["yaw"]])
        v_cmd = SPEED_CMD_CENTRE_MS if self._v_cmd is None else self._v_cmd
        self._est.update(gs * math.cos(course), gs * math.sin(course), yaw, v_cmd)

    def act(self, obs) -> np.ndarray:
        from env.glider_env import SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS
        self._update_estimate(obs)
        for t_log in (5.0, 10.0, 15.0, 20.0):
            if t_log not in self.estimates and self._t >= t_log:
                self.estimates[t_log] = float(np.hypot(*self._est.w))
        if not self.handed_over and self.wind_at_decision is None and self._t >= self.t_decide_s:
            self.wind_at_decision = float(np.hypot(*self._est.w))
            self.handed_over = self.wind_at_decision > self.threshold_ms
        if self.handed_over:
            action = self._baseline.act(obs)
        else:
            action = self.policy.act(obs)
        self._v_cmd = SPEED_CMD_CENTRE_MS + float(action[1]) * SPEED_CMD_SCALE_MS
        self._t += _DT_RL
        return np.asarray(action, dtype=np.float32)

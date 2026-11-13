"""
wind_gated.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

A wind-gated controller: the learned policy flies where it is strong (calm
air) and the scripted baseline (DeterministicRTL) takes over where it is not
(steady wind).

The handover decision is baseline/wind_gate.py's WindGate, which reads only
the sensor observation and the command history (SPEC V24 Phase 3.1). The
policy flies from launch; once the gate fires, the baseline flies to
touchdown (one-way). The policy's frame stack keeps being fed until then,
exactly as when it flies alone.

Both controllers read the same raw observation vector (observation v1 for
the V20-V22 policies), so the actor fits training/benchmark.py's loop. Its
episode_record() is stored with every benchmark episode: the gate state, the
handover time and the share of the flight the baseline flew.

The gate parameters live in training/configs/base.yaml (gate:), frozen there
after tuning on the development seeds (analysis/seeds/dev_seeds.json).
"""

from __future__ import annotations

import numpy as np

from baseline.deterministic_rtl import DeterministicRTL
from baseline.wind_gate import DT_RL, WindGate


class WindGatedActor:
    def __init__(self, policy_actor, threshold_ms: float = 1.0, t_decide_s: float = 15.0,
                 mode: str = "single", t_earliest_s: float = 5.0, t_latest_s: float = 15.0,
                 at_step: int | None = None) -> None:
        self.policy = policy_actor
        self.obs_version = int(getattr(policy_actor, "obs_version", 1))
        self.gate = WindGate(threshold_ms=threshold_ms, t_decide_s=t_decide_s, mode=mode,
                             t_earliest_s=t_earliest_s, t_latest_s=t_latest_s, at_step=at_step)
        if mode == "single":
            self.name = f"gated_{threshold_ms:g}ms_{t_decide_s:g}s"
        elif mode == "window":
            self.name = f"gated_window_{threshold_ms:g}ms_{t_earliest_s:g}-{t_latest_s:g}s"
        else:
            self.name = f"gated_at_step_{at_step}"
        self._baseline = DeterministicRTL()

    @classmethod
    def from_config(cls, policy_actor, gate_cfg: dict, variant: str) -> "WindGatedActor":
        """variant: 'single' or 'window' (training/configs/base.yaml gate:)."""
        g = gate_cfg[variant]
        if variant == "single":
            return cls(policy_actor, threshold_ms=g["threshold_ms"], t_decide_s=g["t_decide_s"])
        return cls(policy_actor, threshold_ms=g["threshold_ms"], mode="window",
                   t_earliest_s=g["t_earliest_s"], t_latest_s=g["t_latest_s"])

    def reset(self, seed: int) -> None:
        self.policy.reset(seed)
        self._baseline.reset()
        self.gate.reset()

    # Read-only views kept for analysis/gate_tune.py.
    @property
    def handed_over(self) -> bool:
        return self.gate.handed_over

    @property
    def wind_at_decision(self):
        return self.gate.wind_at_decision

    @property
    def estimates(self) -> dict[float, float]:
        tr = self.gate.trace
        return {t: tr[int(round(t / DT_RL))] for t in (5.0, 10.0, 15.0, 20.0)
                if int(round(t / DT_RL)) < len(tr)}

    def act(self, obs) -> np.ndarray:
        if self.gate.update(obs):
            action = self._baseline.act(obs)
        else:
            action = self.policy.act(obs)
        self.gate.record_action(action)
        return np.asarray(action, dtype=np.float32)

    def episode_record(self) -> dict:
        g = self.gate
        steps = g.step
        hs = g.handover_step
        return {"mode": g.mode, "handed_over": g.handed_over,
                "handover_time_s": None if hs is None else hs * DT_RL,
                "wind_est_at_decision_ms": g.wind_at_decision,
                "baseline_share_of_flight": 0.0 if hs is None else (steps - hs) / max(steps, 1),
                "steps": steps}

"""
controllers.py -- the controller the Pi flies, chosen in the config file.

  gated     (default) Run 4 policy from the apex; one-way handover to the
            scripted baseline at the first on-board wind estimate above
            threshold inside the window (training/configs/base.yaml
            gate.run4.window: 1.5 m/s, 8-15 s). The vendored
            baseline/wind_gated.py, unchanged.
  policy    the policy alone.
  baseline  the scripted baseline (baseline/deterministic_rtl.py) alone.

Every controller maps observation v1 -> normalised action [bank, speed] in
[-1, 1]; to_setpoint() turns that into the PiSetpoint fields exactly as
env/glider_env.py does: bank = a0 * 45 deg, speed = 12.75 + a1 * 3.25 m/s.
"""

from __future__ import annotations

import numpy as np

from onboard.policy import PolicyActor
from onboard.vendor.constants import BANK_CMD_SCALE_RAD, SPEED_CMD_CENTRE_MS, SPEED_CMD_SCALE_MS
from onboard.vendor.deterministic_rtl import DeterministicRTL
from onboard.vendor.wind_gated import WindGatedActor

KINDS = ("gated", "policy", "baseline")
CTRL_POLICY, CTRL_BASELINE = 1, 2          # PiSetpoint.controller


class BaselineActor:
    name = "baseline"
    obs_version = 1

    def __init__(self) -> None:
        self.rtl = DeterministicRTL()

    def reset(self, seed: int = 0) -> None:
        self.rtl.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.rtl.act(obs), dtype=np.float32)


def build(cfg: dict):
    kind = cfg.get("controller", "gated")
    if kind not in KINDS:
        raise ValueError(f"controller must be one of {KINDS}, got {kind!r}")
    if kind == "baseline":
        return BaselineActor()
    policy = PolicyActor(cfg["model"])
    if kind == "policy":
        return policy
    g = cfg["gate"]
    return WindGatedActor(policy, threshold_ms=g["threshold_ms"], mode="window",
                          t_earliest_s=g["t_earliest_s"], t_latest_s=g["t_latest_s"])


def active_controller(actor) -> int:
    """Which controller produced the last action (PiSetpoint.controller)."""
    if isinstance(actor, BaselineActor):
        return CTRL_BASELINE
    if isinstance(actor, WindGatedActor) and actor.handed_over:
        return CTRL_BASELINE
    return CTRL_POLICY


def to_setpoint(action) -> tuple[float, float]:
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float(a[0] * BANK_CMD_SCALE_RAD), float(SPEED_CMD_CENTRE_MS + a[1] * SPEED_CMD_SCALE_MS)

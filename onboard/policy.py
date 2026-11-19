"""
policy.py -- the trained policy on the Pi, in numpy (no torch).

NumpyPolicy   forward pass of an exported SB3 PPO actor (onboard/export_policy.py):
              Linear -> act -> ... -> action head, deterministic (the mean),
              clipped to the action bounds exactly as SB3's predict() does for a
              Box action space.
PolicyActor   the evaluation observation pipeline, identical to
              training/benchmark.py PolicyActor: frames of obs_dim values,
              n_stack deep, starting as zeros with the first observation in
              the last slot, normalised with the frozen VecNormalize statistics
              (clip((x - mean) / sqrt(var + eps), -clip, clip)).
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np


class NumpyPolicy:
    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.meta = json.loads(bytes(z["meta"]).decode())
        n = self.meta["n_layers"]
        self.layers = [(z[f"w{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32)) for i in range(n)]
        self.w_head, self.b_head = z["w_head"].astype(np.float32), z["b_head"].astype(np.float32)
        self.mean, self.var = z["obs_mean"], z["obs_var"]
        self.act = np.tanh if self.meta["activation"] == "tanh" else (lambda x: np.maximum(x, 0.0))
        self.low = np.asarray(self.meta["action_low"], dtype=np.float32)
        self.high = np.asarray(self.meta["action_high"], dtype=np.float32)

    def normalize(self, stacked: np.ndarray) -> np.ndarray:
        if not self.meta["norm_obs"]:
            return stacked.astype(np.float32)
        c = self.meta["clip_obs"]
        return np.clip((stacked - self.mean) / np.sqrt(self.var + self.meta["epsilon"]), -c, c).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for w, b in self.layers:
            h = self.act(h @ w.T + b)
        a = h @ self.w_head.T + self.b_head
        return np.clip(a, self.low, self.high).astype(np.float32)


class PolicyActor:
    name = "policy"

    def __init__(self, path: str) -> None:
        self.net = NumpyPolicy(path)
        self.obs_version = self.net.meta["obs_version"]
        self.obs_dim = self.net.meta["obs_dim"]
        self.n_stack = self.net.meta["n_stack"]
        self._frames: deque = deque(maxlen=self.n_stack)
        self._first = True

    def reset(self, seed: int = 0) -> None:
        self._first = True

    def observe(self, obs: np.ndarray) -> np.ndarray:
        if np.shape(obs) != (self.obs_dim,):
            raise ValueError(f"expected {self.obs_dim} observation values, got {np.shape(obs)}")
        if self._first:
            self._frames.clear()
            for _ in range(self.n_stack - 1):
                self._frames.append(np.zeros(self.obs_dim, dtype=np.float32))
            self._first = False
        self._frames.append(np.asarray(obs, dtype=np.float32))
        return self.net.normalize(np.concatenate(list(self._frames)).astype(np.float32))

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.net.forward(self.observe(obs))

"""
__init__.py
===========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Gymnasium environment package for the RL-guided fixed-wing glider.
Wraps the custom 6-DOF simulator (sim/) in a Gymnasium-compatible interface
with a hierarchical action space, sensor-corrupted observations, a shaped
reward function, and a curriculum difficulty scheduler.

Public API (available once submodules are implemented):
    GliderEnv           -- Gymnasium.Env subclass  (env/glider_env.py)
    CurriculumScheduler -- Episode difficulty scheduler (env/curriculum.py)
    compute_reward      -- Standalone reward function   (env/reward.py)

Typical usage with Stable-Baselines3:
    from env import GliderEnv
    env = GliderEnv(cfg)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

For vectorised training:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    venv = SubprocVecEnv([lambda: GliderEnv(cfg) for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import importlib

__all__ = [
    "GliderEnv",
    "CurriculumScheduler",
    "compute_reward",
]

# Map of public name -> (module path, attribute name within that module).
# Resolved lazily so this file imports cleanly even while the submodules
# are still stubs — no circular-import or missing-symbol errors.
_LAZY: dict[str, tuple[str, str]] = {
    "GliderEnv":           ("env.glider_env", "GliderEnv"),
    "CurriculumScheduler": ("env.curriculum", "CurriculumScheduler"),
    "compute_reward":      ("env.reward",     "compute_reward"),
}


def __getattr__(name: str):
    """Lazy-load public symbols from their submodules on first access."""
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr)
        # Cache in module namespace so subsequent accesses are direct
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'env' has no attribute {name!r}")

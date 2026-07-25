"""
deterministic_rtl.py
====================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Rule-based return-to-launch controller used as a benchmark and fallback.
Consumes the same 11-element normalised observation and produces the same
2-element normalised action as the RL policy, so training/evaluate.py can
compare both controllers through an identical evaluation harness.

Algorithm:
    1. Compute bearing from current GPS position to home (origin).
    2. Compute heading error = bearing_home - current_course.
    3. Command a bank angle proportional to the heading error (P controller),
       clamped to ±max_bank_deg.
    4. Always command a fixed glide speed (stall-safe, near best-glide).

Observation index contract (must match env/glider_env.py):
    obs[0] : dx_home  (m) -- GPS North relative to home
    obs[1] : dy_home  (m) -- GPS East  relative to home
    obs[3] : course_angle (rad) -- current ground-track heading

Action convention (must match env/glider_env.py denormalisation):
    action[0] = bank_cmd_rad  / radians(45)   -- normalised bank ∈ [-1, 1]
    action[1] = (speed_cmd_ms - 9.0) / 4.0   -- normalised speed ∈ [-1, 1]

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation R maps NED -> BODY: v_body = R @ v_ned

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.math_utils import wrap_pi


# ---------------------------------------------------------------------------
# Action space constants (must match GliderEnv)
# ---------------------------------------------------------------------------

_BANK_SCALE_RAD:   float = np.radians(45.0)   # ±1 -> ±45°
_SPEED_CENTRE_MS:  float = 9.0
_SPEED_SCALE_MS:   float = 4.0                # ±1 -> 5–13 m/s


# ---------------------------------------------------------------------------
# Deterministic RTL controller
# ---------------------------------------------------------------------------

class DeterministicRTL:
    """Rule-based return-to-launch policy.

    Uses a proportional heading controller: command a bank angle proportional
    to the bearing error toward home.  Speed is fixed at a stall-safe glide
    speed close to best-glide.

    Args:
        kp_bank       : proportional gain on heading error (rad bank / rad error)
        glide_speed_ms: commanded airspeed (m/s); default 10.0 ≈ best-glide
        max_bank_deg  : hard limit on commanded bank angle (deg)

    The default gains produce fast, stable homing in calm conditions.
    They are intentionally simple — the RL policy is evaluated against this.
    """

    def __init__(
        self,
        kp_bank:        float = 1.5,
        glide_speed_ms: float = 10.0,
        max_bank_deg:   float = 30.0,
    ) -> None:
        self.kp_bank        = float(kp_bank)
        self.glide_speed_ms = float(glide_speed_ms)
        self.max_bank_rad   = float(np.radians(max_bank_deg))

    # ------------------------------------------------------------------
    # Policy interface (identical signature to an SB3 policy.predict)
    # ------------------------------------------------------------------

    def predict(
        self,
        obs:          NDArray,
        deterministic: bool = True,   # accepted for API compatibility; always deterministic
    ) -> tuple[NDArray, None]:
        """Compute action from a single (unnormalised) observation.

        Mirrors the SB3 policy.predict() interface so evaluate.py can call
        both the RL policy and this controller identically:
            action, _ = policy.predict(obs, deterministic=True)

        Args:
            obs           : shape (11,) or (N, 11) float32 observation vector
            deterministic : ignored; present for interface compatibility

        Returns:
            (action, None) where action has shape (2,) or (N, 2) float32.
        """
        obs = np.asarray(obs, dtype=np.float32)
        batched = obs.ndim == 2
        if not batched:
            obs = obs[np.newaxis, :]   # (1, 11)

        actions = np.stack([self._act_single(o) for o in obs])

        if not batched:
            actions = actions[0]   # (2,)

        return actions, None

    def act(self, obs: NDArray) -> NDArray:
        """Convenience wrapper — returns action array only (no state tuple).

        Args:
            obs : shape (11,) observation vector (single step)

        Returns:
            action : shape (2,) float32 normalised action
        """
        action, _ = self.predict(obs)
        return action

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _act_single(self, obs: NDArray) -> NDArray:
        """Compute normalised action for a single observation row."""
        dx_home      = float(obs[0])   # North offset to home (m)
        dy_home      = float(obs[1])   # East  offset to home (m)
        course_angle = float(obs[3])   # current ground-track heading (rad)

        # Bearing from current position to home.
        # Home is at the origin; glider is at (dx_home, dy_home) relative to
        # home, so the bearing back is atan2(-dy_home, -dx_home).
        bearing_home = float(np.arctan2(-dy_home, -dx_home))
        heading_err  = wrap_pi(bearing_home - course_angle)

        # Proportional bank command
        bank_cmd_rad = float(np.clip(
            self.kp_bank * heading_err,
            -self.max_bank_rad,
            self.max_bank_rad,
        ))

        # Normalise to ±1 action space
        action_bank  = bank_cmd_rad / _BANK_SCALE_RAD
        action_speed = (self.glide_speed_ms - _SPEED_CENTRE_MS) / _SPEED_SCALE_MS

        return np.array(
            [np.clip(action_bank, -1.0, 1.0),
             np.clip(action_speed, -1.0, 1.0)],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Standalone evaluation helper
# ---------------------------------------------------------------------------

def evaluate_rtl(
    n_episodes:    int  = 100,
    stage_idx:     int  = 0,
    seed:          int  = 0,
    render:        bool = False,
    verbose:       bool = True,
) -> dict:
    """Run the deterministic RTL controller for N episodes and report stats.

    Intended for quick sanity checks and as the reference score that the
    RL policy must beat.  For full comparative evaluation use
    training/evaluate.py.

    Args:
        n_episodes : number of episodes to evaluate
        stage_idx  : curriculum stage index (0–3)
        seed       : RNG seed for reproducibility
        render     : unused; kept for API consistency
        verbose    : print per-episode summary if True

    Returns:
        dict with keys:
            success_rate  : fraction of episodes ending in success
            mean_reward   : mean total episode reward
            std_reward    : std  total episode reward
            mean_steps    : mean steps per episode
            n_episodes    : number of episodes evaluated
    """
    from env.glider_env import GliderEnv
    from env.curriculum import STAGES

    cfg = dict(STAGES[stage_idx])
    env = GliderEnv(cfg=cfg)
    controller = DeterministicRTL()

    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    steps:   list[int]   = []
    successes: int = 0

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        total_r = 0.0
        n_steps = 0
        while True:
            action = controller.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            n_steps  += 1
            if terminated or truncated:
                if info.get('success', False):
                    successes += 1
                break

        rewards.append(total_r)
        steps.append(n_steps)

        if verbose:
            outcome = 'SUCCESS' if info.get('success') else ('CRASH' if info.get('crash') else 'TIMEOUT')
            print(f"  ep {ep+1:4d}/{n_episodes}  {outcome:<8}  "
                  f"steps={n_steps:5d}  reward={total_r:8.1f}  "
                  f"dist_home={info.get('dist_home', 0.0):.1f} m")

    result = {
        'success_rate': successes / n_episodes,
        'mean_reward':  float(np.mean(rewards)),
        'std_reward':   float(np.std(rewards)),
        'mean_steps':   float(np.mean(steps)),
        'n_episodes':   n_episodes,
    }

    if verbose:
        print(f"\nDeterministicRTL  stage={stage_idx}  n={n_episodes}")
        print(f"  success_rate : {result['success_rate']:.1%}")
        print(f"  mean_reward  : {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")
        print(f"  mean_steps   : {result['mean_steps']:.1f}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deterministic RTL baseline")
    parser.add_argument("--episodes", type=int, default=20,
                        help="number of evaluation episodes (default: 20)")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3],
                        help="curriculum stage index (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-episode output")
    args = parser.parse_args()

    evaluate_rtl(
        n_episodes = args.episodes,
        stage_idx  = args.stage,
        seed       = args.seed,
        verbose    = not args.quiet,
    )

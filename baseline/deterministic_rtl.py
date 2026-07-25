"""
deterministic_rtl.py
====================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Rule-based return-to-launch controller used as a benchmark and fallback.
Consumes the same 11-element normalised observation and produces the same
2-element normalised action as the RL policy, so training/evaluate.py can
compare both controllers through an identical evaluation harness.

Algorithm (PD heading controller):
    1. Compute bearing from current GPS position to home (origin).
    2. Compute heading error = bearing_home - current_course (GPS ground track).
    3. Command a bank angle = kp_bank*heading_err - kd_bank*yaw_rate, clamped
       to +/-max_bank_deg. The derivative term uses IMU yaw rate, NOT a
       finite-difference of heading_err -- see "Why a PD controller" below.
    4. Always command a fixed glide speed (stall-safe, near best-glide).

Why a PD controller (history -- read before changing gains again):
    An earlier P-only version (kp_bank=1.5, no damping) measured a healthy
    28-29% success rate at Stage 0. That number was measured while
    sim/jsbsim_fdm.py's position_ned had a since-fixed bug (it silently
    reported the UNSIGNED magnitude of displacement along each NED axis,
    not signed displacement -- see position_ned's docstring). Once that bug
    was fixed, the SAME P-only controller dropped to ~4-6% success, and
    tracing individual episodes showed why: bank command was saturated at
    +/-max_bank_deg on 98.9% of steps, mean|heading_err| was 135 degrees
    (should be small once converged), and dist_home grew almost
    monotonically to 300+ m before the glider ran out of altitude -- a
    genuine, non-decaying limit cycle, not a subtle mistuning. Root cause:
    at max bank the glider turns at roughly 30 deg/s; with a pure-P law the
    "deadband" where the command un-saturates is only (max_bank_deg/kp_bank)
    degrees wide, so the glider blows through that narrow window at close
    to full turn rate and overshoots the target heading by a large margin
    every single cycle, forever.
    Fix: add a derivative (damping) term so the bank command backs off as
    the glider's heading APPROACHES the target, the same way the inner
    AttitudeController already damps roll with KD_ROLL*p_rate. The naive
    version of this -- a finite difference of heading_err itself -- doesn't
    work: heading_err is built from GPS position/course (obs[0], obs[1],
    obs[3]), and GPS updates at 5 Hz while this controller is called at
    20 Hz, so heading_err is a staircase signal, frozen for 3 out of every
    4 calls. Differencing a staircase produces zero rate on the frozen
    calls and a 4x-too-large spurious spike on the update call -- pure
    aliasing noise, not damping (confirmed empirically: mean crash distance
    barely moved, ~330m, across a wide P+staircase-D gain sweep). obs[6]
    (yaw, IMU-derived) updates every single call (IMU >> policy rate), so
    differencing THAT instead gives a clean, unaliased rate estimate. With
    kp_bank=0.3 / kd_bank=5.0 this reaches ~20-22% success at Stage 0 --
    a real, stable improvement over the ~4-6% pure-P baseline, and the
    number to trust going forward (not the old 28-29%, which was measured
    under the position_ned bug). See tests/test_dynamics.py's
    test_baseline_reaches_home for the current threshold and README.md's
    baseline performance table for the full corrected numbers.

Statefulness:
    This controller now has internal state (previous yaw, for the
    derivative term) and MUST have reset() called at the start of every new
    episode -- otherwise the first step's derivative is computed against
    the previous episode's final yaw, producing one spurious command.
    predict()/act() support batched (N, 11) observations for N parallel
    environments; reset() clears state for all of them at once. If you add
    a new call site that reuses one DeterministicRTL instance across
    multiple episodes (evaluation loops, benchmarks), call reset() at the
    top of every episode.

Observation index contract (must match env/glider_env.py):
    obs[0] : dx_home  (m) -- GPS North relative to home
    obs[1] : dy_home  (m) -- GPS East  relative to home
    obs[3] : course_angle (rad) -- GPS ground-track heading (updates at 5 Hz)
    obs[6] : yaw (rad) -- IMU attitude heading (updates every policy step)

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

# RL/policy step rate (must match env/glider_env.py DT_RL). Not imported
# directly to avoid baseline/ depending on env/ for a single constant.
_DT_RL: float = 0.050


# ---------------------------------------------------------------------------
# Deterministic RTL controller
# ---------------------------------------------------------------------------

class DeterministicRTL:
    """Rule-based return-to-launch policy.

    Uses a PD heading controller: command a bank angle proportional to the
    bearing error toward home, damped by IMU yaw rate to prevent the
    saturated-bank limit cycle a pure-P law falls into (see module
    docstring's "Why a PD controller" for the full story). Speed is fixed
    at a stall-safe glide speed close to best-glide.

    Args:
        kp_bank       : proportional gain on heading error (rad bank / rad error)
        kd_bank       : derivative gain on IMU yaw rate (rad bank / (rad/s)),
                        subtracted from the proportional term
        glide_speed_ms: commanded airspeed (m/s); default 10.0 ≈ best-glide
        max_bank_deg  : hard limit on commanded bank angle (deg)
        dt            : policy step size (s); must match env/glider_env.py
                        DT_RL, used to compute the yaw-rate derivative

    Intentionally simple — the RL policy is evaluated against this — but
    must actually converge to be a meaningful reference point.
    """

    def __init__(
        self,
        kp_bank:        float = 0.3,
        kd_bank:        float = 5.0,
        glide_speed_ms: float = 10.0,
        max_bank_deg:   float = 30.0,
        dt:             float = _DT_RL,
    ) -> None:
        self.kp_bank        = float(kp_bank)
        self.kd_bank        = float(kd_bank)
        self.glide_speed_ms = float(glide_speed_ms)
        self.max_bank_rad   = float(np.radians(max_bank_deg))
        self.dt             = float(dt)

        # Per-row previous yaw (rad), one entry per parallel environment.
        # NaN means "no previous reading yet" (start of episode) -> zero
        # derivative for that row on the next predict() call. Cleared by
        # reset(), which callers MUST invoke at the start of every episode.
        self._prev_yaw: NDArray | None = None

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the yaw-rate derivative state.

        Call at the start of every new episode (single-env use) or before
        the first predict() call after resetting a batch of parallel
        environments. Without this, the first step after a reset computes
        its derivative against the previous episode's final yaw.
        """
        self._prev_yaw = None

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

        n = obs.shape[0]
        if self._prev_yaw is None or self._prev_yaw.shape[0] != n:
            prev_yaw = np.full(n, np.nan, dtype=np.float64)
        else:
            prev_yaw = self._prev_yaw

        actions      = np.empty((n, 2), dtype=np.float32)
        new_prev_yaw = np.empty(n, dtype=np.float64)
        for i in range(n):
            actions[i], new_prev_yaw[i] = self._act_single(obs[i], float(prev_yaw[i]))
        self._prev_yaw = new_prev_yaw

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

    def _act_single(self, obs: NDArray, prev_yaw: float) -> tuple[NDArray, float]:
        """Compute normalised action for a single observation row.

        Args:
            obs      : shape (11,) observation row
            prev_yaw : this row's yaw (rad) from the previous call, or NaN
                       if this is the first call since reset()

        Returns:
            (action, yaw) -- yaw is returned so the caller can store it as
            next call's prev_yaw.
        """
        dx_home      = float(obs[0])   # North offset to home (m), GPS (5 Hz)
        dy_home      = float(obs[1])   # East  offset to home (m), GPS (5 Hz)
        course_angle = float(obs[3])   # ground-track heading (rad), GPS (5 Hz)
        yaw          = float(obs[6])   # attitude heading (rad), IMU (fresh every call)

        # Bearing from current position to home.
        # Home is at the origin; glider is at (dx_home, dy_home) relative to
        # home, so the bearing back is atan2(-dy_home, -dx_home).
        bearing_home = float(np.arctan2(-dy_home, -dx_home))
        heading_err  = wrap_pi(bearing_home - course_angle)

        # Yaw-rate derivative (NOT a finite difference of heading_err --
        # see module docstring for why that aliases against the 5 Hz GPS
        # update rate). NaN prev_yaw (first call since reset) -> zero rate.
        if np.isnan(prev_yaw):
            yaw_rate = 0.0
        else:
            yaw_rate = wrap_pi(yaw - prev_yaw) / self.dt

        # PD bank command: proportional on heading error, damped by yaw rate.
        bank_cmd_rad = float(np.clip(
            self.kp_bank * heading_err - self.kd_bank * yaw_rate,
            -self.max_bank_rad,
            self.max_bank_rad,
        ))

        # Normalise to ±1 action space
        action_bank  = bank_cmd_rad / _BANK_SCALE_RAD
        action_speed = (self.glide_speed_ms - _SPEED_CENTRE_MS) / _SPEED_SCALE_MS

        action = np.array(
            [np.clip(action_bank, -1.0, 1.0),
             np.clip(action_speed, -1.0, 1.0)],
            dtype=np.float32,
        )
        return action, yaw


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
        controller.reset()   # clear yaw-rate derivative state from the previous episode
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

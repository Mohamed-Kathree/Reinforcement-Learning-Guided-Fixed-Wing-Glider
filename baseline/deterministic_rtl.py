"""
deterministic_rtl.py
====================
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Rule-based three-phase pattern controller used as a benchmark and fallback
for the Phase 4 precision-landing task (see RLGlider_Phase4_Landing_Task_
Spec.md §4; supersedes the earlier stateless P-only heading law, which flew
straight at home at a fixed speed and had no notion of energy management --
under the old "cross R_home_m and stop" task that was enough, but under a
task that is actually SCORED on touchdown quality it would arrive high, fast,
and off-centre with the score collapsing to near zero).

Consumes the same 12-element normalised observation and produces the same
2-element normalised action as the RL policy, so training/evaluate.py can
compare both controllers through an identical evaluation harness.

Algorithm (three phases, advancing forward only -- never reverting mid-episode):
    Phase 1 -- RETURN
        Fly toward home at best-glide speed until dist_home < r_pattern_m.
    Phase 2 -- ENERGY MANAGEMENT (orbit)
        If still higher than the glide slope needs, orbit at a fixed bank
        (direction chosen once, from the sign of the initial heading error,
        and held for the rest of the episode -- flipping it mid-pattern
        wastes altitude) at min-sink speed, burning off excess altitude.
    Phase 3 -- FINAL APPROACH + FLARE
        Heading-P bank command (same law as Phase 1); speed scheduled against
        altitude error vs. the glide-slope target; below h_flare_m (and only
        when the ultrasonic reports a valid reading), blend the speed command
        toward a flare speed proportional to how close to the ground it is.
        This is a CLOSED-LOOP command on the existing speed_cmd channel --
        never a raw elevator/attitude override -- because an open-loop pull
        is exactly what env/glider_env.py's deleted flare shield did, and
        measurement showed that stalls the aircraft instead of helping it
        (see RLGlider_Phase4_Landing_Task_Spec.md §0.2).

Heading-P law history (read before re-adding a derivative term):
    An earlier revision of the Phase 1/3 heading law carried a PD term
    (kp_bank=0.3, kd_bank=5.0) justified by an apparent limit-cycle: the
    P-only law (kp_bank=1.5) measured ~4-6% success at Stage 0, with bank
    saturated on ~99% of steps and dist_home diverging to 300+ m.
    That diagnosis was wrong. The actual root cause was a sign bug in
    sim/sensor_models.py::_refresh_gps() -- it computed v_ned via
    quat_to_rotmat(q).T instead of quat_to_rotmat(q) (see that function's
    corrected docstring), which mirrors obs[3] (GPS course_angle) about
    North and inverts obs[8] (vertical speed). Every "correction" the old
    controller issued from a mirrored course reading pushed the glider
    further off course, which is exactly the divergent limit-cycle that was
    observed and is exactly why detuning the gain looked like an
    improvement: it just made the loop too sluggish to diverge as fast, not
    correct. With the GPS fix in place, the plain P-only heading law
    (kp_bank=1.5, no derivative term) reaches ~100% success at Stage 0 under
    the OLD radius-crossing task. Do not reintroduce a derivative/PD term on
    the heading channel to compensate for a poor success rate under the NEW
    landing task either -- if heading tracking looks bad, the orbit phase
    (§4 above) or the speed/flare schedule are far more likely culprits than
    the P-only heading law, which was never the actual problem.

Statefulness:
    Unlike the old stateless P-only law, this controller IS stateful: it
    tracks the current pattern phase and the orbit direction chosen at the
    start of Phase 2. reset() is no longer a no-op -- every existing call
    site already calls it at the top of each episode (evaluation loops,
    benchmarks, tests), so this requires no caller changes.

Observation index contract (must match env/glider_env.py):
    obs[0]  : dx_home       (m)   -- GPS North relative to home
    obs[1]  : dy_home       (m)   -- GPS East  relative to home
    obs[3]  : course_angle  (rad) -- GPS ground-track heading (updates at 5 Hz)
    obs[7]  : baro_alt      (m)
    obs[9]  : lidar_agl     (m)   -- ultrasonic; 0 unless <4.5 m
    obs[10] : lidar_valid   (0/1) -- ultrasonic

Action convention (must match env/glider_env.py denormalisation):
    action[0] = bank_cmd_rad  / radians(45)   -- normalised bank ∈ [-1, 1]
    action[1] = (speed_cmd_ms - 9.0) / 4.0   -- normalised speed ∈ [-1, 1]

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

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
    """Rule-based three-phase (return / orbit / final-approach-flare) landing
    pattern controller.

    See module docstring for the full phase design. Speed and bank are
    commanded through the same normalised channels the RL policy uses, so
    the flare is closed-loop through the existing speed_cmd/AttitudeController
    path rather than a raw surface override.

    Tuning notes (empirical, not the spec's literal suggested defaults --
    see RLGlider_Phase4_Landing_Task_Spec.md §4 for the original pseudocode):

      - glide_slope=10.0 (steeper than the spec's suggested 6.0, though still
        conservative vs. the measured best L/D ~16.7 / ~12.9 trimmed): with
        6.0, the Phase 2->3 altitude target left far more altitude in reserve
        than the aircraft's real efficiency needed, so Phase 3 routinely flew
        30-50 m past home before it ran out of altitude to lose.

      - The Phase 2->3 transition is gated on h_target computed at the FIXED
        r_pattern_m distance, not the instantaneous (oscillating) dist_home:
        gating on dist_home directly races against the orbit's own distance
        oscillation (dist_home swings from ~0 to ~2*r_pattern_m every lap)
        and tends to fire while swinging outward, handing off to Phase 3 from
        much farther out than intended.

      - The flare (see _act_single) also blends bank_cmd toward 0 (wings
        level), not just speed: without it, the heading-P law kept correcting
        residual heading error right up to touchdown, landing at ~30 deg bank
        in most episodes -- reward-graded roll_td then dominated the quality
        product even though sink/speed/alpha were all clean.

      - Even after both fixes, expect the final touchdown point to land
        tens of metres from centre fairly often: once Phase 3 is down to a
        few metres of altitude, a fixed-wing glider needs forward speed to
        stay flying and will cover roughly glide_ratio*altitude more ground
        track before touching down, regardless of bank angle -- so the very
        closest pass to home (which can be quite close) is generally NOT
        where it touches down. This is a genuine limitation of a simple
        scripted final approach, not a bug, and is exactly the kind of
        precision-vs-energy tradeoff the RL policy is expected to learn to
        do better than a scripted heuristic can.

    Args:
        kp_bank            : proportional gain on heading error (rad bank / rad error)
        max_bank_deg        : hard limit on Phase 1/3 commanded bank angle (deg)
        best_glide_speed_ms : Phase 1/3 nominal commanded airspeed (m/s)
        r_pattern_m         : dist_home (m) below which Phase 1 -> Phase 2/3;
                              also the fixed reference distance for the
                              Phase 2 -> 3 altitude gate (see tuning notes)
        glide_slope         : target altitude(d) = d / glide_slope (m per m);
                              conservative vs. measured best L/D ~16.7
        orbit_bank_deg      : fixed bank angle (deg) commanded during Phase 2
        min_sink_speed_ms   : Phase 2 commanded airspeed (near min-sink)
        altitude_margin_m   : Phase 2 -> Phase 3 hysteresis margin (m)
        speed_alt_gain      : Phase 3 speed correction per metre of altitude
                              error vs. the glide-slope target (m/s per m)
        h_flare_m           : AGL (m) below which the flare speed AND bank
                              (wings-level) blend begins (only once the
                              ultrasonic reports valid, i.e. already <4.5 m
                              -- see sim/sensor_models.py)
        flare_speed_ms      : fully-blended flare-phase commanded airspeed
        dt                  : policy step size (s); accepted for API
                              compatibility with callers that pass
                              env/glider_env.py's DT_RL
    """

    def __init__(
        self,
        kp_bank:             float = 1.5,
        max_bank_deg:        float = 30.0,
        best_glide_speed_ms: float = 10.0,
        r_pattern_m:         float = 40.0,
        glide_slope:         float = 10.0,
        orbit_bank_deg:      float = 25.0,
        min_sink_speed_ms:   float = 9.0,
        altitude_margin_m:   float = 1.0,
        speed_alt_gain:      float = 0.08,
        h_flare_m:           float = 3.0,
        flare_speed_ms:      float = 7.0,
        dt:                  float = _DT_RL,
    ) -> None:
        self.kp_bank             = float(kp_bank)
        self.max_bank_rad        = float(np.radians(max_bank_deg))
        self.best_glide_speed_ms = float(best_glide_speed_ms)
        self.r_pattern_m         = float(r_pattern_m)
        self.glide_slope         = float(glide_slope)
        self.orbit_bank_rad      = float(np.radians(orbit_bank_deg))
        self.min_sink_speed_ms   = float(min_sink_speed_ms)
        self.altitude_margin_m   = float(altitude_margin_m)
        self.speed_alt_gain      = float(speed_alt_gain)
        self.h_flare_m           = float(h_flare_m)
        self.flare_speed_ms      = float(flare_speed_ms)
        self.dt                  = float(dt)

        # Backward-compat aliases some scripts/tests may still reach for.
        self.glide_speed_ms = self.best_glide_speed_ms

        self._phase:     int   = 1
        self._orbit_dir: float = 0.0   # chosen once, from the first heading error

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the pattern state machine for a new episode.

        Unlike the old stateless P-only law, this controller IS stateful
        (current phase + orbit direction) -- see module docstring. Every
        existing call site already calls reset() at the top of each episode.
        """
        self._phase     = 1
        self._orbit_dir = 0.0

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
            obs           : shape (12,) or (N, 12) float32 observation vector
            deterministic : ignored; present for interface compatibility

        Returns:
            (action, None) where action has shape (2,) or (N, 2) float32.
        """
        obs = np.asarray(obs, dtype=np.float32)
        batched = obs.ndim == 2
        if not batched:
            obs = obs[np.newaxis, :]   # (1, 12)

        n = obs.shape[0]
        actions = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            actions[i] = self._act_single(obs[i])

        if not batched:
            actions = actions[0]   # (2,)

        return actions, None

    def act(self, obs: NDArray) -> NDArray:
        """Convenience wrapper — returns action array only (no state tuple).

        Args:
            obs : shape (12,) observation vector (single step)

        Returns:
            action : shape (2,) float32 normalised action
        """
        action, _ = self.predict(obs)
        return action

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _act_single(self, obs: NDArray) -> NDArray:
        """Compute normalised action for a single observation row.

        Args:
            obs : shape (12,) observation row

        Returns:
            action : shape (2,) float32 normalised action
        """
        dx_home      = float(obs[0])   # North offset to home (m), GPS (5 Hz)
        dy_home      = float(obs[1])   # East  offset to home (m), GPS (5 Hz)
        course_angle = float(obs[3])   # ground-track heading (rad), GPS (5 Hz)
        baro_alt     = float(obs[7])
        lidar_agl    = float(obs[9])
        lidar_valid  = bool(obs[10])

        dist_home = float(np.hypot(dx_home, dy_home))

        # Bearing from current position to home.
        # Home is at the origin; glider is at (dx_home, dy_home) relative to
        # home, so the bearing back is atan2(-dy_home, -dx_home).
        bearing_home = float(np.arctan2(-dy_home, -dx_home))
        heading_err  = wrap_pi(bearing_home - course_angle)

        # Orbit direction is chosen once (sign of the FIRST heading error
        # seen this episode) and held -- flipping it mid-pattern wastes
        # altitude re-establishing a turn the other way.
        if self._orbit_dir == 0.0:
            self._orbit_dir = 1.0 if heading_err >= 0.0 else -1.0

        h_target = dist_home / self.glide_slope

        # Phases only ever advance forward within an episode.
        if self._phase == 1 and dist_home < self.r_pattern_m:
            self._phase = 2
        # Phase 2 -> 3 uses h_target AT r_pattern_m (a fixed reference
        # distance), not the instantaneous dist_home: the orbit itself makes
        # dist_home swing between ~0 and ~2*r_pattern_m every lap, so gating
        # on h_target(dist_home) races against that oscillation and tends to
        # fire while swinging outward -- i.e. handing off to Phase 3 from
        # ~40-60 m out instead of from the close pass, which then overshoots
        # home by the same distance on the final glide. Gating on a fixed
        # target distance instead means "orbit until at the right altitude
        # to glide r_pattern_m home", independent of where in the lap the
        # altitude threshold happens to be crossed.
        h_target_pattern = self.r_pattern_m / self.glide_slope
        if self._phase == 2 and baro_alt <= h_target_pattern + self.altitude_margin_m:
            self._phase = 3

        if self._phase == 1:
            # RETURN: fly toward home at best-glide speed.
            bank_cmd_rad = float(np.clip(
                self.kp_bank * heading_err, -self.max_bank_rad, self.max_bank_rad
            ))
            speed_cmd_ms = self.best_glide_speed_ms

        elif self._phase == 2:
            # ENERGY MANAGEMENT: orbit at a fixed bank to burn excess
            # altitude, holding roughly dist_home ~= r_pattern_m.
            bank_cmd_rad = self.orbit_bank_rad * self._orbit_dir
            speed_cmd_ms = self.min_sink_speed_ms

        else:
            # FINAL APPROACH: heading-P bank (same law as Phase 1); speed
            # scheduled on altitude error vs. the glide-slope target (faster
            # if low, slower if high), then closed-loop flare-blended below
            # h_flare_m once the ultrasonic reports valid.
            bank_cmd_rad = float(np.clip(
                self.kp_bank * heading_err, -self.max_bank_rad, self.max_bank_rad
            ))
            alt_err = baro_alt - h_target   # + = high, - = low
            speed_cmd_ms = self.best_glide_speed_ms - self.speed_alt_gain * alt_err

            if lidar_valid and lidar_agl < self.h_flare_m:
                blend = float(np.clip(
                    (self.h_flare_m - lidar_agl) / self.h_flare_m, 0.0, 1.0
                ))
                speed_cmd_ms = speed_cmd_ms * (1.0 - blend) + self.flare_speed_ms * blend
                # Wings-level blend: bleed the heading-P bank command toward
                # 0 as the flare progresses. Without this the heading law
                # keeps correcting residual heading error right up to
                # touchdown, which measured ~30 deg bank at ground contact in
                # most episodes -- reward-graded roll_td (env/reward.py's
                # q_roll factor) then dominated the quality product even
                # though sink/speed/alpha were all clean. Real landings are
                # flown wings-level for exactly this reason; closed-loop on
                # the same bank_cmd channel, same as the speed blend above.
                bank_cmd_rad *= (1.0 - blend)

        # Normalise to ±1 action space
        action_bank  = bank_cmd_rad / _BANK_SCALE_RAD
        action_speed = (speed_cmd_ms - _SPEED_CENTRE_MS) / _SPEED_SCALE_MS

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
            mean_quality  : mean landing-quality grade (0-1) across episodes
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
    qualities: list[float] = []
    successes: int = 0

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        controller.reset()   # clear pattern-phase/orbit-direction state from the previous episode
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
                qualities.append(float(info.get('quality', 0.0)))
                break

        rewards.append(total_r)
        steps.append(n_steps)

        if verbose:
            outcome = 'SUCCESS' if info.get('success') else ('CRASH' if info.get('crash') else 'TIMEOUT')
            print(f"  ep {ep+1:4d}/{n_episodes}  {outcome:<8}  "
                  f"steps={n_steps:5d}  reward={total_r:8.1f}  "
                  f"quality={info.get('quality', 0.0):.2f}  "
                  f"dist_home={info.get('dist_home', 0.0):.1f} m")

    result = {
        'success_rate': successes / n_episodes,
        'mean_reward':  float(np.mean(rewards)),
        'std_reward':   float(np.std(rewards)),
        'mean_steps':   float(np.mean(steps)),
        'mean_quality': float(np.mean(qualities)),
        'n_episodes':   n_episodes,
    }

    if verbose:
        print(f"\nDeterministicRTL  stage={stage_idx}  n={n_episodes}")
        print(f"  success_rate : {result['success_rate']:.1%}")
        print(f"  mean_quality : {result['mean_quality']:.2f}")
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

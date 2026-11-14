"""
curriculum.py
=============
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

Staged difficulty scheduler for RL training.  Starts at Stage 0 (perfect
conditions, wide home radius) and advances automatically when the rolling
success rate over the last N episodes crosses a threshold.

Four stages (CLAUDE.md Section 12):
    0 : calm air, no noise, R_home=25 m   -- learns basic RTL geometry
    1 : light wind, mild noise            -- adds disturbance rejection
    2 : moderate wind + gusts             -- adds robust aero uncertainty
    3 : full domain randomisation         -- final deployment difficulty

Launch altitude and horizontal offset (as-built hardware notes):
    The glider is hand/ground-launched, not tow-released at altitude.
    Realistic peak altitude after the launch zoom-climb is ~15-25 m, so
    alt0_m is held in that band across all stages (unlike wind/noise, this
    is a hardware constant, not something that should vary with difficulty).
    launch_offset_{min,max}_m are scaled to fit inside the glide range
    available from that altitude (L/D ~= 12-14 per validate_glide.py) while
    staying comfortably clear of R_home_m so the task remains well-posed.
    R_home_m is widened versus the original design to account for the
    NEO-6M GPS's ~2.5 m CEP (vs the ~1.5 m originally assumed).

Phase 4 precision-landing keys (sigma_centre_m, sink_bad, roll_bad_deg):
    R_home_m no longer TERMINATES the episode (see env/reward.py's module
    docstring -- only ground contact/fault does); it and these three new keys
    are purely scoring-strictness parameters that widen at easy stages and
    tighten to the final Stage 3 values, same spirit as wind/noise. sink_ok /
    roll_ok_deg / alpha_ok_deg / alpha_bad_deg stay FIXED across all stages
    (in env/reward.py's DEFAULT_REWARD_CFG) so the definition of "clean
    landing" never changes -- only how strictly distance-from-centre and the
    sink/roll BAD thresholds are graded.

Launch speed/pitch randomisation (deliberately NOT a per-stage key here):
    Launch speed and pitch are randomised per episode across a wide range
    (see LAUNCH_SPEED_MIN_MS etc. in env/glider_env.py, or launch.speed_*_ms
    / launch.pitch_*_deg in training/configs/base.yaml) to cover the range of
    energy states the not-yet-built ESP32 apex-detection firmware might hand
    control over at -- a true apex (near-level, ~trim speed) vs. something
    closer to the original release-state assumption (nose-up, near launch
    speed). Unlike wind/noise/R_home/altitude, this does NOT vary by stage:
    it represents a fixed hardware uncertainty to be robust to at every
    difficulty level, not something that gets easier/harder with curriculum
    progress. It replaces the old flat `launch_jitter` scalar (a small
    multiplicative jitter around one fixed release-state value), which
    covered a much narrower and less realistic range and has been removed
    from STAGES.

Advance rule:
    advance_threshold, rolling_window, and max_episodes_at_stage are all
    configured in training/configs/base.yaml's curriculum: section -- that
    file, not this docstring, is the source of truth (values have been
    retuned since this project's early design: see that file's own comments
    for the current numbers and why). Rolling buffer is cleared after each
    advance (earned or forced -- see max_episodes_at_stage below) to
    re-evaluate on the harder stage from scratch.

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
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Sequence

import numpy as np

from sim.mass_properties import load_mass_properties as _load_mass_properties

# ---------------------------------------------------------------------------
# Domain randomisation (SPEC V20 Phase 4)
# ---------------------------------------------------------------------------
# One central table. Each range brackets BOTH independent estimates (the V19
# build estimate and the CAD rebuild). Narrow only on measurement, and record
# any narrowing in aircraft/mass_properties.yaml with its reason.
#
# Mass, CG, roll/pitch inertia and wing area are read from
# aircraft/mass_properties.yaml rather than repeated here, so the YAML stays
# the single source of truth; tests/test_domain_rand.py checks they still
# bracket both estimates. Deliberately absent:
#   izz  -- follows as ixx + iyy (flat-plate upper bound; docs/SPEC_V20_decisions.md)
#   ixz  -- fixed at the YAML midpoint; sensitivity measured in the Phase 5 sweep
#   cm0_offset -- hook exists in the aircraft file, but no range is specified
#
# Key -> aircraft-file hook mapping lives in sim/jsbsim_fdm.py (DR_PROPERTY_MAP).
_MP = _load_mass_properties()


def _yaml_range(section: str, key: str) -> tuple[float, float]:
    """The sampled range: `dr_range` where the YAML narrows it (with its
    `dr_reason`), otherwise the full `range`."""
    entry = _MP[section][key]
    lo, hi = entry.get("dr_range", entry["range"])
    return (float(lo), float(hi))


_SW_NOM = float(_MP["geometry"]["wing_area_m2"]["value"])
_SW_LO, _SW_HI = _yaml_range("geometry", "wing_area_m2")

DR_RANGES: dict[str, tuple[float, float]] = {
    'mass_kg':        _yaml_range("mass", "flying_kg"),     # (1.50, 1.80): V19 1.64, CAD 1.62
    'cg_x_m':         _yaml_range("cg", "residual_m"),      # (-0.013, 0.015) after ballasting to 362 mm; +fwd; dr_range, see YAML
    'ixx':            _yaml_range("inertia", "ixx"),        # (0.14, 0.32): CAD 0.16, V19 0.28
    'iyy':            _yaml_range("inertia", "iyy"),        # (0.110, 0.26): CAD 0.12, V19 0.22; dr_range, see YAML

    # Lift: clmax_scale sets the PEAK (CLmax and stall angle) at constant
    # slope; cl_mult sets the lift-curve SLOPE at constant peak. Orthogonal
    # by construction; see the aircraft file's aerodynamics header.
    'clmax_scale':    (0.85, 1.15),   # V19 stall 8.8 m/s, XML lift table 7.75 m/s
    'cl_mult':        (0.80, 1.20),
    'k_induced':      (0.024, 0.048),
    'cmalpha_mult':   (0.75, 1.25),   # narrow: CG randomisation also moves this
    'cmq_mult':       (0.60, 1.40),
    'clp_mult':       (0.60, 1.40),
    'clbeta_mult':    (0.60, 1.40),
    'cnbeta_mult':    (0.50, 1.50),
    'cnr_mult':       (0.50, 1.50),

    # Control effectiveness, per surface: a servo, linkage or surface-area
    # error scales every term that surface produces, coherently.
    'elev_eff_mult':  (0.60, 1.40),   # Cmde, CLde, elevator term in CLtotal
    'ail_eff_mult':   (0.60, 1.40),   # Clda, Cnda
    'rud_eff_mult':   (0.60, 1.40),   # Cndr, Cldr

    'sw_mult':        (_SW_LO / _SW_NOM, _SW_HI / _SW_NOM),   # (0.95, 1.05): geometry known from CAD
    'trim_bias_roll':  (-0.04, 0.04),   # normalised surface units
    'trim_bias_pitch': (-0.03, 0.03),
}

# One-directional: real drag is never better than the model. Scaled from the
# lower bound, not the midpoint, so 0 (the model as written) is in every stage.
# Narrowed from 0.015 (SPEC V20 Phase 5 gate sweep): at 0.015 the airframe's
# trimmed glide ratio is 9.7 alone and 9.2 at the heaviest-draggiest corner,
# below the glide gate's floor of 10; the corner passes up to 0.0109. 0.010 is
# +40% on CD0 at worst. See docs/SPEC_V20_decisions.md.
DR_ONESIDED: dict[str, tuple[float, float]] = {'cd0_add': (0.0, 0.010)}

# Fraction of the full range used at each curriculum stage.
DR_SCALE: list[float] = [0.25, 0.50, 0.75, 1.00]


def stage_ranges(scale: float) -> dict[str, tuple[float, float]]:
    """Every DR range at `scale` of its full width.

    Two-sided ranges shrink about their midpoint; one-sided ranges shrink
    toward their lower bound.
    """
    out = {}
    for key, (lo, hi) in DR_RANGES.items():
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * scale
        out[key] = (mid - half, mid + half)
    for key, (lo, hi) in DR_ONESIDED.items():
        out[key] = (lo, lo + (hi - lo) * scale)
    return out

# The 4-way episode-outcome bucket used throughout the dashboard
# (backend/recorder.py, training/benchmark.py) -- NOT the 3-way
# success/crash/timeout some older docs describe. "soft_landing" (touched
# down clean-ish but missed the strict success bar) is deliberately kept
# separate from "timeout" (never touched down at all): collapsing the two
# was a real bug, fixed elsewhere in this project, and is not to be
# reintroduced here.
OUTCOME_BUCKETS: tuple[str, ...] = ('success', 'soft_landing', 'crash', 'timeout')


def outcome_bucket(info: dict) -> str:
    """The one definition of an episode's OUTCOME_BUCKETS bucket, from its
    terminal-step info dict. Used by training (CurriculumCallback), the
    dashboard (WebStreamCallback, backend/recorder.py) and the evaluator
    (training/benchmark.py).

    env/reward.py sets info['outcome'] only in its touchdown and fault
    branches, so 'outcome' is None at episode end exactly when the episode
    was truncated without ever touching down: a real timeout. A touchdown
    that misses the success bar is a soft landing, never a timeout.
    """
    if info.get('success'):
        return 'success'
    if info.get('outcome') is None:
        return 'timeout'
    if info.get('crash'):
        return 'crash'
    return 'soft_landing'


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES: list[dict] = [
    # Stage 0 — perfect conditions: learn the geometry with no distractions
    dict(
        wind_speed          = 0.0,
        gust_intensity      = 0.0,
        sensor_noise        = 0.0,
        dropout_prob        = 0.0,
        R_home_m            = 25.0,
        alt0_m              = 25.0,
        launch_offset_min_m = 50.0,
        launch_offset_max_m = 90.0,
        dr_scale            = DR_SCALE[0],
        sigma_centre_m      = 15.0,
        sink_bad            = 3.5,
        roll_bad_deg        = 45.0,
    ),
    # Stage 1 — light wind, mild noise: add first disturbances
    dict(
        wind_speed          = 3.0,
        gust_intensity      = 0.5,
        sensor_noise        = 0.3,
        dropout_prob        = 0.05,
        R_home_m            = 22.0,
        alt0_m              = 23.0,
        launch_offset_min_m = 45.0,
        launch_offset_max_m = 80.0,
        dr_scale            = DR_SCALE[1],
        sigma_centre_m      = 13.0,
        sink_bad            = 3.0,
        roll_bad_deg        = 40.0,
    ),
    # Stage 2 — moderate wind + gusts, meaningful noise
    dict(
        wind_speed          = 6.0,
        gust_intensity      = 1.0,
        sensor_noise        = 0.7,
        dropout_prob        = 0.15,
        R_home_m            = 20.0,
        alt0_m              = 21.0,
        launch_offset_min_m = 40.0,
        launch_offset_max_m = 70.0,
        dr_scale            = DR_SCALE[2],
        sigma_centre_m      = 11.0,
        sink_bad            = 2.75,
        roll_bad_deg        = 37.0,
    ),
    # Stage 3 — full domain randomisation: deployment difficulty
    dict(
        wind_speed          = 9.0,
        gust_intensity      = 2.0,
        sensor_noise        = 1.0,
        dropout_prob        = 0.25,
        R_home_m            = 20.0,
        alt0_m              = 20.0,
        launch_offset_min_m = 40.0,
        launch_offset_max_m = 70.0,
        dr_scale            = DR_SCALE[3],
        sigma_centre_m      = 10.0,
        sink_bad            = 2.5,
        roll_bad_deg        = 35.0,
    ),
]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def episode_stage(current_stage: int, replay_frac: float, rng: np.random.Generator) -> int:
    """The stage one training episode runs at.

    With probability replay_frac (and only once past Stage 0) the episode is
    replayed at an earlier stage, drawn uniformly from 0 .. current_stage-1;
    otherwise it runs at current_stage. Replay keeps earlier skills in the
    training distribution: the V20 run's Stage 0 success fell from 51% at
    2.0M steps to 4% at 18.6M once it trained only at later stages.

    replay_frac == 0 never touches rng, so an environment without replay
    draws exactly the random numbers it did before replay existed.
    """
    if not 0.0 <= replay_frac < 1.0:
        raise ValueError(f"replay_frac must be in [0, 1); got {replay_frac}")
    if replay_frac == 0.0 or current_stage == 0:
        return int(current_stage)
    if rng.random() < replay_frac:
        return int(rng.integers(0, current_stage))
    return int(current_stage)


def validate_stage_weights(weights) -> tuple[float, ...]:
    """curriculum.stage_weights as a normalised tuple, one weight per stage.

    Rejects the wrong length, negative weights and an all-zero list."""
    w = tuple(float(x) for x in weights)
    if len(w) != len(STAGES):
        raise ValueError(f"stage_weights needs {len(STAGES)} values; got {len(w)}")
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"stage_weights must be non-negative with a positive sum; got {w}")
    total = sum(w)
    return tuple(x / total for x in w)


def weighted_episode_stage(weights: tuple[float, ...], rng: np.random.Generator) -> int:
    """The stage of one training episode under a fixed stage mix
    (curriculum.stage_weights, SPEC V24 run 5): drawn with probability
    weights[s], independent of the scheduler. One rng.random() per episode."""
    u = rng.random()
    acc = 0.0
    for s, w in enumerate(weights):
        acc += w
        if u < acc:
            return s
    return max(s for s, w in enumerate(weights) if w > 0)   # float round-off at u ~ 1


class CurriculumScheduler:
    """Tracks rolling episode success rate and advances the difficulty stage.

    Args:
        advance_threshold    : success rate required to advance (default 0.80).
                               Either one value for every stage, or one value
                               per advance (len(STAGES) - 1 entries: entry i
                               is the rate needed to leave stage i). The
                               per-stage form exists because one flat bar
                               does not fit a task whose baseline succeeds
                               27% at Stage 0 and 8% at Stage 3.
        rolling_window       : number of recent episodes used for the rate (default 100)
        max_episodes_at_stage: force an advance after this many episodes at the
                               current stage even if the threshold isn't met
                               (default None = disabled, i.e. the pre-Phase-5
                               behaviour -- never force-advances). Guards
                               against a run stalling at an early stage for
                               its whole timestep budget: eval_venv is pinned
                               at STAGES[-1] (see training/train.py), so a
                               stall means best_model.zip gets selected by a
                               policy that never trained against wind, gusts,
                               sensor noise, or domain randomisation.

    Attributes:
        current_stage          : int, index into STAGES (read-only via property)
        success_rate           : float, rolling success rate over the last window episodes
        current_cfg             : dict, stage config for the current difficulty level
        episodes_seen           : int, total episodes recorded since last reset()
        last_advance_was_forced : bool, True if the most recent advance was a
                                  forced (not earned) promotion
    """

    def __init__(
        self,
        advance_threshold:     float | Sequence[float] = 0.80,
        rolling_window:        int       = 100,
        max_episodes_at_stage: int | None = None,
    ) -> None:
        if isinstance(advance_threshold, (int, float)):
            thresholds = [float(advance_threshold)] * (len(STAGES) - 1)
        else:
            thresholds = [float(t) for t in advance_threshold]
            if len(thresholds) != len(STAGES) - 1:
                raise ValueError(
                    f"advance_threshold needs one value per advance "
                    f"({len(STAGES) - 1}); got {len(thresholds)}")
        for t in thresholds:
            if not 0.0 < t <= 1.0:
                raise ValueError(f"advance_threshold must be in (0, 1]; got {t}")
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1; got {rolling_window}")
        if max_episodes_at_stage is not None and max_episodes_at_stage < 1:
            raise ValueError(f"max_episodes_at_stage must be >= 1; got {max_episodes_at_stage}")

        self._thresholds: list[float] = thresholds
        self._window:    int   = rolling_window
        self._max_episodes_at_stage: int | None = max_episodes_at_stage
        self._stage:     int   = 0
        self._buffer:    Deque[bool] = deque(maxlen=rolling_window)
        self._outcome_buffer: Deque[str] = deque(maxlen=rolling_window)
        self.episodes_seen: int = 0
        self._episodes_at_stage: int = 0
        self.last_advance_was_forced: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_episode(self, success: bool, outcome: str | None = None) -> bool:
        """Record the outcome of one episode and check for a stage advance.

        Args:
            success : True if the episode ended with the glider reaching home
            outcome : optional 4-way outcome bucket ('success', 'soft_landing',
                      'crash', or 'timeout' -- see OUTCOME_BUCKETS) for the
                      rolling outcome_counts breakdown below. None (default)
                      skips outcome tallying entirely -- existing callers that
                      only track success/fail are unaffected.

        Returns:
            True if the stage was advanced this call, False otherwise.
            Callers can use this to log stage transitions.
        """
        self._buffer.append(bool(success))
        if outcome is not None:
            self._outcome_buffer.append(str(outcome))
        self.episodes_seen += 1
        self._episodes_at_stage += 1
        return self._maybe_advance()

    def state_dict(self) -> dict:
        """Serialise scheduler state for checkpointing (plain JSON-safe types).

        Written alongside every checkpoint pair by training/train.py's
        VecNormCheckpointCallback so a --resume can restore the curriculum
        stage instead of silently restarting at Stage 0 (see the training
        runbook's Phase B -- a resumed run that regresses the curriculum
        produces a policy that looks like it's succeeding because it's back
        on easy episodes, with nothing in the logs saying why).
        """
        return {
            'current_stage':           self._stage,
            'episodes_at_stage':       self._episodes_at_stage,
            'episodes_seen':           self.episodes_seen,
            'last_advance_was_forced': self.last_advance_was_forced,
            'buffer':                  list(self._buffer),
            'outcome_buffer':          list(self._outcome_buffer),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore scheduler state from state_dict(). Tolerant of missing
        keys (warns and keeps the default for that field) so a checkpoint
        written by an older version of this class does not crash a resume.
        An empty dict is a no-op: the scheduler is left at its current
        (default: Stage 0) state.
        """
        missing = [k for k in
                   ('current_stage', 'episodes_at_stage', 'episodes_seen',
                    'last_advance_was_forced', 'buffer', 'outcome_buffer')
                   if k not in d]
        if missing:
            print(f"[Curriculum] WARNING: state_dict missing keys {missing}; "
                  f"using defaults for those fields.")

        stage = d.get('current_stage')
        if stage is not None:
            if 0 <= stage < len(STAGES):
                self._stage = stage
            else:
                print(f"[Curriculum] WARNING: state_dict current_stage={stage} "
                      f"out of range [0, {len(STAGES)-1}]; keeping stage "
                      f"{self._stage}.")

        self._episodes_at_stage = d.get('episodes_at_stage', self._episodes_at_stage)
        self.episodes_seen      = d.get('episodes_seen', self.episodes_seen)
        self.last_advance_was_forced = d.get('last_advance_was_forced',
                                              self.last_advance_was_forced)

        buffer = d.get('buffer')
        if buffer is not None:
            self._buffer = deque((bool(v) for v in buffer), maxlen=self._window)

        outcome_buffer = d.get('outcome_buffer')
        if outcome_buffer is not None:
            self._outcome_buffer = deque((str(v) for v in outcome_buffer), maxlen=self._window)

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
        self._outcome_buffer.clear()
        self.episodes_seen = 0
        self._episodes_at_stage = 0
        self.last_advance_was_forced = False

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

    @property
    def episodes_at_stage(self) -> int:
        """Episodes recorded at the current stage since the last advance."""
        return self._episodes_at_stage

    @property
    def advance_threshold(self) -> float:
        """Rolling success rate required to leave the current stage (at the
        final stage, which never advances, the last configured value)."""
        return self._thresholds[min(self._stage, len(self._thresholds) - 1)]

    @property
    def advance_thresholds(self) -> list[float]:
        """Every configured threshold, entry i for leaving stage i."""
        return list(self._thresholds)

    @property
    def outcome_counts(self) -> dict[str, int]:
        """4-way outcome tally (see OUTCOME_BUCKETS) over the rolling window.

        Only episodes recorded with an explicit `outcome=` (see
        record_episode()) are counted -- callers that never pass it see all
        zeros here, not an error. Always returns all four bucket keys, even
        at zero, so a consumer (e.g. the dashboard stream) never has to guard
        against a missing key.
        """
        counts = Counter(self._outcome_buffer)
        return {bucket: counts.get(bucket, 0) for bucket in OUTCOME_BUCKETS}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_advance(self) -> bool:
        """Advance stage if the rolling success rate meets the threshold, or
        (if enabled) force an advance after max_episodes_at_stage episodes.

        Only advances on the threshold path once the buffer is full
        (rolling_window episodes recorded at the current stage) to avoid
        premature promotion on small samples at the start of training. The
        force-advance path does not require a full buffer -- it exists
        precisely to unstick a stage that isn't accumulating enough
        successes to fill/pass the threshold at all.

        Returns True if the stage was advanced.
        """
        if self.at_final_stage:
            return False

        if len(self._buffer) >= self._window and self.success_rate >= self.advance_threshold:
            self._stage += 1
            self._buffer.clear()   # re-evaluate from scratch on the new stage
            self._episodes_at_stage = 0
            self.last_advance_was_forced = False
            return True

        if (self._max_episodes_at_stage is not None
                and self._episodes_at_stage >= self._max_episodes_at_stage):
            self._stage += 1
            self._buffer.clear()
            self._episodes_at_stage = 0
            self.last_advance_was_forced = True
            return True

        return False

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        at_stage = (
            f"at_stage={self._episodes_at_stage}/{self._max_episodes_at_stage}"
            if self._max_episodes_at_stage is not None
            else f"at_stage={self._episodes_at_stage}"
        )
        return (
            f"CurriculumScheduler("
            f"stage={self._stage}/{len(STAGES)-1}, "
            f"success_rate={self.success_rate:.2%}, "
            f"window={len(self._buffer)}/{self._window}, "
            f"threshold={self.advance_threshold:.0%}, "
            f"{at_stage}, "
            f"last_advance={'forced' if self.last_advance_was_forced else 'earned'})"
        )

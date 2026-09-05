"""
callbacks/web_stream_callback.py
=================================
SB3 callback that streams live training metrics to a per-run JSONL file
under data/runs/ (V16 Phase A §A5 -- one file per run; see
backend/config.py::RUNS_DIR). Replaces the old single data/train_stream.jsonl
every run used to truncate, destroying run history. The FastAPI WebSocket in
backend/routers/training.py tails a run's file from byte 0 on every
connection -- the training subprocess and the web server are different
processes with no shared memory, so a JSONL file on disk is the IPC. Tailing
from 0 also means a client connecting after training has already progressed
(or finished) replays the full history instantly, rather than needing to
have been watching live.

Added to training/train.py's callback list ADDITIVELY, alongside
CurriculumCallback / CheckpointCallback / EvalCallback -- never replacing
them.

Also owns the V16 Phase A §A6 / Phase B §B7 "live rollout dump"
(data/live_episode.json): the true-state trajectory of the most recently
completed episode, for the training panel's live top-down ground-track view.
Only possible when the training env is a DummyVecEnv -- true-state FDM/wind
accessors live on the actual GliderEnv instance, reachable in-process only
when envs run in-process (DummyVecEnv), not across a SubprocVecEnv's process
boundary. The dashboard's own launch path (backend/routers/training.py's
start_training()) already forces --dummy-vec (n_envs=1), so this covers the
one case that actually matters here; a CLI run using the default
SubprocVecEnv simply doesn't get a live ground track, gracefully (checked
once at _on_training_start(), not re-checked every step).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env.curriculum import CurriculumScheduler, OUTCOME_BUCKETS

from .. import config
from ..recorder import _extract_true_state

# Matches env/glider_env.py's DT_RL (20 Hz policy step) -- mirrored rather
# than imported, same "don't couple backend/ to env/'s private module
# constants" convention backend/recorder.py already established for the
# same value.
_DT_RL = 0.050

# SB3 diagnostics pulled from self.model.logger.name_to_value each emit
# (V16 Phase A §A1). record() only overwrites this dict; dump() does not
# clear it -- so a read at any point returns "most recently recorded",
# which is the correct live-dashboard semantics (a value between two
# rollout-end dumps is simply the last one, not stale-to-the-point-of-wrong).
# Emitted field name is everything after the '/'. Keys absent this early in
# a run (before the first update/episode) emit as null, never fabricated.
_SB3_KEYS = (
    'rollout/ep_rew_mean',
    'rollout/ep_len_mean',
    'train/explained_variance',
    'train/approx_kl',
    'train/clip_fraction',
    'train/entropy_loss',
    'train/value_loss',
    'train/policy_gradient_loss',
    'train/learning_rate',
)

# Mirrors env/glider_env.py's _REWARD_COMPONENT_KEYS. Not imported directly
# -- backend/ intentionally only reads scalars CurriculumCallback has
# already logged into the shared SB3 Logger (see training/train.py), rather
# than importing env/'s private naming constant, keeping this module's only
# coupling to training/ the documented try/except import at the top of
# training/train.py.
_REWARD_COMPONENT_NAMES = (
    'r_path', 'penalty_unreach', 'penalty_stall', 'penalty_bank',
    'penalty_smooth', 'r_terminal', 'penalty_truncate',
)


class WebStreamCallback(BaseCallback):
    """Appends a TrainingMetric JSON line every `emit_freq` _on_step() calls
    to this run's own data/runs/<run_id>.jsonl file, and writes a sibling
    data/runs/<run_id>.meta.json once at training start.

    Args:
        scheduler : the same CurriculumScheduler passed to CurriculumCallback
                    -- read-only here, just observed for stage/success_rate.
        seed      : this run's RNG seed -- written into the run's meta file
                    and used to build the run id.
        total_timesteps : this run's configured target timestep count --
                    written into the run's meta file.
        emit_freq : emit every N _on_step() calls, NOT N environment
                    timesteps -- with n_envs parallel workers, num_timesteps
                    advances by n_envs each call.
        dump_freq : write data/live_episode.json every `dump_freq` emit-gated
                    boundaries (i.e. every `emit_freq * dump_freq` steps), if
                    a completed episode is buffered -- "every 10th emit" per
                    the V16 spec. Only takes effect when live-rollout dumping
                    is reachable at all (see class docstring).
    """

    def __init__(
        self,
        scheduler:       CurriculumScheduler,
        seed:            int,
        total_timesteps: int,
        emit_freq:       int = 200,
        dump_freq:       int = 10,
        verbose:         int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.scheduler       = scheduler
        self.seed            = seed
        self.total_timesteps = total_timesteps
        self.emit_freq       = emit_freq
        self.dump_freq       = dump_freq
        self._run_path       = None   # set in _on_training_start()

        # Live rollout dump (A6/B7) state.
        self._rollout_env               = None    # raw GliderEnv, env index 0, if reachable
        self._rollout_dump_enabled      = False
        self._current_episode_frames: list[dict] = []
        self._last_completed_episode: dict | None = None
        self._emits_since_last_dump     = 0

    def _on_training_start(self) -> None:
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_seed{self.seed}"
        self._run_path = config.RUNS_DIR / f"{run_id}.jsonl"
        meta_path      = config.RUNS_DIR / f"{run_id}.meta.json"

        # Best-effort -- a run must still start cleanly if git isn't
        # available (e.g. a plain extracted archive rather than a clone).
        git_hash = None
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(config.GLIDER_REPO_ROOT),
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip() or None
        except Exception:
            pass

        n_envs = getattr(self.training_env, 'num_envs', None)

        meta = dict(
            start_time      = datetime.now(timezone.utc).isoformat(),
            seed            = self.seed,
            total_timesteps = self.total_timesteps,
            git_hash        = git_hash,
            n_envs          = n_envs,
        )
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Detect live-rollout-dump reachability ONCE, not every step.
        # self.training_env is the VecNormalize wrapping the actual vec env;
        # `.venv` is what VecNormalize wraps. True-state accessors (JSBSimFDM/
        # WindModel) only live in-process when that's a DummyVecEnv.
        vec_env = getattr(self.training_env, 'venv', None)
        if isinstance(vec_env, DummyVecEnv) and len(vec_env.envs) > 0:
            self._rollout_env = vec_env.envs[0].unwrapped
            self._rollout_dump_enabled = True
        else:
            self._rollout_dump_enabled = False

    def _build_record(self, name_to_value: dict[str, Any], info: dict[str, Any] | None = None) -> dict:
        """Pure record-construction, factored out of _on_step() so it can be
        exercised directly against a synthetic name_to_value fixture for
        verification -- see V16 Phase A Gate A item 2. Never runs a training
        loop; `name_to_value` is just a plain dict here, real or fake.

        `info` (V16 Phase C2 §C2.1) is the single step's info dict at the
        moment of this emission -- the same one compute_reward() returns,
        read the same way _accumulate_rollout_frame() already reads it
        (self.locals['infos'][0]). None (the default) keeps this callable
        exactly as Gate A's synthetic-fixture verification already exercises
        it, with no info dict at all.
        """
        info = info or {}
        record: dict[str, Any] = dict(
            timesteps    = int(self.num_timesteps),
            stage        = int(self.scheduler.current_stage),
            success_rate = float(self.scheduler.success_rate),
        )

        for key in _SB3_KEYS:
            field = key.split('/', 1)[1]
            record[field] = name_to_value.get(key)

        record['advance_threshold'] = name_to_value.get('curriculum/advance_threshold')
        record['episodes_at_stage'] = name_to_value.get('curriculum/episodes_at_stage')

        outcome_counts = {
            bucket: name_to_value.get(f'curriculum/outcome_{bucket}')
            for bucket in OUTCOME_BUCKETS
        }
        record['outcome_counts'] = (
            outcome_counts if any(v is not None for v in outcome_counts.values()) else None
        )

        reward_components = {
            name: name_to_value.get(f'reward/{name}')
            for name in _REWARD_COMPONENT_NAMES
        }
        record['reward_components'] = (
            reward_components if any(v is not None for v in reward_components.values()) else None
        )

        record['stall_violation'] = info.get('stall_violation')
        record['bank_violation'] = info.get('bank_violation')
        record['unreach_violation'] = info.get('unreach_violation')
        record['alpha_deg'] = info.get('alpha_deg')
        record['roll_deg'] = info.get('roll_deg')

        return record

    @staticmethod
    def _outcome_bucket(info: dict) -> str:
        """4-way outcome bucket (env/curriculum.py's OUTCOME_BUCKETS) from a
        terminal-step info dict -- duplicated from training/train.py's
        CurriculumCallback._outcome_bucket() rather than imported, since
        training/train.py imports THIS module (see the try/except at its
        top), so the reverse import would be circular. Both copies derive
        the bucket the same way from the same info-dict shape (see
        CurriculumCallback._outcome_bucket()'s fuller docstring for the
        reasoning); keep them in sync if that rule ever changes.
        """
        if info.get('success'):
            return 'success'
        if info.get('outcome') is None:
            return 'timeout'
        if info.get('crash'):
            return 'crash'
        return 'soft_landing'

    def _accumulate_rollout_frame(self) -> None:
        """Append one frame for env index 0 and, on episode end, snapshot the
        completed episode -- called every raw step (not gated by emit_freq),
        since a trajectory needs every frame. Cheap when disabled (a single
        bool check); see class docstring for why this is DummyVecEnv-only.
        """
        if not self._rollout_dump_enabled:
            return

        state = _extract_true_state(self._rollout_env)
        self._current_episode_frames.append(dict(
            t=len(self._current_episode_frames) * _DT_RL,
            pos_ned=state['pos_ned'],
        ))

        info0 = self.locals.get('infos', [{}])[0]
        if 'episode' not in info0:
            return

        self._last_completed_episode = dict(
            frames     = self._current_episode_frames,
            outcome    = self._outcome_bucket(info0),
            quality    = float(info0.get('quality', 0.0)),
            dist_home  = float(info0.get('dist_home', float('nan'))),
            R_home     = float(self.scheduler.current_cfg.get('R_home_m', 20.0)),
            stage      = int(self.scheduler.current_stage),
            updated_at = datetime.now(timezone.utc).isoformat(),
        )
        self._current_episode_frames = []

    def _maybe_dump_live_episode(self) -> None:
        self._emits_since_last_dump += 1
        if self._emits_since_last_dump < self.dump_freq:
            return
        self._emits_since_last_dump = 0

        if self._last_completed_episode is None:
            return
        with open(config.DATA_DIR / "live_episode.json", "w") as f:
            json.dump(self._last_completed_episode, f)

    def _on_step(self) -> bool:
        self._accumulate_rollout_frame()

        if self.n_calls % self.emit_freq != 0:
            return True

        info0 = self.locals.get('infos', [{}])[0]
        record = self._build_record(self.model.logger.name_to_value, info0)
        with open(self._run_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self._rollout_dump_enabled:
            self._maybe_dump_live_episode()

        return True

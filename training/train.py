"""
train.py
========
Part of: RL-Guided Return-to-Launch Fixed-Wing Glider

PPO training entry point using Stable-Baselines3.

Architecture:
    SubprocVecEnv  -- n_envs parallel GliderEnv workers (separate processes)
    VecNormalize   -- running obs/reward normalisation (mandatory for 12-D obs)
    PPO            -- MlpPolicy, hyperparams from training/configs/base.yaml
    CurriculumCallback -- advances difficulty stage when rolling success >= threshold
    CheckpointCallback -- periodic model + VecNormalize snapshots
    EvalCallback   -- periodic deterministic evaluation; saves best model

Usage:
    # Train from scratch with default config:
    python -m training.train

    # Override config values on the command line:
    python -m training.train --config training/configs/base.yaml \\
                              --set ppo.n_envs=4 ppo.total_timesteps=5000000

    # Resume from a checkpoint:
    python -m training.train --resume checkpoints/ppo_glider_1000000_steps

Outputs (all under logging.checkpoint_dir from the config):
    ppo_glider_<step>_steps.zip   -- periodic policy snapshots
    vecnormalize_<step>.pkl       -- matching VecNormalize stats
    best_model.zip                -- best evaluation policy so far
    vecnormalize_best.pkl         -- VecNormalize stats for best model

TensorBoard:
    tensorboard --logdir runs/

Coordinate frames used:
    NED : North-East-Down inertial frame (world)
    BODY: Forward-Right-Down body frame (FRD, attached to glider)
    WIND: Stability/wind frame (x into relative wind)

Quaternion convention: [q0, q1, q2, q3] where q0 is the scalar component.
Rotation quat_to_rotmat(q) maps BODY -> NED: v_ned = R @ v_body

Units: SI throughout (m, m/s, rad, rad/s, kg, N, N*m)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
from typing import Any

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.glider_env import GliderEnv
from env.curriculum import CurriculumScheduler, STAGES

# eval_venv is held at the FINAL curriculum stage for the entire run (see
# train()) so that best_model.zip -- the artefact actually deployed -- is
# selected at deployment difficulty, not whatever stage training happens to
# be on. Do not call eval_venv.env_method('set_stage', ...) with anything
# other than STAGES[-1]; CurriculumCallback only advances the training venv.

# Optional dashboard hook: streams live metrics to data/train_stream.jsonl
# for the showcase frontend's Training tab (backend/routers/training.py's
# WebSocket tails it). Additive only -- training must run identically with
# or without the dashboard backend present, so this never raises.
try:
    from backend.callbacks.web_stream_callback import WebStreamCallback
except ImportError:
    WebStreamCallback = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict:
    """Load and return the YAML config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' overrides to a nested config dict.

    Example:
        apply_overrides(cfg, ['ppo.n_envs=4', 'reward.w_land=600'])
    """
    for item in overrides:
        key_path, _, raw_value = item.partition('=')
        if not _:
            raise ValueError(f"Override must be 'key=value', got: {item!r}")
        parts = key_path.strip().split('.')
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        leaf = parts[-1]
        # Attempt numeric coercion; fall back to string
        try:
            value: Any = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
        node[leaf] = value
    return cfg


_LAUNCH_KEY_MAP = {
    'speed_min_ms':  'launch_speed_min_ms',
    'speed_max_ms':  'launch_speed_max_ms',
    'pitch_min_deg': 'launch_pitch_min_deg',
    'pitch_max_deg': 'launch_pitch_max_deg',
}


def flat_env_cfg(cfg: dict) -> dict:
    """Flatten the reward + launch config sections into one dict for GliderEnv.

    Only launch.speed_{min,max}_ms / launch.pitch_{min,max}_deg are pulled in
    here (renamed to the launch_speed_min_ms/etc. keys GliderEnv reads) --
    they represent a fixed hardware uncertainty (the not-yet-built ESP32
    apex-detection firmware's unknown hand-off energy state), not something
    curriculum-stage-dependent. launch.alt0_m is deliberately NOT included:
    that IS curriculum-stage-dependent and is applied via set_stage() (see
    train(), which applies STAGES[0] immediately after building the
    environments, then CurriculumCallback applies later stages as training
    progresses).
    """
    flat = dict(cfg['reward'])
    launch = cfg.get('launch', {})
    for yaml_key, cfg_key in _LAUNCH_KEY_MAP.items():
        if yaml_key in launch:
            flat[cfg_key] = launch[yaml_key]
    return flat


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _extract_step_count(checkpoint_path: str) -> int | None:
    """Parse the timestep count out of a checkpoint path such as
    'checkpoints/ppo_glider_1000000_steps' or '...ppo_glider_1000000_steps.zip'.

    Returns None if the '_<digits>_steps' suffix isn't found, e.g. for
    'ppo_glider_final'  (there is no matching curriculum_final.json).
    """
    stem = pathlib.Path(checkpoint_path).stem
    m = re.search(r'_(\d+)_steps$', stem)
    return int(m.group(1)) if m else None


def _vecnorm_path_for_checkpoint(checkpoint_path: str) -> pathlib.Path:
    """Map a checkpoint path to its matching VecNormalize stats file.

    CheckpointCallback names periodic snapshots 'ppo_glider_<N>_steps.zip'
    while VecNormCheckpointCallback names the matching stats
    'vecnormalize_<N>.pkl' -- no '_steps' suffix -- so a naive
    ppo_glider->vecnormalize substring swap keeps the stray '_steps' and
    never matches. Found while verifying Phase B's resume behaviour: every
    --resume from a periodic checkpoint (the shape this runbook itself
    hands off, e.g. after an interruption) silently reset the running
    normalisation stats instead of restoring them.

        ppo_glider_<N>_steps[.zip]  -> vecnormalize_<N>.pkl
        ppo_glider_final[.zip]      -> vecnormalize_final.pkl
        best_model[.zip]            -> vecnormalize_best.pkl
    """
    p = pathlib.Path(checkpoint_path)
    stem = p.stem if p.suffix == '.zip' else p.name
    step_num = _extract_step_count(checkpoint_path)
    if step_num is not None:
        name = f"vecnormalize_{step_num}.pkl"
    elif stem == 'best_model':
        name = "vecnormalize_best.pkl"
    else:
        name = stem.replace('ppo_glider', 'vecnormalize') + '.pkl'
    return p.parent / name


def make_env(cfg: dict, rank: int, seed: int = 0):
    """Return a callable that creates a single monitored GliderEnv.

    Each worker gets a unique seed derived from the base seed + rank,
    guaranteeing diverse episode starts across parallel environments.
    """
    def _init() -> Monitor:
        env = GliderEnv(cfg=flat_env_cfg(cfg))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_venv(cfg: dict, seed: int = 0, force_dummy: bool = False) -> VecNormalize:
    """Create a VecNormalize-wrapped vectorised environment.

    Uses SubprocVecEnv for multi-process parallelism (default) or
    DummyVecEnv for single-process debugging (force_dummy=True or n_envs=1).

    Args:
        cfg         : full config dict
        seed        : base RNG seed
        force_dummy : use DummyVecEnv even if n_envs > 1 (useful for debugging)

    Returns:
        VecNormalize wrapping the vectorised env
    """
    n_envs = int(cfg['ppo']['n_envs'])
    fns    = [make_env(cfg, rank=i, seed=seed) for i in range(n_envs)]

    if force_dummy or n_envs == 1:
        venv = DummyVecEnv(fns)
    else:
        venv = SubprocVecEnv(fns, start_method='spawn')

    venv = VecNormalize(
        venv,
        norm_obs    = bool(cfg['ppo']['normalize_obs']),
        norm_reward = bool(cfg['ppo']['normalize_reward']),
        clip_obs    = float(cfg['ppo']['clip_obs']),
        gamma       = float(cfg['ppo']['gamma']),
    )
    return venv


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Advances the curriculum stage when the rolling success rate crosses
    the threshold, then propagates the new stage config to all VecEnv workers.

    Episode outcomes are read from the Monitor 'is_success' key in the
    episode info dict.  If the key is absent (env didn't set it), the
    callback falls back to checking whether episode reward > success_threshold.

    Args:
        scheduler          : CurriculumScheduler instance (shared state)
        venv               : the VecNormalize-wrapped env (workers are set_stage'd)
        success_reward_min : fallback: episode counts as success if ep_reward >= this
        verbose            : 1 to log stage advances
    """

    def __init__(
        self,
        scheduler:           CurriculumScheduler,
        venv:                VecNormalize,
        success_reward_min:  float = 400.0,
        verbose:             int   = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.scheduler          = scheduler
        self.venv               = venv
        self.success_reward_min = success_reward_min
        # Per-env running sums of the dense reward components, for the
        # reward/* TensorBoard scalars below. compute_reward() returns
        # r_path/penalty_unreach/r_terminal in info on EVERY step (not just
        # at episode end), so these must be accumulated across the episode
        # here rather than read once from the terminal step's info dict.
        n_envs = venv.num_envs
        self._ep_r_path_sum          = np.zeros(n_envs, dtype=np.float64)
        self._ep_penalty_unreach_sum = np.zeros(n_envs, dtype=np.float64)
        self._ep_r_terminal_sum      = np.zeros(n_envs, dtype=np.float64)

    def _on_step(self) -> bool:
        # SB3 populates self.locals['infos'] with a list of info dicts,
        # one per env.  'episode' key appears when the episode ends
        # (injected by Monitor wrapper).
        for env_idx, info in enumerate(self.locals.get('infos', [])):
            if 'r_path' in info:
                self._ep_r_path_sum[env_idx]          += info['r_path']
            if 'penalty_unreach' in info:
                self._ep_penalty_unreach_sum[env_idx]  += info['penalty_unreach']
            if 'r_terminal' in info:
                self._ep_r_terminal_sum[env_idx]       += info['r_terminal']

            ep = info.get('episode')
            if ep is None:
                continue

            # Determine success: prefer explicit flag, fall back to reward
            if 'success' in info:
                success = bool(info['success'])
            else:
                success = float(ep['r']) >= self.success_reward_min

            # Task/curriculum/reward-budget scalars -- the only way to see
            # curriculum stage, success rate, touchdown distance, and the
            # reward-component split live in TensorBoard over a multi-day
            # run instead of only as stdout prints that scroll away.
            self.logger.record('curriculum/stage',             self.scheduler.current_stage)
            self.logger.record('curriculum/success_rate',      self.scheduler.success_rate)
            self.logger.record('curriculum/episodes_at_stage', self.scheduler.episodes_at_stage)
            self.logger.record('task/dist_home_final', info.get('dist_home', float('nan')))
            self.logger.record('task/quality',          info.get('quality', 0.0))
            self.logger.record('reward/r_terminal',      self._ep_r_terminal_sum[env_idx])
            self.logger.record('reward/penalty_unreach', self._ep_penalty_unreach_sum[env_idx])
            self.logger.record('reward/r_path',          self._ep_r_path_sum[env_idx])

            self._ep_r_path_sum[env_idx]          = 0.0
            self._ep_penalty_unreach_sum[env_idx] = 0.0
            self._ep_r_terminal_sum[env_idx]      = 0.0

            advanced = self.scheduler.record_episode(success=success)

            if advanced:
                stage     = self.scheduler.current_stage
                stage_cfg = self.scheduler.current_cfg
                if self.verbose >= 1:
                    if self.scheduler.last_advance_was_forced:
                        # A forced advance means the policy was promoted
                        # while still failing the success-rate threshold --
                        # this is information the user needs while reading
                        # TensorBoard, not a detail to bury alongside earned
                        # advances (RLGlider_Phase5_Reward_Rescaling_Spec.md
                        # §3.1).
                        print(f"\n[Curriculum] WARNING: FORCED advance to Stage {stage} "
                              f"(success_rate={self.scheduler.success_rate:.1%} did not "
                              f"reach threshold after max_episodes_at_stage episodes; "
                              f"{self.scheduler.episodes_seen} episodes seen)")
                    else:
                        print(f"\n[Curriculum] Advanced to Stage {stage}  "
                              f"({self.scheduler.episodes_seen} episodes seen)")

                # Propagate to all parallel workers via set_attr (works for
                # both DummyVecEnv and SubprocVecEnv).
                self.venv.env_method('set_stage', stage_cfg)

                # Also update the VecNormalize's wrapped env reference
                # so any future env-level cfg reads are consistent.
                for attr_name, val in stage_cfg.items():
                    try:
                        self.venv.set_attr(attr_name, val)
                    except Exception:
                        pass  # not all attrs are settable via VecEnv

        return True  # do not stop training


# ---------------------------------------------------------------------------
# Checkpoint + VecNormalize save callback
# ---------------------------------------------------------------------------

class VecNormCheckpointCallback(BaseCallback):
    """Saves a VecNormalize stats file alongside each model checkpoint.

    SB3's built-in CheckpointCallback only saves the policy zip; this
    companion callback saves the matching normalisation stats so that
    evaluation with --resume loads both files together.

    Also writes curriculum_<num_timesteps>.json alongside each .pkl, so a
    --resume can restore the curriculum stage instead of silently
    restarting at Stage 0 (training runbook Phase B.1).

    Args:
        checkpoint_dir : directory where stats files are written
        save_freq      : timesteps between saves (should match CheckpointCallback)
        venv           : the VecNormalize instance to save
        scheduler      : the CurriculumScheduler whose state is saved alongside
        name_prefix    : filename prefix (default: 'vecnormalize')
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq:      int,
        venv:           VecNormalize,
        scheduler:      CurriculumScheduler,
        name_prefix:    str = 'vecnormalize',
    ) -> None:
        super().__init__()
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.save_freq      = save_freq
        self.venv           = venv
        self.scheduler       = scheduler
        self.name_prefix    = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.checkpoint_dir / f"{self.name_prefix}_{self.num_timesteps}.pkl"
            self.venv.save(str(path))
            curriculum_path = self.checkpoint_dir / f"curriculum_{self.num_timesteps}.json"
            with open(curriculum_path, 'w') as f:
                json.dump(self.scheduler.state_dict(), f)
            if self.verbose >= 1:
                print(f"[VecNorm] Saved {path}")
                print(f"[Curriculum] Saved {curriculum_path}")
        return True


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """Saves VecNormalize stats whenever EvalCallback records a new best model.

    EvalCallback saves best_model.zip on its own but has no equivalent for the
    matching normalisation stats -- this module's own docstring has always
    promised a vecnormalize_best.pkl output, but nothing ever produced it
    (found during the Phase 5 throughput smoke-check; see
    RLGlider_Phase5_Reward_Rescaling_Spec.md §5). Without this, best_model.zip
    -- the artefact actually meant for deployment -- has no reliable
    observation-normalisation stats to load it with; the periodic
    vecnormalize_<step>.pkl snapshots are not guaranteed to land on the same
    timestep as a new-best event.

    Passed as EvalCallback's callback_on_new_best, so SB3 invokes it (via
    BaseCallback.on_step -> _on_step) immediately after best_model.zip is
    saved, at which point `venv` (the training VecNormalize) has already been
    synced into eval_venv by EvalCallback itself -- either one's stats are
    equivalent at this instant, so this saves `venv` directly.

    Args:
        checkpoint_dir : directory to write vecnormalize_best.pkl into
        venv           : the training VecNormalize instance
    """

    def __init__(
        self,
        checkpoint_dir: str,
        venv:           VecNormalize,
        verbose:        int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.venv           = venv

    def _on_step(self) -> bool:
        path = self.checkpoint_dir / 'vecnormalize_best.pkl'
        self.venv.save(str(path))
        if self.verbose >= 1:
            print(f"[VecNorm] Saved {path} (new best model)")
        return True


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None, seed: int = 0) -> None:
    """Build and run the PPO training loop.

    Args:
        cfg    : full config dict (from base.yaml + CLI overrides)
        resume : path to a .zip checkpoint to resume from (no extension needed)
        seed   : global RNG seed
    """
    ppo_cfg  = cfg['ppo']
    log_cfg  = cfg['logging']
    cur_cfg  = cfg['curriculum']
    eval_cfg = cfg['eval']

    checkpoint_dir = pathlib.Path(log_cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = log_cfg['tensorboard_log']
    pathlib.Path(tb_log_dir).mkdir(parents=True, exist_ok=True)

    # --- Build environments ----------------------------------------------
    print(f"Building {ppo_cfg['n_envs']} parallel environments …")
    venv = build_venv(cfg, seed=seed)

    # Separate eval env (single, deterministic, no reward normalisation)
    eval_venv = build_venv(cfg, seed=seed + 10000, force_dummy=True)
    eval_venv.training       = False   # freeze running stats during eval
    eval_venv.norm_reward    = False   # eval on raw rewards

    # --- Curriculum scheduler --------------------------------------------
    scheduler = CurriculumScheduler(
        advance_threshold     = float(cur_cfg['advance_threshold']),
        rolling_window        = int(cur_cfg['rolling_window']),
        max_episodes_at_stage = (
            int(cur_cfg['max_episodes_at_stage'])
            if cur_cfg.get('max_episodes_at_stage') is not None else None
        ),
    )

    # On resume, restore the curriculum stage BEFORE applying it to venv
    # below -- otherwise a run interrupted at Stage 3 would resume at
    # Stage 0 (no wind/gusts/noise/domain-rand) with no indication why the
    # success rate suddenly looks great (training runbook Phase B.1).
    if resume:
        step_num = _extract_step_count(resume)
        curriculum_resume = (
            pathlib.Path(resume).parent / f"curriculum_{step_num}.json"
            if step_num is not None else None
        )
        if curriculum_resume is not None and curriculum_resume.exists():
            with open(curriculum_resume) as f:
                scheduler.load_state_dict(json.load(f))
            print(f"[Curriculum] RESTORED to Stage {scheduler.current_stage} "
                  f"(episodes_at_stage={scheduler.episodes_at_stage}, "
                  f"episodes_seen={scheduler.episodes_seen}) from {curriculum_resume}")
        else:
            print("[Curriculum] WARNING: no matching curriculum_<steps>.json found "
                  "for this resume checkpoint -- restarting the curriculum at "
                  "Stage 0. This is almost certainly not what you want if "
                  "training had already progressed past Stage 0.")

    # make_env() only seeds each worker with the reward section (flat_reward_cfg);
    # it does not know about STAGES[0]. Without this, training silently starts
    # with none of Stage 0's curriculum settings applied (wind, R_home_m,
    # alt0_m, launch offsets, domain-rand ranges) until/unless the scheduler
    # happens to advance past stage 0, at which point set_stage first fires
    # with Stage 1's config. Apply the initial stage explicitly here, the same
    # way CurriculumCallback applies every later stage advance.
    venv.env_method('set_stage', scheduler.current_cfg)

    # eval_venv is held PERMANENTLY at the final (hardest) stage -- not the
    # scheduler's current stage. CurriculumCallback only ever calls set_stage
    # on `venv` (the training env); if eval_venv started at Stage 0 it would
    # stay there for the whole run, so best_model.zip would be selected on
    # the easiest difficulty forever instead of deployment difficulty.
    eval_venv.env_method('set_stage', dict(STAGES[-1]))

    # --- Policy ----------------------------------------------------------
    policy_kwargs = dict(net_arch=list(ppo_cfg['net_arch']))

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        model = PPO.load(
            resume,
            env         = venv,
            tensorboard_log = tb_log_dir,
            verbose     = int(log_cfg['verbose']),
        )
        # Restore VecNormalize stats if a matching .pkl exists
        vecnorm_resume = _vecnorm_path_for_checkpoint(resume)
        if vecnorm_resume.exists():
            venv = VecNormalize.load(str(vecnorm_resume), venv.venv)
            model.set_env(venv)
            print(f"Loaded VecNormalize stats from {vecnorm_resume}")
        else:
            print(f"Warning: no matching VecNormalize stats found at "
                  f"{vecnorm_resume}; continuing with fresh running stats.")
    else:
        model = PPO(
            policy          = ppo_cfg['policy'],
            env             = venv,
            n_steps         = int(ppo_cfg['n_steps']),
            batch_size      = int(ppo_cfg['batch_size']),
            n_epochs        = int(ppo_cfg['n_epochs']),
            gae_lambda      = float(ppo_cfg['gae_lambda']),
            gamma           = float(ppo_cfg['gamma']),
            ent_coef        = float(ppo_cfg['ent_coef']),
            vf_coef         = float(ppo_cfg['vf_coef']),
            max_grad_norm   = float(ppo_cfg['max_grad_norm']),
            learning_rate   = float(ppo_cfg['learning_rate']),
            clip_range      = float(ppo_cfg['clip_range']),
            policy_kwargs   = policy_kwargs,
            tensorboard_log = tb_log_dir,
            verbose         = int(log_cfg['verbose']),
            seed            = seed,
        )

    # --- Callbacks -------------------------------------------------------
    checkpoint_freq = int(log_cfg['checkpoint_freq'])

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(checkpoint_freq // int(ppo_cfg['n_envs']), 1),
        save_path   = str(checkpoint_dir),
        name_prefix = 'ppo_glider',
        verbose     = 1,
    )

    vecnorm_cb = VecNormCheckpointCallback(
        checkpoint_dir = str(checkpoint_dir),
        save_freq      = max(checkpoint_freq // int(ppo_cfg['n_envs']), 1),
        venv           = venv,
        scheduler      = scheduler,
        name_prefix    = 'vecnormalize',
    )

    curriculum_cb = CurriculumCallback(
        scheduler = scheduler,
        venv      = venv,
        verbose   = 1,
    )

    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path = str(checkpoint_dir),
        log_path             = str(checkpoint_dir / 'eval_logs'),
        eval_freq            = max(int(eval_cfg['eval_freq']) // int(ppo_cfg['n_envs']), 1),
        n_eval_episodes      = int(eval_cfg['n_episodes']),
        deterministic        = bool(eval_cfg['deterministic']),
        callback_on_new_best = SaveVecNormalizeOnBestCallback(
            checkpoint_dir = str(checkpoint_dir),
            venv           = venv,
            verbose        = 1,
        ),
        verbose              = 1,
    )

    callback_list = [curriculum_cb, checkpoint_cb, vecnorm_cb, eval_cb]
    if WebStreamCallback is not None:
        callback_list.append(WebStreamCallback(scheduler=scheduler))
    callbacks = CallbackList(callback_list)

    # --- Train -----------------------------------------------------------
    total_steps = int(ppo_cfg['total_timesteps'])
    print(f"Training for {total_steps:,} timesteps …")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir}")
    print(f"  Checkpoints: {checkpoint_dir}/")

    t0 = time.time()
    model.learn(
        total_timesteps = total_steps,
        callback        = callbacks,
        reset_num_timesteps = (resume is None),
        progress_bar    = False,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f} h  ({elapsed:.0f} s)")

    # --- Save final model + VecNormalize stats ---------------------------
    final_model_path  = str(checkpoint_dir / 'ppo_glider_final')
    final_vecnorm_path = str(checkpoint_dir / 'vecnormalize_final.pkl')
    model.save(final_model_path)
    venv.save(final_vecnorm_path)
    print(f"Saved final model  : {final_model_path}.zip")
    print(f"Saved VecNormalize : {final_vecnorm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RL-guided glider RTL policy with PPO"
    )
    parser.add_argument(
        '--config', default='training/configs/base.yaml',
        help='path to YAML config file (default: training/configs/base.yaml)',
    )
    parser.add_argument(
        '--set', nargs='*', default=[],
        metavar='SECTION.KEY=VALUE',
        help='override config values, e.g. --set ppo.n_envs=4 ppo.total_timesteps=1000000',
    )
    parser.add_argument(
        '--resume', default=None,
        metavar='CHECKPOINT',
        help='path to .zip checkpoint to resume training from',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='global RNG seed (default: 0)',
    )
    parser.add_argument(
        '--dummy-vec', action='store_true',
        help='use DummyVecEnv instead of SubprocVecEnv (useful on Windows/debugging)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    # Windows SubprocVecEnv requires the spawn start method and a
    # __main__ guard — both are satisfied here.
    if args.dummy_vec:
        cfg['ppo']['n_envs'] = 1   # DummyVecEnv handles only 1 env elegantly

    train(cfg=cfg, resume=args.resume, seed=args.seed)

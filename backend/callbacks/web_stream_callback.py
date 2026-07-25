"""
callbacks/web_stream_callback.py
=================================
SB3 callback that streams live training metrics to data/train_stream.jsonl
as newline-delimited JSON (one line per emit). The FastAPI WebSocket in
backend/routers/training.py tails this file from byte 0 on every connection
-- the training subprocess and the web server are different processes with
no shared memory, so a JSONL file on disk is the IPC (see
FRONTEND_BUILD_CONTEXT.md Section 3). Tailing from 0 also means a client
connecting after training has already progressed (or finished) replays the
full history instantly, rather than needing to have been watching live.

Added to training/train.py's callback list ADDITIVELY, alongside
CurriculumCallback / CheckpointCallback / EvalCallback -- never replacing
them (see Section 5, TODO 5).
"""
from __future__ import annotations

import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.curriculum import CurriculumScheduler

from .. import config


class WebStreamCallback(BaseCallback):
    """Appends a TrainingMetric JSON line every `emit_freq` _on_step() calls.

    Args:
        scheduler : the same CurriculumScheduler passed to CurriculumCallback
                    -- read-only here, just observed for stage/success_rate.
        emit_freq : emit every N _on_step() calls, NOT N environment
                    timesteps -- with n_envs parallel workers, num_timesteps
                    advances by n_envs each call.
    """

    def __init__(self, scheduler: CurriculumScheduler, emit_freq: int = 200, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.scheduler = scheduler
        self.emit_freq = emit_freq

    def _on_training_start(self) -> None:
        config.TRAIN_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so this run's stream doesn't mix with a stale previous one.
        config.TRAIN_STREAM_PATH.write_text("")

    def _on_step(self) -> bool:
        if self.n_calls % self.emit_freq != 0:
            return True

        ep_rew_mean = None
        if len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))

        record = dict(
            timesteps=int(self.num_timesteps),
            stage=int(self.scheduler.current_stage),
            success_rate=float(self.scheduler.success_rate),
            ep_rew_mean=ep_rew_mean,
        )
        with open(config.TRAIN_STREAM_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        return True

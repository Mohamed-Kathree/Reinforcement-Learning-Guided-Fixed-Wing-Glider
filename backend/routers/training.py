"""
routers/training.py
====================
POST /api/training/start  -- launches `python -m training.train` as a real
                              subprocess (not a fake progress bar).
POST /api/training/stop   -- terminates the running subprocess, if any.
GET  /api/training/status -- whether a training subprocess is currently running.
WS   /ws/training          -- tails data/train_stream.jsonl from byte 0 on
                              every connection and pushes new lines to the
                              client, polling every 0.5s (the trainer and
                              server are separate processes -- see
                              FRONTEND_BUILD_CONTEXT.md Section 3 and
                              backend/callbacks/web_stream_callback.py).
                              Tailing from 0 doubles as "load a completed
                              run's stream": reconnecting after training
                              has finished replays the whole history at once.
"""
from __future__ import annotations

import asyncio
import subprocess
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import config
from ..schemas import TrainingStatus

router = APIRouter(tags=["training"])

_process: Optional[subprocess.Popen] = None


def _status() -> TrainingStatus:
    running = _process is not None and _process.poll() is None
    return TrainingStatus(running=running, pid=_process.pid if running else None, total_timesteps=None)


@router.post("/api/training/start", response_model=TrainingStatus)
def start_training(total_timesteps: int = 50_000, dummy_vec: bool = True, seed: int = 0) -> TrainingStatus:
    global _process
    if _process is not None and _process.poll() is None:
        return _status()

    cmd = [
        str(config.GLIDER_PYTHON), "-m", "training.train",
        "--set", f"ppo.total_timesteps={total_timesteps}",
        "--seed", str(seed),
    ]
    if dummy_vec:
        cmd.append("--dummy-vec")

    _process = subprocess.Popen(cmd, cwd=str(config.GLIDER_REPO_ROOT))
    return TrainingStatus(running=True, pid=_process.pid, total_timesteps=total_timesteps)


@router.post("/api/training/stop", response_model=TrainingStatus)
def stop_training() -> TrainingStatus:
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
    return _status()


@router.get("/api/training/status", response_model=TrainingStatus)
def training_status() -> TrainingStatus:
    return _status()


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    await websocket.accept()
    path = config.TRAIN_STREAM_PATH
    last_pos = 0
    try:
        while True:
            if path.is_file():
                with open(path, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                for line in new_lines:
                    line = line.strip()
                    if line:
                        await websocket.send_text(line)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

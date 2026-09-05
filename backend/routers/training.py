"""
routers/training.py
====================
POST /api/training/start  -- launches `python -m training.train` as a real
                              subprocess (not a fake progress bar).
POST /api/training/stop   -- terminates the running subprocess, if any.
GET  /api/training/status -- whether a training subprocess is currently running.
GET  /api/training/runs   -- lists past/current runs (data/runs/*.meta.json),
                              newest first -- V16 Phase A §A5.
WS   /ws/training          -- tails one run's JSONL file (data/runs/, see
                              backend/callbacks/web_stream_callback.py) from
                              byte 0 on every connection and pushes new lines
                              to the client, polling every 0.5s (the trainer
                              and server are separate processes, so a file on
                              disk is the IPC). `run` query param selects
                              which run's file by its JSONL filename;
                              omitted (the default) resolves to the newest
                              run in data/runs/. Tailing from 0 doubles as
                              "load a completed run's stream": reconnecting
                              (to either the live run or a past one) replays
                              the whole history at once before continuing to
                              stream any new lines.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import config
from ..schemas import LiveEpisode, TrainingRunMeta, TrainingStatus

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


def _list_run_metas() -> list[tuple[str, TrainingRunMeta]]:
    """(jsonl_filename, meta) pairs for every run in data/runs/, newest first."""
    runs: list[tuple[str, TrainingRunMeta]] = []
    for meta_path in config.RUNS_DIR.glob("*.meta.json"):
        jsonl_name = meta_path.name.removesuffix(".meta.json") + ".jsonl"
        try:
            with open(meta_path) as f:
                raw = json.load(f)
            runs.append((jsonl_name, TrainingRunMeta(run_id=jsonl_name.removesuffix(".jsonl"), **raw)))
        except (OSError, json.JSONDecodeError, TypeError):
            continue   # skip a malformed/partially-written meta file rather than 500ing the whole list
    runs.sort(key=lambda pair: pair[1].start_time, reverse=True)
    return runs


@router.get("/api/training/runs", response_model=list[TrainingRunMeta])
def list_runs() -> list[TrainingRunMeta]:
    return [meta for _, meta in _list_run_metas()]


@router.get("/api/training/live_episode", response_model=LiveEpisode)
def live_episode() -> LiveEpisode:
    """The most recently completed episode's ground track (V16 §A6/§B7) --
    only ever populated for a dashboard-launched (DummyVecEnv) run; see
    backend/callbacks/web_stream_callback.py's class docstring. 404 until
    the first episode of such a run has completed."""
    path = config.DATA_DIR / "live_episode.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no live episode recorded yet")
    with open(path) as f:
        return LiveEpisode.model_validate_json(f.read())


def _resolve_run_path(run: Optional[str]) -> Path:
    if run is not None:
        # `run` is a bare run_id (TrainingRunMeta.run_id / _list_run_metas
        # strip the .jsonl suffix for display), so it has to be added back
        # here to find the actual file on disk.
        path = config.RUNS_DIR / f"{run}.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"run not found: {run}")
        return path

    candidates = sorted(config.RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        # No run has ever started -- return a path that simply never exists
        # yet; the WS loop below tolerates that (it just waits).
        return config.RUNS_DIR / "__no_run_yet__.jsonl"
    return candidates[-1]


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket, run: Optional[str] = None) -> None:
    await websocket.accept()
    path = _resolve_run_path(run)
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

"""
routers/episodes.py
====================
POST /api/episode/record  -- runs one episode, records its TRUE-state
                              trajectory (baseline controller only for now).
GET  /api/episode         -- lists recorded episode ids.
GET  /api/episode/{id}    -- fetches a previously recorded trajectory.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from .. import config, recorder
from ..schemas import Trajectory

router = APIRouter(prefix="/api/episode", tags=["episode"])


@router.post("/record", response_model=Trajectory)
def record(
    controller: str = "baseline",
    stage: int = 0,
    seed: Optional[int] = None,
    wind_speed: Optional[float] = None,
    gust_intensity: Optional[float] = None,
    sensor_noise: Optional[float] = None,
    dropout_prob: Optional[float] = None,
    alt0_m: Optional[float] = None,
    launch_offset_m: Optional[float] = None,
) -> Trajectory:
    try:
        return recorder.record_episode(
            controller=controller,
            stage=stage,
            seed=seed,
            wind_speed=wind_speed,
            gust_intensity=gust_intensity,
            sensor_noise=sensor_noise,
            dropout_prob=dropout_prob,
            alt0_m=alt0_m,
            launch_offset_m=launch_offset_m,
        )
    except (NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=list[str])
def list_episodes() -> list[str]:
    return sorted((p.stem for p in config.EPISODES_DIR.glob("*.json")), reverse=True)


@router.get("/{episode_id}", response_model=Trajectory)
def get_episode(episode_id: str) -> Trajectory:
    try:
        return recorder.load_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"episode not found: {episode_id}")

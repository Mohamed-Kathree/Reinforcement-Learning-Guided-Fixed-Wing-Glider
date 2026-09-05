"""
routers/episodes.py
====================
POST /api/episode/record    -- runs one episode, records its TRUE-state
                                trajectory (baseline or rl controller).
GET  /api/episode           -- lists recorded episode ids.
GET  /api/episode/summaries -- lightweight per-episode summaries for the
                                analysis views (V16 Phase D §D1/§D3) --
                                touchdown position + outcome/quality/stage/
                                controller, without parsing full frame
                                arrays. Registered BEFORE /{episode_id}
                                below so "summaries" doesn't get swallowed
                                as a literal episode_id path param.
GET  /api/episode/{id}      -- fetches a previously recorded trajectory.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from .. import config, recorder
from ..schemas import EpisodeRecordSummary, Trajectory

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
    # V16 Phase D added *.summary.json sidecars alongside each episode's
    # full *.json -- exclude them here or they'd show up as bogus fake
    # episode ids (Path.stem only strips one suffix, so "<id>.summary.json"
    # would otherwise list as "<id>.summary").
    return sorted(
        (p.stem for p in config.EPISODES_DIR.glob("*.json") if not p.name.endswith(".summary.json")),
        reverse=True,
    )


_summaries_cache: tuple[float, list[EpisodeRecordSummary]] | None = None


def _episodes_signature() -> float:
    """Cheap proxy for "has data/episodes/ changed since the last summaries
    call" -- stat() every file (fast: metadata only, no content read/parse)
    rather than re-validating 850+ full sidecar JSON files on every call.
    The archive only changes when an episode is recorded/removed, and the
    Analysis tab (which calls this) is always mounted per App.tsx's design,
    so this endpoint gets hit on every single page load whether or not a
    user ever opens that tab -- worth not redoing the real work every time.
    """
    total = 0.0
    for p in config.EPISODES_DIR.glob("*.json"):
        try:
            total += p.stat().st_mtime
        except OSError:
            pass
    return total


@router.get("/summaries", response_model=list[EpisodeRecordSummary])
def list_summaries() -> list[EpisodeRecordSummary]:
    """Lightweight per-episode summaries for the analysis views (V16 Phase D
    §D1/§D3). Reads each episode's *.summary.json sidecar (written by
    recorder.record_episode() going forward); an episode recorded before
    this existed has none, so its full Trajectory is read ONCE here and the
    sidecar is written for next time -- a self-healing backfill, not a
    separate migration script. The first call after this ships is slow
    (backfilling the whole archive); every call after is fast, and cached
    in-process (see _episodes_signature()) so repeat calls with nothing
    recorded/removed in between don't re-read the whole archive again.
    """
    global _summaries_cache
    signature = _episodes_signature()
    if _summaries_cache is not None and _summaries_cache[0] == signature:
        return _summaries_cache[1]

    summaries: list[EpisodeRecordSummary] = []
    for json_path in sorted(config.EPISODES_DIR.glob("*.json")):
        if json_path.name.endswith(".summary.json"):
            continue
        episode_id = json_path.stem
        summary_path = config.EPISODES_DIR / f"{episode_id}.summary.json"
        if summary_path.is_file():
            try:
                summaries.append(EpisodeRecordSummary.model_validate_json(summary_path.read_text()))
                continue
            except ValueError:
                pass  # corrupt/partial sidecar -- fall through and rebuild it below

        try:
            traj = recorder.load_episode(episode_id)
        except (FileNotFoundError, ValueError):
            continue  # skip an unreadable/corrupt episode rather than 500ing the whole list

        summary = EpisodeRecordSummary(
            episode_id=episode_id,
            stage=traj.meta.stage,
            controller=traj.meta.controller,
            outcome=traj.meta.outcome,
            quality=traj.meta.quality,
            seed=traj.meta.seed,
            R_home=traj.meta.R_home,
            touchdown_ned=traj.frames[-1].pos_ned,
            home_ned=traj.home_ned,
        )
        summary_path.write_text(summary.model_dump_json())
        summaries.append(summary)

    _summaries_cache = (signature, summaries)
    return summaries


@router.get("/{episode_id}", response_model=Trajectory)
def get_episode(episode_id: str) -> Trajectory:
    try:
        return recorder.load_episode(episode_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"episode not found: {episode_id}")

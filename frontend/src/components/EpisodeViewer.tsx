import { useEffect, useState } from "react";
import FlightScene from "./FlightScene";
import AttitudeIndicator from "./AttitudeIndicator";
import type { Trajectory } from "../types";

const API_BASE = "http://localhost:8000";

interface EpisodeViewerProps {
  initialEpisodeId?: string | null;
  label?: string;
}

export default function EpisodeViewer({ initialEpisodeId, label }: EpisodeViewerProps) {
  const [episodeIds, setEpisodeIds] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(initialEpisodeId ?? null);
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stage, setStage] = useState(0);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refreshList(preferId?: string) {
    fetch(`${API_BASE}/api/episode`)
      .then((r) => r.json())
      .then((ids: string[]) => {
        setEpisodeIds(ids);
        if (preferId) setSelected(preferId);
        else if (!selected && ids.length > 0) setSelected(ids[0]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refreshList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (initialEpisodeId) setSelected(initialEpisodeId);
  }, [initialEpisodeId]);

  useEffect(() => {
    if (!selected) return;
    setError(null);
    fetch(`${API_BASE}/api/episode/${selected}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((traj: Trajectory) => {
        setTrajectory(traj);
        setFrameIndex(0);
        setPlaying(false);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [selected]);

  useEffect(() => {
    if (!playing || !trajectory) return;
    const id = setInterval(() => {
      setFrameIndex((i) => {
        if (i >= trajectory.frames.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 50); // matches DT_RL = 20 Hz policy step
    return () => clearInterval(id);
  }, [playing, trajectory]);

  async function handleRecord() {
    setRecording(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/episode/record?controller=baseline&stage=${stage}`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const traj: Trajectory = await res.json();
      refreshList(traj.episode_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRecording(false);
    }
  }

  const frame = trajectory?.frames[frameIndex];

  return (
    <div className="episode-viewer">
      <div className="panel-header">
        {label && <h3 className="viewer-label">{label}</h3>}
        <div className="replay-controls">
          <select value={stage} onChange={(e) => setStage(Number(e.target.value))}>
            {[0, 1, 2, 3].map((s) => (
              <option key={s} value={s}>Stage {s}</option>
            ))}
          </select>
          <button onClick={handleRecord} disabled={recording}>
            {recording ? "Recording…" : "Record baseline"}
          </button>
          <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value)}>
            {episodeIds.length === 0 && <option value="">No recorded episodes</option>}
            {episodeIds.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      {trajectory && (
        <>
          <div className="replay-meta">
            <span>Stage {trajectory.meta.stage}</span>
            <span className={`outcome outcome-${trajectory.meta.outcome}`}>
              {trajectory.meta.outcome}
            </span>
            <span>{trajectory.meta.controller}</span>
            <span>seed {trajectory.meta.seed}</span>
          </div>

          <div className="scene-wrap">
            <FlightScene
              frames={trajectory.frames}
              currentIndex={frameIndex}
              rHome={trajectory.meta.R_home}
            />
            {frame && (
              <div className="hud-overlay">
                <AttitudeIndicator roll={frame.euler[0]} pitch={frame.euler[1]} />
                <div className="hud-readout">
                  <div><span>AGL</span><strong>{frame.agl.toFixed(1)} m</strong></div>
                  <div><span>Home</span><strong>{frame.dist_home.toFixed(1)} m</strong></div>
                </div>
              </div>
            )}
          </div>

          <div className="scrubber-row">
            <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
            <input
              type="range"
              min={0}
              max={trajectory.frames.length - 1}
              value={frameIndex}
              onChange={(e) => {
                setPlaying(false);
                setFrameIndex(Number(e.target.value));
              }}
            />
            <span>{frame ? `t=${frame.t.toFixed(2)}s` : ""}</span>
          </div>

          {frame && (
            <div className="telemetry-grid">
              <div><span>AGL</span><strong>{frame.agl.toFixed(1)} m</strong></div>
              <div><span>Dist home</span><strong>{frame.dist_home.toFixed(1)} m</strong></div>
              <div><span>Roll</span><strong>{((frame.euler[0] * 180) / Math.PI).toFixed(0)}°</strong></div>
              <div><span>Pitch</span><strong>{((frame.euler[1] * 180) / Math.PI).toFixed(0)}°</strong></div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

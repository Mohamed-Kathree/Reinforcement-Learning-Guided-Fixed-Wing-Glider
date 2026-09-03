import { useEffect, useState } from "react";
import FlightScene from "./FlightScene";
import OrientationSphere from "./OrientationSphere";
import type { Controller, Frame, Trajectory } from "../types";

const API_BASE = "http://localhost:8000";

interface EpisodeViewerProps {
  initialEpisodeId?: string | null;
  label?: string;
}

const RAD2DEG = 180 / Math.PI;
const DT_RL = 0.05; // matches env/glider_env.py's policy step (20 Hz)

// Ground/vertical speed aren't recorded directly (only body-frame velocity
// is) -- derive them from consecutive recorded positions instead of doing
// a body->NED rotation client-side. Falls back to the next frame at t=0
// where there's no previous one yet.
function kinematicsAt(frames: Frame[], index: number): { groundSpeed: number; verticalSpeed: number } {
  const a = frames[index > 0 ? index - 1 : index];
  const b = frames[index > 0 ? index : Math.min(index + 1, frames.length - 1)];
  if (!a || !b || a === b) return { groundSpeed: 0, verticalSpeed: 0 };
  const dn = b.pos_ned[0] - a.pos_ned[0];
  const de = b.pos_ned[1] - a.pos_ned[1];
  const dd = b.pos_ned[2] - a.pos_ned[2];
  return {
    groundSpeed: Math.hypot(dn, de) / DT_RL,
    verticalSpeed: -dd / DT_RL, // positive = climbing
  };
}

export default function EpisodeViewer({ initialEpisodeId, label }: EpisodeViewerProps) {
  const [episodeIds, setEpisodeIds] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(initialEpisodeId ?? null);
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stage, setStage] = useState(0);
  const [controller, setController] = useState<Controller>("baseline");
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Custom flight conditions -- when off, the chosen stage's own preset is
  // used unchanged (pre-existing behaviour). When on, these override that
  // preset's wind/gust/noise/dropout/altitude/launch-distance for one
  // episode; R_home/aero-mass-randomisation/landing-grading-strictness
  // still come from `stage` (see backend/recorder.py's record_episode).
  const [useCustom, setUseCustom] = useState(false);
  const [windSpeed, setWindSpeed] = useState(3.0);
  const [gustIntensity, setGustIntensity] = useState(0.5);
  const [sensorNoise, setSensorNoise] = useState(0.3);
  const [dropoutProb, setDropoutProb] = useState(0.05);
  const [alt0, setAlt0] = useState(23);
  const [launchOffset, setLaunchOffset] = useState(60);

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
      const params = new URLSearchParams({ controller, stage: String(stage) });
      if (useCustom) {
        params.set("wind_speed", String(windSpeed));
        params.set("gust_intensity", String(gustIntensity));
        params.set("sensor_noise", String(sensorNoise));
        params.set("dropout_prob", String(dropoutProb));
        params.set("alt0_m", String(alt0));
        params.set("launch_offset_m", String(launchOffset));
      }
      const res = await fetch(
        `${API_BASE}/api/episode/record?${params.toString()}`,
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
          <select value={controller} onChange={(e) => setController(e.target.value as Controller)}>
            <option value="baseline">Baseline (DeterministicRTL)</option>
            <option value="rl">RL policy</option>
          </select>
          <button onClick={handleRecord} disabled={recording}>
            {recording ? "Recording…" : `Record ${controller}`}
          </button>
          <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value)}>
            {episodeIds.length === 0 && <option value="">No recorded episodes</option>}
            {episodeIds.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>
      </div>

      <label className="custom-conditions-toggle">
        <input
          type="checkbox"
          checked={useCustom}
          onChange={(e) => setUseCustom(e.target.checked)}
        />
        Custom flight conditions
      </label>

      {useCustom && (
        <div className="telemetry-box custom-conditions-box">
          <div className="condition-rows">
            <label>
              <span>Wind (max) {windSpeed.toFixed(1)} m/s</span>
              <input type="range" min={0} max={15} step={0.5} value={windSpeed}
                onChange={(e) => setWindSpeed(Number(e.target.value))} />
            </label>
            <label>
              <span>Gusts (max) {gustIntensity.toFixed(1)}</span>
              <input type="range" min={0} max={3} step={0.1} value={gustIntensity}
                onChange={(e) => setGustIntensity(Number(e.target.value))} />
            </label>
            <label>
              <span>Sensor noise (max) {sensorNoise.toFixed(2)}</span>
              <input type="range" min={0} max={1.5} step={0.05} value={sensorNoise}
                onChange={(e) => setSensorNoise(Number(e.target.value))} />
            </label>
            <label>
              <span>Sensor dropout (max) {(dropoutProb * 100).toFixed(0)}%</span>
              <input type="range" min={0} max={0.5} step={0.01} value={dropoutProb}
                onChange={(e) => setDropoutProb(Number(e.target.value))} />
            </label>
            <label>
              <span>Launch altitude {alt0.toFixed(0)} m</span>
              <input type="range" min={10} max={30} step={1} value={alt0}
                onChange={(e) => setAlt0(Number(e.target.value))} />
            </label>
            <label>
              <span>Launch distance {launchOffset.toFixed(0)} m</span>
              <input type="range" min={10} max={150} step={5} value={launchOffset}
                onChange={(e) => setLaunchOffset(Number(e.target.value))} />
            </label>
          </div>
          <p className="hint-text">
            Wind/gusts/noise/dropout are the maximum for this episode — GliderEnv draws the
            actual value uniformly between 0 and this, same as a curriculum stage preset.
            Launch distance is fixed at this value (not randomised).
          </p>
        </div>
      )}

      {error && <p className="error-text">{error}</p>}

      {trajectory && (
        <>
          <div className="replay-meta">
            <span>Stage {trajectory.meta.stage}</span>
            <span className={`outcome outcome-${trajectory.meta.outcome}`}>
              {trajectory.meta.outcome.replace("_", " ")}
            </span>
            <span>quality {trajectory.meta.quality.toFixed(2)}</span>
            <span>final dist {trajectory.meta.final_dist_home.toFixed(1)} m</span>
            <span>{trajectory.meta.controller}</span>
            <span>seed {trajectory.meta.seed}</span>
          </div>
          <div className="replay-meta replay-meta-conditions">
            <span>wind ≤{trajectory.meta.conditions.wind_speed.toFixed(1)} m/s</span>
            <span>gusts ≤{trajectory.meta.conditions.gust_intensity.toFixed(1)}</span>
            <span>noise ≤{trajectory.meta.conditions.sensor_noise.toFixed(2)}</span>
            <span>dropout ≤{(trajectory.meta.conditions.dropout_prob * 100).toFixed(0)}%</span>
            <span>launch {trajectory.meta.alt0.toFixed(0)}m alt, {trajectory.meta.conditions.launch_offset_min_m.toFixed(0)}
              {trajectory.meta.conditions.launch_offset_min_m !== trajectory.meta.conditions.launch_offset_max_m
                ? `–${trajectory.meta.conditions.launch_offset_max_m.toFixed(0)}` : ""}m out</span>
          </div>

          <div className="replay-layout">
            <div className="replay-visual">
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

              <div className="scene-wrap">
                <FlightScene
                  frames={trajectory.frames}
                  currentIndex={frameIndex}
                  rHome={trajectory.meta.R_home}
                />
              </div>
            </div>

            <div className="replay-side">
              {frame && (
                <OrientationSphere roll={frame.euler[0]} pitch={frame.euler[1]} yaw={frame.euler[2]} />
              )}

              {frame && (() => {
                const { groundSpeed, verticalSpeed } = kinematicsAt(trajectory.frames, frameIndex);
                return (
                  <div className="telemetry-box">
                    <h4>Aircraft</h4>
                    <div className="telemetry-rows">
                      <div><span>Time</span><strong>{frame.t.toFixed(2)} s</strong></div>
                      <div><span>AGL</span><strong>{frame.agl.toFixed(1)} m</strong></div>
                      <div><span>Dist home</span><strong>{frame.dist_home.toFixed(1)} m</strong></div>
                      <div><span>Ground speed</span><strong>{groundSpeed.toFixed(1)} m/s</strong></div>
                      <div><span>Vertical speed</span><strong>{verticalSpeed.toFixed(1)} m/s</strong></div>
                      <div><span>Roll</span><strong>{(frame.euler[0] * RAD2DEG).toFixed(0)}°</strong></div>
                      <div><span>Pitch</span><strong>{(frame.euler[1] * RAD2DEG).toFixed(0)}°</strong></div>
                      <div><span>Yaw</span><strong>{(frame.euler[2] * RAD2DEG).toFixed(0)}°</strong></div>
                      <div><span>Body vel (u,v,w)</span><strong>
                        {frame.v_body[0].toFixed(1)}, {frame.v_body[1].toFixed(1)}, {frame.v_body[2].toFixed(1)} m/s
                      </strong></div>
                      <div><span>Aileron</span><strong>{(frame.ctrl.ail * RAD2DEG).toFixed(0)}°</strong></div>
                      <div><span>Elevator</span><strong>{(frame.ctrl.elev * RAD2DEG).toFixed(0)}°</strong></div>
                      <div><span>Rudder</span><strong>{(frame.ctrl.rud * RAD2DEG).toFixed(0)}°</strong></div>
                    </div>
                  </div>
                );
              })()}

              {frame && (() => {
                const [wn, we, wd] = frame.wind_ned ?? [0, 0, 0];
                const windSpeed = Math.hypot(wn, we);
                const bearingDeg = ((Math.atan2(we, wn) * RAD2DEG) + 360) % 360;
                return (
                  <div className="telemetry-box">
                    <h4>Environment</h4>
                    <div className="telemetry-rows">
                      <div><span>Wind speed</span><strong>{windSpeed.toFixed(1)} m/s</strong></div>
                      <div><span>Wind bearing (to)</span><strong>{bearingDeg.toFixed(0)}°</strong></div>
                      <div><span>Vertical wind</span><strong>{(-wd).toFixed(2)} m/s</strong></div>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

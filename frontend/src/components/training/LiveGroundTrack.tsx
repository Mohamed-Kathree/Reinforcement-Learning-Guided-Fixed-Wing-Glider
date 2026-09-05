// LiveGroundTrack.tsx
// ====================
// V16 Phase B §B7 (data from §A6, backend/callbacks/web_stream_callback.py)
// -- a compact top-down (North/East plane) view of the most recently
// completed training episode: ground track, home marker, R_home circle,
// touchdown point coloured by outcome. Only ever populated for a
// dashboard-launched (DummyVecEnv) run -- see the backend's own docstring
// for why a CLI SubprocVecEnv run can't supply this; shown as an empty
// state here, not an error.
//
// Plain SVG, not three.js/FlightScene.tsx -- this is a small always-visible
// panel widget, not the main replay viewer, and only needs a 2D top-down
// projection. Reuses the same North/East-plane convention nedThree.ts
// documents for the 3D view (x = East, y = -North for "north-up"), just
// without going through a WebGL scene for it.
import { useEffect, useRef, useState } from "react";
import ExportButtons from "../ExportButtons";
import type { LiveEpisode } from "../../types";

const API_BASE = "http://localhost:8000";
const POLL_MS = 3000;
const SIZE = 220;

const OUTCOME_COLORS: Record<string, string> = {
  success: "var(--success)",
  soft_landing: "var(--accent)",
  crash: "var(--danger)",
  timeout: "var(--warning)",
};

export default function LiveGroundTrack() {
  const [episode, setEpisode] = useState<LiveEpisode | null>(null);
  const [notFound, setNotFound] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      fetch(`${API_BASE}/api/training/live_episode`)
        .then((r) => {
          if (r.status === 404) {
            if (!cancelled) {
              setNotFound(true);
              setEpisode(null);
            }
            return null;
          }
          if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
          return r.json();
        })
        .then((data: LiveEpisode | null) => {
          if (!cancelled && data) {
            setNotFound(false);
            setEpisode(data);
          }
        })
        .catch(() => {
          /* transient fetch error -- keep showing the last-known episode */
        });
    }
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (notFound && !episode) {
    return (
      <div className="telemetry-box live-ground-track">
        <h4>Live rollout</h4>
        <p className="placeholder">
          No episode recorded yet — this view only populates for a dashboard-launched run.
        </p>
      </div>
    );
  }

  if (!episode || episode.frames.length === 0) {
    return (
      <div className="telemetry-box live-ground-track">
        <h4>Live rollout</h4>
        <p className="placeholder">Waiting for the first completed episode…</p>
      </div>
    );
  }

  // NED -> this widget's 2D plane: x = East, y = -North (so "up" on screen
  // is North, matching a conventional map), scaled to fit SIZE with a
  // margin, centred on home (origin).
  const positions = episode.frames.map((f) => ({ n: f.pos_ned[0], e: f.pos_ned[1] }));
  const maxAbs = Math.max(
    episode.R_home,
    ...positions.map((p) => Math.abs(p.n)),
    ...positions.map((p) => Math.abs(p.e)),
    1,
  );
  const margin = 12;
  const scale = (SIZE / 2 - margin) / maxAbs;
  const toXY = (n: number, e: number) => ({
    x: SIZE / 2 + e * scale,
    y: SIZE / 2 - n * scale,
  });

  const pathD = positions
    .map((p, i) => `${i === 0 ? "M" : "L"} ${toXY(p.n, p.e).x.toFixed(1)} ${toXY(p.n, p.e).y.toFixed(1)}`)
    .join(" ");
  const last = positions[positions.length - 1];
  const lastXY = toXY(last.n, last.e);
  const homeXY = toXY(0, 0);
  const color = episode.outcome ? (OUTCOME_COLORS[episode.outcome] ?? "var(--text-dim)") : "var(--text-dim)";

  return (
    <div className="telemetry-box live-ground-track">
      <h4>Live rollout</h4>
      <svg ref={svgRef} viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}>
        <circle
          cx={homeXY.x}
          cy={homeXY.y}
          r={episode.R_home * scale}
          fill="none"
          stroke="var(--success)"
          strokeOpacity={0.4}
          strokeWidth={1}
        />
        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
        <circle cx={homeXY.x} cy={homeXY.y} r={3} fill="var(--text-dim)" />
        <circle cx={lastXY.x} cy={lastXY.y} r={4} fill={color} />
      </svg>
      <div className="telemetry-rows">
        <div><span>Outcome</span><strong className={`outcome outcome-${episode.outcome ?? "timeout"}`}>
          {(episode.outcome ?? "timeout").replace("_", " ")}
        </strong></div>
        <div><span>Quality</span><strong>{episode.quality.toFixed(2)}</strong></div>
        <div><span>Dist home</span><strong>{episode.dist_home.toFixed(1)} m</strong></div>
        <div><span>Stage</span><strong>{episode.stage}</strong></div>
      </div>
      <ExportButtons filename="live-rollout-ground-track" getSvg={() => svgRef.current} />
    </div>
  );
}

// TouchdownScatter.tsx
// =====================
// V16 Phase D §D1 -- every recorded episode's touchdown point relative to
// home, with the R_home circle(s) drawn. The most honest single visual of
// the actual objective, and far more informative than a success percentage
// (per the spec). Same North=up/East=right top-down convention
// frontend/src/components/training/LiveGroundTrack.tsx already established.
// `summaries` is fetched once by the parent AnalysisPanel and shared with
// FailureGallery, not re-fetched here -- see AnalysisPanel.tsx's docstring.
import { useRef, useState } from "react";
import ExportButtons from "./ExportButtons";
import type { EpisodeRecordSummary } from "../types";

const SIZE = 420;
const PADDING = 40; // px reserved for axis labels/ticks

const OUTCOME_VAR: Record<string, string> = {
  success: "var(--success)",
  soft_landing: "var(--accent)",
  crash: "var(--danger)",
  timeout: "var(--warning)",
};

interface TouchdownScatterProps {
  summaries: EpisodeRecordSummary[] | null;
  error: string | null;
}

export default function TouchdownScatter({ summaries, error }: TouchdownScatterProps) {
  const [stageFilter, setStageFilter] = useState<number | "all">("all");
  const svgRef = useRef<SVGSVGElement>(null);

  if (error) return <p className="error-text">{error}</p>;
  if (!summaries) return <p className="placeholder">Loading episode archive…</p>;
  if (summaries.length === 0) {
    return <p className="placeholder">No recorded episodes yet — record one from Baseline or Flight Replay.</p>;
  }

  const filtered = stageFilter === "all" ? summaries : summaries.filter((s) => s.stage === stageFilter);
  const points = filtered.map((s) => ({
    north: s.touchdown_ned[0] - s.home_ned[0],
    east: s.touchdown_ned[1] - s.home_ned[1],
    outcome: s.outcome,
    id: s.episode_id,
  }));
  const radii = Array.from(new Set(filtered.map((s) => s.R_home))).sort((a, b) => a - b);

  // Scale to fit every point plus every R_home circle, with a little margin.
  const maxAbs = Math.max(
    10,
    ...points.map((p) => Math.max(Math.abs(p.north), Math.abs(p.east))),
    ...radii,
  ) * 1.15;
  const plotSize = SIZE - 2 * PADDING;
  const scale = plotSize / (2 * maxAbs);
  const cx = SIZE / 2;
  const cy = SIZE / 2;
  const toXY = (north: number, east: number) => ({ x: cx + east * scale, y: cy - north * scale });

  const counts = points.reduce<Record<string, number>>((acc, p) => {
    acc[p.outcome] = (acc[p.outcome] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="analysis-chart">
      <div className="analysis-chart-header">
        <label>
          Stage{" "}
          <select value={stageFilter} onChange={(e) => setStageFilter(e.target.value === "all" ? "all" : Number(e.target.value))}>
            <option value="all">All</option>
            {[0, 1, 2, 3].map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <ExportButtons filename="touchdown-scatter" getSvg={() => svgRef.current} />
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE} className="touchdown-scatter-svg">
        <rect x={0} y={0} width={SIZE} height={SIZE} className="scatter-bg" />
        {radii.map((r) => (
          <circle key={r} cx={cx} cy={cy} r={r * scale} className="scatter-r-home" />
        ))}
        <line x1={PADDING} y1={cy} x2={SIZE - PADDING} y2={cy} className="scatter-axis" />
        <line x1={cx} y1={PADDING} x2={cx} y2={SIZE - PADDING} className="scatter-axis" />
        <text x={SIZE - PADDING + 4} y={cy + 4} className="scatter-axis-label">E (m)</text>
        <text x={cx + 4} y={PADDING - 6} className="scatter-axis-label">N (m)</text>
        <text x={SIZE / 2} y={SIZE - 8} className="scatter-axis-label" textAnchor="middle">
          home at centre · R_home {radii.map((r) => `${r.toFixed(0)}m`).join(" / ")}
        </text>
        {points.map((p) => {
          const { x, y } = toXY(p.north, p.east);
          return (
            <circle
              key={p.id}
              cx={x}
              cy={y}
              r={3.5}
              fill={OUTCOME_VAR[p.outcome] ?? "var(--text-faint)"}
              opacity={0.75}
            >
              <title>{`${p.id} — ${p.outcome}`}</title>
            </circle>
          );
        })}
      </svg>

      <div className="analysis-chart-legend">
        {Object.entries(counts).map(([outcome, n]) => (
          <span key={outcome} className={`outcome outcome-${outcome}`}>
            {outcome.replace("_", " ")} × {n}
          </span>
        ))}
      </div>
    </div>
  );
}

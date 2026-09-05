// MetricsGrid.tsx
// ================
// V16 Phase B §B3 -- one chart per metric, all sharing an x-axis and cursor
// (via UplotChart's shared sync group). Each tile: name, current value as a
// large tabular-numeral readout, sparkline-scale chart. Clicking a tile
// expands it to full width.
import { useRef, useState } from "react";
import UplotChart, { type Series } from "./UplotChart";
import ExportButtons from "../ExportButtons";
import { METRIC_KEYS, type MetricKey, type RunSnapshot } from "./useMetricsStream";

const METRIC_LABELS: Record<MetricKey, string> = {
  success_rate: "Success rate",
  ep_rew_mean: "Episode return",
  ep_len_mean: "Episode length",
  explained_variance: "Explained variance",
  approx_kl: "Approx KL",
  clip_fraction: "Clip fraction",
  entropy_loss: "Entropy loss",
  value_loss: "Value loss",
};

const RUN_COLORS = ["#22d3ee", "#4caf50", "#e0a930", "#f4543c", "#d946ef", "#4dd0e1"];

function formatValue(key: MetricKey, v: number | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (key === "success_rate") return `${(v * 100).toFixed(0)}%`;
  if (key === "ep_len_mean") return v.toFixed(0);
  return v.toFixed(v < 1 && v > -1 ? 4 : 2);
}

interface MetricsGridProps {
  runs: RunSnapshot[]; // runs[0] is the primary/live run; the rest are ghost overlays (§B8)
}

export default function MetricsGrid({ runs }: MetricsGridProps) {
  const [expanded, setExpanded] = useState<MetricKey | null>(null);
  const primary = runs[0];
  // V16 Phase D §D4 -- one ref per tile's chart-mount wrapper, so the export
  // button can reach in and grab uPlot's own <canvas> (uPlot draws its axis
  // labels/units directly onto that canvas, so a PNG snapshot of it already
  // stands alone).
  const tileRefs = useRef<Partial<Record<MetricKey, HTMLDivElement | null>>>({});

  const visibleKeys = expanded ? [expanded] : METRIC_KEYS;

  return (
    <div className={`metrics-grid${expanded ? " metrics-grid-expanded" : ""}`}>
      {visibleKeys.map((key) => {
        const series: Series[] = runs.map((run, i) => ({
          label: run.runId === "" ? "live" : run.runId,
          points: run.points[key] ?? [],
          color: RUN_COLORS[i % RUN_COLORS.length],
          dashed: i > 0,
        }));
        const currentValue = primary?.latest
          ? ((primary.latest as unknown as Record<string, number | null>)[key] ?? null)
          : null;

        return (
          <div key={key} className="metric-tile" onClick={() => setExpanded(expanded === key ? null : key)}>
            <div className="metric-tile-header">
              <span className="metric-tile-label">{METRIC_LABELS[key]}</span>
              <span className="metric-tile-value">{formatValue(key, currentValue)}</span>
            </div>
            <div ref={(el) => { tileRefs.current[key] = el; }}>
              <UplotChart series={series} stageBands={primary?.stageBands} height={expanded ? 360 : 140} />
            </div>
            {/* stopPropagation -- otherwise a click here also bubbles to the
                tile's own onClick and toggles the expand/collapse state. */}
            <div onClick={(e) => e.stopPropagation()}>
              <ExportButtons
                filename={`training-${key}`}
                getCanvas={() => tileRefs.current[key]?.querySelector("canvas") ?? null}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

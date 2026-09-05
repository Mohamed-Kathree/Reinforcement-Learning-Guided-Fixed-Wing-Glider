// UplotChart.tsx
// ===============
// V16 Phase B §B2 -- thin uPlot wrapper, replacing the hand-rolled SVG chart
// (D4: no axes, ticks, tooltips, or cursor) with a real charting library
// that handles thousands of points redrawn at 4 Hz without falling over.
//
// Mirrors FlightScene.tsx's own mount-once-in-a-useEffect / cleanup-on-
// unmount pattern already established in this codebase -- the uPlot
// instance is built ONCE per mount and updated via .setData()/.setSize() on
// prop changes, never destroyed and rebuilt per update (that repeated
// destroy/rebuild was the old SVG approach's actual per-message cost, not
// just "SVG is slow").
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Point } from "./lttb";
import type { StageBand } from "./useMetricsStream";
import { stageBandsPlugin } from "./stageBandsPlugin";

// One shared sync group so every chart in the grid shares a single cursor/
// zoom (§B3's "shared cursor synchronised across every chart").
const SYNC_KEY = "rl-glider-training-charts";

export interface Series {
  label: string;
  points: Point[];
  color: string;
  dashed?: boolean; // ghost/overlay series (V16 §B8 multi-run overlay)
}

interface UplotChartProps {
  series: Series[];
  stageBands?: StageBand[];
  height?: number;
  valueFormatter?: (v: number) => string;
}

export default function UplotChart({ series, stageBands, height = 180, valueFormatter }: UplotChartProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const bandsRef = useRef<StageBand[]>(stageBands ?? []);
  bandsRef.current = stageBands ?? [];

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const opts: uPlot.Options = {
      width: mount.clientWidth || 300,
      height,
      cursor: { sync: { key: SYNC_KEY } },
      legend: { show: true },
      scales: { x: { time: false } },
      axes: [
        { stroke: "#a7a8b3", grid: { stroke: "#26262f", width: 1 } },
        {
          stroke: "#a7a8b3",
          grid: { stroke: "#26262f", width: 1 },
          values: (_u, ticks) => ticks.map((t) => (valueFormatter ? valueFormatter(t) : String(t))),
        },
      ],
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: 1.5,
          dash: s.dashed ? [4, 3] : undefined,
          points: { show: false },
        })),
      ],
      plugins: [stageBandsPlugin(() => bandsRef.current)],
    };

    const data = buildUplotData(series);
    const plot = new uPlot(opts, data, mount);
    plotRef.current = plot;

    const resizeObserver = new ResizeObserver(() => {
      if (mount.clientWidth > 0) plot.setSize({ width: mount.clientWidth, height });
    });
    resizeObserver.observe(mount);

    return () => {
      resizeObserver.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Series identity (which metrics, in what order) changing requires a
  // rebuild; data updates within the same series set use setData().
  const seriesKey = series.map((s) => s.label).join("|");
  const prevSeriesKeyRef = useRef(seriesKey);
  useEffect(() => {
    if (!plotRef.current) return;
    if (prevSeriesKeyRef.current !== seriesKey) {
      // Series set changed (e.g. overlay runs added/removed) -- rebuild via
      // remount rather than uPlot's own addSeries/delSeries bookkeeping,
      // simpler and this only happens on an explicit user action (B8
      // select/deselect), not on the hot data-update path.
      prevSeriesKeyRef.current = seriesKey;
      return;
    }
    plotRef.current.setData(buildUplotData(series));
  }, [series, seriesKey]);

  return <div ref={mountRef} className="uplot-chart" style={{ width: "100%" }} />;
}

function buildUplotData(series: Series[]): uPlot.AlignedData {
  // uPlot wants one shared x-array; our series may have been LTTB-sampled
  // independently and so can have different x positions. Union + sort the
  // x values, then re-index each series against that union (null where a
  // series has no point at that x) -- correct or not, per-series
  // downsampling already trades exactness for a bounded point budget, so a
  // slightly different LTTB selection per series here is consistent with
  // that, not a new approximation.
  const xSet = new Set<number>();
  for (const s of series) for (const p of s.points) xSet.add(p.x);
  const xs = Array.from(xSet).sort((a, b) => a - b);

  const ys = series.map((s) => {
    const byX = new Map(s.points.map((p) => [p.x, p.y]));
    return xs.map((x) => byX.get(x) ?? null);
  });

  return [xs, ...ys] as unknown as uPlot.AlignedData;
}

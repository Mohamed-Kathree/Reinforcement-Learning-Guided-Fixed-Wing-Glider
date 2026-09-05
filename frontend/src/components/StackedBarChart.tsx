// StackedBarChart.tsx
// =====================
// V16 Phase D §D4 -- SVG rebuild of BaselinePanel's outcome-breakdown bar
// (was plain HTML/CSS divs, not exportable). Same 4-way outcome colours as
// the rest of the app (bar-success/soft_landing/crash/timeout in App.css),
// now expressed as real SVG so it can carry axis labels/units standalone
// when exported (per the spec's D4 requirement).
import { useRef } from "react";
import ExportButtons from "./ExportButtons";

interface Segment {
  key: string;
  value: number;
  label: string;
  colorVar: string;
}

interface StackedBarChartProps {
  segments: Segment[];
  title: string;
  width?: number;
  exportName: string;
}

export default function StackedBarChart({ segments, title, width = 520, exportName }: StackedBarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  const barHeight = 22;
  const height = barHeight + 22;

  let x = 0;
  const rects = segments.map((s) => {
    const w = (s.value / total) * width;
    const rect = (
      <rect key={s.key} x={x} y={20} width={Math.max(w, 0)} height={barHeight} fill={s.colorVar}>
        <title>{`${s.label}: ${s.value} (${((s.value / total) * 100).toFixed(0)}%)`}</title>
      </rect>
    );
    x += w;
    return rect;
  });

  return (
    <div className="stacked-bar-chart">
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="stacked-bar-svg">
        <text x={0} y={12} className="scatter-axis-label">{title} (n={total})</text>
        {rects}
      </svg>
      <ExportButtons filename={exportName} getSvg={() => svgRef.current} />
    </div>
  );
}

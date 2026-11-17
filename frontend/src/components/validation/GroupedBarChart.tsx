// GroupedBarChart.tsx
// ===================
// Grouped vertical bars, one group per wind stage, one bar per controller.
// Plain SVG (no chart dependency), scaled by viewBox so it fills its card at
// any width. Every bar is focusable and has a tooltip and an aria-label, so
// values are reachable without a mouse and without relying on colour.
import { useEffect, useRef, useState } from "react";
import ExportButtons from "../ExportButtons";

export interface Series {
  key: string;
  label: string;
  color: string;        // CSS colour, usually a var(--series-*)
  selected?: boolean;   // drawn with value labels and a heavier outline
}

export interface Group {
  key: string;
  label: string;
  sublabel?: string;
  values: Record<string, number>;               // series key -> fraction 0..1
  ci?: Record<string, [number, number]>;        // optional 95% interval per series
}

interface Props {
  series: Series[];
  groups: Group[];
  yMax?: number;          // fraction; default 1
  yTitle: string;
  title: string;          // used for the accessible name and export
  exportName: string;
  valueFormat?: (v: number) => string;
}

const H = 280;
const M = { top: 18, right: 8, bottom: 44, left: 44 };
const MIN_W = 300;

export default function GroupedBarChart({
  series, groups, yMax = 1, yTitle, title, exportName,
  valueFormat = (v) => `${Math.round(v * 100)}%`,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  // Lay out at the container's real width, so labels keep their size on a
  // phone instead of shrinking with a scaled viewBox.
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(MIN_W, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const [tip, setTip] = useState<{ x: number; y: number; text: string; value: string } | null>(null);

  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;
  const groupW = innerW / groups.length;
  const groupPad = groupW * 0.18;
  const barW = (groupW - groupPad * 2) / series.length;
  const y = (v: number) => M.top + innerH * (1 - Math.min(v, yMax) / yMax);
  const ticks = Array.from({ length: 5 }, (_, i) => (yMax * i) / 4);

  function show(e: { currentTarget: SVGRectElement }, text: string, value: string) {
    const svg = svgRef.current;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const r = e.currentTarget.getBoundingClientRect();
    setTip({ x: r.left - box.left + r.width / 2, y: r.top - box.top, text, value });
  }

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="grouped-bar-svg"
        role="img"
        aria-label={title}
        onMouseLeave={() => setTip(null)}
      >
        <title>{title}</title>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gb-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="gb-tick" x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {valueFormat(t)}
            </text>
          </g>
        ))}
        <line className="gb-axis" x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} />
        <text
          className="gb-axis-title"
          transform={`translate(12 ${M.top + innerH / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yTitle}
        </text>

        {groups.map((g, gi) => {
          const gx = M.left + gi * groupW;
          return (
            <g key={g.key} className="gb-group">
              {series.map((s, si) => {
                const v = g.values[s.key];
                const x = gx + groupPad + si * barW;
                const top = y(v);
                const h = Math.max(y(0) - top, v > 0 ? 1.5 : 0);
                const ci = g.ci?.[s.key];
                const text = `${g.label} · ${s.label}`;
                const value = valueFormat(v) + (ci ? ` (95% CI ${valueFormat(ci[0])}–${valueFormat(ci[1])})` : "");
                return (
                  <g key={s.key}>
                    <rect
                      className="gb-bar"
                      x={x + 1.5}
                      y={y(0) - h}
                      width={barW - 3}
                      height={h}
                      rx={3}
                      fill={s.color}
                      tabIndex={0}
                      aria-label={`${text}: ${value}`}
                      onMouseEnter={(e) => show(e, text, value)}
                      onFocus={(e) => show(e, text, value)}
                      onBlur={() => setTip(null)}
                    />
                    {v === 0 && (
                      <line className="gb-axis" x1={x + 3} x2={x + barW - 3} y1={y(0) - 1} y2={y(0) - 1}
                        stroke={s.color} strokeWidth={2} />
                    )}
                    {ci && (
                      <g aria-hidden="true">
                        <line className="gb-ci" x1={x + barW / 2} x2={x + barW / 2} y1={y(ci[1])} y2={y(ci[0])} />
                        <line className="gb-ci" x1={x + barW / 2 - 3} x2={x + barW / 2 + 3} y1={y(ci[1])} y2={y(ci[1])} />
                        <line className="gb-ci" x1={x + barW / 2 - 3} x2={x + barW / 2 + 3} y1={y(ci[0])} y2={y(ci[0])} />
                      </g>
                    )}
                    {s.selected && (
                      <text
                        className="gb-value gb-value-selected"
                        x={x + barW / 2}
                        y={(ci ? y(ci[1]) : top) - 5}
                        textAnchor="middle"
                        aria-hidden="true"
                      >
                        {valueFormat(v)}
                      </text>
                    )}
                  </g>
                );
              })}
              <text className="gb-cat" x={gx + groupW / 2} y={H - M.bottom + 18} textAnchor="middle">
                {g.label}
              </text>
              {g.sublabel && (
                <text className="gb-cat-sub" x={gx + groupW / 2} y={H - M.bottom + 33} textAnchor="middle">
                  {g.sublabel}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="chart-tooltip" style={{ left: tip.x, top: tip.y }} role="presentation">
          {tip.text}: <strong>{tip.value}</strong>
        </div>
      )}
      <ExportButtons filename={exportName} getSvg={() => svgRef.current} />
    </div>
  );
}

export function ChartLegend({ series }: { series: Series[] }) {
  return (
    <ul className="chart-legend">
      {series.map((s) => (
        <li key={s.key} className={s.selected ? "legend-selected" : undefined}>
          <span className="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
          {s.label}
          {s.selected && <span className="badge badge-accent">Selected</span>}
        </li>
      ))}
    </ul>
  );
}

// GateTraceChart.tsx
// ==================
// The wind gate's view of one recorded flight: the on-board wind estimate
// (frames[].wind_est, recorded for "gated" flights) against the threshold,
// with the decision window shaded and the handover marked. A playhead follows
// the replay; clicking or using arrow keys on the chart seeks.
import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";
import type { Frame, GateRecord } from "../types";
import { GATE } from "../data/validation";

interface Props {
  frames: Frame[];
  currentIndex: number;
  gate: GateRecord;
  onSeek?: (index: number) => void;
}

const H = 150;
const M = { top: 12, right: 12, bottom: 26, left: 40 };

export default function GateTraceChart({ frames, currentIndex, gate, onSeek }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(600);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pts = frames.map((f, i) => ({ i, t: f.t, w: f.wind_est })).filter((p) => p.w != null) as
    { i: number; t: number; w: number }[];
  if (pts.length === 0) return null;

  const tMax = Math.max(frames[frames.length - 1].t, GATE.t_latest_s + 2);
  const wMax = Math.max(GATE.threshold_ms * 2, ...pts.map((p) => p.w)) * 1.08;
  const x = (t: number) => M.left + ((W - M.left - M.right) * t) / tMax;
  const y = (w: number) => M.top + (H - M.top - M.bottom) * (1 - w / wMax);
  const path = pts.map((p, k) => `${k ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.w).toFixed(1)}`).join("");
  const now = frames[Math.min(currentIndex, frames.length - 1)];
  const nowW = now?.wind_est;
  const yTicks = [0, GATE.threshold_ms, Math.round(wMax)].filter((v, i, a) => a.indexOf(v) === i && v <= wMax);
  const tStep = tMax > 40 ? 10 : 5;
  const tTicks = Array.from({ length: Math.floor(tMax / tStep) + 1 }, (_, k) => k * tStep);

  function seekToT(t: number) {
    if (!onSeek) return;
    let best = 0;
    for (let k = 0; k < frames.length; k++) if (Math.abs(frames[k].t - t) < Math.abs(frames[best].t - t)) best = k;
    onSeek(best);
  }
  function onClick(e: MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    seekToT(((px - M.left) / (W - M.left - M.right)) * tMax);
  }
  function onKey(e: KeyboardEvent<SVGSVGElement>) {
    if (!onSeek) return;
    const step = e.shiftKey ? 20 : 4;
    if (e.key === "ArrowRight") onSeek(Math.min(frames.length - 1, currentIndex + step));
    else if (e.key === "ArrowLeft") onSeek(Math.max(0, currentIndex - step));
    else return;
    e.preventDefault();
  }

  return (
    <div className="gate-trace" ref={wrapRef}>
      <div className="gate-trace-head">
        <h4>Wind gate</h4>
        <span>
          Estimate now{" "}
          <strong>{nowW == null ? "–" : `${nowW.toFixed(2)} m/s`}</strong>
          {" · "}threshold {GATE.threshold_ms} m/s
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="gate-trace-svg"
        role="slider"
        tabIndex={onSeek ? 0 : -1}
        aria-label="On-board wind estimate over the flight; click or use arrow keys to seek"
        aria-valuemin={0}
        aria-valuemax={Math.round(tMax)}
        aria-valuenow={now ? Math.round(now.t) : 0}
        onClick={onClick}
        onKeyDown={onKey}
        style={{ cursor: onSeek ? "pointer" : "default" }}
      >
        <rect className="gt-window" x={x(GATE.t_earliest_s)} y={M.top} width={x(GATE.t_latest_s) - x(GATE.t_earliest_s)}
          height={H - M.top - M.bottom} />
        {yTicks.map((v) => (
          <g key={v}>
            <line className="gb-grid" x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} />
            <text className="gb-tick" x={M.left - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        <line className="gt-threshold" x1={M.left} x2={W - M.right} y1={y(GATE.threshold_ms)} y2={y(GATE.threshold_ms)} />
        {tTicks.map((t) => (
          <text key={t} className="gb-tick" x={x(t)} y={H - 8} textAnchor="middle">{t} s</text>
        ))}
        <path d={path} className="gt-line" />
        {gate.handed_over && gate.handover_time_s != null && (
          <g>
            <line className="gt-handover" x1={x(gate.handover_time_s)} x2={x(gate.handover_time_s)} y1={M.top} y2={H - M.bottom} />
            <text className="gt-label" x={x(gate.handover_time_s) + 4} y={M.top + 10}>handover</text>
          </g>
        )}
        {now && <line className="gt-playhead" x1={x(now.t)} x2={x(now.t)} y1={M.top} y2={H - M.bottom} />}
        {now && nowW != null && <circle className="gt-dot" cx={x(now.t)} cy={y(nowW)} r={3.5} />}
      </svg>
      <ul className="chart-legend" style={{ margin: "var(--space-1) 0 0" }}>
        <li><span className="legend-swatch" style={{ background: "var(--series-selected)" }} />Wind estimate</li>
        <li><span className="legend-swatch" style={{ background: "var(--warning-fill)" }} />Threshold</li>
        <li><span className="legend-swatch" style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-fill)" }} />Window {GATE.t_earliest_s}–{GATE.t_latest_s} s</li>
      </ul>
    </div>
  );
}

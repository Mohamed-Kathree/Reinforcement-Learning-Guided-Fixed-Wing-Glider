// StatusBar.tsx
// ==============
// V16 Phase B §B6 -- running/idle, timesteps-vs-target progress, throughput,
// ETA, current stage, success rate vs advance threshold, elapsed wall-clock.
import { useEffect, useRef, useState } from "react";
import type { TrainingMetric, TrainingRunMeta, TrainingStatus } from "../../types";

interface StatusBarProps {
  status: TrainingStatus | null;
  runMeta: TrainingRunMeta | null;
  latest: TrainingMetric | null;
}

function useThroughput(latest: TrainingMetric | null): number | null {
  const historyRef = useRef<{ t: number; steps: number }[]>([]);
  const [sps, setSps] = useState<number | null>(null);

  useEffect(() => {
    if (!latest) return;
    const now = performance.now();
    const history = historyRef.current;
    history.push({ t: now, steps: latest.timesteps });
    // Keep ~10s of history for a lightly smoothed rate.
    while (history.length > 2 && now - history[0].t > 10_000) history.shift();
    if (history.length >= 2) {
      const dt = (now - history[0].t) / 1000;
      const dSteps = latest.timesteps - history[0].steps;
      if (dt > 0.5) setSps(dSteps / dt);
    }
  }, [latest]);

  return sps;
}

function formatElapsed(startIso: string | undefined): string {
  if (!startIso) return "—";
  const ms = Date.now() - new Date(startIso).getTime();
  if (ms < 0) return "—";
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}

export default function StatusBar({ status, runMeta, latest }: StatusBarProps) {
  const sps = useThroughput(latest);
  const target = runMeta?.total_timesteps ?? status?.total_timesteps ?? null;
  const timesteps = latest?.timesteps ?? 0;
  const progressPct = target ? Math.min((timesteps / target) * 100, 100) : 0;
  const remaining = target ? Math.max(target - timesteps, 0) : null;
  const etaSec = sps && sps > 0 && remaining !== null ? remaining / sps : null;

  return (
    <div className="status-bar">
      <div className="status-bar-row">
        <span className={`status-pill ${status?.running ? "status-running" : "status-idle"}`}>
          {status?.running ? "● running" : "○ idle"}
        </span>
        <span>
          {timesteps.toLocaleString()} {target ? `/ ${target.toLocaleString()}` : ""} steps
        </span>
        <span>{sps ? `${sps.toFixed(0)} steps/s` : "— steps/s"}</span>
        <span>ETA {etaSec !== null ? formatEta(etaSec) : "—"}</span>
        <span>elapsed {formatElapsed(runMeta?.start_time)}</span>
      </div>
      <div className="status-bar-row">
        <span>stage {latest?.stage ?? "—"}</span>
        <span>
          success {latest ? `${(latest.success_rate * 100).toFixed(0)}%` : "—"}
          {latest?.advance_threshold != null && ` / threshold ${(latest.advance_threshold * 100).toFixed(0)}%`}
        </span>
        <span>{latest?.episodes_at_stage != null ? `${latest.episodes_at_stage} episodes at stage` : ""}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
      </div>
    </div>
  );
}

function formatEta(sec: number): string {
  if (!Number.isFinite(sec)) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

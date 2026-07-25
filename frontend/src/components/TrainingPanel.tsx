import { useEffect, useRef, useState } from "react";
import type { TrainingMetric, TrainingStatus } from "../types";

const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/training";

export default function TrainingPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [totalTimesteps, setTotalTimesteps] = useState(50_000);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  function refreshStatus() {
    fetch(`${API_BASE}/api/training/status`)
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => {});
  }

  useEffect(() => {
    refreshStatus();

    // Connecting tails data/train_stream.jsonl from byte 0 -- this both
    // streams a live run AND replays a completed one's full history if the
    // tab is opened afterwards (see backend/routers/training.py).
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const metric: TrainingMetric = JSON.parse(ev.data);
        setMetrics((prev) => [...prev, metric]);
      } catch {
        // ignore malformed lines
      }
    };
    ws.onerror = () => setError("WebSocket error — is the backend running?");

    const statusPoll = setInterval(refreshStatus, 2000);
    return () => {
      ws.close();
      clearInterval(statusPoll);
    };
  }, []);

  async function handleStart() {
    setError(null);
    setMetrics([]);
    try {
      const res = await fetch(
        `${API_BASE}/api/training/start?total_timesteps=${totalTimesteps}&dummy_vec=true`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setStatus(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleStop() {
    const res = await fetch(`${API_BASE}/api/training/stop`, { method: "POST" });
    setStatus(await res.json());
  }

  const latest = metrics[metrics.length - 1];
  const maxTimesteps = metrics.length > 0 ? metrics[metrics.length - 1].timesteps : 1;

  return (
    <section className="panel training-panel">
      <div className="panel-header">
        <h2>Training Monitor</h2>
        <div className="replay-controls">
          <label>
            Total timesteps{" "}
            <input
              type="number"
              min={1000}
              step={1000}
              value={totalTimesteps}
              onChange={(e) => setTotalTimesteps(Number(e.target.value))}
              style={{ width: "6rem" }}
            />
          </label>
          <button onClick={handleStart} disabled={status?.running}>Start</button>
          <button onClick={handleStop} disabled={!status?.running}>Stop</button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      <div className="replay-meta">
        <span>{status?.running ? "● running" : "○ idle"}</span>
        {latest && <span>stage {latest.stage}</span>}
        {latest && <span>{latest.timesteps.toLocaleString()} steps</span>}
        {latest && <span>success rate {(latest.success_rate * 100).toFixed(0)}%</span>}
      </div>

      {metrics.length > 0 ? (
        <TrainingChart metrics={metrics} maxTimesteps={maxTimesteps} />
      ) : (
        <p className="placeholder">
          No data yet — start a training run, or reopen this tab while one is already running.
        </p>
      )}
    </section>
  );
}

function TrainingChart({ metrics, maxTimesteps }: { metrics: TrainingMetric[]; maxTimesteps: number }) {
  const width = 800;
  const height = 220;
  const padding = 32;

  const xScale = (t: number) => padding + (t / Math.max(maxTimesteps, 1)) * (width - 2 * padding);
  const yScaleRate = (r: number) => height - padding - r * (height - 2 * padding);

  const rewardPoints = metrics.filter(
    (m): m is TrainingMetric & { ep_rew_mean: number } =>
      m.ep_rew_mean !== null && m.ep_rew_mean !== undefined,
  );
  const maxAbsReward = Math.max(1, ...rewardPoints.map((m) => Math.abs(m.ep_rew_mean)));
  const yScaleReward = (r: number) =>
    height - padding - ((r + maxAbsReward) / (2 * maxAbsReward)) * (height - 2 * padding);

  const successPath = metrics
    .map((m, i) => `${i === 0 ? "M" : "L"} ${xScale(m.timesteps)} ${yScaleRate(m.success_rate)}`)
    .join(" ");
  const rewardPath = rewardPoints
    .map((m, i) => `${i === 0 ? "M" : "L"} ${xScale(m.timesteps)} ${yScaleReward(m.ep_rew_mean)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="training-chart">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#333" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#333" />
      <path d={successPath} fill="none" stroke="#4caf50" strokeWidth={2} />
      {rewardPath && <path d={rewardPath} fill="none" stroke="#64b5f6" strokeWidth={2} strokeDasharray="4 3" />}
      <text x={padding} y={16} fill="#4caf50" fontSize={11}>— success rate</text>
      <text x={padding + 140} y={16} fill="#64b5f6" fontSize={11}>- - ep_rew_mean</text>
    </svg>
  );
}

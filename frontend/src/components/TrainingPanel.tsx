// TrainingPanel.tsx
// ===================
// V16 Phase B -- composition root for the rebuilt training monitor. The old
// single-file version (unbounded metrics array, hand-rolled SVG chart --
// D1/D4) is now StatusBar + RunBrowser + MetricsGrid + RewardDecomposition +
// LiveGroundTrack, all driven by one useMetricsStream() call. See
// frontend/src/components/training/useMetricsStream.ts's module docstring
// for the actual D1 fix (message-rate/render-rate decoupling).
import { useEffect, useState } from "react";
import type { TrainingRunMeta, TrainingStatus } from "../types";
import { LIVE_RUN_ID, useMetricsStream } from "./training/useMetricsStream";
import MetricsGrid from "./training/MetricsGrid";
import RewardDecomposition from "./training/RewardDecomposition";
import StatusBar from "./training/StatusBar";
import LiveGroundTrack from "./training/LiveGroundTrack";
import RunBrowser from "./training/RunBrowser";
import AnnunciatorRow from "./AnnunciatorRow";

const API_BASE = "http://localhost:8000";

export default function TrainingPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [runs, setRuns] = useState<TrainingRunMeta[]>([]);
  const [totalTimesteps, setTotalTimesteps] = useState(50_000);
  const [error, setError] = useState<string | null>(null);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([LIVE_RUN_ID]);

  const snapshots = useMetricsStream(selectedRunIds);
  const runSnapshots = selectedRunIds.map((id) => snapshots[id]).filter((s) => s !== undefined);

  function refreshStatus() {
    fetch(`${API_BASE}/api/training/status`)
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => {});
  }

  function refreshRuns() {
    fetch(`${API_BASE}/api/training/runs`)
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => {});
  }

  useEffect(() => {
    refreshStatus();
    refreshRuns();
    const statusPoll = setInterval(refreshStatus, 2000);
    const runsPoll = setInterval(refreshRuns, 15_000);
    return () => {
      clearInterval(statusPoll);
      clearInterval(runsPoll);
    };
  }, []);

  async function handleStart() {
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/training/start?total_timesteps=${totalTimesteps}&dummy_vec=true`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setStatus(await res.json());
      setSelectedRunIds([LIVE_RUN_ID]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleStop() {
    const res = await fetch(`${API_BASE}/api/training/stop`, { method: "POST" });
    setStatus(await res.json());
  }

  const primary = runSnapshots[0];
  const primaryRunMeta =
    selectedRunIds[0] === LIVE_RUN_ID
      ? runs[0] ?? null // newest run -- GET /api/training/runs returns newest first
      : runs.find((r) => r.run_id === selectedRunIds[0]) ?? null;

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

      <StatusBar status={status} runMeta={primaryRunMeta} latest={primary?.latest ?? null} />

      {primary?.latest && (
        <AnnunciatorRow
          stall={primary.latest.stall_violation}
          bank={primary.latest.bank_violation}
          reach={primary.latest.unreach_violation}
        />
      )}

      <div className="training-layout">
        <div className="training-main">
          {runSnapshots.length > 0 && runSnapshots.some((s) => s.count > 0) ? (
            <MetricsGrid runs={runSnapshots} />
          ) : (
            <p className="placeholder">
              No data yet — start a training run, select a past run below, or reopen this tab
              while one is already running.
            </p>
          )}
        </div>
        <div className="training-side">
          <RunBrowser selectedRunIds={selectedRunIds} onChange={setSelectedRunIds} />
          <LiveGroundTrack />
          <RewardDecomposition latest={primary?.latest ?? null} />
        </div>
      </div>
    </section>
  );
}

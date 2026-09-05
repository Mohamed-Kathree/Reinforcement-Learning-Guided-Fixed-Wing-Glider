// RunBrowser.tsx
// ===============
// V16 Phase B §B8 -- lists past runs from GET /api/training/runs, newest
// first. Selecting one loads it into the SAME useMetricsStream data path
// the live run uses (see useMetricsStream.ts's module docstring -- there's
// no separate code path for "the live run" vs "a past run"). Multi-select
// overlays additional runs as ghost/dashed series on the same charts.
import { useEffect, useState } from "react";
import type { TrainingRunMeta } from "../../types";
import { LIVE_RUN_ID } from "./useMetricsStream";

const API_BASE = "http://localhost:8000";

interface RunBrowserProps {
  selectedRunIds: string[]; // includes LIVE_RUN_ID when the live run is selected
  onChange: (runIds: string[]) => void;
}

export default function RunBrowser({ selectedRunIds, onChange }: RunBrowserProps) {
  const [runs, setRuns] = useState<TrainingRunMeta[]>([]);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    fetch(`${API_BASE}/api/training/runs`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(setRuns)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15_000); // new runs can appear without a reload
    return () => clearInterval(id);
  }, []);

  function toggle(runId: string) {
    if (selectedRunIds.includes(runId)) {
      onChange(selectedRunIds.filter((id) => id !== runId));
    } else {
      onChange([...selectedRunIds, runId]);
    }
  }

  return (
    <div className="run-browser">
      <div className="run-browser-header">
        <h4>Runs</h4>
        <button onClick={refresh} className="run-browser-refresh">Refresh</button>
      </div>
      {error && <p className="error-text">{error}</p>}
      <div className="run-list">
        <button
          className={`run-chip${selectedRunIds.includes(LIVE_RUN_ID) ? " run-chip-selected" : ""}`}
          onClick={() => toggle(LIVE_RUN_ID)}
        >
          live
        </button>
        {runs.map((run) => (
          <button
            key={run.run_id}
            className={`run-chip${selectedRunIds.includes(run.run_id) ? " run-chip-selected" : ""}`}
            title={`seed ${run.seed} · ${run.total_timesteps.toLocaleString()} steps${
              run.git_hash ? ` · ${run.git_hash.slice(0, 7)}` : ""
            }`}
            onClick={() => toggle(run.run_id)}
          >
            {formatRunLabel(run)}
          </button>
        ))}
        {runs.length === 0 && <p className="placeholder">No past runs yet.</p>}
      </div>
    </div>
  );
}

function formatRunLabel(run: TrainingRunMeta): string {
  const date = new Date(run.start_time);
  const datePart = Number.isNaN(date.getTime()) ? run.run_id : date.toLocaleString();
  return `${datePart} (seed ${run.seed})`;
}

import { useState } from "react";
import StackedBarChart from "./StackedBarChart";
import type { BaselineRunResult } from "../types";

const API_BASE = "http://localhost:8000";

interface BaselinePanelProps {
  onSelectEpisode: (episodeId: string) => void;
}

export default function BaselinePanel({ onSelectEpisode }: BaselinePanelProps) {
  const [nEpisodes, setNEpisodes] = useState(20);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BaselineRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/baseline/run?n_episodes=${nEpisodes}`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="panel baseline-panel">
      <div className="panel-header">
        <h2>Baseline (Deterministic RTL)</h2>
        <div className="replay-controls">
          <label>
            Episodes/stage{" "}
            <input
              type="number"
              min={1}
              max={200}
              value={nEpisodes}
              onChange={(e) => setNEpisodes(Number(e.target.value))}
              style={{ width: "4rem" }}
            />
          </label>
          <button onClick={handleRun} disabled={running}>
            {running ? "Running…" : "Run baseline"}
          </button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}
      {running && <p className="placeholder">Running {nEpisodes} episodes × 4 stages — this can take a little while.</p>}

      {result && (
        <div className="baseline-stages">
          {result.stages.map((s) => {
            return (
              <div key={s.stage} className="baseline-stage">
                <div className="baseline-stage-header">
                  <strong>Stage {s.stage}</strong>
                  <span>
                    {(s.success_rate * 100).toFixed(0)}% success · mean quality{" "}
                    {s.mean_quality.toFixed(2)} ({s.n_episodes} episodes)
                  </span>
                </div>
                <StackedBarChart
                  title={`Stage ${s.stage} outcomes`}
                  exportName={`baseline-stage${s.stage}-outcomes`}
                  segments={[
                    { key: "success", value: s.n_success, label: "success", colorVar: "var(--success)" },
                    { key: "soft_landing", value: s.n_soft_landing, label: "soft landing", colorVar: "var(--accent)" },
                    { key: "crash", value: s.n_crash, label: "crash", colorVar: "var(--danger)" },
                    { key: "timeout", value: s.n_timeout, label: "timeout", colorVar: "var(--warning)" },
                  ]}
                />
                <div className="episode-chips">
                  {s.episodes.map((ep, i) => (
                    <button
                      key={ep.episode_id}
                      className={`chip chip-${ep.outcome}`}
                      title={`${ep.episode_id} — ${ep.outcome} (quality ${ep.quality.toFixed(2)})`}
                      onClick={() => onSelectEpisode(ep.episode_id)}
                    >
                      #{i + 1}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

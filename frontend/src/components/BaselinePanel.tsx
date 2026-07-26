import { useState } from "react";
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
            const total = s.n_episodes || 1;
            const successPct = (100 * s.n_success) / total;
            const softLandingPct = (100 * s.n_soft_landing) / total;
            const crashPct = (100 * s.n_crash) / total;
            const timeoutPct = (100 * s.n_timeout) / total;
            return (
              <div key={s.stage} className="baseline-stage">
                <div className="baseline-stage-header">
                  <strong>Stage {s.stage}</strong>
                  <span>
                    {(s.success_rate * 100).toFixed(0)}% success · mean quality{" "}
                    {s.mean_quality.toFixed(2)} ({s.n_episodes} episodes)
                  </span>
                </div>
                <div className="stacked-bar">
                  <div className="bar-seg bar-success" style={{ width: `${successPct}%` }} title={`success ${s.n_success}`} />
                  <div className="bar-seg bar-soft_landing" style={{ width: `${softLandingPct}%` }} title={`soft landing ${s.n_soft_landing}`} />
                  <div className="bar-seg bar-crash" style={{ width: `${crashPct}%` }} title={`crash ${s.n_crash}`} />
                  <div className="bar-seg bar-timeout" style={{ width: `${timeoutPct}%` }} title={`timeout ${s.n_timeout}`} />
                </div>
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

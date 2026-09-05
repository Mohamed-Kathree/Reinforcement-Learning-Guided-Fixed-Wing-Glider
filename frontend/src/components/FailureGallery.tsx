// FailureGallery.tsx
// ====================
// V16 Phase D §D3 -- auto-collected episodes as a browsable grid, filterable
// by stage and outcome ("failure mode" per the spec's wording -- outcome is
// what env/reward.py actually buckets episodes into, see backend/schemas.py's
// Outcome type). Reuses the same "click a card to open it in Flight Replay"
// pattern BaselinePanel.tsx's episode chips already established.
// `summaries` is fetched once by the parent AnalysisPanel and shared with
// TouchdownScatter, not re-fetched here -- see AnalysisPanel.tsx's docstring.
import { useState } from "react";
import type { EpisodeRecordSummary } from "../types";

const OUTCOMES = ["crash", "timeout", "soft_landing", "success"] as const;

interface FailureGalleryProps {
  summaries: EpisodeRecordSummary[] | null;
  error: string | null;
  onSelectEpisode: (episodeId: string) => void;
}

export default function FailureGallery({ summaries, error, onSelectEpisode }: FailureGalleryProps) {
  const [outcomeFilter, setOutcomeFilter] = useState<(typeof OUTCOMES)[number]>("crash");
  const [stageFilter, setStageFilter] = useState<number | "all">("all");

  if (error) return <p className="error-text">{error}</p>;
  if (!summaries) return <p className="placeholder">Loading episode archive…</p>;

  const filtered = summaries
    .filter((s) => s.outcome === outcomeFilter)
    .filter((s) => stageFilter === "all" || s.stage === stageFilter);

  return (
    <div className="analysis-chart">
      <div className="analysis-chart-header">
        <label>
          Outcome{" "}
          <select value={outcomeFilter} onChange={(e) => setOutcomeFilter(e.target.value as (typeof OUTCOMES)[number])}>
            {OUTCOMES.map((o) => <option key={o} value={o}>{o.replace("_", " ")}</option>)}
          </select>
        </label>
        <label>
          Stage{" "}
          <select value={stageFilter} onChange={(e) => setStageFilter(e.target.value === "all" ? "all" : Number(e.target.value))}>
            <option value="all">All</option>
            {[0, 1, 2, 3].map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <span className="hint-text">{filtered.length} episode{filtered.length === 1 ? "" : "s"}</span>
      </div>

      {filtered.length === 0 ? (
        <p className="placeholder">No episodes match this filter.</p>
      ) : (
        <div className="failure-gallery-grid">
          {filtered.map((s) => (
            <button
              key={s.episode_id}
              className={`failure-card outcome-${s.outcome}`}
              onClick={() => onSelectEpisode(s.episode_id)}
              title={s.episode_id}
            >
              <div className="failure-card-id">{s.episode_id}</div>
              <div className="failure-card-meta">
                <span>stage {s.stage}</span>
                <span>{s.controller}</span>
                <span>quality {s.quality.toFixed(2)}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

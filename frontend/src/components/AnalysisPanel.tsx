// AnalysisPanel.tsx
// ===================
// V16 Phase D -- houses the two analysis views (touchdown scatter §D1,
// failure gallery §D3) together. Neither belongs inside Baseline (which is
// specifically about the baseline benchmark, not the whole cross-controller
// episode archive) or Replay.
//
// GET /api/episode/summaries is fetched ONCE here and passed to both views,
// not once per view -- this endpoint reads/validates every recorded
// episode's summary sidecar (850+ at last count, growing), so having both
// children fetch it independently doubled real, non-trivial backend work on
// every single page load (this tab is always mounted, per App.tsx's
// keep-every-tab-alive design, whether or not it's the active one).
import { useEffect, useState } from "react";
import TouchdownScatter from "./TouchdownScatter";
import FailureGallery from "./FailureGallery";
import type { EpisodeRecordSummary } from "../types";
import { API_BASE } from "../config";


interface AnalysisPanelProps {
  onSelectEpisode: (episodeId: string) => void;
}

export default function AnalysisPanel({ onSelectEpisode }: AnalysisPanelProps) {
  const [summaries, setSummaries] = useState<EpisodeRecordSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/episode/summaries`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(setSummaries)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  return (
    <section className="panel analysis-panel">
      <div className="panel-header">
        <div>
          <h1>Analysis</h1>
          <p className="page-subtitle">
            Every recorded flight in the archive, across controllers: where it touched down, and the
            failures worth replaying.
          </p>
        </div>
      </div>

      <div className="home-section">
        <h2>Touchdown scatter</h2>
        <TouchdownScatter summaries={summaries} error={error} />
      </div>

      <div className="home-section">
        <h2>Failure gallery</h2>
        <FailureGallery summaries={summaries} error={error} onSelectEpisode={onSelectEpisode} />
      </div>
    </section>
  );
}

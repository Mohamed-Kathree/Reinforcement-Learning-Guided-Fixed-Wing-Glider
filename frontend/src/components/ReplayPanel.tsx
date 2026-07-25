import { useEffect, useState } from "react";
import EpisodeViewer from "./EpisodeViewer";

const API_BASE = "http://localhost:8000";

interface ReplayPanelProps {
  episodeId?: string | null;
}

export default function ReplayPanel({ episodeId }: ReplayPanelProps) {
  const [compareMode, setCompareMode] = useState(false);
  const [hasRlEpisode, setHasRlEpisode] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/episode`)
      .then((r) => r.json())
      .then((ids: string[]) => setHasRlEpisode(ids.some((id) => id.startsWith("rl-"))))
      .catch(() => {});
  }, [episodeId, compareMode]);

  return (
    <section className="panel replay-panel">
      <div className="panel-header">
        <h2>Flight Replay</h2>
        <button className="mode-toggle" onClick={() => setCompareMode((v) => !v)}>
          {compareMode ? "Single view" : "Compare two episodes"}
        </button>
      </div>

      {compareMode && !hasRlEpisode && (
        <p className="hint-text">
          No RL episode has been recorded yet (needs a trained model — see Milestone 6 TODO 6).
          Comparing two baseline runs for now, e.g. a Stage 3 crash against a Stage 0 success.
        </p>
      )}

      {compareMode ? (
        <div className="compare-grid">
          <EpisodeViewer initialEpisodeId={episodeId} label="A" />
          <EpisodeViewer label="B" />
        </div>
      ) : (
        <EpisodeViewer initialEpisodeId={episodeId} />
      )}
    </section>
  );
}

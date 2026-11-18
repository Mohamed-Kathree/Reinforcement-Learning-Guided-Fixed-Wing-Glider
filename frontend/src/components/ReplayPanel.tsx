import { useEffect, useState } from "react";
import EpisodeViewer from "./EpisodeViewer";
import { API_BASE } from "../config";


interface ReplayPanelProps {
  episodeId?: string | null;
  episodeIdB?: string | null;   // set: open both side by side (paired benchmark flights)
}

export default function ReplayPanel({ episodeId, episodeIdB }: ReplayPanelProps) {
  const [compareMode, setCompareMode] = useState(Boolean(episodeIdB));
  useEffect(() => {
    if (episodeIdB) setCompareMode(true);
  }, [episodeIdB]);
  const [hasRlEpisode, setHasRlEpisode] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/episode`)
      .then((r) => r.json())
      .then((ids: string[]) => setHasRlEpisode(ids.some((id) => id.startsWith("gated-") || id.startsWith("rl-"))))
      .catch(() => {});
  }, [episodeId, compareMode]);

  return (
    <section className="panel replay-panel">
      <div className="panel-header">
        <div>
          <h1>Flight Replay</h1>
          <p className="page-subtitle">
            Fly the selected controller in the simulator, then replay the true aircraft state frame by
            frame. Recordings are single simulated flights, not validation evidence.
          </p>
        </div>
        <button className="mode-toggle" onClick={() => setCompareMode((v) => !v)} aria-pressed={compareMode}>
          {compareMode ? "Single view" : "Compare two flights"}
        </button>
      </div>

      {compareMode && (
        <p className="hint-text">
          {hasRlEpisode
            ? "Set the same seed in both viewers to fly two controllers on an identical launch."
            : "No learned-controller flight recorded yet. Record one with Run 4 – Gated in either viewer, using the same seed in both for a like-for-like comparison."}
        </p>
      )}

      {compareMode ? (
        <div className="compare-grid">
          <EpisodeViewer initialEpisodeId={episodeId} label="A" />
          <EpisodeViewer initialEpisodeId={episodeIdB} label="B" />
        </div>
      ) : (
        <EpisodeViewer initialEpisodeId={episodeId} />
      )}
    </section>
  );
}

import { useEffect, useState, type ComponentType } from "react";
import "./App.css";
import HomePanel from "./components/HomePanel";
import ValidationPanel from "./components/ValidationPanel";
import ReplayPanel from "./components/ReplayPanel";
import BaselinePanel from "./components/BaselinePanel";
import TrainingPanel from "./components/TrainingPanel";
import AnalysisPanel from "./components/AnalysisPanel";
import { IconHome, IconCheck, IconBarChart, IconActivity, IconScatter, IconPlay } from "./icons";
import { useHashRoute, type Tab } from "./useHashRoute";

const NAV: { id: Tab; label: string; icon: ComponentType<{ className?: string }> }[] = [
  { id: "home", label: "Overview", icon: IconHome },
  { id: "validation", label: "Validation", icon: IconCheck },
  { id: "baseline", label: "Baseline", icon: IconBarChart },
  { id: "training", label: "Training", icon: IconActivity },
  { id: "analysis", label: "Analysis", icon: IconScatter },
  { id: "replay", label: "Flight Replay", icon: IconPlay },
];

const API_BASE = "http://localhost:8000";

function App() {
  const { tab, episodeId: selectedEpisodeId, navigate } = useHashRoute();
  const activeNav = NAV.find((n) => n.id === tab);

  // Real backend connectivity, not decorative -- reuses the same
  // GET /api/health this app's other panels already poll for their own
  // status, just surfaced once here so it's visible from every tab.
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    function check() {
      fetch(`${API_BASE}/api/health`)
        .then((r) => {
          if (!cancelled) setBackendOk(r.ok);
        })
        .catch(() => {
          if (!cancelled) setBackendOk(false);
        });
    }
    check();
    const id = setInterval(check, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  function handleSelectEpisode(episodeId: string) {
    navigate("replay", episodeId);
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-title">RL Glider</div>
            <div className="brand-subtitle">RTL Dashboard</div>
          </div>
        </div>

        <nav className="side-nav">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={tab === id ? "side-nav-item active" : "side-nav-item"}
              onClick={() => navigate(id)}
            >
              <Icon className="side-nav-icon" />
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">COS 731/732 · UWC</div>
      </aside>

      <main className={tab === "replay" ? "content content-wide" : "content"}>
        <header className="content-header">
          <div className="content-header-crumb">
            <span>RL Glider</span>
            <span className="content-header-sep">/</span>
            <strong>{activeNav?.label}</strong>
          </div>
          <div className={`content-header-status${backendOk ? " content-header-status-ok" : ""}`}>
            <span className="live-dot" aria-hidden="true" />
            {backendOk === null ? "connecting…" : backendOk ? "backend online" : "backend unreachable"}
          </div>
        </header>

        {/* All tabs are always mounted, just hidden with CSS when inactive --
            conditionally rendering (unmounting) them on tab switch used to
            wipe every panel's local state (Replay's selected episode/frame,
            Training's live WebSocket + accumulated metrics, etc.) every time
            the user navigated away and back. */}
        <div className={tab === "home" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <HomePanel />
        </div>
        <div className={tab === "validation" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <ValidationPanel />
        </div>
        <div className={tab === "baseline" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <BaselinePanel onSelectEpisode={handleSelectEpisode} />
        </div>
        <div className={tab === "training" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <TrainingPanel />
        </div>
        <div className={tab === "analysis" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <AnalysisPanel onSelectEpisode={handleSelectEpisode} />
        </div>
        <div className={tab === "replay" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <ReplayPanel episodeId={selectedEpisodeId} />
        </div>
      </main>
    </div>
  );
}

export default App;

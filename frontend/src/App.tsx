import { useState, type ComponentType } from "react";
import "./App.css";
import HomePanel from "./components/HomePanel";
import ValidationPanel from "./components/ValidationPanel";
import ReplayPanel from "./components/ReplayPanel";
import BaselinePanel from "./components/BaselinePanel";
import TrainingPanel from "./components/TrainingPanel";
import { IconHome, IconCheck, IconBarChart, IconActivity, IconPlay } from "./icons";

type Tab = "home" | "validation" | "baseline" | "training" | "replay";

const NAV: { id: Tab; label: string; icon: ComponentType<{ className?: string }> }[] = [
  { id: "home", label: "Overview", icon: IconHome },
  { id: "validation", label: "Validation", icon: IconCheck },
  { id: "baseline", label: "Baseline", icon: IconBarChart },
  { id: "training", label: "Training", icon: IconActivity },
  { id: "replay", label: "Flight Replay", icon: IconPlay },
];

function App() {
  const [tab, setTab] = useState<Tab>("home");
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<string | null>(null);

  function handleSelectEpisode(episodeId: string) {
    setSelectedEpisodeId(episodeId);
    setTab("replay");
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
              onClick={() => setTab(id)}
            >
              <Icon className="side-nav-icon" />
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">COS 731/732 · UWC</div>
      </aside>

      <main className={tab === "replay" ? "content content-wide" : "content"}>
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
        <div className={tab === "replay" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
          <ReplayPanel episodeId={selectedEpisodeId} />
        </div>
      </main>
    </div>
  );
}

export default App;

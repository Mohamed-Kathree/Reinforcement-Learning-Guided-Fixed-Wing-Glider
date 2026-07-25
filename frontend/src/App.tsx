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

      <main className="content">
        {tab === "home" && <HomePanel />}
        {tab === "validation" && <ValidationPanel />}
        {tab === "baseline" && <BaselinePanel onSelectEpisode={handleSelectEpisode} />}
        {tab === "training" && <TrainingPanel />}
        {tab === "replay" && <ReplayPanel episodeId={selectedEpisodeId} />}
      </main>
    </div>
  );
}

export default App;

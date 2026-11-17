import { useEffect, useState, type ComponentType } from "react";
import "./App.css";
import HomePanel from "./components/HomePanel";
import ValidationPanel from "./components/ValidationPanel";
import ReplayPanel from "./components/ReplayPanel";
import BaselinePanel from "./components/BaselinePanel";
import TrainingPanel from "./components/TrainingPanel";
import AnalysisPanel from "./components/AnalysisPanel";
import {
  IconActivity, IconBarChart, IconCheck, IconGlider, IconHome, IconMoon, IconPlay, IconScatter, IconSun,
} from "./icons";
import { useHashRoute, type Tab } from "./useHashRoute";
import { CONTROLLER_NAME, GATE } from "./data/validation";

interface NavItem {
  id: Tab;
  label: string;
  icon: ComponentType<{ className?: string }>;
}

// Grouped by what a visitor is trying to do: understand the selected
// controller, fly it, or dig into the research behind it.
const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Controller",
    items: [
      { id: "home", label: "Overview", icon: IconHome },
      { id: "validation", label: "Validation", icon: IconCheck },
    ],
  },
  {
    label: "Simulation",
    items: [
      { id: "replay", label: "Flight Replay", icon: IconPlay },
      { id: "baseline", label: "Baseline", icon: IconBarChart },
    ],
  },
  {
    label: "Research",
    items: [
      { id: "training", label: "Training", icon: IconActivity },
      { id: "analysis", label: "Analysis", icon: IconScatter },
    ],
  },
];

const API_BASE = "http://localhost:8000";
const THEME_KEY = "rl-glider-theme";

type Theme = "light" | "dark";

function readTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function App() {
  const { tab, episodeId: selectedEpisodeId, navigate } = useHashRoute();
  const activeGroup = NAV_GROUPS.find((g) => g.items.some((i) => i.id === tab));
  const activeNav = activeGroup?.items.find((i) => i.id === tab);

  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* private mode: the choice just isn't remembered */
    }
  }, [theme]);

  // Real backend connectivity, polled once here so it is visible from every tab.
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

  useEffect(() => {
    document.title = `${activeNav?.label ?? "Overview"} · RL Glider`;
  }, [activeNav]);

  function handleSelectEpisode(episodeId: string) {
    navigate("replay", episodeId);
  }

  const statusClass =
    backendOk === null ? "" : backendOk ? " content-header-status-ok" : " content-header-status-down";

  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <IconGlider width={18} height={18} />
          </span>
          <div>
            <div className="brand-title">RL Glider</div>
            <div className="brand-subtitle">Return-to-launch autonomy</div>
          </div>
        </div>

        <nav className="side-nav" aria-label="Main">
          {NAV_GROUPS.map((group) => (
            <div className="nav-group" key={group.label} role="group" aria-label={group.label}>
              <div className="nav-group-label" aria-hidden="true">{group.label}</div>
              {group.items.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  className={tab === id ? "side-nav-item active" : "side-nav-item"}
                  aria-current={tab === id ? "page" : undefined}
                  onClick={() => navigate(id)}
                >
                  <Icon className="side-nav-icon" />
                  {label}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-controller">
          <div className="sidebar-controller-label">Selected controller</div>
          <div className="sidebar-controller-name">{CONTROLLER_NAME}</div>
          <div className="sidebar-controller-detail">
            Window gate · {GATE.threshold_ms} m/s · {GATE.t_earliest_s}–{GATE.t_latest_s} s
          </div>
        </div>

        <div className="sidebar-footer">COS 731/732 · University of the Western Cape</div>
      </aside>

      <main id="main" className={tab === "replay" ? "content content-wide" : "content"}>
        <header className="content-header">
          <div className="content-header-crumb">
            <span className="crumb-root">{activeGroup?.label}</span>
            <span className="content-header-sep" aria-hidden="true">/</span>
            <strong>{activeNav?.label}</strong>
          </div>
          <div className="content-header-actions">
            <div className={`content-header-status${statusClass}`} role="status">
              <span className="live-dot" aria-hidden="true" />
              {backendOk === null ? "Connecting…" : backendOk ? "Simulator online" : "Simulator offline"}
            </div>
            <button
              className="icon-button"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              aria-label={theme === "light" ? "Switch to dark appearance" : "Switch to light appearance"}
              title={theme === "light" ? "Dark appearance" : "Light appearance"}
            >
              {theme === "light" ? <IconMoon width={16} height={16} /> : <IconSun width={16} height={16} />}
            </button>
          </div>
        </header>

        {/* All tabs stay mounted and are hidden with CSS when inactive, so
            each panel keeps its state (Replay's episode and frame, Training's
            live stream) across navigation. */}
        <div className="content-inner">
          <div className={tab === "home" ? "tab-pane" : "tab-pane tab-pane-hidden"}>
            <HomePanel onNavigate={navigate} />
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
        </div>
      </main>
    </div>
  );
}

export default App;

// pages.tsx -- the two flight-test pages built from the four panels.
import type { Tab } from "../useHashRoute";
import { IconAlert } from "../icons";
import AircraftPanel from "./AircraftPanel";
import CalibrationWizard from "./CalibrationWizard";
import ConnectionPanel from "./ConnectionPanel";
import LogReplayPanel from "./LogReplayPanel";
import PanelBoundary from "./PanelBoundary";
import ServoPanel from "./ServoPanel";
import SetupChecklist from "./SetupChecklist";
import TelemetryPanel from "./TelemetryPanel";
import { ft, useFlightTest } from "./store";

function BackendBanner() {
  const s = useFlightTest();
  return (
    <>
      {s.backend === "missing" && (
        <div className="callout callout-danger" role="alert" style={{ marginBottom: "var(--space-5)" }}>
          <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
          <div>
            <div className="callout-title">The running backend has no flight-test endpoints</div>
            It was started before they existed. Restart it: <code>python -m uvicorn backend.main:app --port 8000</code>
          </div>
        </div>
      )}
      {s.backend === "down" && (
        <div className="callout callout-danger" role="alert" style={{ marginBottom: "var(--space-5)" }}>
          <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
          <div><div className="callout-title">Backend unreachable</div>Start it from the repo root: <code>python -m uvicorn backend.main:app --port 8000</code></div>
        </div>
      )}
      {s.error && (
        <div className="callout callout-warning" role="alert" style={{ marginBottom: "var(--space-5)" }}>
          <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
          <div>
            <div className="callout-title">Last action failed</div>
            {s.error}{" "}
            <button className="btn-plain" onClick={() => ft.clearError()}>Dismiss</button>
          </div>
        </div>
      )}
    </>
  );
}

export function GroundStationPage() {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Ground Station</h1>
          <p className="page-subtitle">
            Connect to the aircraft, check it is ready, calibrate the control surfaces and watch live telemetry. A
            simulated stub glider stands in when no hardware is connected; it is labelled wherever it appears.
          </p>
        </div>
      </div>
      <BackendBanner />
      <PanelBoundary name="checklist"><SetupChecklist /></PanelBoundary>
      <PanelBoundary name="connection"><ConnectionPanel /></PanelBoundary>
      <PanelBoundary name="flight computer"><AircraftPanel /></PanelBoundary>
      <PanelBoundary name="calibration"><CalibrationWizard /></PanelBoundary>
      <PanelBoundary name="live telemetry"><TelemetryPanel /></PanelBoundary>
      <PanelBoundary name="servo tester"><ServoPanel /></PanelBoundary>
    </section>
  );
}

export function FlightLogsPage({ onNavigate }: { onNavigate: (tab: Tab, a?: string | null) => void }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h1>Flight Logs</h1>
          <p className="page-subtitle">
            Full-rate logs from the Pi's SD card, the record of every flight. Replay them in 3D and, on the bench,
            play the control-surface movements back on the real servos.
          </p>
        </div>
      </div>
      <BackendBanner />
      <PanelBoundary name="flight log replay"><LogReplayPanel onNavigate={onNavigate} /></PanelBoundary>
    </section>
  );
}

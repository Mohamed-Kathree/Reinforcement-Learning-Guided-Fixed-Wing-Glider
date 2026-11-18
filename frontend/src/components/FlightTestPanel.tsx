// FlightTestPanel.tsx
// ===================
// Flight-test readiness: go/no-go limits taken from the validation results,
// a pre-flight checklist (remembered per browser), known sim-to-real gaps,
// and the as-wired avionics with open hardware issues. The live flight-test
// tools (connection, control-surface tester, telemetry, replay with servo
// playback) live on the Ground Station and Flight Logs pages (src/flighttest/).
import { useEffect, useState } from "react";
import { Callout, SectionHead } from "./validation/parts";
import { AVIONICS, HARDWARE_ISSUES } from "../data/hardware";
import { GATE, STAGES, pct } from "../data/validation";
import { IconAlert, IconCheck } from "../icons";

const CHECK_KEY = "rl-glider-preflight";

const CHECKLIST: { group: string; items: { id: string; text: string }[] }[] = [
  {
    group: "Airframe",
    items: [
      { id: "weigh", text: "Weigh the complete aircraft and record it (simulation assumes ≈ 1.64 kg)." },
      { id: "cg", text: "Balance: CG at 362 mm (30% MAC); record the ballast used." },
      { id: "throws", text: "Control throws and directions checked against the calibration file." },
    ],
  },
  {
    group: "Sensors",
    items: [
      { id: "imu", text: "IMU bench checks pass: level az ≈ −1.00 g, nose up → +pitch, right wing down → +roll, nose right → +gz. Transform determinant +1." },
      { id: "baro", text: "Barometer warmed up at least 2 minutes; ground reference captured on arm, not at boot." },
      { id: "gps", text: "GPS 3D fix; satellite count recorded." },
      { id: "sonar", text: "Sonar reads ground distance correctly and reports invalid above ~4.5 m." },
    ],
  },
  {
    group: "Power and links",
    items: [
      { id: "rail-cal", text: "Rail divider (GPIO34) calibrated against a multimeter; scale factor saved in the calibration file." },
      { id: "rail", text: "5 V rail minimum measured under full servo load (all four MG90S): at least 4.75 V. Record the value." },
      { id: "link", text: "ESP-NOW telemetry packet loss measured at walking distance." },
      { id: "log", text: "Pi writes the full-rate log to SD and it can be retrieved over WiFi on the ground." },
    ],
  },
  {
    group: "Safety",
    items: [
      { id: "override", text: "Manual override / abort tested: surfaces centre and controller disarms." },
      { id: "timeout", text: "Ground-link timeout tested in BENCH: no command for 150 ms centres the surfaces." },
      { id: "pi-timeout", text: "Pi-link timeout tested in FLIGHT on the bench: stopping the Pi's setpoints enters failsafe glide within 150 ms." },
      { id: "airborne", text: "Airborne test checked: BENCH refused for 60 s after FLIGHT, and while walking with a GPS fix." },
      { id: "wind", text: "Wind measured at launch height with an anemometer and within the limit below." },
      { id: "baseline-first", text: "Deterministic baseline flown first, in calm air, before the learned controller." },
    ],
  },
];

function load(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(CHECK_KEY) ?? "{}");
  } catch {
    return {};
  }
}

export default function FlightTestPanel() {
  const [done, setDone] = useState<Record<string, boolean>>(load);
  useEffect(() => {
    try {
      localStorage.setItem(CHECK_KEY, JSON.stringify(done));
    } catch {
      /* not remembered in private mode */
    }
  }, [done]);

  const total = CHECKLIST.reduce((n, g) => n + g.items.length, 0);
  const complete = Object.values(done).filter(Boolean).length;
  const blocking = HARDWARE_ISSUES.filter((i) => i.severity === "blocking").length;
  const [s0, s1, s2, s3] = STAGES;

  return (
    <section className="panel flighttest-panel">
      <div className="panel-header">
        <div>
          <h1>Flight Test</h1>
          <p className="page-subtitle">
            What has to be true before the glider flies, and the limits the simulation results support.
            Nothing here has been verified on the real aircraft yet.
          </p>
        </div>
      </div>

      <Callout tone={blocking ? "danger" : "warning"} title="Not cleared for flight">
        {blocking > 0 && `${blocking} blocking hardware issue${blocking === 1 ? "" : "s"}, `}
        {HARDWARE_ISSUES.length - blocking} hardware item{HARDWARE_ISSUES.length - blocking === 1 ? "" : "s"} to check,{" "}
        {total - complete} of {total} pre-flight checks open. The flight-test tools (Ground Station, Flight Logs)
        are built and tested against a simulated stub glider, but none of them has run against the real
        aircraft, and the firmware side is host-tested only.
      </Callout>

      <section className="section" aria-labelledby="limits-h" style={{ marginTop: "var(--space-6)" }}>
        <SectionHead id="limits-h" kicker="Limits" title="Go / no-go by measured wind">
          From the simulation benchmark. Measure wind at launch height; gusty air counts as the next band up.
        </SectionHead>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Measured wind</th>
                <th>Controller</th>
                <th>Decision</th>
                <th>Simulation evidence</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Calm, below ~1.5 m/s</td>
                <td>Baseline first, then Run 4 – Gated</td>
                <td><span className="badge badge-success"><IconCheck width={12} height={12} aria-hidden="true" />Go</span></td>
                <td className="cell-wrap">Gated {pct(s0.gated.success)} success, {pct(s0.gated.crash)} crashes (baseline {pct(s0.baseline.success)}). The only significant result.</td>
              </tr>
              <tr>
                <td>Light, up to ~3 m/s, gentle gusts</td>
                <td>Run 4 – Gated</td>
                <td><span className="badge badge-warning">Go with care</span></td>
                <td className="cell-wrap">{pct(s1.gated.success)} success, {pct(s1.gated.crash)} crashes; not significantly better than the baseline. The gate hands over in {pct(s1.handover_share_of_episodes)} of flights.</td>
              </tr>
              <tr>
                <td>Moderate, 3–6 m/s</td>
                <td>Baseline only</td>
                <td><span className="badge badge-danger"><IconAlert width={12} height={12} aria-hidden="true" />No-go for the learned controller</span></td>
                <td className="cell-wrap">Gated crashes {pct(s2.gated.crash)} against the baseline's {pct(s2.baseline.crash)}.</td>
              </tr>
              <tr>
                <td>Strong, above 6 m/s</td>
                <td>None</td>
                <td><span className="badge badge-danger"><IconAlert width={12} height={12} aria-hidden="true" />No-go</span></td>
                <td className="cell-wrap">{pct(s3.gated.crash)} gated and {pct(s3.baseline.crash)} baseline crashes at up to 9 m/s.</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="provenance" style={{ marginTop: "var(--space-2)" }}>
          The gate threshold ({GATE.threshold_ms} m/s) is on the on-board estimate, which lags the real wind for the first
          seconds of flight. That is why calm air, not the threshold, is the limit for first flights.
        </p>
      </section>

      <section className="section" aria-labelledby="check-h">
        <div className="section-row">
          <SectionHead id="check-h" kicker="Checklist" title={`Pre-flight checks · ${complete} of ${total}`}>
            Ticks are remembered in this browser only. Record every measurement, not just pass or fail.
          </SectionHead>
          <button className="btn-plain" onClick={() => setDone({})} disabled={complete === 0}>Reset</button>
        </div>
        <div className="checklist-grid">
          {CHECKLIST.map((g) => (
            <fieldset className="surface checklist" key={g.group}>
              <legend>{g.group}</legend>
              {g.items.map((it) => (
                <label key={it.id} className={done[it.id] ? "check-item done" : "check-item"}>
                  <input type="checkbox" checked={!!done[it.id]}
                    onChange={(e) => setDone((d) => ({ ...d, [it.id]: e.target.checked }))} />
                  <span>{it.text}</span>
                </label>
              ))}
            </fieldset>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="gaps-h">
        <SectionHead id="gaps-h" kicker="Sim to real" title="Known gaps between simulation and the aircraft" />
        <ul className="surface evidence-list">
          <li><span>Airspeed</span><div>The simulated attitude loop holds true airspeed. The glider has no pitot and can only use GPS ground speed, which differs from airspeed by the wind.</div></li>
          <li><span>Mass properties</span><div>Only the fuselage and tail have been weighed. Mass, CG and inertia in the simulator are estimates.</div></li>
          <li><span>IMU frame</span><div>Axis signs are being re-derived for the current mount; attitude in the simulator is exact.</div></li>
          <li><span>Barometer</span><div>Real drift (~0.3 m) and warm-up (~2 m) are larger than a fresh simulated reading; capture the reference on arm.</div></li>
          <li><span>Power</span><div>The rail's minimum voltage under servo load has not been measured (budget ~5.6 A against ~5 A); the simulator has no brownouts.</div></li>
        </ul>
      </section>

      <section className="section" aria-labelledby="hw-h">
        <SectionHead id="hw-h" kicker="Hardware" title="Avionics as wired">
          Transcribed from the wiring diagram. Issues are listed first.
        </SectionHead>
        <div className="issue-list">
          {HARDWARE_ISSUES.map((i) => (
            <Callout key={i.title} tone={i.severity === "blocking" ? "danger" : "warning"}
              title={`${i.severity === "blocking" ? "Blocking" : "Check"}: ${i.title}`}>
              {i.detail}
            </Callout>
          ))}
        </div>
        <div className="table-wrap" style={{ marginTop: "var(--space-4)" }}>
          <table className="avionics-table">
            <thead>
              <tr><th>Part</th><th>Role</th><th>Bus</th><th>Pins</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {AVIONICS.map((c) => (
                <tr key={c.part}>
                  <td className="cell-strong">{c.part}</td>
                  <td className="cell-wrap">{c.role}</td>
                  <td>{c.bus}</td>
                  <td className="cell-wrap">{c.pins}</td>
                  <td className="cell-wrap">{c.note ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}

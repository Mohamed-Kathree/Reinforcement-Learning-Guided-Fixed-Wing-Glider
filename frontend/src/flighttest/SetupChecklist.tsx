// SetupChecklist.tsx -- "is the aircraft ready?" at a glance, top of the Ground
// Station page. Every line is green / amber / red with a one-line fix, derived
// from the live snapshot (telemetry status bits and the 1 Hz Info packet).
import { useFlightTest } from "./store";

type Level = "ok" | "warn" | "bad";
interface Item { label: string; level: Level; detail: string }

const ICON: Record<Level, string> = { ok: "✓", warn: "!", bad: "✕" };
const COLOR: Record<Level, string> = { ok: "var(--success, #30a14e)", warn: "var(--warning, #e08600)", bad: "var(--danger, #e5383b)" };

export default function SetupChecklist() {
  const s = useFlightTest();
  const snap = s.snapshot;
  const g = snap?.ground;
  const t = snap?.telemetry;
  const st = t?.status;
  const info = snap?.info;
  const stub = g?.kind === "stub";
  const fileId = snap?.file_cal_id_hex;

  const items: Item[] = [];
  items.push(!g?.connected
    ? { label: "Ground station", level: "bad", detail: "Not connected: plug it into this PC and press Connect to the aircraft." }
    : stub ? { label: "Ground station", level: "warn", detail: "Connected to the simulated stub glider, not the real aircraft." }
    : { label: "Ground station", level: "ok", detail: `Connected (${g.port}).` });

  const talking = !!g?.receiving;
  items.push(!g?.connected ? { label: "Flight computer", level: "bad", detail: "Waiting for the ground station." }
    : !talking ? { label: "Flight computer", level: "bad", detail: "No telemetry: switch the aircraft on (press EN on the ESP32 if it does not start)." }
    : { label: "Flight computer", level: "ok", detail: `Talking: ${g.rate_hz.toFixed(1)} Hz${info ? `, control loop ${info.loop_hz} Hz` : ""}.` });

  items.push(!talking || stub ? { label: "Raspberry Pi", level: stub ? "warn" : "bad", detail: stub ? "Not part of the stub." : "Unknown until the flight computer talks." }
    : st?.pi_link ? { label: "Raspberry Pi", level: "ok", detail: "Sending setpoints to the flight computer." }
    : { label: "Raspberry Pi", level: "bad", detail: "No setpoints: check the Pi has power. Its program starts by itself about a minute after power-up." });

  const calOk = info && info.cal_servos_measured && info.cal_from_nvs && info.cal_id_hex === fileId;
  items.push(!info ? { label: "Servo calibration", level: stub ? "warn" : "bad", detail: stub ? "Not reported by the stub." : "Unknown until the flight computer reports." }
    : calOk ? { label: "Servo calibration", level: "ok", detail: `Measured and on the aircraft (${info.cal_id_hex}).` }
    : !info.cal_servos_measured ? { label: "Servo calibration", level: "bad", detail: "Not measured: run Calibrate the control surfaces below, then Send to aircraft." }
    : { label: "Servo calibration", level: "warn", detail: "The aircraft has a different calibration from the saved file: Send to aircraft." });

  items.push(!st ? { label: "GPS", level: "bad", detail: "No telemetry." }
    : st.gps_fix === 2 ? { label: "GPS", level: "ok", detail: `3D fix, ${st.gps_sats} satellites${st.gps_ubx ? ", 5 Hz" : ""}.` }
    : { label: "GPS", level: "bad", detail: "No 3D fix: go outside with a clear view of the sky; the first fix can take a few minutes." });

  items.push(!st ? { label: "Barometer", level: "bad", detail: "No telemetry." }
    : st.baro_warm ? { label: "Barometer", level: "ok", detail: "Warmed up." }
    : { label: "Barometer", level: "warn", detail: "Warming up: arming is allowed 2 minutes after switching on." });

  const blocking = items.filter((i) => i.level === "bad");
  const ready = blocking.length === 0 && !stub && talking;
  items.push(ready
    ? { label: "Ready to arm", level: "ok", detail: "Everything above is green. Arm FLIGHT from the Flight computer panel." }
    : { label: "Ready to arm", level: stub ? "warn" : "bad", detail: stub ? "Stub only." : `Not yet: ${blocking.map((b) => b.label).join(", ")}.` });

  return (
    <section className="section" aria-labelledby="chk-h">
      <div className="section-head">
        <div className="section-kicker">Status</div>
        <h2 id="chk-h">Aircraft checklist</h2>
      </div>
      <ul className="surface" style={{ listStyle: "none", margin: 0, padding: "var(--space-3) var(--space-4)" }}>
        {items.map((i) => (
          <li key={i.label} style={{ display: "flex", gap: "var(--space-3)", alignItems: "baseline", padding: "4px 0" }}>
            <span role="img" aria-label={i.level === "ok" ? "ok" : i.level === "warn" ? "warning" : "not ready"}
                  style={{ color: COLOR[i.level], fontWeight: 700, width: "1.2em", textAlign: "center" }}>{ICON[i.level]}</span>
            <strong style={{ minWidth: "9.5em" }}>{i.label}</strong>
            <span>{i.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

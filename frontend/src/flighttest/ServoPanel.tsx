// ServoPanel.tsx -- flight-test panel 2: control-surface tester.
// Arming is an explicit action and nothing moves without it. Sliders span each
// servo's sweep range from the calibration file (travel limits shown, enforced
// again by the backend and the firmware). What the ground asked for and what the
// glider reports are shown separately: the firmware decides.
import { useEffect, useState } from "react";
import { IconAlert } from "../icons";
import { ft, SERVO_GPIO, SERVO_LABELS, SERVO_NAMES, useFlightTest, type Calibration } from "./store";

export function GliderModeLine() {
  const { snapshot } = useFlightTest();
  const t = snapshot?.telemetry;
  const sess = snapshot?.session;
  const end = snapshot?.last_session_end;
  return (
    <div className="ft-modeline" role="status">
      <span>Requested: <strong>{sess ? `${sess.mode}` : "nothing"}</strong>
        {sess && <small> · expires in {Math.ceil(sess.expires_in_ms / 1000)} s without a UI action</small>}</span>
      <span>Glider reports: <strong>{t ? `${t.mode}${t.status.armed ? " · armed" : ""}` : "no telemetry"}</strong></span>
      {t?.status.cmd_rejected_airborne ? (
        <span className="badge badge-danger"><IconAlert width={12} height={12} aria-hidden="true" />Refused: glider thinks it is airborne</span>
      ) : null}
      {t?.status.timeout_disarmed ? <span className="badge badge-warning">Last disarm: 150 ms link timeout</span> : null}
      {!sess && end && <span className="muted">Last session ended: {end.reason}</span>}
    </div>
  );
}

export default function ServoPanel() {
  const s = useFlightTest();
  const sess = s.snapshot?.session;
  const connected = !!s.snapshot?.ground.connected;
  const armedBench = sess?.mode === "BENCH";
  const otherSession = sess && sess.mode !== "BENCH";
  const [cal, setCal] = useState<Calibration | null>(null);
  const [draft, setDraft] = useState<Calibration | null>(null);
  const [values, setValues] = useState<number[]>([1500, 1500, 1500, 1500]);
  const [calError, setCalError] = useState<string | null>(null);

  async function loadCal() {
    try {
      const r = await ft.calibration();
      setCal(r.calibration);
      setDraft(structuredClone(r.calibration));
      setCalError(r.problems.length ? r.problems.join("; ") : null);
    } catch (e) {
      setCalError(e instanceof Error ? e.message : String(e));
    }
  }
  useEffect(() => {
    if (s.backend === "ok" && !cal) loadCal();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.backend]);

  const neutrals = cal ? SERVO_NAMES.map((n) => cal.servos[n].neutral_us) : [1500, 1500, 1500, 1500];

  async function toggleArm() {
    if (armedBench) {
      await ft.disarm();
      return;
    }
    setValues(neutrals);
    const ok = await ft.arm("bench");
    if (ok) ft.setServos(neutrals);
  }

  function move(i: number, v: number) {
    const next = values.map((x, k) => (k === i ? v : x));
    setValues(next);
    ft.setServos(next);
  }

  function centreAll() {
    setValues(neutrals);
    ft.setServos(neutrals);
  }

  function editServo(name: string, patch: Record<string, unknown>) {
    if (!draft) return;
    setDraft({ ...draft, servos: { ...draft.servos, [name]: { ...draft.servos[name], ...patch } } });
  }

  async function saveCal() {
    if (!draft) return;
    const r = await ft.saveCalibration(draft);
    if (r) await loadCal();
  }

  const dirty = JSON.stringify(cal) !== JSON.stringify(draft);
  const today = new Date().toISOString().slice(0, 10);

  return (
    <section className="section" aria-labelledby="servo-h">
      <div className="section-row">
        <div className="section-head">
          <div className="section-kicker">Bench test</div>
          <h2 id="servo-h">Control surface tester</h2>
          <p>Moves the real servos in BENCH mode. Keep hands and propeller-free surfaces clear. The firmware refuses BENCH while it believes it is airborne.</p>
        </div>
        <div className="ft-arm">
          <button className={armedBench ? "btn-danger" : "btn-primary"} onClick={toggleArm}
            disabled={!connected || !!otherSession} aria-pressed={armedBench}>
            {armedBench ? "Disarm" : "Arm BENCH"}
          </button>
          {armedBench && <button onClick={() => ft.keepalive()}>Keep armed</button>}
        </div>
      </div>

      <GliderModeLine />
      {!connected && <p className="hint-text">Connect the ground station first (panel 1).</p>}
      {otherSession && <p className="hint-text">A {sess?.mode} session is active (flight-log replay). Stop it first.</p>}

      <div className="surface ft-servos">
        {SERVO_NAMES.map((name, i) => {
          const c = cal?.servos[name];
          const lo = c?.sweep_min_us ?? 700;
          const hi = c?.sweep_max_us ?? 2200;
          return (
            <div className="ft-servo" key={name}>
              <div className="ft-servo-name">
                <strong>{SERVO_LABELS[name]}</strong>
                <span>GPIO{SERVO_GPIO[name]} · {c?.type ?? "–"}</span>
              </div>
              <div className="ft-servo-slider">
                <input type="range" min={lo} max={hi} step={5} value={values[i]}
                  disabled={!armedBench} aria-label={`${SERVO_LABELS[name]} pulse width`}
                  list={`neutral-${name}`} onChange={(e) => move(i, Number(e.target.value))} />
                <datalist id={`neutral-${name}`}><option value={c?.neutral_us ?? 1500} /></datalist>
                <div className="ft-servo-scale">
                  <span>{lo}</span>
                  <span>neutral {c?.neutral_us ?? "–"}</span>
                  <span>{hi}</span>
                </div>
              </div>
              <div className="ft-servo-value">
                <span><strong>{values[i]}</strong> µs</span>
                <small>limits {c?.min_us ?? "–"}–{c?.max_us ?? "–"}</small>
              </div>
            </div>
          );
        })}
        <div className="ft-servo-actions">
          <button onClick={centreAll} disabled={!armedBench}>Centre all</button>
          <span className="hint-text" style={{ margin: 0 }}>
            Commanded values only: telemetry does not report servo positions. Sent at most 20 times a second.
          </span>
        </div>
      </div>

      <h3 className="sub-head">Calibration (stored in flighttest/config/calibration.json)</h3>
      {calError && <p className="error-text">{calError}</p>}
      {draft && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Servo</th><th className="num">Neutral (µs)</th><th>Direction</th>
                <th className="num">Sweep</th><th className="num">Travel limits</th><th>Measured</th><th />
              </tr>
            </thead>
            <tbody>
              {SERVO_NAMES.map((name, i) => {
                const d = draft.servos[name];
                return (
                  <tr key={name}>
                    <td className="cell-strong">{SERVO_LABELS[name]}</td>
                    <td className="num">
                      <input type="number" value={d.neutral_us} min={d.sweep_min_us} max={d.sweep_max_us} step={1}
                        style={{ width: "6rem" }} aria-label={`${SERVO_LABELS[name]} neutral`}
                        onChange={(e) => editServo(name, { neutral_us: Number(e.target.value) })} />
                    </td>
                    <td>
                      <select value={d.direction} aria-label={`${SERVO_LABELS[name]} direction`}
                        onChange={(e) => editServo(name, { direction: Number(e.target.value) })}>
                        <option value={1}>+1 normal</option>
                        <option value={-1}>−1 reversed</option>
                      </select>
                    </td>
                    <td className="num">{d.sweep_min_us}–{d.sweep_max_us}</td>
                    <td className="num">{d.min_us}–{d.max_us}</td>
                    <td>{d.measured ? `yes, ${d.measured_on ?? ""}` : <span className="badge badge-warning">placeholder</span>}</td>
                    <td>
                      <button className="btn-plain" disabled={!armedBench}
                        title="Use the slider's current position as this servo's neutral"
                        onClick={() => editServo(name, { neutral_us: values[i], measured: true, measured_on: today })}>
                        Set from slider
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="ft-servo-actions" style={{ marginTop: "var(--space-3)" }}>
        <button className="btn-primary" disabled={!dirty} onClick={saveCal}>Save calibration</button>
        <button disabled={!dirty} onClick={() => setDraft(structuredClone(cal))}>Discard changes</button>
        <span className="hint-text" style={{ margin: 0 }}>
          Saved values are validated (neutral inside the sweep, limits inside the servo type). Pushing them to the
          ESP32's flash is not implemented yet.
        </span>
      </div>
    </section>
  );
}

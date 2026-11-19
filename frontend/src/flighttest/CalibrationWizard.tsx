// CalibrationWizard.tsx -- guided servo calibration, one surface at a time.
//
//   1 Centre     nudge the servo until the surface is flush (sets neutral_us)
//   2 Direction  "Test" moves the surface the way the simulator calls positive
//                (roll right / nose down / nose left); the user says whether it
//                moved the way the picture shows; "wrong way" flips direction
//   3 Travel     step to each end stop, back off, set the limits; full
//                deflection defaults to the nearer limit
//   Review       validated like flighttest/calibration.py, Save, Send to aircraft
//
// Moves the real servos in a BENCH session (the firmware refuses it while
// armed for flight or airborne). Pulses go out through the store's 50 ms
// throttle; the wizard asks for the servo type's full range ("envelope") so it
// can find end stops beyond the current limits. Kept armed while open.
import { useEffect, useMemo, useState } from "react";
import { ft, SERVO_LABELS, SERVO_NAMES, useFlightTest, type Calibration, type ServoCal } from "./store";

type Step = "centre" | "direction" | "travel";
const STEPS: { id: Step; label: string }[] = [
  { id: "centre", label: "1 Centre" }, { id: "direction", label: "2 Direction" }, { id: "travel", label: "3 Travel" },
];
const ENVELOPE: Record<string, [number, number]> = { MG90S: [600, 2300], MG996R: [1000, 2000] };
const TEST_US = 150;

// What the simulator's POSITIVE command does to each surface (see firmware/flight/README.md).
const POSITIVE: Record<string, { move: string; why: string; arrow: string }> = {
  aileron_l: { move: "Left aileron trailing edge goes DOWN", why: "positive aileron rolls the aircraft right", arrow: "↓" },
  aileron_r: { move: "Right aileron trailing edge goes UP", why: "positive aileron rolls the aircraft right", arrow: "↑" },
  elevator: { move: "Elevator trailing edge goes DOWN", why: "positive elevator pitches the nose down", arrow: "↓" },
  rudder: { move: "Rudder trailing edge goes LEFT (seen from behind)", why: "positive rudder yaws the nose left", arrow: "←" },
};

interface Done { centre: boolean; direction: boolean; travel: boolean }

function problems(name: string, c: ServoCal): string[] {
  const out: string[] = [];
  const [lo, hi] = ENVELOPE[c.type] ?? [600, 2300];
  if (c.min_us < lo || c.max_us > hi) out.push(`limits ${c.min_us}–${c.max_us} µs are outside the ${c.type} range ${lo}–${hi}`);
  if (!(c.min_us <= c.neutral_us && c.neutral_us <= c.max_us)) out.push("centre must lie between the two limits");
  if (c.us_full_deflection < 50 || c.us_full_deflection > 900) out.push("full deflection must be 50–900 µs");
  else if (c.neutral_us - c.us_full_deflection < c.min_us || c.neutral_us + c.us_full_deflection > c.max_us)
    out.push("centre ± full deflection runs past a limit");
  return out.map((p) => `${SERVO_LABELS[name]}: ${p}`);
}

export default function CalibrationWizard() {
  const s = useFlightTest();
  const snap = s.snapshot;
  const connected = !!snap?.ground.connected;
  const sess = snap?.session;
  const active = sess?.mode === "BENCH";
  const st = snap?.telemetry?.status;
  const push = snap?.calibration_push ?? null;
  const [saved, setSaved] = useState<Calibration | null>(null);
  const [draft, setDraft] = useState<Calibration | null>(null);
  const [servo, setServo] = useState<string>(SERVO_NAMES[0]);
  const [step, setStep] = useState<Step | "review">("centre");
  const [cmd, setCmd] = useState<number>(1500);          // commanded µs of the servo being calibrated
  const [testing, setTesting] = useState(false);
  const [done, setDone] = useState<Record<string, Done>>(
    Object.fromEntries(SERVO_NAMES.map((n) => [n, { centre: false, direction: false, travel: false }])));
  const [savedOk, setSavedOk] = useState(false);

  async function load() {
    const r = await ft.calibration();
    setSaved(r.calibration);
    setDraft(structuredClone(r.calibration));
  }
  useEffect(() => { if (s.backend === "ok" && !draft) load().catch(() => {}); }, [s.backend]);  // eslint-disable-line react-hooks/exhaustive-deps

  // stay armed while the wizard is open (the backend expires a session after 60 s without a UI action)
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => ft.keepalive(), 10000);
    return () => clearInterval(id);
  }, [active]);

  const sv = draft?.servos[servo];

  // Send all four: the one being calibrated at `us`, the others at their draft neutral.
  function drive(us: number) {
    if (!draft) return;
    setCmd(us);
    ft.setServos(SERVO_NAMES.map((n) => (n === servo ? us : draft.servos[n].neutral_us)), true);
  }
  function patch(p: Partial<ServoCal>) {
    if (!draft) return;
    setDraft({ ...draft, servos: { ...draft.servos, [servo]: { ...draft.servos[servo], ...p } } });
    setSavedOk(false);
  }
  function mark(k: keyof Done) { setDone({ ...done, [servo]: { ...done[servo], [k]: true } }); }

  async function start() {
    if (!draft) await load();
    const ok = await ft.arm("bench");
    if (ok && draft) drive(draft.servos[servo].neutral_us);
  }
  async function finish() { await ft.disarm(); setTesting(false); }

  function choose(name: string, next: Step | "review" = "centre") {
    setServo(name); setStep(next); setTesting(false);
    if (draft && active) {
      const n = draft.servos[name].neutral_us;
      setCmd(n);
      ft.setServos(SERVO_NAMES.map((m) => draft.servos[m].neutral_us), true);
    }
  }
  function goStep(next: Step | "review") {
    setStep(next); setTesting(false);
    if (sv && active) drive(sv.neutral_us);
  }

  const allProblems = useMemo(() => (draft ? SERVO_NAMES.flatMap((n) => problems(n, draft.servos[n])) : []), [draft]);
  const today = new Date().toISOString().slice(0, 10);

  async function save() {
    if (!draft) return;
    const out = structuredClone(draft);
    for (const n of SERVO_NAMES) {
      const c = out.servos[n];
      c.sweep_min_us = c.min_us;
      c.sweep_max_us = c.max_us;
      if (done[n].centre && done[n].direction && done[n].travel) { c.measured = true; c.measured_on = today; }
    }
    const r = await ft.saveCalibration(out);
    if (r) { await load(); setSavedOk(true); }
  }

  const nudge = (d: number) => { if (!sv) return; const n = sv.neutral_us + d; patch({ neutral_us: n }); drive(n); };
  const step10 = (d: number) => drive(cmd + d);
  const env = sv ? ENVELOPE[sv.type] ?? [600, 2300] : [600, 2300];

  return (
    <section className="section" aria-labelledby="wiz-h">
      <div className="section-row">
        <div className="section-head">
          <div className="section-kicker">Set up</div>
          <h2 id="wiz-h">Calibrate the control surfaces</h2>
          <p>
            A guided walk through each surface: centre it, check which way it moves, find its end stops. Takes a few
            minutes per surface. The servos move: keep fingers clear of the horns and linkages.
          </p>
        </div>
        <div className="ft-arm">
          {active
            ? <button className="btn-danger" onClick={finish}>Finish (disarm)</button>
            : <button className="btn-primary" onClick={start} disabled={!connected || !!sess || !!st?.armed || !!st?.airborne}>
                Start calibration
              </button>}
        </div>
      </div>

      {!connected && <p className="hint-text">Connect the ground station first.</p>}
      {connected && (st?.armed || st?.airborne) && !active && (
        <p className="hint-text">Disarm the aircraft first: calibration only runs on the ground, disarmed.</p>)}
      {sess && !active && <p className="hint-text">A {sess.mode} session is running. Stop it first.</p>}

      {draft && (
        <>
          <div className="seg" role="tablist" aria-label="Surface" style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", margin: "var(--space-3) 0" }}>
            {SERVO_NAMES.map((n) => {
              const d = done[n];
              const all = d.centre && d.direction && d.travel;
              return (
                <button key={n} role="tab" aria-selected={servo === n && step !== "review"}
                        className={servo === n && step !== "review" ? "btn-primary" : ""} onClick={() => choose(n)}>
                  {all ? "✓ " : ""}{SERVO_LABELS[n]}
                </button>
              );
            })}
            <button role="tab" aria-selected={step === "review"} className={step === "review" ? "btn-primary" : ""}
                    onClick={() => setStep("review")}>Review &amp; save</button>
          </div>

          {step !== "review" && sv && (
            <div className="surface" style={{ padding: "var(--space-4)" }}>
              <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", marginBottom: "var(--space-3)" }}>
                {STEPS.map((x) => (
                  <button key={x.id} className={step === x.id ? "btn-primary" : "btn-plain"} onClick={() => goStep(x.id)}>
                    {done[servo][x.id] ? "✓ " : ""}{x.label}
                  </button>
                ))}
              </div>
              {!active && <p className="hint-text">Press <strong>Start calibration</strong> to move the servos.</p>}

              {step === "centre" && (
                <div>
                  <h3 style={{ marginTop: 0 }}>{SERVO_LABELS[servo]}: centre</h3>
                  <p>Nudge until the surface lines up with the wing or tail (flush, no deflection).</p>
                  <div style={{ fontSize: "2rem", fontVariantNumeric: "tabular-nums", margin: "var(--space-2) 0" }}>{sv.neutral_us} µs</div>
                  <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
                    {[-10, -1, 1, 10].map((d) => (
                      <button key={d} disabled={!active} onClick={() => nudge(d)}>{d > 0 ? `+${d}` : d}</button>))}
                  </div>
                  <div style={{ marginTop: "var(--space-3)" }}>
                    <button className="btn-primary" disabled={!active} onClick={() => { mark("centre"); goStep("direction"); }}>
                      It's centred, next
                    </button>
                  </div>
                </div>
              )}

              {step === "direction" && (
                <div>
                  <h3 style={{ marginTop: 0 }}>{SERVO_LABELS[servo]}: direction</h3>
                  <p>
                    Press <strong>Test</strong>. The surface should move like this:
                  </p>
                  <div className="surface" style={{ padding: "var(--space-3)", display: "flex", gap: "var(--space-3)", alignItems: "center" }}>
                    <span style={{ fontSize: "2.5rem", lineHeight: 1 }} aria-hidden="true">{POSITIVE[servo].arrow}</span>
                    <div><strong>{POSITIVE[servo].move}</strong><br /><small>({POSITIVE[servo].why})</small></div>
                  </div>
                  <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", marginTop: "var(--space-3)" }}>
                    <button className="btn-primary" disabled={!active}
                            onClick={() => { drive(sv.neutral_us + sv.direction * TEST_US); setTesting(true); }}>Test</button>
                    <button disabled={!active || !testing} onClick={() => { drive(sv.neutral_us); setTesting(false); }}>Back to centre</button>
                  </div>
                  {testing && (
                    <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", marginTop: "var(--space-3)" }}>
                      <button className="btn-primary" onClick={() => { mark("direction"); drive(sv.neutral_us); goStep("travel"); }}>
                        It moved that way
                      </button>
                      <button onClick={() => {
                        const dir = (sv.direction === 1 ? -1 : 1) as 1 | -1;
                        patch({ direction: dir });
                        drive(sv.neutral_us + dir * TEST_US);
                      }}>
                        It moved the other way (flip, and test again)
                      </button>
                    </div>
                  )}
                  <p className="hint-text">Direction is now <strong>{sv.direction === 1 ? "normal (+1)" : "reversed (−1)"}</strong>.</p>
                </div>
              )}

              {step === "travel" && (
                <div>
                  <h3 style={{ marginTop: 0 }}>{SERVO_LABELS[servo]}: travel limits</h3>
                  <p>
                    Step towards each end until the surface stops moving or the linkage starts to strain, then back off one
                    step and set the limit. Never leave a servo pushing against a stop: it heats up and draws heavy current.
                  </p>
                  <div style={{ fontSize: "1.6rem", fontVariantNumeric: "tabular-nums", margin: "var(--space-2) 0" }}>
                    now {cmd} µs <small className="muted">(servo range {env[0]}–{env[1]})</small>
                  </div>
                  <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
                    {[-50, -10, 10, 50].map((d) => (
                      <button key={d} disabled={!active} onClick={() => step10(d)}>{d > 0 ? `+${d}` : d}</button>))}
                    <button disabled={!active} onClick={() => drive(sv.neutral_us)}>Centre</button>
                  </div>
                  <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", marginTop: "var(--space-3)" }}>
                    <button disabled={!active || cmd >= sv.neutral_us} onClick={() => patch({ min_us: cmd })}>
                      Set LOW limit here ({sv.min_us})
                    </button>
                    <button disabled={!active || cmd <= sv.neutral_us} onClick={() => patch({ max_us: cmd })}>
                      Set HIGH limit here ({sv.max_us})
                    </button>
                  </div>
                  <div style={{ marginTop: "var(--space-3)" }}>
                    <label>
                      Full deflection{" "}
                      <input type="number" min={50} max={900} step={10} value={sv.us_full_deflection} style={{ width: "6rem" }}
                             onChange={(e) => patch({ us_full_deflection: Number(e.target.value) })} /> µs from centre
                    </label>{" "}
                    <button className="btn-plain" onClick={() =>
                      patch({ us_full_deflection: Math.max(50, Math.min(900, Math.min(sv.neutral_us - sv.min_us, sv.max_us - sv.neutral_us))) })}>
                      Use the nearer limit
                    </button>
                    <p className="hint-text">
                      The pulse offset at the simulator's maximum deflection ({servo === "elevator" ? "20°" : "25°"}). If you can
                      measure the angle, drive to it and enter the offset; otherwise "use the nearer limit".
                    </p>
                  </div>
                  <button className="btn-primary" disabled={!active || problems(servo, sv).length > 0}
                          onClick={() => {
                            mark("travel"); drive(sv.neutral_us);
                            const i = SERVO_NAMES.indexOf(servo as typeof SERVO_NAMES[number]);
                            if (i < SERVO_NAMES.length - 1) choose(SERVO_NAMES[i + 1]); else setStep("review");
                          }}>
                    Done with {SERVO_LABELS[servo]}, next
                  </button>
                  {problems(servo, sv).map((p) => <p key={p} className="error-text">{p}</p>)}
                </div>
              )}
            </div>
          )}

          {step === "review" && (
            <div className="surface" style={{ padding: "var(--space-4)" }}>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Surface</th><th className="num">Centre</th><th>Direction</th><th className="num">Limits</th><th className="num">Full defl.</th><th>Steps done</th></tr></thead>
                  <tbody>
                    {SERVO_NAMES.map((n) => {
                      const c = draft.servos[n]; const d = done[n];
                      return (
                        <tr key={n}>
                          <td className="cell-strong">{SERVO_LABELS[n]}</td>
                          <td className="num">{c.neutral_us}</td>
                          <td>{c.direction === 1 ? "normal" : "reversed"}</td>
                          <td className="num">{c.min_us}–{c.max_us}</td>
                          <td className="num">{c.us_full_deflection}</td>
                          <td>{[d.centre && "centre", d.direction && "direction", d.travel && "travel"].filter(Boolean).join(", ") || <span className="badge badge-warning">none</span>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {allProblems.map((p) => <p key={p} className="error-text">{p}</p>)}
              <p className="hint-text">
                A surface counts as <strong>measured</strong> only when all three steps are done; the aircraft refuses to arm
                for flight until every surface is measured.
              </p>
              <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", alignItems: "center" }}>
                <button className="btn-primary" disabled={allProblems.length > 0 || JSON.stringify(saved) === JSON.stringify(draft) && savedOk}
                        onClick={save}>1. Save</button>
                <button className="btn-primary" disabled={!savedOk || !connected || !!st?.armed || !!st?.airborne || push?.state === "sending"}
                        onClick={async () => { if (active) await ft.disarm(); ft.pushCalibration(); }}>
                  2. Send to aircraft
                </button>
                <span className="hint-text" style={{ margin: 0 }}>
                  {push?.state === "confirmed" ? `✓ The aircraft is using calibration ${push.cal_id_hex}.`
                    : push?.state === "sending" ? `Sending ${push.cal_id_hex}… (attempt ${push.sends} of 3)`
                    : push?.state === "not confirmed" ? "Not confirmed: check the aircraft is on and disarmed, then send again."
                    : savedOk ? "Saved. Now send it to the aircraft." : ""}
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}

// LogReplayPanel.tsx -- flight-test panel 4: replay a flight log, optionally
// driving the real servos in sync (REPLAY mode).
//
// Shares FlightScene with the simulator's Flight Replay, so a real flight and
// a simulated one are viewed the same way. Servo playback:
//   - only after an explicit "Drive servos" arm, and only in REPLAY mode, which
//     the firmware refuses while it believes it is airborne;
//   - scrubbing and playback send through the store's 50 ms throttle, and the
//     backend caps the radio at 20 Hz, so scrubbing never floods the link;
//   - what happens on pause is a visible choice: hold the last position, or
//     centre the surfaces.
import { useEffect, useMemo, useRef, useState } from "react";
import FlightScene from "../components/FlightScene";
import type { Frame } from "../types";
import type { Tab } from "../useHashRoute";
import { ft, SERVO_LABELS, SERVO_NAMES, useFlightTest, type Calibration, type ParsedLog } from "./store";
import { GliderModeLine } from "./ServoPanel";

const DEG = 180 / Math.PI;
const SPEEDS = [0.25, 0.5, 1, 2, 4];

function toSceneFrames(log: ParsedLog): Frame[] {
  let last: [number, number, number] = [0, 0, -(log.frames[0]?.alt_baro ?? 0)];
  return log.frames.map((f) => {
    if (f.pos_ned) last = f.pos_ned;
    const pos: [number, number, number] = f.pos_ned ?? [last[0], last[1], -f.alt_baro];
    return {
      t: f.t, pos_ned: pos, euler: f.euler, v_body: [0, 0, 0], ctrl: { ail: 0, elev: 0, rud: 0 },
      dist_home: Math.hypot(pos[0], pos[1]), agl: f.alt_baro, wind_ned: [0, 0, 0],
      stall_violation: false, bank_violation: false, unreach_violation: false,
      alpha_deg: 0, roll_deg: f.euler[0] * DEG, airspeed: 0,
    } as Frame;
  });
}

export default function LogReplayPanel({ onNavigate }: { onNavigate: (tab: Tab, a?: string | null) => void }) {
  const s = useFlightTest();
  const sess = s.snapshot?.session;
  const driving = sess?.mode === "REPLAY";
  const [logs, setLogs] = useState<{ name: string; size_bytes: number }[]>([]);
  const [name, setName] = useState<string | null>(null);
  const [log, setLog] = useState<ParsedLog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [onPause, setOnPause] = useState<"hold" | "centre">("hold");
  const [cal, setCal] = useState<Calibration | null>(null);
  const [showSkipped, setShowSkipped] = useState(false);

  async function refresh() {
    try {
      setLogs(await ft.localLogs());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  useEffect(() => {
    if (s.backend === "ok") {
      refresh();
      ft.calibration().then((r) => setCal(r.calibration)).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.backend, s.logsVersion]);

  useEffect(() => {
    if (!name) return;
    setError(null);
    setPlaying(false);
    setIdx(0);
    ft.log(name).then(setLog).catch((e) => { setLog(null); setError(e instanceof Error ? e.message : String(e)); });
  }, [name]);

  const frames = useMemo(() => (log ? toSceneFrames(log) : []), [log]);
  const frame = log?.frames[idx];

  // Playback on the log's own clock.
  const clock = useRef<{ wall: number; t: number } | null>(null);
  useEffect(() => {
    if (!playing || !log || log.frames.length < 2) {
      clock.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      if (!clock.current) clock.current = { wall: now, t: log.frames[idx].t };
      const t = clock.current.t + ((now - clock.current.wall) / 1000) * speed;
      let i = idx;
      while (i < log.frames.length - 1 && log.frames[i + 1].t <= t) i++;
      if (i >= log.frames.length - 1 && t >= log.frames[log.frames.length - 1].t) {
        setIdx(log.frames.length - 1);
        setPlaying(false);
        return;
      }
      if (i !== idx) setIdx(i);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, log, idx]);

  // Servo playback follows the frame shown (throttled in the store).
  useEffect(() => {
    if (driving && frame) ft.setServos(frame.servo_us);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, driving]);

  // Pause behaviour.
  const prevPlaying = useRef(playing);
  useEffect(() => {
    if (prevPlaying.current && !playing && driving && onPause === "centre" && cal) {
      ft.setServos(SERVO_NAMES.map((n) => cal.servos[n].neutral_us));
    }
    prevPlaying.current = playing;
  }, [playing, driving, onPause, cal]);

  async function toggleDrive() {
    if (driving) {
      await ft.disarm();
      return;
    }
    setPlaying(false);
    const ok = await ft.arm("replay");
    if (ok && frame) ft.setServos(frame.servo_us);
  }

  const src = String(log?.header.source ?? "");
  const isStub = src.startsWith("STUB");
  const simId = /simulator recording (\S+)/.exec(src)?.[1] ?? null;
  const connected = !!s.snapshot?.ground.connected;

  return (
    <section className="section" aria-labelledby="replay-h">
      <div className="section-head">
        <div className="section-kicker">Replay</div>
        <h2 id="replay-h">Flight log replay</h2>
        <p>Replays a full-rate log fetched from the Pi, in the same 3D view as the simulator replays. Optionally drives the real servos in step with the log (REPLAY mode).</p>
      </div>

      <div className="surface replay-toolbar">
        <label className="toolbar-field" style={{ minWidth: "16rem", flex: 1 }}>
          Fetched log
          <select value={name ?? ""} onChange={(e) => setName(e.target.value || null)}>
            <option value="">{logs.length ? "Choose a log…" : "No logs fetched yet (panel 1)"}</option>
            {logs.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
          </select>
        </label>
        <button onClick={refresh}>Refresh</button>
        <div className="toolbar-spacer" />
        <button className={driving ? "btn-danger" : "btn-primary"} disabled={!log || !connected || (!!sess && !driving)}
          onClick={toggleDrive} aria-pressed={driving}>
          {driving ? "Stop driving servos" : "Drive servos (REPLAY)"}
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}

      {log && (
        <>
          {isStub && (
            <div className="callout callout-warning" style={{ marginBottom: "var(--space-3)" }}>
              <div className="callout-title">Not a real flight</div>
              <div>{src}. {String(log.header.note ?? "")}</div>
            </div>
          )}
          <div className="replay-meta">
            <span>{log.summary.n_frames} frames</span>
            <span>{log.summary.duration_s.toFixed(1)} s</span>
            <span>{log.summary.rate_hz ? `${log.summary.rate_hz.toFixed(1)} Hz` : "–"}</span>
            <span>max alt {log.summary.max_alt_m?.toFixed(1) ?? "–"} m</span>
            <span>min rail {log.summary.min_rail_v == null ? "not measured" : `${log.summary.min_rail_v.toFixed(2)} V`}</span>
            {log.summary.frames_without_position > 0 && <span className="badge badge-warning">{log.summary.frames_without_position} frames without GPS position</span>}
            {log.summary.skipped_lines > 0 && (
              <button className="btn-plain" onClick={() => setShowSkipped((v) => !v)}>
                {log.summary.skipped_lines} unreadable lines {showSkipped ? "▴" : "▾"}
              </button>
            )}
            {simId && <button className="btn-plain" onClick={() => onNavigate("replay", simId)}>Open the simulator flight</button>}
          </div>
          {showSkipped && (
            <ul className="hint-text">{log.skipped.map((k) => <li key={k.line}>line {k.line}: {k.reason}</li>)}</ul>
          )}

          {driving && <GliderModeLine />}

          <div className="replay-layout">
            <div className="replay-visual">
              <div className="scrubber-row">
                <button onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>{playing ? "Pause" : "Play"}</button>
                <input type="range" aria-label="Log time" min={0} max={Math.max(0, log.frames.length - 1)} value={idx}
                  onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }} />
                <span>{frame ? `t=${frame.t.toFixed(2)} s` : ""}</span>
                <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">
                  {SPEEDS.map((v) => <option key={v} value={v}>{v}×</option>)}
                </select>
              </div>
              {frames.length > 0 && <div className="scene-wrap"><FlightScene frames={frames} currentIndex={idx} rHome={20} /></div>}
            </div>

            <div className="replay-side">
              <div className="telemetry-box" role="radiogroup" aria-label="When paused">
                <h4>When paused, the servos</h4>
                <label className="control-label"><input type="radio" name="onpause" checked={onPause === "hold"}
                  onChange={() => setOnPause("hold")} />hold the last position</label>
                <label className="control-label"><input type="radio" name="onpause" checked={onPause === "centre"}
                  onChange={() => setOnPause("centre")} />return to neutral</label>
                <p className="hint-text" style={{ margin: "var(--space-2) 0 0" }}>
                  {driving ? (playing ? "Playing: servos follow the log." : onPause === "hold" ? "Paused: holding." : "Paused: centred.")
                    : "Servos are not being driven."}
                </p>
              </div>
              {frame && (
                <div className="telemetry-box">
                  <h4>Logged at this moment</h4>
                  <div className="telemetry-rows">
                    <div><span>Mode</span><strong>{frame.mode}{frame.armed ? " · armed" : ""}</strong></div>
                    <div><span>Altitude</span><strong>{frame.alt_baro.toFixed(1)} m</strong></div>
                    <div><span>Roll / pitch</span><strong>{(frame.euler[0] * DEG).toFixed(0)}° / {(frame.euler[1] * DEG).toFixed(0)}°</strong></div>
                    <div><span>Sonar</span><strong>{frame.sonar == null ? "invalid" : `${frame.sonar.toFixed(2)} m`}</strong></div>
                    <div><span>Rail</span><strong>{frame.rail_v == null ? "not measured" : `${frame.rail_v.toFixed(2)} V`}</strong></div>
                    <div><span>GPS</span><strong>{["none", "2D", "3D"][frame.gps_fix] ?? "?"} · {frame.gps_sats} sats</strong></div>
                    {SERVO_NAMES.map((n, i) => (
                      <div key={n}><span>{SERVO_LABELS[n]}</span><strong>{frame.servo_us[i]} µs</strong></div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
      {!log && !error && <div className="empty-state"><strong>No log selected</strong>Fetch a log from the Pi in panel 1, then choose it here.</div>}
    </section>
  );
}

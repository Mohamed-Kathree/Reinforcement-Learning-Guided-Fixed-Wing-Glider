// AircraftPlayback.tsx -- "Play on aircraft": the real servos follow a simulated
// episode's surface movements in step with the replay.
//
// Each replay frame carries the deflections the simulated ESP32 commanded
// (frame.ctrl, rad). They go to POST /api/ft/surfaces, which maps them through
// the saved calibration exactly as the flight firmware does (calibration.py
// deflection_to_us == inner_loop.c rlg_deflection_to_us), in a REPLAY session:
//   - only after an explicit "Play on aircraft" arm;
//   - the aircraft refuses REPLAY while armed for flight or airborne;
//   - pulses are clamped to the calibrated limits; sent at most every 50 ms,
//     and the radio carries 20 commands/s, the same rate as the replay (20 Hz);
//   - when playback pauses or ends: hold the last position, or centre (a choice).
import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../config";
import { ft, useFlightTest } from "../flighttest/store";
import type { Frame } from "../types";

interface Summary { episode_id: string; stage: number; controller: string; outcome: string; quality: number }

function pickDemos(list: Summary[]): { label: string; why: string; id: string }[] {
  const ok = list.filter((s) => s.outcome === "success");
  const best = (xs: Summary[]) => xs.sort((a, b) => b.quality - a.quality)[0];
  const learned = (s: Summary) => s.controller !== "baseline";
  const picks: { label: string; why: string; s?: Summary }[] = [
    { label: "Clean landing", why: "calm air, the learned controller, best touchdown", s: best(ok.filter((s) => s.stage === 0 && learned(s))) },
    { label: "Landing in wind", why: "stage 2-3 wind and gusts, best touchdown", s: best(ok.filter((s) => s.stage >= 2)) },
    { label: "Strong wind, gated controller", why: "stage 3: the gate may hand over to the baseline mid-flight", s: best(list.filter((s) => s.stage === 3 && s.controller.startsWith("gated"))) },
  ];
  return picks.filter((p) => p.s).map((p) => ({ label: p.label, why: p.why, id: p.s!.episode_id }));
}

export default function AircraftPlayback({ frame, playing, onPick }:
  { frame: Frame | undefined; playing: boolean; onPick: (id: string) => void }) {
  const s = useFlightTest();
  const snap = s.snapshot;
  const connected = !!snap?.ground.connected;
  const sess = snap?.session;
  const driving = sess?.mode === "REPLAY";
  const st = snap?.telemetry?.status;
  const info = snap?.info;
  const [onPause, setOnPause] = useState<"hold" | "centre">("hold");
  const [demos, setDemos] = useState<{ label: string; why: string; id: string }[]>([]);
  const wasPlaying = useRef(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/episode/summaries`).then((r) => r.json()).then((xs: Summary[]) => setDemos(pickDemos(xs))).catch(() => {});
  }, []);

  // follow the replay
  useEffect(() => {
    if (driving && frame && (playing || onPause === "hold")) ft.surfaces(frame.ctrl.ail, frame.ctrl.elev, frame.ctrl.rud);
  }, [driving, frame, playing, onPause]);

  // paused or finished: centre if asked
  useEffect(() => {
    if (driving && wasPlaying.current && !playing && onPause === "centre") ft.surfaces(0, 0, 0);
    wasPlaying.current = playing;
  }, [playing, driving, onPause]);

  // a paused replay is not a UI action: keep the session alive
  useEffect(() => {
    if (!driving) return;
    const id = setInterval(() => ft.keepalive(), 10000);
    return () => clearInterval(id);
  }, [driving]);

  const blocked = !connected ? "Connect the ground station (Ground Station page)."
    : st?.armed && !driving ? "The aircraft is armed: disarm it first."
    : st?.airborne ? "The aircraft thinks it is airborne; it refuses replays."
    : sess && !driving ? `A ${sess.mode} session is running (Ground Station page). Stop it first.`
    : null;

  return (
    <div className="surface" style={{ padding: "var(--space-3)", margin: "var(--space-3) 0" }}>
      <div style={{ display: "flex", gap: "var(--space-3)", alignItems: "center", flexWrap: "wrap" }}>
        <strong>Play on aircraft</strong>
        {driving
          ? <button className="btn-danger" onClick={() => ft.disarm()}>Stop (servos to centre)</button>
          : <button className="btn-primary" disabled={!!blocked} onClick={() => ft.arm("replay")}>Play on aircraft</button>}
        <label style={{ display: "flex", gap: "var(--space-2)", alignItems: "center" }}>
          When paused:
          <select value={onPause} onChange={(e) => setOnPause(e.target.value as "hold" | "centre")}>
            <option value="hold">hold position</option>
            <option value="centre">centre the surfaces</option>
          </select>
        </label>
        <span className="hint-text" style={{ margin: 0 }}>
          {blocked ?? (driving
            ? (playing ? "The servos are following the replay." : "Ready: press Play. Scrubbing moves the surfaces too.")
            : "The real control surfaces will follow this flight, in real time.")}
        </span>
      </div>
      {connected && info && !info.cal_servos_measured && (
        <p className="hint-text" style={{ marginBottom: 0 }}>
          The aircraft's servo calibration is not measured yet: surfaces may move the wrong way. Run the calibration
          wizard on the Ground Station page first.
        </p>
      )}
      {demos.length > 0 && (
        <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap", marginTop: "var(--space-2)", alignItems: "center" }}>
          <span className="muted">Good demo flights:</span>
          {demos.map((d) => (
            <button key={d.id} className="btn-plain" title={d.why} onClick={() => onPick(d.id)}>{d.label}</button>))}
        </div>
      )}
    </div>
  );
}

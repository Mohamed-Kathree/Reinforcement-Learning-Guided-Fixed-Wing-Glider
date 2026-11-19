// AircraftPanel.tsx -- the real flight computer through the ESP32 ground station.
//
//  - flight phase, control-loop rate, Pi link, GPS rate (Info packet, 1 Hz);
//  - calibration: what the aircraft is using vs the saved file, and "Send to
//    aircraft" (CalibrationSet; the aircraft applies it only DISARMED and on the
//    ground, stores it in flash, and the push counts only when it reports the
//    new calibration back);
//  - console: the flight computer's text output, quick commands, and FLIGHT
//    arming with a confirmation (the ground app never arms FLIGHT on its own).
import { useState } from "react";
import { IconAlert } from "../icons";
import { ft, useFlightTest } from "./store";

const DEG = 180 / Math.PI;
const APEX = ["-", "baro descent", "timeout", "at throw"];
const QUICK = ["report", "disarm", "help"];

export default function AircraftPanel() {
  const s = useFlightTest();
  const snap = s.snapshot;
  const info = snap?.info ?? null;
  const t = snap?.telemetry ?? null;
  const st = t?.status;
  const push = snap?.calibration_push ?? null;
  const connected = !!snap?.ground.connected;
  const fileId = snap?.file_cal_id_hex ?? null;
  const calMatches = info && fileId && info.cal_id_hex === fileId;
  const [line, setLine] = useState("");
  const [confirmArm, setConfirmArm] = useState(false);
  const lines = snap?.console ?? [];

  const send = (cmd: string) => {
    if (cmd.trim().toLowerCase() === "arm") { setConfirmArm(true); return; }
    ft.console(cmd);
  };

  return (
    <section className="section" aria-labelledby="ac-h">
      <div className="section-head">
        <div className="section-kicker">Aircraft</div>
        <h2 id="ac-h">Flight computer</h2>
        <p>
          The ESP32 flight computer as it reports itself through the ground station: flight phase, loop rate, Pi link,
          and the servo calibration it is flying with. Calibration edits in the tester below are saved to the file;
          <strong> Send to aircraft</strong> applies them on the aircraft.
        </p>
      </div>

      {!connected && (
        <div className="callout callout-warning" role="status" style={{ marginBottom: "var(--space-4)" }}>
          <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
          <div>Connect the ground station (serial) above to talk to the aircraft.</div>
        </div>
      )}

      <div className="ft-tiles">
        <div className="surface ft-tile">
          <div className="ft-tile-label">Phase</div>
          <div className="ft-tile-value">{t?.phase ?? "-"}</div>
          <div className="ft-tile-note">
            {t ? `${t.mode}${st?.pi_failsafe ? " · failsafe glide" : ""}${st?.home_set ? " · home set" : ""}${st?.yaw_aligned ? " · yaw aligned" : ""}` : "no telemetry"}
            {info && info.apex_cause > 0 ? ` · apex: ${APEX[info.apex_cause] ?? info.apex_cause}` : ""}
          </div>
        </div>
        <div className="surface ft-tile">
          <div className="ft-tile-label">Control loop</div>
          <div className="ft-tile-value">{info ? `${info.loop_hz} Hz` : "-"}</div>
          <div className="ft-tile-note">
            {info ? `setpoint roll ${(info.roll_sp * DEG).toFixed(1)}°, pitch ${(info.pitch_sp * DEG).toFixed(1)}°` : "waiting for Info (1 Hz)"}
          </div>
        </div>
        <div className="surface ft-tile">
          <div className="ft-tile-label">Pi link</div>
          <div className="ft-tile-value">{st ? (st.pi_link ? "OK" : "No setpoints") : "-"}</div>
          <div className="ft-tile-note">
            {info ? `gaps ${info.pi_rx_gaps} · bad ${info.pi_rx_bad} · Pi saw ${info.pi_tlm_gaps} telemetry gaps` : ""}
          </div>
        </div>
        <div className="surface ft-tile">
          <div className="ft-tile-label">GPS</div>
          <div className="ft-tile-value">{st ? (st.gps_ubx ? "UBX" : "NMEA") : "-"}</div>
          <div className="ft-tile-note">{info ? `${info.gps_rate_hz.toFixed(1)} Hz${st?.gps_ubx ? "" : " · fallback, no vertical speed"}` : ""}</div>
        </div>
      </div>

      <div className="surface" style={{ padding: "var(--space-4)", marginTop: "var(--space-4)" }}>
        <h3 style={{ marginTop: 0 }}>Servo calibration on the aircraft</h3>
        {info ? (
          <p>
            Aircraft is using <code>{info.cal_id_hex}</code> ({info.cal_from_nvs ? "pushed, stored in flash" : "compiled into the firmware"};
            servos {info.cal_servos_measured ? "measured" : <strong>not measured</strong>}, IMU {info.cal_imu_verified ? "verified" : <strong>not verified</strong>}).
            Saved file is <code>{fileId}</code>: {calMatches ? <strong>same</strong> : <strong>different</strong>}.
          </p>
        ) : <p>No Info packet from the aircraft yet.</p>}
        <div style={{ display: "flex", gap: "var(--space-3)", alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn-primary" disabled={!connected || !!st?.armed || !!st?.airborne || push?.state === "sending"}
                  onClick={() => ft.pushCalibration()}>
            Send to aircraft
          </button>
          <span className="ft-tile-note">
            {st?.armed || st?.airborne ? "Disarm and land first: the aircraft refuses a calibration while armed or airborne."
              : push ? (push.state === "confirmed" ? `Confirmed: aircraft is flying ${push.cal_id_hex}.`
                : push.state === "sending" ? `Sending ${push.cal_id_hex} (attempt ${push.sends} of 3)...`
                : `Not confirmed after ${push.sends} attempts: check range, and that the aircraft is DISARMED.`) : ""}
          </span>
        </div>
      </div>

      <div className="surface" style={{ padding: "var(--space-4)", marginTop: "var(--space-4)" }}>
        <h3 style={{ marginTop: 0 }}>Console</h3>
        <pre className="ft-console" aria-live="polite"
             style={{ maxHeight: 260, overflow: "auto", fontSize: 12, margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
          {lines.length ? lines.map((l) => (l.sent ? `${l.text}\n` : `${l.text}\n`)).join("") : "(no output yet)"}
        </pre>
        <div style={{ display: "flex", gap: "var(--space-2)", marginTop: "var(--space-3)", flexWrap: "wrap" }}>
          {QUICK.map((c) => (
            <button key={c} disabled={!connected} onClick={() => send(c)}>{c}</button>
          ))}
          <button className="btn-danger" disabled={!connected || !!st?.armed} onClick={() => setConfirmArm(true)}>
            Arm FLIGHT...
          </button>
          <form style={{ display: "flex", gap: "var(--space-2)", flex: "1 1 220px" }}
                onSubmit={(e) => { e.preventDefault(); if (line.trim()) { send(line); setLine(""); } }}>
            <input aria-label="Console command" placeholder="command (help)" value={line} disabled={!connected}
                   onChange={(e) => setLine(e.target.value)} style={{ flex: 1, minWidth: 0 }} />
            <button type="submit" disabled={!connected}>Send</button>
          </form>
        </div>
        {confirmArm && (
          <div className="callout callout-danger" role="alertdialog" aria-labelledby="arm-h" style={{ marginTop: "var(--space-3)" }}>
            <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
            <div>
              <div className="callout-title" id="arm-h">Arm FLIGHT?</div>
              The aircraft captures home and the barometric zero now, flies the failsafe glide, takes over at the
              detected launch and hands control to the Pi at the apex. Surfaces will move. It refuses without a 3D GPS
              fix, before the 2-minute barometer warm-up, or with an unmeasured calibration.
              <div style={{ display: "flex", gap: "var(--space-2)", marginTop: "var(--space-3)" }}>
                <button className="btn-danger" onClick={() => { ft.console("arm"); setConfirmArm(false); }}>Arm FLIGHT</button>
                <button onClick={() => setConfirmArm(false)}>Cancel</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

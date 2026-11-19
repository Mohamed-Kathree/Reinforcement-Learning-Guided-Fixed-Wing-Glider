// ConnectionPanel.tsx -- flight-test panel 1.
// Two independent indicators, never merged: the ground station (ESP-NOW via
// the ground ESP32 on USB serial) and the Pi (WiFi, ground only). Either can
// be up while the other is down.
import { useEffect, useState } from "react";
import { ft, useFlightTest, type GroundState, type PiStatus } from "./store";

type Light = "up" | "warn" | "down";

function groundLight(g: GroundState | undefined): { light: Light; text: string } {
  if (!g || !g.connected) return { light: "down", text: "Not connected" };
  if (!g.receiving) return { light: "warn", text: "Connected, no telemetry" };
  if ((g.loss_pct ?? 0) > 20) return { light: "warn", text: "Receiving, heavy loss" };
  return { light: "up", text: "Receiving telemetry" };
}

function piLight(p: PiStatus | null): { light: Light; text: string } {
  if (!p) return { light: "down", text: "Unknown" };
  return p.reachable ? { light: "up", text: "Reachable" } : { light: "down", text: "Unreachable" };
}

function fmtBytes(n?: number) {
  if (n == null) return "–";
  if (n > 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(0)} kB`;
}

function StatusLight({ light, text }: { light: Light; text: string }) {
  return (
    <span className={`ft-light ft-light-${light}`} role="status">
      <span className="ft-light-dot" aria-hidden="true" />
      {text}
    </span>
  );
}

export default function ConnectionPanel() {
  const s = useFlightTest();
  const g = s.snapshot?.ground;
  const gl = groundLight(g);
  const pl = piLight(s.pi);
  const [transport, setTransport] = useState<"stub" | "serial">("stub");
  const [ports, setPorts] = useState<{ device: string; description: string }[]>([]);
  const [portsNote, setPortsNote] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [port, setPort] = useState("");
  const [piUrl, setPiUrl] = useState("");
  const [fetching, setFetching] = useState<string | null>(null);

  useEffect(() => {
    if (transport !== "serial") return;
    ft.ports().then((r) => {
      setPorts(r.ports);
      setPortsNote(r.available ? (r.ports.length ? null : "No serial ports found.") : r.reason);
      if (r.ports[0]) setPort(r.ports[0].device);
    }).catch((e) => setPortsNote(String(e)));
  }, [transport]);

  return (
    <section className="section" aria-labelledby="conn-h">
      <div className="section-head">
        <div className="section-kicker">Links</div>
        <h2 id="conn-h">Connection</h2>
        <p>Two separate links. The ground station carries live telemetry and commands over ESP-NOW; the Pi is reached over WiFi on the ground only, for logs.</p>
      </div>

      <div className="grid-2">
        <div className={`surface surface-pad ft-link ft-link-${gl.light}`}>
          <div className="ft-link-head">
            <h3>Ground station</h3>
            <StatusLight {...gl} />
          </div>
          <dl className="ft-kv">
            <div><dt>Transport</dt><dd>{g?.connected ? (g.kind === "stub" ? "Stub (simulated glider)" : "USB serial") : "–"}</dd></div>
            <div><dt>Port</dt><dd>{g?.port ?? "–"}</dd></div>
            <div><dt>Packet rate</dt><dd>{g?.connected ? `${g.rate_hz.toFixed(1)} Hz` : "–"} <small>expected 5 Hz</small></dd></div>
            <div><dt>Loss</dt><dd>{g?.loss_pct == null ? "–" : `${g.loss_pct.toFixed(1)}%`} <small>from sequence gaps</small></dd></div>
            <div><dt>CRC drops</dt><dd>{g?.connected ? g.crc_drops : "–"}</dd></div>
            <div><dt>RSSI</dt><dd>{g?.rssi_dbm == null ? <span className="muted">not reported by the bridge</span> : `${g.rssi_dbm} dBm`}</dd></div>
            <div><dt>Last packet</dt><dd>{g?.last_packet_age_ms == null ? "–" : `${(g.last_packet_age_ms / 1000).toFixed(1)} s ago`}</dd></div>
          </dl>
          {g?.connected ? (
            <button onClick={() => ft.disconnect()}>Disconnect</button>
          ) : (
            <div className="ft-connect">
              <button className="btn-primary" disabled={searching}
                onClick={async () => { setSearching(true); await ft.connectAuto(); setSearching(false); }}>
                {searching ? "Searching the USB ports…" : "Connect to the aircraft"}
              </button>
              <p className="hint-text" style={{ width: "100%" }}>
                Finds the ground station by itself. The aircraft must be switched on (it looks for live telemetry).
                Or choose manually:
              </p>
              <label className="toolbar-field">
                Transport
                <select value={transport} onChange={(e) => setTransport(e.target.value as "stub" | "serial")}>
                  <option value="stub">Stub (no hardware)</option>
                  <option value="serial">USB serial (ground ESP32)</option>
                </select>
              </label>
              {transport === "serial" && (
                <label className="toolbar-field">
                  Port
                  <select value={port} onChange={(e) => setPort(e.target.value)} disabled={!ports.length}>
                    {ports.map((p) => <option key={p.device} value={p.device}>{p.device} · {p.description}</option>)}
                  </select>
                </label>
              )}
              <button disabled={transport === "serial" && !port}
                onClick={() => ft.connect(transport, transport === "serial" ? port : undefined)}>
                Connect
              </button>
              {transport === "serial" && portsNote && <p className="hint-text">{portsNote}</p>}
            </div>
          )}
        </div>

        <div className={`surface surface-pad ft-link ft-link-${pl.light}`}>
          <div className="ft-link-head">
            <h3>Raspberry Pi</h3>
            <StatusLight {...pl} />
          </div>
          <dl className="ft-kv">
            <div><dt>Address</dt><dd>{s.pi?.address ?? "–"}</dd></div>
            <div><dt>Host / IP</dt><dd>{s.pi?.hostname ? `${s.pi.hostname} · ${s.pi.ip}` : "–"}</dd></div>
            <div><dt>Free disk</dt><dd>{s.pi?.reachable ? `${fmtBytes(s.pi.disk_free_bytes)} of ${fmtBytes(s.pi.disk_total_bytes)}` : "–"}</dd></div>
          </dl>
          {s.pi && !s.pi.reachable && s.pi.error && <p className="error-text">{s.pi.error}</p>}
          <div className="ft-connect">
            <label className="toolbar-field" style={{ flex: 1 }}>
              Pi address
              <input type="text" placeholder="http://raspberrypi.local:8090 (blank = stub)" value={piUrl}
                onChange={(e) => setPiUrl(e.target.value)} />
            </label>
            <button onClick={() => ft.setPi(piUrl.trim() || null)}>{piUrl.trim() ? "Use this Pi" : "Use stub Pi"}</button>
          </div>
          <h4 className="ft-subhead">Logs on the SD card</h4>
          {s.pi?.reachable && s.pi.logs?.length ? (
            <ul className="ft-loglist">
              {s.pi.logs.map((l) => (
                <li key={l.name}>
                  <div>
                    <strong>{l.name}</strong>
                    <span>{fmtBytes(l.size_bytes)} · {l.modified_utc.replace("T", " ").replace("+00:00", " UTC")}</span>
                  </div>
                  <button className="btn-plain" disabled={fetching !== null}
                    onClick={async () => { setFetching(l.name); await ft.fetchLog(l.name); setFetching(null); }}>
                    {fetching === l.name ? "Fetching…" : "Fetch"}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="hint-text">{s.pi?.reachable ? "No logs on the Pi." : "Connect to the Pi to list its logs."}</p>
          )}
        </div>
      </div>

      {s.stub && <StubControls />}
    </section>
  );
}

function StubControls() {
  const { stub } = useFlightTest();
  if (!stub) return null;
  return (
    <div className="surface surface-pad ft-stub" role="group" aria-label="Stub glider controls">
      <div className="ft-link-head">
        <h3>Stub glider controls</h3>
        <span className="badge badge-warning">Simulation only</span>
      </div>
      <p className="hint-text">These change the simulated aircraft so every panel can be exercised without hardware. They do not exist on the real glider.</p>
      <div className="ft-stub-row">
        <label className="control-label"><input type="checkbox" checked={stub.flying}
          onChange={(e) => ft.stub({ flying: e.target.checked })} />Flying (10 m up, 9 m/s)</label>
        <label className="control-label"><input type="checkbox" checked={stub.pi_alive}
          onChange={(e) => ft.stub({ pi_alive: e.target.checked })} />Pi sending setpoints</label>
        <label className="control-label"><input type="checkbox" checked={stub.rail_sensor}
          onChange={(e) => ft.stub({ rail_sensor: e.target.checked })} />Rail divider fitted</label>
        <label className="control-label">Link loss {Math.round(stub.loss * 100)}%
          <input type="range" min={0} max={0.6} step={0.01} value={stub.loss}
            onChange={(e) => ft.stub({ loss: Number(e.target.value) })} /></label>
        <button onClick={() => ft.stub({ flight_arm: true })}>Simulate FLIGHT arm</button>
        <span className="hint-text" style={{ margin: 0 }}>Firmware airborne decision: <strong>{stub.airborne ? "airborne" : "grounded"}</strong></span>
      </div>
    </div>
  );
}

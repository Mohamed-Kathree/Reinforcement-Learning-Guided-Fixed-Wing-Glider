// TelemetryPanel.tsx -- flight-test panel 3: live telemetry.
// The 5 Hz ESP-NOW stream: "is it flying, should I abort". Not for analysis;
// the full-rate log on the Pi's SD card is the record.
// Rail voltage is shown first and largest: below 4.75 V the Pi's SD card can
// corrupt. It always says whether the reading is measured, indicative
// (uncalibrated divider) or calibrated.
import { useMemo } from "react";
import UplotChart, { type Series } from "../components/training/UplotChart";
import { IconAlert } from "../icons";
import { useFlightTest, type HistoryPoint } from "./store";

const DEG = 180 / Math.PI;
const FIX = ["No fix", "2D fix", "3D fix", "?"];

function series(h: HistoryPoint[], label: string, color: string, f: (p: HistoryPoint) => number | null): Series {
  const t1 = h.length ? h[h.length - 1].t_ms : 0;
  return { label, color, points: h.map((p) => ({ x: (p.t_ms - t1) / 1000, y: f(p) })) };
}

export default function TelemetryPanel() {
  const s = useFlightTest();
  const t = s.snapshot?.telemetry ?? null;
  const g = s.snapshot?.ground;
  const h = s.history;
  const stale = !g?.receiving;

  const charts = useMemo(() => ({
    attitude: [series(h, "roll °", "#0a84ff", (p) => p.roll * DEG), series(h, "pitch °", "#e08600", (p) => p.pitch * DEG)],
    rate: [series(h, "yaw rate °/s", "#8e5bd8", (p) => p.r * DEG)],
    alt: [series(h, "baro alt m", "#30a14e", (p) => p.alt_baro), series(h, "sonar m", "#1fa2b8", (p) => p.sonar)],
    rail: [series(h, "rail V", "#e5383b", (p) => p.rail_v)],
    link: [series(h, "loss %", "#8e8e93", (p) => p.loss_pct), series(h, "GPS sats", "#0a84ff", (p) => p.gps_sats)],
  }), [h]);

  const rail = t?.rail;
  const railCls = !rail?.measured ? "unknown" : rail.level ?? "unknown";

  return (
    <section className="section" aria-labelledby="tel-h">
      <div className="section-head">
        <div className="section-kicker">Telemetry</div>
        <h2 id="tel-h">Live telemetry</h2>
        <p>The decimated 5 Hz ESP-NOW stream, for deciding whether to continue or abort. Packets will drop in flight; the SD-card log is the record.</p>
      </div>

      {stale && (
        <div className="callout callout-warning" role="status" style={{ marginBottom: "var(--space-4)" }}>
          <IconAlert className="callout-icon" width={18} height={18} aria-hidden="true" />
          <div>{g?.connected ? `No telemetry for ${((g.last_packet_age_ms ?? 0) / 1000).toFixed(1)} s. Values below are stale.` : "Ground station not connected. Values below are not live."}</div>
        </div>
      )}

      <div className="ft-tiles">
        <div className={`surface ft-tile ft-rail ft-rail-${railCls}`}>
          <div className="ft-tile-label">5 V rail</div>
          <div className="ft-tile-value">{rail?.volts == null ? "Not measured" : `${rail.volts.toFixed(2)} V`}</div>
          <div className="ft-tile-note">
            {!rail?.measured
              ? "No reading: divider on GPIO34 not reporting."
              : rail.level === "brownout" ? `BROWNOUT: below ${rail.brownout_v} V, the Pi's SD card can corrupt.`
              : rail.level === "low" ? `Low: within 0.15 V of the ${rail.brownout_v} V brownout.`
              : `Above the ${rail.brownout_v} V brownout.`}
            {rail?.measured && !rail.calibrated && <span className="badge badge-warning">Indicative · divider not calibrated</span>}
          </div>
        </div>

        <div className="surface ft-tile">
          <div className="ft-tile-label">Flight mode (glider)</div>
          <div className="ft-tile-value">{t?.mode ?? "–"}</div>
          <div className="ft-tile-note">
            {t ? (
              <>
                <span className={`badge ${t.status.armed ? "badge-accent" : ""}`}>{t.status.armed ? "Armed" : "Not armed"}</span>
                <span className={`badge ${t.status.airborne ? "badge-warning" : ""}`}>{t.status.airborne ? "Airborne" : "Grounded"}</span>
                {t.status.pi_failsafe ? <span className="badge badge-danger">Pi link lost · failsafe glide</span> : null}
              </>
            ) : "–"}
          </div>
        </div>

        <div className="surface ft-tile">
          <div className="ft-tile-label">Altitude above launch</div>
          <div className="ft-tile-value">{t && t.alt_baro != null ? `${t.alt_baro.toFixed(1)} m` : t ? "no reference yet" : "–"}</div>
          <div className="ft-tile-note">
            {t && (t.status.baro_ref_captured ? "Reference captured on arm" : t.status.baro_warm ? "Warm; reference not captured yet" : "Barometer warming up (~2 min)")}
            {" · "}sonar {t?.sonar == null ? "invalid (above ~4.5 m)" : `${t.sonar.toFixed(2)} m`}
          </div>
        </div>

        <div className="surface ft-tile">
          <div className="ft-tile-label">GPS</div>
          <div className="ft-tile-value">{t ? FIX[t.status.gps_fix] : "–"}</div>
          <div className="ft-tile-note">{t ? `${t.status.gps_sats} satellites${t.vn != null && t.ve != null ? ` · ${Math.hypot(t.vn, t.ve).toFixed(1)} m/s` : ""}` : "–"}</div>
        </div>

        <div className="surface ft-tile">
          <div className="ft-tile-label">Attitude</div>
          <div className="ft-tile-value">{t ? `${(t.roll * DEG).toFixed(0)}° / ${(t.pitch * DEG).toFixed(0)}°` : "–"}</div>
          <div className="ft-tile-note">roll / pitch · yaw rate {t ? `${(t.r * DEG).toFixed(0)}°/s` : "–"}</div>
        </div>

        <div className="surface ft-tile">
          <div className="ft-tile-label">Link</div>
          <div className="ft-tile-value">{g?.loss_pct == null ? "–" : `${g.loss_pct.toFixed(0)}% loss`}</div>
          <div className="ft-tile-note">{g?.connected ? `${g.rate_hz.toFixed(1)} Hz · ${g.crc_drops} CRC drops` : "not connected"}</div>
        </div>
      </div>

      <div className="ft-charts">
        {([
          ["Attitude (°)", charts.attitude],
          ["Yaw rate (°/s)", charts.rate],
          ["Altitude (m)", charts.alt],
          ["Rail voltage (V)", charts.rail],
          ["Link loss (%) and GPS satellites", charts.link],
        ] as [string, Series[]][]).map(([title, ss]) => (
          <div className="surface ft-chart" key={title}>
            <h4>{title}</h4>
            {h.length ? <UplotChart series={ss} height={140} valueFormatter={(v) => v.toFixed(1)} />
              : <p className="hint-text" style={{ margin: 0 }}>Waiting for telemetry.</p>}
          </div>
        ))}
      </div>
      <p className="hint-text">Charts show about the last minute (300 packets); the x axis is seconds before now.</p>
    </section>
  );
}

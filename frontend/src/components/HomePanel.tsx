// HomePanel.tsx
// =============
// Overview. Says, in this order: what the system does, what the validation
// shows, and what it does not show. The hero loops the best of the most
// recent recorded flights (FlightScene autoplay). Headline figures come from
// src/data/validation.ts, the same source as the Validation page.
//
// "Best" is picked from the most recent RECENT_SCAN_LIMIT recordings, not
// the whole data/episodes/ archive (800+ files): fetching every trajectory
// client-side just to rank them isn't viable, and recent recordings avoid
// mixing pre- and post-fix data.
import { useEffect, useState } from "react";
import FlightScene from "./FlightScene";
import type { Trajectory } from "../types";
import type { Tab } from "../useHashRoute";
import { controllerShortLabel } from "../controllers";
import { CONTROLLER_NAME, GATE, STAGES, formatP, pct } from "../data/validation";
import { IconAlert, IconArrowRight, IconCheck, IconInfo } from "../icons";

const API_BASE = "http://localhost:8000";
const RECENT_SCAN_LIMIT = 20;

function pickBest(trajectories: Trajectory[]): Trajectory | null {
  if (trajectories.length === 0) return null;
  // Prefer flights of the selected controller, then fall back to any.
  const gated = trajectories.filter((t) => t.meta.controller === "gated");
  const pool0 = gated.length > 0 ? gated : trajectories;
  const successes = pool0.filter((t) => t.meta.outcome === "success");
  const softLandings = pool0.filter((t) => t.meta.outcome === "soft_landing");
  const pool = successes.length > 0 ? successes : softLandings.length > 0 ? softLandings : pool0;
  return pool.reduce((best, t) => (t.meta.quality > best.meta.quality ? t : best));
}

const [s0, s1, s2, s3] = STAGES;

export default function HomePanel({ onNavigate }: { onNavigate: (tab: Tab) => void }) {
  // undefined = loading, null = nothing to show (empty archive or fetch error)
  const [best, setBest] = useState<Trajectory | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const idsRes = await fetch(`${API_BASE}/api/episode`);
        if (!idsRes.ok) throw new Error(`${idsRes.status} ${idsRes.statusText}`);
        const ids = (await idsRes.json()) as string[]; // newest first
        const recent = ids.slice(0, RECENT_SCAN_LIMIT);
        const trajectories = await Promise.all(
          recent.map((id) =>
            fetch(`${API_BASE}/api/episode/${id}`).then((r) => {
              if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
              return r.json() as Promise<Trajectory>;
            }),
          ),
        );
        if (!cancelled) setBest(pickBest(trajectories));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setBest(null);
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="home">
      <section className="home-intro">
        <p className="eyebrow">COS 731/732 Honours Project · University of the Western Cape</p>
        <h1>Return-to-launch autonomy for a hand-launched glider</h1>
        <p className="lede">
          A fixed-wing glider with no motor must find its own way back to where it was launched and land
          there, using only on-board sensors. The selected controller, <strong>{CONTROLLER_NAME}</strong>,
          lets a reinforcement-learning policy fly and hands control to a deterministic baseline when the
          on-board wind estimate says conditions are beyond it.
        </p>
      </section>

      <section className="home-hero" aria-label="Recorded flight">
        {best === undefined ? (
          <div className="home-hero-loading" role="status">Loading recent flight data…</div>
        ) : best === null ? (
          <div className="home-hero-empty">
            {error
              ? `Recorded flights are unavailable (${error}). Start the backend to see one here.`
              : "No flights recorded yet. Record one from Flight Replay to see it here."}
          </div>
        ) : (
          <>
            <FlightScene frames={best.frames} currentIndex={0} rHome={best.meta.R_home} autoplay />
            <div className="home-hero-caption">Recorded simulation flight</div>
            <div className="home-hero-overlay">
              <div className="home-hero-stat">
                <span>Controller</span>
                <strong>{controllerShortLabel(best.meta.controller)}</strong>
              </div>
              <div className="home-hero-stat">
                <span>Stage</span>
                <strong>{best.meta.stage}</strong>
              </div>
              <div className={`home-hero-stat outcome-${best.meta.outcome}`}>
                <span>Outcome</span>
                <strong>{best.meta.outcome.replace("_", " ")}</strong>
              </div>
              <div className="home-hero-stat">
                <span>Landing quality</span>
                <strong>{(best.meta.quality * 100).toFixed(0)}%</strong>
              </div>
              <div className="home-hero-stat">
                <span>Miss distance</span>
                <strong>{best.meta.final_dist_home.toFixed(0)} m</strong>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="home-section" aria-labelledby="findings-h">
        <h2 id="findings-h">What the validation shows</h2>
        <p className="section-note">
          One evaluation on 100 paired simulated launches per wind stage, against the deterministic baseline.
        </p>
        <div className="summary-grid">
          <div className="surface summary-item">
            <span className="badge badge-success"><IconCheck width={12} height={12} aria-hidden="true" />Statistically significant</span>
            <div className="summary-figure">
              {pct(s0.gated.success)}<small>vs {pct(s0.baseline.success)} baseline</small>
            </div>
            <h3>Calm air: more than double</h3>
            <p>Return-to-launch success in calm air, with no crashes (p {formatP(s0.paired_success.p)}).</p>
          </div>
          <div className="surface summary-item">
            <span className="badge badge-warning"><IconInfo width={12} height={12} aria-hidden="true" />Not significant</span>
            <div className="summary-figure">
              {pct(s1.gated.success)} · {pct(s2.gated.success)}
              <small>vs {pct(s1.baseline.success)} · {pct(s2.baseline.success)}</small>
            </div>
            <h3>Moderate wind: encouraging</h3>
            <p>
              Highest numerical success of any controller tested at Stages 1 and 2, but too few launches to
              rule out chance (p {formatP(s1.paired_success.p)} and {formatP(s2.paired_success.p)}).
            </p>
          </div>
          <div className="surface summary-item">
            <span className="badge badge-danger"><IconAlert width={12} height={12} aria-hidden="true" />Limitation</span>
            <div className="summary-figure">
              {pct(s3.gated.crash)}<small>crashes vs {pct(s3.baseline.crash)}</small>
            </div>
            <h3>Strong wind: safety not established</h3>
            <p>
              At up to 9 m/s the gated controller crashes more often than the baseline (not significant,
              p {formatP(s3.paired_crash.p)}) and does not land more often.
            </p>
          </div>
        </div>
        <div className="link-row">
          <button className="btn-plain" onClick={() => onNavigate("validation")}>
            Full validation results <IconArrowRight width={14} height={14} aria-hidden="true" />
          </button>
        </div>
      </section>

      <section className="home-section" aria-labelledby="arch-h">
        <h2 id="arch-h">Control architecture</h2>
        <p className="section-note">
          Layers separated by rate and responsibility, so stabilisation never depends on the learned policy.
        </p>
        <div className="surface arch-stack">
          <div className="arch-layer arch-layer-gate">
            <div className="arch-layer-head">
              <span className="arch-rate">20 Hz</span>
              <strong>Wind gate</strong>
              <span className="arch-target">on-board inputs only</span>
            </div>
            <p>
              Estimates the wind from GPS ground speed and heading. Between {GATE.t_earliest_s} and{" "}
              {GATE.t_latest_s} s after launch, the first estimate above {GATE.threshold_ms} m/s hands control
              to the baseline for the rest of the flight.
            </p>
          </div>
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">20 Hz</span>
              <strong>Guidance: Run 4 policy or baseline</strong>
              <span className="arch-target">Raspberry Pi 5 (target)</span>
            </div>
            <p>
              PPO policy (Stable-Baselines3, 128 × 128 MLP) on 8 stacked frames of a 12-value observation, or
              the deterministic baseline. Both output <code>[bank, airspeed]</code> commands.
            </p>
          </div>
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">200 Hz</span>
              <strong>Attitude controller</strong>
              <span className="arch-target">ESP32 (simulated)</span>
            </div>
            <p>Roll PD, speed-scheduled pitch and coordinated rudder. A safety shield limits bank and pulls up near the ground.</p>
          </div>
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">200 Hz</span>
              <strong>JSBSim physics</strong>
              <span className="arch-target">6-DOF flight model</span>
            </div>
            <p>Validated aerodynamics, actuator lag and rate limits, mean wind plus Ornstein–Uhlenbeck gusts.</p>
          </div>
        </div>
      </section>

      <section className="home-section" aria-labelledby="problem-h">
        <h2 id="problem-h">The problem</h2>
        <p>
          With no thrust and no airbrakes, a hand-launched glider gets one attempt. A per-launch
          feasibility analysis shows home is within gliding reach on {STAGES.reduce((a, s) => a + s.feasibility.reachable, 0)} of
          400 benchmark launches, usually with several times the energy needed. The difficulty is shedding
          that surplus precisely enough to touch down gently near the launch point, especially in gusts.
        </p>
      </section>

      <section className="home-section home-grid">
        <div className="surface">
          <h2>Aircraft and sensors</h2>
          <ul className="spec-list">
            <li><span>Flying mass</span><strong>≈ 1.64 kg (est. 1.50–1.80)</strong></li>
            <li><span>Wingspan / area</span><strong>2.0 m / 0.38 m²</strong></li>
            <li><span>Glide ratio at trim</span><strong>≈ 14</strong></li>
            <li><span>Launch</span><strong>by hand, 20–25 m height</strong></li>
            <li><span>GPS</span><strong>5 Hz</strong></li>
            <li><span>Barometer</span><strong>25 Hz</strong></li>
            <li><span>Ground sensing</span><strong>ultrasonic, 20 Hz, &lt; 4.5 m</strong></li>
            <li><span>Airspeed sensor</span><strong>none (no pitot)</strong></li>
          </ul>
        </div>
        <div className="surface">
          <h2>Policy and training</h2>
          <ul className="spec-list">
            <li><span>Algorithm</span><strong>PPO, MLP 128 × 128</strong></li>
            <li><span>Observation</span><strong>12 values × 8 stacked frames</strong></li>
            <li><span>Action</span><strong>bank ± 45°, airspeed 9.5–16 m/s</strong></li>
            <li><span>Discount γ / GAE λ</span><strong>0.999 / 0.95</strong></li>
            <li><span>Rollout / batch</span><strong>8 envs × 2048 / 256</strong></li>
            <li><span>Curriculum advance</span><strong>35 / 30 / 15% rolling success</strong></li>
            <li><span>Run 4 checkpoint</span><strong>best model, 15.5M steps</strong></li>
          </ul>
        </div>
      </section>

      <section className="home-section" aria-labelledby="curr-h">
        <h2 id="curr-h">Wind stages</h2>
        <p className="section-note">
          Four stages of increasing wind, gusts, sensor noise and airframe variation. Validation results are
          reported per stage.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th className="num">Max wind</th>
                <th className="num">Home radius</th>
                <th className="num">Launch height</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>0 · Calm</td><td className="num">0 m/s</td><td className="num">25 m</td><td className="num">25 m</td></tr>
              <tr><td>1 · Light wind</td><td className="num">3 m/s</td><td className="num">22 m</td><td className="num">23 m</td></tr>
              <tr><td>2 · Moderate wind</td><td className="num">6 m/s</td><td className="num">20 m</td><td className="num">21 m</td></tr>
              <tr><td>3 · Strong wind, full variation</td><td className="num">9 m/s</td><td className="num">20 m</td><td className="num">20 m</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="home-section" aria-labelledby="gates-h">
        <h2 id="gates-h">Physics validation</h2>
        <p className="section-note">
          Five correctness gates must pass before any training run. Run them live from the Validation page.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Gate</th><th>Checks</th><th>Threshold</th></tr>
            </thead>
            <tbody>
              <tr><td>1 · Energy conservation</td><td>Zero-drag glide, 10 s</td><td>drift &lt; 0.1%</td></tr>
              <tr><td>2 · Glide ratio</td><td>Trimmed glide, 5 s</td><td>10 &lt; L/D &lt; 20</td></tr>
              <tr><td>3 · Stall behaviour</td><td>CL sweep past α = 12°</td><td>CL drops post-stall</td></tr>
              <tr><td>4 · Trim stability</td><td>Trimmed flight, 15 s</td><td>|q| &lt; 0.1 rad/s</td></tr>
              <tr><td>5 · Attitude integrity</td><td>Free flight, 30 s</td><td>Euler angles finite</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="home-section" aria-labelledby="reward-h">
        <h2 id="reward-h">Reward</h2>
        <pre className="reward-formula">{`quality  =  q_sink · q_roll · q_speed · q_alpha          (each ∈ [0,1], graded at touchdown)

r_step   =  w_path  · v_ground · dt
  −  w_unreach · max(0, d_home/h − L/D_usable)     (h ≥ h_barrier only, capped)
  −  w_stall  · max(0, α − α_safe)
  −  w_bank   · max(0, |φ| − φ_soft)
  −  w_smooth · ‖a − a_prev‖²

r_touchdown = quality · ( w_land · σ/(σ + d_home)  +  w_bullseye · exp(−(d_home/2)²) )
            − (1 − quality) · w_crash`}</pre>
      </section>

      <footer className="home-footer">
        Mohamed Yusuf Kathree · 4253340 · Department of Computer Science, University of the Western Cape
      </footer>
    </div>
  );
}

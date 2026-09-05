// HomePanel.tsx
// =============
// V16 Phase C1 §C1.7 -- fixes D5 (the old version was a static text wall,
// the weakest possible first impression). Hero is a full-bleed looping 3D
// ground track of the best recently recorded episode (FlightScene in
// autoplay mode); everything the old static version had is kept below the
// fold, restyled, not deleted -- it's real reference material, just no
// longer the first thing a visitor sees.
//
// "Best" is picked from the most recent RECENT_SCAN_LIMIT recordings, not
// the whole data/episodes/ archive (837 files at last count) -- fetching
// every trajectory client-side just to rank them isn't viable, and this
// project has its own standing warning about mixing pre- and post-fix
// recordings (see the verification checklist), so scanning only the most
// recent ones sidesteps both problems at once.
import { useEffect, useState } from "react";
import FlightScene from "./FlightScene";
import type { Trajectory } from "../types";

const API_BASE = "http://localhost:8000";
const RECENT_SCAN_LIMIT = 20;

function pickBest(trajectories: Trajectory[]): Trajectory | null {
  if (trajectories.length === 0) return null;
  const successes = trajectories.filter((t) => t.meta.outcome === "success");
  const softLandings = trajectories.filter((t) => t.meta.outcome === "soft_landing");
  const pool = successes.length > 0 ? successes : softLandings.length > 0 ? softLandings : trajectories;
  return pool.reduce((best, t) => (t.meta.quality > best.meta.quality ? t : best));
}

export default function HomePanel() {
  // undefined = still loading, null = nothing to show (empty archive or a
  // fetch error), a Trajectory once one's picked.
  const [best, setBest] = useState<Trajectory | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const idsRes = await fetch(`${API_BASE}/api/episode`);
        if (!idsRes.ok) throw new Error(`${idsRes.status} ${idsRes.statusText}`);
        const ids = (await idsRes.json()) as string[]; // already newest-first
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
      <section className="home-hero">
        {best === undefined ? (
          <div className="home-hero-loading placeholder">Loading recent flight data…</div>
        ) : best === null ? (
          <div className="home-hero-empty placeholder">
            {error
              ? `Could not load recorded episode data (${error}).`
              : "No flights recorded yet — record one from the Baseline or Flight Replay tab to see it here."}
          </div>
        ) : (
          <>
            <FlightScene frames={best.frames} currentIndex={0} rHome={best.meta.R_home} autoplay />
            <div className="home-hero-overlay">
              <div className="home-hero-stat">
                <span>Controller</span>
                <strong>{best.meta.controller}</strong>
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
                <span>Quality</span>
                <strong>{(best.meta.quality * 100).toFixed(0)}%</strong>
              </div>
              <div className="home-hero-stat">
                <span>R_home</span>
                <strong>{best.meta.R_home.toFixed(0)} m</strong>
              </div>
            </div>
          </>
        )}
      </section>

      <section className="home-intro">
        <p className="eyebrow">COS 731/732 Honours Project · University of the Western Cape</p>
        <h1>RL-Guided Return-to-Launch for a Hand-Launched Fixed-Wing Glider</h1>
        <p className="lede">
          A 1.1&nbsp;kg hand-launched glider that, after losing power, must find its own way back
          to the launch point using only onboard sensors. A PPO reinforcement-learning policy is
          trained entirely in a physics-validated JSBSim simulation and is designed to transfer to
          a Raspberry Pi 5 / ESP32 hardware stack without retuning.
        </p>
      </section>

      <section className="home-section">
        <h2>The problem</h2>
        <p>
          Hand-launched gliders face highly variable launch energy and outdoor wind disturbances
          that make hand-tuned guidance brittle. The task is fundamentally energy management: the
          glider can only trade altitude for airspeed while continuously losing energy to drag, so
          the return-to-launch policy must optimise a flight path home while respecting hard safety
          margins on stall angle, bank angle, and height above ground.
        </p>
      </section>

      <section className="home-section">
        <h2>Control architecture</h2>
        <p className="section-note">
          Three layers, separated by timing rate and responsibility, so safety-critical
          stabilisation never depends on the learned policy.
        </p>
        <div className="arch-stack">
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">20 Hz</span>
              <strong>RL Policy</strong>
              <span className="arch-target">Raspberry Pi 5</span>
            </div>
            <p>11-element noisy observation in, <code>[bank_cmd, speed_cmd]</code> out. PPO (Stable-Baselines3).</p>
          </div>
          <div className="arch-arrow">↓</div>
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">200 Hz</span>
              <strong>Attitude Controller</strong>
              <span className="arch-target">ESP32 (simulated)</span>
            </div>
            <p>Roll PD + speed-scheduled pitch + coordinated rudder. Safety shield clips bank and pulls up below minimum AGL.</p>
          </div>
          <div className="arch-arrow">↓</div>
          <div className="arch-layer">
            <div className="arch-layer-head">
              <span className="arch-rate">200 Hz</span>
              <strong>JSBSim Physics</strong>
              <span className="arch-target">6-DOF FDM</span>
            </div>
            <p>Validated aero tables, actuator lag/rate limits, wind = mean NED + Ornstein–Uhlenbeck gusts.</p>
          </div>
        </div>
      </section>

      <section className="home-section">
        <h2>Physics validation</h2>
        <p className="section-note">
          Five correctness gates must pass before any training run is permitted — each targets a
          distinct class of simulation error. Run live from the Validation tab.
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

      <section className="home-section home-grid">
        <div>
          <h2>Aircraft &amp; sensors</h2>
          <ul className="spec-list">
            <li><span>Mass</span><strong>1.1 kg</strong></li>
            <li><span>Wingspan</span><strong>2.0 m</strong></li>
            <li><span>Wing area</span><strong>0.38 m²</strong></li>
            <li><span>Glide ratio (L/D)</span><strong>≈ 12–14</strong></li>
            <li><span>Launch</span><strong>hand/ground, ~15–25 m peak alt.</strong></li>
            <li><span>GPS</span><strong>NEO-6M, ~5 Hz, ~2.5 m CEP</strong></li>
            <li><span>Ground sensing</span><strong>ultrasonic, 20 Hz, valid &lt; 4.5 m</strong></li>
            <li><span>No pitot, no LiDAR</span><strong>dropped from earlier design</strong></li>
          </ul>
        </div>
        <div>
          <h2>Policy &amp; training</h2>
          <ul className="spec-list">
            <li><span>Algorithm</span><strong>PPO, MlpPolicy [128, 128]</strong></li>
            <li><span>Observation</span><strong>11-element, float32</strong></li>
            <li><span>Action</span><strong>[bank ±45°, speed 5–13 m/s]</strong></li>
            <li><span>Parallel envs</span><strong>8 × SubprocVecEnv</strong></li>
            <li><span>Rollout / batch</span><strong>2048 / 256</strong></li>
            <li><span>Discount γ</span><strong>0.999</strong></li>
            {/* Verified against training/configs/base.yaml + env/curriculum.py
                during V16 Phase A (Gate A) -- the previous "80% / 100
                episodes" here was stale. */}
            <li><span>Curriculum advance</span><strong>60% success / 50-episode window</strong></li>
            <li><span>Target run</span><strong>20M timesteps</strong></li>
          </ul>
        </div>
      </section>

      <section className="home-section">
        <h2>Curriculum</h2>
        <p className="section-note">
          Four stages of increasing wind, sensor noise, and aerodynamic domain randomisation. Launch
          altitude and home radius are hardware-derived, not arbitrary — see the Baseline tab for
          measured performance per stage.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Stage</th><th>Max wind</th><th>Home radius</th><th>Launch alt.</th></tr>
            </thead>
            <tbody>
              <tr><td>0 — calm</td><td>0 m/s</td><td>25 m</td><td>25 m</td></tr>
              <tr><td>1 — light wind</td><td>3 m/s</td><td>22 m</td><td>23 m</td></tr>
              <tr><td>2 — moderate</td><td>6 m/s</td><td>20 m</td><td>21 m</td></tr>
              <tr><td>3 — full domain rand.</td><td>9 m/s</td><td>20 m</td><td>20 m</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="home-section">
        <h2>Reward shaping</h2>
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

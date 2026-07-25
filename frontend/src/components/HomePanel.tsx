export default function HomePanel() {
  return (
    <div className="home">
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
            <li><span>Curriculum advance</span><strong>80% success / 100 episodes</strong></li>
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
        <pre className="reward-formula">{`r =  w_progress · Δd_home
  +  w_airtime  · Δt
  −  w_agl      · max(0, h_min − h)²
  −  w_stall    · max(0, α − α_safe)
  −  w_bank     · max(0, |φ| − φ_soft)
  −  w_smooth   · ‖a − a_prev‖²
  +  terminal:  +500 success  /  −100 crash`}</pre>
      </section>

      <footer className="home-footer">
        Mohamed Yusuf Kathree · 4253340 · Department of Computer Science, University of the Western Cape
      </footer>
    </div>
  );
}

// types.ts
// ========
// Mirrors backend/schemas.py field-for-field. The data contract is sacred
// (FRONTEND_BUILD_CONTEXT.md Section 4): change a field here, change it
// there in the same commit.

// ---------------------------------------------------------------------------
// Validation panel (Milestone 1)
// ---------------------------------------------------------------------------

export interface GateResult {
  gate: number;
  name: string;
  passed: boolean;
  detail: string;
}

export interface UnitTestResult {
  name: string;
  passed: boolean;
}

export interface EnvProbeResult {
  passed: boolean;
  obs_shape: number[];
  action_shape: number[];
  detail: string;
}

export interface TestRunResult {
  physics_gates: GateResult[];
  unit_tests: UnitTestResult[];
  env_probe: EnvProbeResult;
  all_passed: boolean;
  raw_stdout: string;
}

// ---------------------------------------------------------------------------
// Flight trajectory -- the core data contract (Milestones 2-4)
// ---------------------------------------------------------------------------

export type Controller = "baseline" | "rl";

// "soft_landing" reflects the Phase 4 precision-landing task (env/reward.py):
// dist_home < R_home no longer ends the episode by itself, and most episodes
// that don't hit the strict "success" bar (quality > 0.5 and centred) still
// land cleanly, just off-centre -- they are NOT timeouts. "timeout" means
// what it says: truncated without ever touching down.
export type Outcome = "success" | "soft_landing" | "crash" | "timeout";

// The env-config values actually applied for this episode. wind_speed/
// gust_intensity/sensor_noise/dropout_prob are each the UPPER BOUND
// GliderEnv.reset() draws the episode's actual value from, not a fixed
// exact number -- same convention curriculum stages already use.
export interface FlightConditions {
  wind_speed: number;
  gust_intensity: number;
  sensor_noise: number;
  dropout_prob: number;
  launch_offset_min_m: number;
  launch_offset_max_m: number;
}

export interface TrajectoryMeta {
  stage: number;
  controller: Controller;
  outcome: Outcome;
  R_home: number;
  alt0: number;
  seed: number;
  n_frames: number;
  duration_s: number;
  quality: number;           // landing-quality grade [0,1]; 0 for a real timeout
  final_dist_home: number;   // dist_home (m) at the final frame
  conditions: FlightConditions;
}

export interface ControlSurfaces {
  ail: number;
  elev: number;
  rud: number;
}

export interface Frame {
  t: number;
  pos_ned: [number, number, number];
  euler: [number, number, number];
  v_body: [number, number, number];
  ctrl: ControlSurfaces;
  dist_home: number;
  agl: number;
  wind_ned: [number, number, number];   // [wn, we, wd] m/s, total (mean+gust)
  // V16 Phase C2 -- straight from env/reward.py's compute_reward() info dict,
  // ground-truth, not recomputed client-side. Defaults (false/0.0) keep
  // pre-this-field recordings loadable, showing "no violation, ever".
  stall_violation: boolean;
  bank_violation: boolean;
  unreach_violation: boolean;
  alpha_deg: number;
  roll_deg: number;
  airspeed: number;
}

export interface Trajectory {
  episode_id: string;
  meta: TrajectoryMeta;
  home_ned: [number, number, number];
  frames: Frame[];
}

// V16 Phase D §D1/§D3 -- lightweight per-episode summary for the analysis
// views, mirrors backend/schemas.py's EpisodeRecordSummary.
export interface EpisodeRecordSummary {
  episode_id: string;
  stage: number;
  controller: string;
  outcome: string;
  quality: number;
  seed: number;
  R_home: number;
  touchdown_ned: [number, number, number];
  home_ned: [number, number, number];
}

// ---------------------------------------------------------------------------
// Baseline panel (Milestone 4)
// ---------------------------------------------------------------------------

export interface EpisodeSummary {
  episode_id: string;
  outcome: Outcome;
  quality: number;
}

export interface StageBaselineResult {
  stage: number;
  n_episodes: number;
  n_success: number;
  n_soft_landing: number;
  n_crash: number;
  n_timeout: number;
  success_rate: number;
  mean_quality: number;
  episodes: EpisodeSummary[];
}

export interface BaselineRunResult {
  stages: StageBaselineResult[];
}

// ---------------------------------------------------------------------------
// Training monitor (Milestone 5)
// ---------------------------------------------------------------------------

export interface TrainingStatus {
  running: boolean;
  pid: number | null;
  total_timesteps: number | null;
}

export interface TrainingMetric {
  timesteps: number;
  stage: number;
  success_rate: number;
  ep_rew_mean: number | null;
  // SB3 diagnostics (V16 Phase A §A1) -- null until the first rollout/
  // episode completes, never fabricated.
  ep_len_mean: number | null;
  explained_variance: number | null;
  approx_kl: number | null;
  clip_fraction: number | null;
  entropy_loss: number | null;
  value_loss: number | null;
  policy_gradient_loss: number | null;
  learning_rate: number | null;
  // Curriculum detail (§A2)
  advance_threshold: number | null;
  episodes_at_stage: number | null;
  // 4-way rolling outcome breakdown (§A4) -- success/soft_landing/crash/
  // timeout, matching Outcome above (not the V16 spec's literal 3-way text
  // -- see backend/schemas.py's TrainingMetric for why).
  outcome_counts: Record<string, number> | null;
  // Per-component reward breakdown for the most recently completed episode
  // (§A3), keyed by env/reward.py's component names plus 'penalty_truncate'.
  reward_components: Record<string, number> | null;
  // V16 Phase C2 §C2.1 -- live annunciator-lamp state, sampled from the
  // single step this record was emitted at (see backend/schemas.py).
  stall_violation: boolean | null;
  bank_violation: boolean | null;
  unreach_violation: boolean | null;
  alpha_deg: number | null;
  roll_deg: number | null;
}

// One entry in GET /api/training/runs -- mirrors a data/runs/<run_id>.meta.json
// file written once by WebStreamCallback at training start (V16 Phase A §A5).
export interface TrainingRunMeta {
  run_id: string;
  start_time: string;
  seed: number;
  total_timesteps: number;
  git_hash: string | null;
  n_envs: number | null;
}

// The most recently completed episode's true-state ground track, for the
// training panel's live top-down view (V16 §A6/§B7). NOT the full
// Trajectory/TrajectoryMeta shape used by recorded-episode replay -- a live
// training rollout has no meaningful seed/controller/conditions the way a
// manually recorded evaluation episode does.
export interface LiveFrame {
  t: number;
  pos_ned: [number, number, number];
}

export interface LiveEpisode {
  frames: LiveFrame[];
  outcome: Outcome | null;
  quality: number;
  dist_home: number;
  R_home: number;
  stage: number;
  updated_at: string;
}

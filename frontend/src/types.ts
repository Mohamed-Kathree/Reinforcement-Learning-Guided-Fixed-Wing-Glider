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
}

export interface Trajectory {
  episode_id: string;
  meta: TrajectoryMeta;
  home_ned: [number, number, number];
  frames: Frame[];
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
}

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
export type Outcome = "success" | "crash" | "timeout";

export interface TrajectoryMeta {
  stage: number;
  controller: Controller;
  outcome: Outcome;
  R_home: number;
  alt0: number;
  seed: number;
  n_frames: number;
  duration_s: number;
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
}

export interface StageBaselineResult {
  stage: number;
  n_episodes: number;
  n_success: number;
  n_crash: number;
  n_timeout: number;
  success_rate: number;
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

// validation.ts
// =============
// Typed access to validation.json -- the dashboard's single source of truth
// for benchmark figures. validation.json is GENERATED, never hand-edited:
//
//     python analysis/export_dashboard_data.py
//
// reads results/gated_v24_run4_benchmark.json (run 4 + window gate, run 4
// alone, baseline), results/gated_v24_benchmark.json (V22 + window gate),
// results/feasibility_v24.json and the frozen gate in base.yaml. Every
// number on the Overview and Validation pages comes from here.
import raw from "./validation.json";

export interface Rates {
  success: number;
  crash: number;
}

export interface StageResult {
  stage: number;
  label: string;
  max_wind_ms: number;
  n: number;
  gated: Rates & {
    success_ci95: [number, number];
    crash_ci95: [number, number];
    median_miss_m: number;
  };
  baseline: Rates & { median_miss_m: number };
  run4_alone: Rates;
  v22_gated: Rates;
  paired_success: { both: number; gated_only: number; baseline_only: number; neither: number; p: number };
  paired_crash: { gated_only: number; baseline_only: number; p: number };
  handover_share_of_episodes: number;
  baseline_share_of_flight_time: number;
  handover_time_median_s: number | null;
  crash_timing: {
    total: number;
    after_handover: number;
    policy_only_after_window_opened: number;
    before_window_opened: number;
  };
  feasibility: { reachable: number; landed_by_any_controller: number };
  flights: BenchmarkFlight[];
}

export type FlightOutcome = "success" | "soft_landing" | "crash" | "timeout";

/** One benchmark launch: what the gated controller and the baseline did on it. */
export interface BenchmarkFlight {
  seed: number;
  outcome: FlightOutcome;
  miss_m: number;
  handover_s: number | null;
  baseline_outcome: FlightOutcome;
}

export interface ValidationData {
  generated_by: string;
  sources: string[];
  controller: {
    name: string;
    policy_checkpoint: string;
    gate: { threshold_ms: number; t_earliest_s: number; t_latest_s: number; mode: string };
  };
  evaluation: {
    seed_set: string;
    n_seeds: number;
    evaluations: number;
    tuning_seed_set: string;
    test: string;
    alpha: number;
  };
  stages: StageResult[];
}

export const VALIDATION = raw as unknown as ValidationData;
export const STAGES = VALIDATION.stages;
export const GATE = VALIDATION.controller.gate;
export const ALPHA = VALIDATION.evaluation.alpha;
export const CONTROLLER_NAME = VALIDATION.controller.name;

/** Controllers compared on the dashboard. `selected` marks the deployed one. */
export const CONTROLLERS = [
  { key: "gated", label: "Run 4 – Gated", selected: true },
  { key: "baseline", label: "Baseline", selected: false },
  { key: "v22_gated", label: "V22 – Gated", selected: false },
  { key: "run4_alone", label: "Run 4 alone", selected: false },
] as const;
export type ControllerKey = (typeof CONTROLLERS)[number]["key"];

export function rateOf(s: StageResult, key: ControllerKey, metric: "success" | "crash"): number {
  return s[key][metric];
}

export type Verdict = "significant" | "higher-not-significant" | "no-improvement";

/** Success-rate verdict for the selected controller against the baseline. */
export function successVerdict(s: StageResult): Verdict {
  if (s.gated.success > s.baseline.success && s.paired_success.p < ALPHA) return "significant";
  if (s.gated.success > s.baseline.success) return "higher-not-significant";
  return "no-improvement";
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  significant: "Statistically significant",
  "higher-not-significant": "Numerically higher, not significant",
  "no-improvement": "No demonstrated improvement",
};

export function pct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** p-value as reported: "< 0.001" below that, otherwise two decimals. */
export function formatP(p: number): string {
  if (p < 0.001) return "< 0.001";
  if (p >= 0.995) return "1.00";
  return p.toFixed(2);
}

/** Percentage-point difference, signed. */
export function pp(a: number, b: number): string {
  const d = Math.round((a - b) * 100);
  return `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d)} pts`;
}

export const SIGNIFICANT_STAGES = STAGES.filter((s) => successVerdict(s) === "significant").map((s) => s.stage);

/** The benchmark record behind a benchmark replay id
 *  ("<controller>-s<stage>-seed<seed>-bench", backend/recorder.py), if it is one. */
export function lookupBenchmark(episodeId: string):
  { controller: "gated" | "baseline"; stage: number; flight: BenchmarkFlight } | null {
  const m = /^(gated|baseline)-s(\d)-seed(\d+)-bench$/.exec(episodeId);
  if (!m) return null;
  const stage = Number(m[2]);
  const flight = STAGES[stage]?.flights.find((f) => f.seed === Number(m[3]));
  return flight ? { controller: m[1] as "gated" | "baseline", stage, flight } : null;
}

// research.ts
// ===========
// Typed access to research.json (generated with validation.json by
// analysis/export_dashboard_data.py): every training run since V22 with its
// benchmark results, the cross-run findings, and the per-stage feasibility
// analysis. Never hand-edited.
import raw from "./research.json";

export interface StageRates {
  success: number;
  crash: number;
}

export interface RunCheckpoint {
  label: string;
  source: string;
  stages: StageRates[];
}

export interface Run {
  id: string;
  name: string;
  change: string;
  verdict: "reference" | "negative" | "selected" | "mixed";
  finding: string;
  checkpoints: RunCheckpoint[];
}

export interface Bin {
  lo: number;
  hi: number;
  n: number;
  landed: number;
}

export interface FeasibilityStage {
  stage: number;
  n: number;
  reachable: number;
  landed_by_any_controller: number;
  gated_success: number;
  baseline_success: number;
  energy_margin_median: number;
  energy_margin_p10: number;
  by_gust: Bin[];
  by_along_wind: Bin[];
}

export interface ResearchData {
  generated_by: string;
  sources: string[];
  baseline: StageRates[];
  runs: Run[];
  findings: { title: string; body: string }[];
  feasibility: FeasibilityStage[];
}

export const RESEARCH = raw as unknown as ResearchData;

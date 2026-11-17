// controllers.ts
// ==============
// Display names for the controllers the backend can fly
// (backend/recorder.py record_episode). One place, so every page names
// them the same way.
import type { Controller } from "./types";

export const DEFAULT_CONTROLLER: Controller = "gated";

export const CONTROLLER_OPTIONS: { value: Controller; label: string; description: string }[] = [
  {
    value: "gated",
    label: "Run 4 – Gated Controller",
    description:
      "Selected controller. Run 4's policy flies from launch; the wind gate can hand over to the baseline between 8 and 15 s.",
  },
  {
    value: "baseline",
    label: "Baseline (deterministic RTL)",
    description: "Hand-designed return-to-launch guidance, and the gated controller's fallback.",
  },
  {
    value: "rl",
    label: "Legacy RL policy (V14)",
    description: "The archived deployment/best_model checkpoint from the V14 retrain, kept for comparison with old recordings.",
  },
];

export function controllerLabel(c: string): string {
  return CONTROLLER_OPTIONS.find((o) => o.value === c)?.label ?? c;
}

export function controllerShortLabel(c: string): string {
  return c === "gated" ? "Run 4 – Gated" : c === "baseline" ? "Baseline" : c === "rl" ? "Legacy RL (V14)" : c;
}

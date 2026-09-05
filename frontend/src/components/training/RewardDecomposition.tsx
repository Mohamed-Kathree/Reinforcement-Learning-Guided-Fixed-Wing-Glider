// RewardDecomposition.tsx
// ========================
// V16 Phase B §B5 -- live horizontal bar chart of the mean per-component
// contribution to episode return, plus each component's share of total
// absolute contribution. The dominance warning (>80% of the budget in one
// term) is the exact failure mode that went undetected three separate times
// in this project's reward-design history (the reachability barrier eating
// ~98% of return pre-Phase-5-fix; penalty_stall/penalty_bank firing at 20x
// their intended rate pre-Phase-F-fix) -- this view exists specifically so
// the next occurrence is visible immediately instead of costing a multi-day
// training run and a post-hoc audit script to find.
import type { TrainingMetric } from "../../types";

const COMPONENT_LABELS: Record<string, string> = {
  r_path: "r_path",
  penalty_unreach: "penalty_unreach",
  penalty_stall: "penalty_stall",
  penalty_bank: "penalty_bank",
  penalty_smooth: "penalty_smooth",
  r_terminal: "r_terminal",
  penalty_truncate: "penalty_truncate",
};

const DOMINANCE_THRESHOLD = 0.8;

interface RewardDecompositionProps {
  latest: TrainingMetric | null;
}

export default function RewardDecomposition({ latest }: RewardDecompositionProps) {
  const components = latest?.reward_components;

  if (!components) {
    return (
      <div className="telemetry-box reward-decomposition">
        <h4>Reward decomposition</h4>
        <p className="placeholder">No episode has completed yet.</p>
      </div>
    );
  }

  const entries = Object.entries(components);
  const totalAbs = entries.reduce((sum, [, v]) => sum + Math.abs(v), 0) || 1;
  const shares = entries
    .map(([name, value]) => ({ name, value, share: Math.abs(value) / totalAbs }))
    .sort((a, b) => b.share - a.share);

  const dominant = shares.find((s) => s.share > DOMINANCE_THRESHOLD);

  return (
    <div className="telemetry-box reward-decomposition">
      <h4>Reward decomposition</h4>
      {dominant && (
        <p className="dominance-warning">
          ⚠ {COMPONENT_LABELS[dominant.name] ?? dominant.name} is {(dominant.share * 100).toFixed(0)}%
          of the reward budget — check for a scaling/unit bug before trusting this run.
        </p>
      )}
      <div className="reward-bars">
        {shares.map(({ name, value, share }) => (
          <div key={name} className={`reward-bar-row${share > DOMINANCE_THRESHOLD ? " reward-bar-dominant" : ""}`}>
            <span className="reward-bar-label">{COMPONENT_LABELS[name] ?? name}</span>
            <div className="reward-bar-track">
              <div
                className="reward-bar-fill"
                style={{ width: `${Math.min(share * 100, 100)}%` }}
              />
            </div>
            <span className="reward-bar-value">
              {value >= 0 ? "+" : ""}
              {value.toFixed(2)} ({(share * 100).toFixed(0)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

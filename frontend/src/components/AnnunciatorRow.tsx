// AnnunciatorRow.tsx
// ===================
// V16 Phase C2 §C2.1 -- three lamps wired to the constraint-violation flags
// env/reward.py's compute_reward() actually computes (read the module's own
// docstring: the old standalone AGL penalty was deleted in the Phase 4
// redesign, so there is no fourth "AGL" constraint today -- the closest
// thing is the reachability/glide-range barrier, labelled REACH here rather
// than inventing a lamp for a constraint that no longer exists).
//
// Dumb component: the caller decides where the booleans come from (a Frame
// during replay, synced to the scrubber for free; a TrainingMetric's latest
// record during live training). Unlit is always --lamp-off; lit reuses the
// existing danger/warning semantic colours plus the same capped glow C1
// established for readouts.
interface AnnunciatorRowProps {
  stall: boolean | null | undefined;
  bank: boolean | null | undefined;
  reach: boolean | null | undefined;
}

const LAMPS: { key: keyof AnnunciatorRowProps; label: string }[] = [
  { key: "stall", label: "STALL" },
  { key: "bank", label: "BANK" },
  { key: "reach", label: "REACH" },
];

export default function AnnunciatorRow({ stall, bank, reach }: AnnunciatorRowProps) {
  const values = { stall, bank, reach };
  return (
    <div className="telemetry-box annunciator-row">
      <h4>Constraints</h4>
      <div className="annunciator-lamps">
        {LAMPS.map(({ key, label }) => (
          <div key={key} className={`annunciator-lamp${values[key] ? " annunciator-lamp-lit" : ""}`}>
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}

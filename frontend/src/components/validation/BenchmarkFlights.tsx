// BenchmarkFlights.tsx
// ====================
// All 100 benchmark launches of one stage as a grid of tiles. Tile colour = what
// the selected controller did; corner mark = what the baseline did on the same
// launch. Clicking a tile records both flights exactly as the benchmark flew
// them (backend recorder, benchmark=true, reused if already recorded) and
// opens them side by side in Flight Replay.
import { useState } from "react";
import { API_BASE } from "../../config";
import { STAGES, type BenchmarkFlight, type FlightOutcome } from "../../data/validation";

type Filter = "all" | FlightOutcome | "differs";

const OUTCOME: Record<FlightOutcome, { label: string; color: string }> = {
  success: { label: "Success", color: "var(--success-fill)" },
  soft_landing: { label: "Soft landing", color: "var(--series-alone)" },
  crash: { label: "Crash", color: "var(--danger-fill)" },
  timeout: { label: "Timeout", color: "var(--warning-fill)" },
};

async function record(controller: "gated" | "baseline", stage: number, seed: number): Promise<string> {
  const q = new URLSearchParams({ controller, stage: String(stage), seed: String(seed), benchmark: "true" });
  const r = await fetch(`${API_BASE}/api/episode/record?${q}`, { method: "POST" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return ((await r.json()) as { episode_id: string }).episode_id;
}

function matches(f: BenchmarkFlight, filter: Filter): boolean {
  if (filter === "all") return true;
  if (filter === "differs") return (f.outcome === "success") !== (f.baseline_outcome === "success");
  return f.outcome === filter;
}

export default function BenchmarkFlights({ onOpen }: { onOpen: (idA: string, idB: string) => void }) {
  const [stage, setStage] = useState(3);
  const [filter, setFilter] = useState<Filter>("all");
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const flights = STAGES[stage].flights;

  const count = (f: Filter) => flights.filter((x) => matches(x, f)).length;
  const filters: { key: Filter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "success", label: "Success" },
    { key: "soft_landing", label: "Soft landing" },
    { key: "crash", label: "Crash" },
    { key: "differs", label: "Success differs from baseline" },
  ];

  async function open(f: BenchmarkFlight) {
    setBusy(f.seed);
    setError(null);
    try {
      const [a, b] = await Promise.all([record("gated", stage, f.seed), record("baseline", stage, f.seed)]);
      onOpen(a, b);
    } catch (err) {
      setError(`Could not open the flight (${err instanceof Error ? err.message : String(err)}). Is the simulator backend running?`);
    } finally {
      setBusy(null);
    }
  }

  function describe(f: BenchmarkFlight): string {
    const hand = f.handover_s == null ? "no handover" : `handover at ${f.handover_s.toFixed(1)} s`;
    return `Seed ${f.seed}: Run 4 gated ${OUTCOME[f.outcome].label.toLowerCase()}, ${f.miss_m.toFixed(1)} m, ${hand}; ` +
      `baseline ${OUTCOME[f.baseline_outcome].label.toLowerCase()}. Open both in Flight Replay.`;
  }

  return (
    <div className="surface surface-pad bench-flights">
      <div className="bench-toolbar">
        <div className="segmented" role="tablist" aria-label="Wind stage">
          {STAGES.map((s) => (
            <button key={s.stage} role="tab" aria-selected={stage === s.stage}
              className={stage === s.stage ? "seg active" : "seg"} onClick={() => setStage(s.stage)}>
              Stage {s.stage}
            </button>
          ))}
        </div>
        <div className="filter-chips" role="group" aria-label="Filter flights">
          {filters.map((f) => (
            <button key={f.key} aria-pressed={filter === f.key}
              className={filter === f.key ? "filter-chip active" : "filter-chip"} onClick={() => setFilter(f.key)}>
              {f.label} <span>{count(f.key)}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="bench-grid" role="list" aria-label={`Stage ${stage} benchmark flights`}>
        {flights.map((f) => {
          const on = matches(f, filter);
          return (
            <button
              key={f.seed}
              role="listitem"
              className={`bench-tile${on ? "" : " dim"}${busy === f.seed ? " busy" : ""}`}
              style={{ background: OUTCOME[f.outcome].color }}
              title={describe(f)}
              aria-label={describe(f)}
              disabled={busy !== null || !on}
              onClick={() => open(f)}
            >
              <span className="bench-tile-corner" style={{ background: OUTCOME[f.baseline_outcome].color }} />
              {f.handover_s != null && <span className="bench-tile-hand" aria-hidden="true" />}
            </button>
          );
        })}
      </div>

      <ul className="chart-legend" style={{ marginTop: "var(--space-3)" }}>
        {(["success", "soft_landing", "crash"] as FlightOutcome[]).map((o) => (
          <li key={o}><span className="legend-swatch" style={{ background: OUTCOME[o].color }} />{OUTCOME[o].label}</li>
        ))}
        <li><span className="legend-corner" aria-hidden="true" />Corner: baseline on the same launch</li>
        <li><span className="legend-hand" aria-hidden="true" />Bar: gate handed over</li>
      </ul>
      {busy !== null && <p className="hint-text" role="status">Flying seed {busy} with both controllers…</p>}
      {error && <p className="error-text">{error}</p>}
      <p className="hint-text" style={{ marginBottom: 0 }}>
        Replays are re-simulated from the benchmark seed and are checked against the stored result when they open.
      </p>
    </div>
  );
}

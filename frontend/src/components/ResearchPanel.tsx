// ResearchPanel.tsx
// =================
// How the selected controller was arrived at: every training run since the
// V22 reference with its benchmark results (negative results included), the
// findings that hold across runs, and why the windy stages are hard
// (per-launch feasibility). Data: src/data/research.ts, generated from the
// committed results by analysis/export_dashboard_data.py.
import { Fragment } from "react";
import GroupedBarChart, { ChartLegend, type Series } from "./validation/GroupedBarChart";
import { Callout, SectionHead } from "./validation/parts";
import { RESEARCH, type Run } from "../data/research";
import { pct } from "../data/validation";

const VERDICT: Record<Run["verdict"], { label: string; cls: string }> = {
  reference: { label: "Reference", cls: "badge" },
  negative: { label: "Negative result", cls: "badge badge-danger" },
  selected: { label: "Selected policy", cls: "badge badge-accent" },
  mixed: { label: "Mixed result", cls: "badge badge-warning" },
};

const FORGET_SERIES: Series[] = [
  { key: "best", label: "Best checkpoint", color: "var(--series-selected)", selected: true },
  { key: "final", label: "Final checkpoint", color: "var(--series-baseline)" },
];

const FEAS_SERIES: Series[] = [
  { key: "reachable", label: "Physically reachable", color: "var(--chart-axis)" },
  { key: "any", label: "Landed by any controller", color: "var(--series-v22)" },
  { key: "gated", label: "Run 4 – Gated", color: "var(--series-selected)", selected: true, badge: "Selected" },
  { key: "baseline", label: "Baseline", color: "var(--series-baseline)" },
];

const GUST_SERIES: Series[] = [
  { key: "landed", label: "Landed by any controller", color: "var(--series-v22)", selected: true },
];

export default function ResearchPanel() {
  const F = RESEARCH.feasibility;
  const s3 = F[3];
  return (
    <section className="panel research-panel">
      <div className="panel-header">
        <div>
          <h1>Research</h1>
          <p className="page-subtitle">
            Every training run since the V22 reference, negative results included, and what the runs show
            together. Simulation benchmarks on the same 100 paired launches per stage.
          </p>
        </div>
      </div>

      <section className="section" aria-labelledby="find-h">
        <SectionHead id="find-h" kicker="Findings" title="What the runs show together" />
        <div className="finding-grid">
          {RESEARCH.findings.map((f) => (
            <div className="surface finding" key={f.title}>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="runs-h">
        <SectionHead id="runs-h" kicker="Runs" title="Training runs and their benchmarks">
          Success rate by stage, crash rate below. The best checkpoint is chosen during training on separate
          selection seeds, never on these benchmark seeds.
        </SectionHead>
        <div className="table-wrap">
          <table className="runs-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Checkpoint</th>
                {[0, 1, 2, 3].map((s) => <th key={s} className="num">Stage {s}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr className="row-muted">
                <td className="cell-strong">Baseline</td>
                <td>deterministic</td>
                {RESEARCH.baseline.map((r, i) => <RateCell key={i} r={r} />)}
              </tr>
              {RESEARCH.runs.map((run) => (
                <Fragment key={run.id}>
                  {run.checkpoints.map((c, ci) => (
                    <tr key={c.label} className={run.verdict === "selected" && ci === 0 ? "row-selected" : undefined}>
                      {ci === 0 && (
                        <td rowSpan={run.checkpoints.length} className="cell-wrap run-cell">
                          <div className="run-name">{run.name}</div>
                          <span className={VERDICT[run.verdict].cls}>{VERDICT[run.verdict].label}</span>
                          <p>{run.change}</p>
                          <p className="run-finding">{run.finding}</p>
                        </td>
                      )}
                      <td>{c.label}</td>
                      {c.stages.map((r, i) => <RateCell key={i} r={r} />)}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="forget-h">
        <SectionHead id="forget-h" kicker="Forgetting" title="Calm-air success, best against final checkpoint">
          Calm-air skill peaks early and is lost once training moves to windier conditions, under every
          curriculum tried.
        </SectionHead>
        <div className="surface chart-card">
          <ChartLegend series={FORGET_SERIES} />
          <GroupedBarChart
            series={FORGET_SERIES}
            groups={RESEARCH.runs.map((r) => ({
              key: r.id,
              label: r.name.replace("V24 ", ""),
              values: {
                best: r.checkpoints[0].stages[0].success,
                final: r.checkpoints[r.checkpoints.length - 1].stages[0].success,
              },
            }))}
            yMax={0.8}
            yTitle="Stage 0 success"
            title="Calm-air (stage 0) success at each run's best and final checkpoint"
            exportName="research-calm-air-forgetting"
          />
        </div>
      </section>

      <section className="section" aria-labelledby="feas-h">
        <SectionHead id="feas-h" kicker="Feasibility" title="Why the windy stages are hard">
          For every benchmark launch: is home within gliding reach (launch state, wind and randomised airframe,
          JSBSim glide polar, best speed-to-fly), and has any controller landed it?
        </SectionHead>
        <div className="chart-grid">
          <div className="surface chart-card">
            <div className="chart-card-head">
              <div>
                <h3>Reachable, landable and landed</h3>
                <p>Share of the 100 launches per stage.</p>
              </div>
            </div>
            <ChartLegend series={FEAS_SERIES} />
            <GroupedBarChart
              series={FEAS_SERIES}
              groups={F.map((f) => ({
                key: String(f.stage),
                label: `Stage ${f.stage}`,
                values: {
                  reachable: f.reachable / f.n,
                  any: f.landed_by_any_controller / f.n,
                  gated: f.gated_success / 100,
                  baseline: f.baseline_success / 100,
                },
              }))}
              yTitle="Share of launches"
              title="Physically reachable launches, launches landed by any controller, and success of the gated controller and baseline, by stage"
              exportName="research-feasibility"
            />
          </div>
          <div className="surface chart-card">
            <div className="chart-card-head">
              <div>
                <h3>Stage 3: gusts decide</h3>
                <p>Share of launches any controller has landed, by gust strength (σ of the gust model).</p>
              </div>
            </div>
            <ChartLegend series={GUST_SERIES} />
            <GroupedBarChart
              series={GUST_SERIES}
              groups={s3.by_gust.map((b) => ({
                key: String(b.lo),
                label: `σ ${b.lo}–${b.hi}`,
                sublabel: `${b.landed} of ${b.n}`,
                values: { landed: b.n ? b.landed / b.n : 0 },
              }))}
              yMax={0.8}
              yTitle="Landed"
              title="Stage 3: share of launches landed by any controller, by gust strength"
              exportName="research-stage3-gusts"
            />
          </div>
        </div>
        <div className="table-wrap" style={{ marginTop: "var(--space-4)" }}>
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th className="num">Reachable</th>
                <th className="num">Energy surplus, median</th>
                <th className="num">Energy surplus, worst 10%</th>
                <th className="num">Landed by any controller</th>
              </tr>
            </thead>
            <tbody>
              {F.map((f) => (
                <tr key={f.stage}>
                  <td>Stage {f.stage}</td>
                  <td className="num">{f.reachable} / {f.n}</td>
                  <td className="num">{f.energy_margin_median.toFixed(1)}×</td>
                  <td className="num">{f.energy_margin_p10.toFixed(1)}×</td>
                  <td className="num">{f.landed_by_any_controller} / {f.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="explainer">
          <Callout tone="info" title="How to read this">
            "Reachable" is an optimistic bound (no gusts, perfect speed-to-fly), so an unreachable launch is
            certainly lost while a reachable one is only possible. "Landed by any controller" pools every
            controller benchmarked at the time of the analysis, so it is a lower bound on what is achievable.
            The gap between the two is control, not energy.
          </Callout>
        </div>
      </section>
    </section>
  );
}

function RateCell({ r }: { r: { success: number; crash: number } }) {
  return (
    <td className="num rate-cell">
      <strong>{pct(r.success)}</strong>
      <span className={r.crash >= 0.3 ? "rate-crash high" : "rate-crash"}>{pct(r.crash)} crash</span>
    </td>
  );
}

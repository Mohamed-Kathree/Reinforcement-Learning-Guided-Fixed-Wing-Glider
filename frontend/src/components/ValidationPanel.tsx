// ValidationPanel.tsx
// ===================
// Validation dashboard for the selected controller (Run 4 – Gated). Every
// figure comes from src/data/validation.ts (generated from the committed
// result files by analysis/export_dashboard_data.py). The live simulator
// checks (physics gates, unit tests, env probe) are kept at the bottom.
import { useState } from "react";
import { runTests } from "../api";
import type { TestRunResult } from "../types";
import GroupedBarChart, { ChartLegend, type Group, type Series } from "./validation/GroupedBarChart";
import { Callout, Meter, SectionHead, VerdictBadge } from "./validation/parts";
import BenchmarkFlights from "./validation/BenchmarkFlights";
import { IconAlert, IconCheck, IconXCircle } from "../icons";
import {
  ALPHA, CONTROLLER_NAME, GATE, SIGNIFICANT_STAGES, STAGES, VALIDATION,
  formatP, pct, pp, successVerdict, type StageResult,
} from "../data/validation";

const S = STAGES;
const s0 = S[0];
const s3 = S[3];

const COMPARE_SERIES: Series[] = [
  { key: "gated", label: "Run 4 – Gated", color: "var(--series-selected)", selected: true, badge: "Selected" },
  { key: "baseline", label: "Baseline", color: "var(--series-baseline)" },
  { key: "v22_gated", label: "V22 – Gated", color: "var(--series-v22)" },
];

const ALONE_SERIES: Series[] = [
  { key: "gated", label: "Run 4 – Gated", color: "var(--series-selected)", selected: true, badge: "Selected" },
  { key: "run4_alone", label: "Run 4 alone (no gate)", color: "var(--series-alone)" },
];

function stageGroups(metric: "success" | "crash", keys: string[], withCi: boolean): Group[] {
  return S.map((s) => ({
    key: String(s.stage),
    label: `Stage ${s.stage}`,
    sublabel: s.max_wind_ms === 0 ? "Calm" : `≤ ${s.max_wind_ms} m/s`,
    values: Object.fromEntries(keys.map((k) => [k, s[k as "gated"][metric]])),
    ci: withCi ? { gated: metric === "success" ? s.gated.success_ci95 : s.gated.crash_ci95 } : undefined,
  }));
}

/** One short engineering reading per stage, built from that stage's numbers. */
function interpret(s: StageResult): string {
  const g = s.gated, b = s.baseline;
  const crash = g.crash === 0 && b.crash === 0
    ? "Neither controller crashed."
    : `Crash rate ${pct(g.crash)} against ${pct(b.crash)} (paired p = ${formatP(s.paired_crash.p)}).`;
  switch (successVerdict(s)) {
    case "significant":
      return `Success rises from ${pct(b.success)} to ${pct(g.success)}, a statistically significant gain. ` +
        `The gate hands over in only ${pct(s.handover_share_of_episodes)} of these flights, so this is ` +
        `the learned policy's own result. ${crash}`;
    case "higher-not-significant":
      return `Success is ${pp(g.success, b.success)} above the baseline, but 100 paired flights cannot ` +
        `distinguish this from chance (p = ${formatP(s.paired_success.p)}). ${crash}`;
    default:
      return `No success advantage (${pct(g.success)} against ${pct(b.success)}). ${crash} ` +
        `The difference is not significant, but the direction is unfavourable.`;
  }
}

export default function ValidationPanel({ onOpenFlights }: { onOpenFlights: (idA: string, idB: string) => void }) {
  return (
    <div className="validation-dashboard">
      <header className="page-header">
        <div>
          <h1>{CONTROLLER_NAME.replace("Controller", "Controller Validation")}</h1>
          <p className="page-subtitle">
            Benchmark evaluation of a learned return-to-launch controller with deterministic baseline
            fallback. Simulation only: these results are not evidence of real-world flight safety.
          </p>
        </div>
      </header>

      <Overview />
      <Performance />
      <Significance />
      <WindStages />
      <section className="section" aria-labelledby="flights-h">
        <SectionHead id="flights-h" kicker="Evidence" title="Every benchmark flight">
          Each tile is one launch. Open any of them to watch the gated controller and the baseline fly
          that exact launch side by side.
        </SectionHead>
        <BenchmarkFlights onOpen={onOpenFlights} />
      </section>
      <GateMechanism />
      <Conclusions />
      <SimulatorChecks />
    </div>
  );
}

/* A. ------------------------------------------------------------------ */

function Overview() {
  const sig = SIGNIFICANT_STAGES.length === 1 ? `Stage ${SIGNIFICANT_STAGES[0]} only`
    : SIGNIFICANT_STAGES.map((s) => `Stage ${s}`).join(", ") || "None";
  return (
    <section className="section" aria-labelledby="overview-h">
      <h2 id="overview-h" className="sr-only">Validation overview</h2>
      <div className="surface kpi-strip">
        <div className="kpi">
          <div className="kpi-label">Benchmark seeds</div>
          <div className="kpi-value">{VALIDATION.evaluation.n_seeds}</div>
          <div className="kpi-note">per wind stage, paired</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Selected controller</div>
          <div className="kpi-value kpi-value-sm">Run 4 Gated</div>
          <div className="kpi-note">baseline fallback</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Wind threshold</div>
          <div className="kpi-value">{GATE.threshold_ms} m/s</div>
          <div className="kpi-note">on-board estimate</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Gating window</div>
          <div className="kpi-value">{GATE.t_earliest_s}–{GATE.t_latest_s} s</div>
          <div className="kpi-note">after launch</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Significant result</div>
          <div className="kpi-value kpi-value-sm">{sig}</div>
          <div className="kpi-note">success, p &lt; {ALPHA}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Evaluation status</div>
          <div className="kpi-value kpi-value-sm">
            <span className="badge badge-success"><IconCheck width={12} height={12} aria-hidden="true" />Completed</span>
          </div>
          <div className="kpi-note">{VALIDATION.evaluation.evaluations} evaluation, frozen settings</div>
        </div>
      </div>
      <p className="provenance">
        Gate settings were chosen on {VALIDATION.evaluation.tuning_seed_set} and frozen before the
        single evaluation on the 100 benchmark seeds. Test: {VALIDATION.evaluation.test}. Figures are
        generated from <code>{VALIDATION.sources[0]}</code> by <code>{VALIDATION.generated_by}</code>.
      </p>
    </section>
  );
}

/* B. ------------------------------------------------------------------ */

function Performance() {
  return (
    <section className="section" aria-labelledby="perf-h">
      <SectionHead id="perf-h" kicker="Performance" title="Success and crash rates by wind stage">
        All rates are shares of the same 100 launches per stage. Error bars on the selected
        controller are 95% Wilson intervals.
      </SectionHead>
      <div className="chart-grid">
        <div className="surface chart-card">
          <div className="chart-card-head">
            <div>
              <h3>Return-to-launch success</h3>
              <p>Landed within the home radius with landing quality above 0.5. Higher is better.</p>
            </div>
          </div>
          <ChartLegend series={COMPARE_SERIES} />
          <GroupedBarChart
            series={COMPARE_SERIES}
            groups={stageGroups("success", ["gated", "baseline", "v22_gated"], true)}
            yMax={0.8}
            yTitle="Success rate"
            title="Return-to-launch success rate by wind stage: Run 4 gated, baseline and V22 gated"
            exportName="validation-success-by-stage"
          />
        </div>
        <div className="surface chart-card">
          <div className="chart-card-head">
            <div>
              <h3>Crash rate</h3>
              <p>Touchdown too hard to survive (landing quality 0). Lower is better.</p>
            </div>
          </div>
          <ChartLegend series={COMPARE_SERIES} />
          <GroupedBarChart
            series={COMPARE_SERIES}
            groups={stageGroups("crash", ["gated", "baseline", "v22_gated"], true)}
            yMax={0.8}
            yTitle="Crash rate"
            title="Crash rate by wind stage: Run 4 gated, baseline and V22 gated"
            exportName="validation-crash-by-stage"
          />
        </div>
        <div className="surface chart-card chart-card-wide">
          <div className="chart-card-head">
            <div>
              <h3>What the gate changes: Run 4 alone against Run 4 gated</h3>
              <p>
                The same policy with and without the fallback. The gate gives up a little calm-air
                success and removes most of the policy's excess crashes in wind.
              </p>
            </div>
          </div>
          <ChartLegend series={ALONE_SERIES} />
          <div className="grid-2">
            <GroupedBarChart
              series={ALONE_SERIES}
              groups={stageGroups("success", ["gated", "run4_alone"], false)}
              yMax={0.8}
              yTitle="Success rate"
              title="Success rate: Run 4 gated against Run 4 alone"
              exportName="validation-run4-alone-vs-gated-success"
            />
            <GroupedBarChart
              series={ALONE_SERIES}
              groups={stageGroups("crash", ["gated", "run4_alone"], false)}
              yMax={0.8}
              yTitle="Crash rate"
              title="Crash rate: Run 4 gated against Run 4 alone"
              exportName="validation-run4-alone-vs-gated-crash"
            />
          </div>
        </div>
      </div>
    </section>
  );
}

/* C. ------------------------------------------------------------------ */

function Significance() {
  const s1 = S[1];
  return (
    <section className="section" aria-labelledby="sig-h">
      <SectionHead id="sig-h" kicker="Statistics" title="Paired comparison against the baseline">
        Both controllers flew the same 100 launches (same airframe, wind and launch state) at each
        stage. The exact McNemar test uses only the launches where exactly one controller succeeded.
      </SectionHead>
      <div className="table-wrap">
        <table>
          <caption className="sr-only">
            Success-rate significance by stage
          </caption>
          <thead>
            <tr>
              <th>Stage</th>
              <th className="num">Run 4 gated</th>
              <th className="num">Baseline</th>
              <th className="num">Difference</th>
              <th>Discordant launches</th>
              <th className="num">p-value</th>
              <th className="num">p &lt; {ALPHA}?</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            {S.map((s) => {
              const ps = s.paired_success;
              const n = ps.gated_only + ps.baseline_only || 1;
              const sig = s.paired_success.p < ALPHA;
              return (
                <tr key={s.stage}>
                  <td className="cell-strong">Stage {s.stage}</td>
                  <td className="num cell-strong">{pct(s.gated.success)}</td>
                  <td className="num">{pct(s.baseline.success)}</td>
                  <td className="num">{pp(s.gated.success, s.baseline.success)}</td>
                  <td>
                    <div className="paired-bar" role="img"
                      aria-label={`${ps.gated_only} launches only the gated controller landed, ${ps.baseline_only} only the baseline landed`}>
                      <span style={{ width: `${(ps.gated_only / n) * 100}%`, background: "var(--series-selected)" }} />
                      <span style={{ width: `${(ps.baseline_only / n) * 100}%`, background: "var(--series-baseline)" }} />
                    </div>
                    <div className="kpi-note">{ps.gated_only} gated only · {ps.baseline_only} baseline only</div>
                  </td>
                  <td className="num cell-strong">{formatP(ps.p)}</td>
                  <td className="num">{sig ? "Yes" : "No"}</td>
                  <td><VerdictBadge stage={s} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="explainer grid-2">
        <Callout tone="info" title="Why a higher rate is not enough">
          At Stage {s1.stage} the gated controller landed {s1.paired_success.gated_only} launches the
          baseline missed, and the baseline landed {s1.paired_success.baseline_only} the gated controller
          missed. If the two controllers were equally good, a split at least this uneven would still
          happen by chance with probability {formatP(s1.paired_success.p)}. Results are called
          significant only below {ALPHA}. A numerical lead of {pp(s1.gated.success, s1.baseline.success)}{" "}
          on 100 launches is encouraging, not established.
        </Callout>
        <Callout tone="warning" title="Crash rates, tested the same way">
          {S.filter((s) => s.paired_crash.gated_only + s.paired_crash.baseline_only > 0).map((s) => (
            <span key={s.stage}>
              Stage {s.stage}: {pct(s.gated.crash)} against {pct(s.baseline.crash)}, p ={" "}
              {formatP(s.paired_crash.p)}.{" "}
            </span>
          ))}
          None of the crash differences is significant, so the data neither show the gated controller
          is less safe nor that it is as safe as the baseline in wind.
        </Callout>
      </div>
    </section>
  );
}

/* D. ------------------------------------------------------------------ */

function WindStages() {
  return (
    <section className="section" aria-labelledby="stages-h">
      <SectionHead id="stages-h" kicker="Wind conditions" title="How performance changes as the wind strengthens">
        Each stage draws a mean wind between 0 and its maximum from a random direction, with gusts,
        sensor noise and airframe variation that also grow by stage.
      </SectionHead>
      <div className="surface stage-list">
        {S.map((s) => (
          <article key={s.stage} className="stage-row">
            <div className="stage-name">
              <h3>Stage {s.stage}</h3>
              <span className="stage-wind">{s.label}</span>
              <VerdictBadge stage={s} />
            </div>
            <div className="stage-metrics">
              <div>
                <div className="metric-pair-label">Success</div>
                <Meter label="Run 4 gated" value={s.gated.success} color="var(--series-selected)" strong />
                <Meter label="Baseline" value={s.baseline.success} color="var(--series-baseline)" />
              </div>
              <div>
                <div className="metric-pair-label">Crash</div>
                <Meter label="Run 4 gated" value={s.gated.crash} color="var(--danger-fill)" strong />
                <Meter label="Baseline" value={s.baseline.crash} color="var(--series-baseline)" />
              </div>
            </div>
            <p className="stage-interpretation">{interpret(s)}</p>
          </article>
        ))}
      </div>
      <p className="provenance" style={{ marginTop: "var(--space-3)" }}>
        Task difficulty for reference: home is physically reachable on{" "}
        {S.map((s) => `${s.feasibility.reachable}`).join(" / ")} of 100 launches per stage, and{" "}
        {S.map((s) => `${s.feasibility.landed_by_any_controller}`).join(" / ")} have been landed
        successfully by at least one controller tested in this project (results/feasibility_v24.json).
      </p>
    </section>
  );
}

/* E. ------------------------------------------------------------------ */

function GateMechanism() {
  const ct = s3.crash_timing;
  return (
    <section className="section" aria-labelledby="gate-h">
      <SectionHead id="gate-h" kicker="Gating" title="How the gate decides who flies">
        The learned policy flies from launch. The gate can only hand control to the baseline, never
        back, and only inside the window.
      </SectionHead>

      <div className="flow">
        <div className="flow-step">
          <div className="flow-step-index">1</div>
          <h4>Wind estimation</h4>
          <p>Estimated on board from GPS ground speed, course and heading. The gate never sees the true wind.</p>
        </div>
        <div className="flow-step flow-step-gate">
          <div className="flow-step-index">2</div>
          <h4>Gating decision</h4>
          <p>
            Between {GATE.t_earliest_s} s and {GATE.t_latest_s} s after launch: the first time the
            estimate exceeds {GATE.threshold_ms} m/s, hand over. One-way.
          </p>
        </div>
        <div className="flow-step">
          <div className="flow-step-index">3</div>
          <h4>Policy or baseline</h4>
          <div className="flow-branch">
            <span className="badge badge-accent">Below threshold: Run 4 policy keeps flying</span>
            <span className="badge">Above threshold: deterministic baseline</span>
          </div>
        </div>
        <div className="flow-step">
          <div className="flow-step-index">4</div>
          <h4>Flight control</h4>
          <p>Bank and airspeed commands go to the 200 Hz attitude controller, whichever controller issued them.</p>
        </div>
      </div>

      <div className="grid-2">
        <div className="surface surface-pad">
          <h3 className="sub-head" style={{ marginTop: 0 }}>Handover statistics</h3>
          <div className="table-wrap handover-table" style={{ boxShadow: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th className="num">Handed over</th>
                  <th className="num">Baseline time</th>
                  <th className="num">Median at</th>
                </tr>
              </thead>
              <tbody>
                {S.map((s) => (
                  <tr key={s.stage}>
                    <td>Stage {s.stage}</td>
                    <td className="num"><strong>{pct(s.handover_share_of_episodes)}</strong></td>
                    <td className="num">{pct(s.baseline_share_of_flight_time)}</td>
                    <td className="num">{s.handover_time_median_s === null ? "–" : `${s.handover_time_median_s.toFixed(1)} s`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint-text" style={{ marginTop: "var(--space-3)", marginBottom: 0 }}>
            Handed over: share of flights where the gate switched to the baseline. Baseline time: share
            of all flight time flown by the baseline. In calm air the policy flies {pct(1 - s0.baseline_share_of_flight_time)} of the flight time. In
            strong wind most handovers happen at {GATE.t_earliest_s} s, the earliest moment the window allows.
          </p>
        </div>

        <div className="surface surface-pad">
          <h3 className="sub-head" style={{ marginTop: 0 }}>The Stage 3 timing problem</h3>
          <GateTimeline />
          <p className="hint-text" style={{ marginBottom: 0 }}>
            Schematic, not to scale. Of the {ct.total} Stage 3 crashes, {ct.after_handover} happened after
            handover (the baseline flying on from the state the policy left it in),{" "}
            {ct.policy_only_after_window_opened} in flights the gate never handed over, and{" "}
            {ct.before_window_opened} before the window opened.
          </p>
        </div>
      </div>

      <Callout tone="warning" title="Why the gate does not protect the strongest-wind flights">
        The policy flies the first {GATE.t_earliest_s} s of every flight while the wind estimate settles.
        At Stage 3 it hands over in {pct(s3.handover_share_of_episodes)} of flights, yet {ct.after_handover} of
        the {ct.total} crashes follow a handover. A likely contributor is that the policy's first seconds
        leave the aircraft in a worse approach configuration than the baseline would have reached on its
        own. This is consistent with the data but has not been tested directly.
      </Callout>
    </section>
  );
}

function GateTimeline() {
  const W = 560, X0 = 112, X1 = 540, T = 24;
  const x = (t: number) => X0 + ((X1 - X0) * t) / T;
  const rows = [
    { label: "Calm air", sub: "no handover", policyEnd: 22, baseEnd: null as number | null, crash: false },
    { label: "Strong wind", sub: "handover in window", policyEnd: GATE.t_earliest_s, baseEnd: 20, crash: false },
    { label: "Strong wind", sub: "fails before 8 s", policyEnd: 6.5, baseEnd: null, crash: true },
  ];
  const rowY = (i: number) => 40 + i * 34;
  return (
    <div className="timeline">
      <svg viewBox={`0 0 ${W} 158`} role="img"
        aria-label={`Timeline: the policy flies from launch; the gate may hand over between ${GATE.t_earliest_s} and ${GATE.t_latest_s} seconds`}>
        <rect className="tl-window" x={x(GATE.t_earliest_s)} y={22} width={x(GATE.t_latest_s) - x(GATE.t_earliest_s)} height={112} rx={4} />
        <line className="tl-window-edge" x1={x(GATE.t_earliest_s)} x2={x(GATE.t_earliest_s)} y1={22} y2={134} />
        <line className="tl-window-edge" x1={x(GATE.t_latest_s)} x2={x(GATE.t_latest_s)} y1={22} y2={134} strokeDasharray="3 3" />
        <text className="tl-label-strong" x={(x(GATE.t_earliest_s) + x(GATE.t_latest_s)) / 2} y={15} textAnchor="middle">
          Gate window
        </text>
        {rows.map((r, i) => (
          <g key={i}>
            <text className="tl-label-strong" x={0} y={rowY(i) + 4}>{r.label}</text>
            <text className="tl-label" x={0} y={rowY(i) + 17}>{r.sub}</text>
            <rect className="tl-policy" x={x(0)} y={rowY(i) - 6} width={x(r.policyEnd) - x(0)} height={12} rx={3} />
            {r.baseEnd !== null && (
              <rect className="tl-baseline" x={x(r.policyEnd) + 1} y={rowY(i) - 6} width={x(r.baseEnd) - x(r.policyEnd) - 1} height={12} rx={3} />
            )}
            {r.crash && (
              <g transform={`translate(${x(r.policyEnd) + 9} ${rowY(i)})`}>
                <circle r={7} className="tl-crash" />
                <path d="M-3 -3 L3 3 M3 -3 L-3 3" stroke="#fff" strokeWidth={1.6} />
              </g>
            )}
          </g>
        ))}
        <line className="tl-axis" x1={X0} x2={X1} y1={140} y2={140} />
        {[0, 4, 8, 12, 16, 20, 24].map((t) => (
          <text key={t} className="tl-tick" x={x(t)} y={154} textAnchor="middle">{t} s</text>
        ))}
      </svg>
      <ul className="chart-legend" style={{ marginBottom: 0 }}>
        <li><span className="legend-swatch" style={{ background: "var(--series-selected)" }} />Run 4 policy flying</li>
        <li><span className="legend-swatch" style={{ background: "var(--series-baseline)" }} />Baseline flying</li>
        <li><span className="legend-swatch" style={{ background: "var(--danger-fill)" }} />Crash</li>
      </ul>
    </div>
  );
}

/* F. ------------------------------------------------------------------ */

function Conclusions() {
  return (
    <section className="section" aria-labelledby="concl-h">
      <SectionHead id="concl-h" kicker="Engineering assessment" title="Conclusions and operating limits" />
      <div className="surface surface-pad prose">
        <p>
          <strong style={{ color: "var(--text)" }}>{CONTROLLER_NAME} is the strongest candidate tested so far.</strong>{" "}
          It has a statistically significant advantage in calm air ({pct(s0.gated.success)} against{" "}
          {pct(s0.baseline.success)}) and the highest numerical success rates in moderate wind
          ({pct(S[1].gated.success)} and {pct(S[2].gated.success)} at Stages 1 and 2). The evaluation does
          not establish a significant improvement in moderate wind, and the controller has a higher observed
          crash rate than the baseline in the strongest wind stage ({pct(s3.gated.crash)} against{" "}
          {pct(s3.baseline.crash)}).
        </p>
      </div>

      <div className="recommendation-grid">
        <div className="surface recommendation recommendation-go">
          <h3><IconCheck width={18} height={18} aria-hidden="true" />Calm to light wind</h3>
          <p>
            Below about 3 m/s with gentle gusts (Stages 0–1): preferred operating conditions for the gated
            controller. Crash rate {pct(s0.gated.crash)}–{pct(S[1].gated.crash)} in simulation.
          </p>
        </div>
        <div className="surface recommendation recommendation-nogo">
          <h3><IconAlert width={18} height={18} aria-hidden="true" />Stronger wind</h3>
          <p>
            Use the deterministic baseline as the preferred operational option. The learned controller has
            not shown an advantage there, and its observed crash rate is higher.
          </p>
        </div>
      </div>

      <Callout tone="danger" title="Safety limitation">
        Stage 3 performance is not sufficient to claim strong-wind safety: {pct(s3.gated.success)} success
        and {pct(s3.gated.crash)} crashes against the baseline's {pct(s3.baseline.success)} and{" "}
        {pct(s3.baseline.crash)} (crash difference not significant, p = {formatP(s3.paired_crash.p)}).
      </Callout>

      <h3 className="sub-head">Evidence and its limits</h3>
      <ul className="surface evidence-list">
        <li>
          <span>Simulated benchmark</span>
          <div>One evaluation on 100 paired launches per stage in JSBSim, with airframe, sensor and wind variation.</div>
        </li>
        <li>
          <span>Statistical evidence</span>
          <div>Significant success gain at Stage {SIGNIFICANT_STAGES.join(", ")} only. Stages 1–2 numerically higher; Stage 3 no improvement.</div>
        </li>
        <li>
          <span>Reproducibility</span>
          <div>Run 4 flown alone reproduced its earlier benchmark exactly at all four stages. Run 4 itself replicates the V22 configuration on a second training seed.</div>
        </li>
        <li>
          <span>Real-world flight</span>
          <div>
            <span className="badge badge-warning"><IconXCircle width={12} height={12} aria-hidden="true" />Not yet validated</span>{" "}
            No flight tests. The simulator's speed hold uses true airspeed; the glider has no pitot and would
            hold speed from GPS, which differs in wind.
          </div>
        </li>
        <li>
          <span>Sample size</span>
          <div>100 launches per stage resolves large effects only. Smaller differences, including the crash-rate gaps, need more seeds to confirm or rule out.</div>
        </li>
      </ul>
    </section>
  );
}

/* Live simulator checks (kept from the previous Validation page) -------- */

type Status = "idle" | "running" | "done" | "error";

function SimulatorChecks() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<TestRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  async function handleRun() {
    setStatus("running");
    setError(null);
    try {
      setResult(await runTests());
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }

  return (
    <section className="section" aria-labelledby="sim-h">
      <div className="section-row">
        <SectionHead id="sim-h" kicker="Simulator integrity" title="Live physics and software checks">
          Runs the five physics gates, the unit tests and an environment probe on the backend. These
          confirm the simulator is sound; they do not evaluate the controller.
        </SectionHead>
        <button className="btn-primary" onClick={handleRun} disabled={status === "running"}>
          {status === "running" ? "Running checks…" : "Run checks"}
        </button>
      </div>

      {status === "error" && (
        <Callout tone="danger" title="Checks could not run">
          {error}. Start the backend (<code>uvicorn backend.main:app</code>) and try again.
        </Callout>
      )}
      {status === "idle" && !result && (
        <div className="empty-state">
          <strong>No checks run in this session</strong>
          Run them to confirm the simulator before recording or comparing flights.
        </div>
      )}
      {status === "running" && <p className="placeholder" role="status">Running physics gates and unit tests…</p>}

      {result && (
        <>
          <div className="overall-banner" data-passed={result.all_passed} role="status">
            {result.all_passed ? <IconCheck width={16} height={16} aria-hidden="true" /> : <IconXCircle width={16} height={16} aria-hidden="true" />}
            {result.all_passed ? "All checks passed" : "Some checks failed"}
          </div>

          <h3 className="sub-head">Physics gates</h3>
          <div className="card-grid">
            {result.physics_gates.map((g) => (
              <CheckCard key={g.gate} passed={g.passed} title={`Gate ${g.gate} — ${g.name}`} detail={g.detail} />
            ))}
          </div>

          <h3 className="sub-head">Unit tests</h3>
          <div className="card-grid">
            {result.unit_tests.map((t) => (
              <CheckCard key={t.name} passed={t.passed} title={t.name} />
            ))}
          </div>

          <h3 className="sub-head">Environment probe</h3>
          <div className="card-grid">
            <CheckCard
              passed={result.env_probe.passed}
              title="GliderEnv reset() + step()"
              detail={`obs shape [${result.env_probe.obs_shape.join(", ")}], action shape [${result.env_probe.action_shape.join(", ")}] — ${result.env_probe.detail}`}
            />
          </div>

          <button className="btn-plain" onClick={() => setShowRaw((v) => !v)} aria-expanded={showRaw}>
            {showRaw ? "Hide raw output" : "Show raw output"}
          </button>
          {showRaw && <pre className="raw-output">{result.raw_stdout}</pre>}
        </>
      )}
    </section>
  );
}

function CheckCard({ passed, title, detail }: { passed: boolean; title: string; detail?: string }) {
  return (
    <div className="card" data-passed={passed}>
      <span className="card-status">
        {passed ? <IconCheck width={16} height={16} aria-label="Passed" /> : <IconXCircle width={16} height={16} aria-label="Failed" />}
      </span>
      <div className="card-title">{title}</div>
      {detail && <div className="card-detail">{detail}</div>}
    </div>
  );
}

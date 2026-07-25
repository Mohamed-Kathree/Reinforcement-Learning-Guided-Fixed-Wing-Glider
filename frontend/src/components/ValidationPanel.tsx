import { useState } from "react";
import { runTests } from "../api";
import type { TestRunResult } from "../types";

type Status = "idle" | "running" | "done" | "error";

export default function ValidationPanel() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<TestRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  async function handleRun() {
    setStatus("running");
    setError(null);
    try {
      const r = await runTests();
      setResult(r);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }

  return (
    <section className="panel validation-panel">
      <div className="panel-header">
        <h2>Validation</h2>
        <button onClick={handleRun} disabled={status === "running"}>
          {status === "running" ? "Running…" : "Run tests"}
        </button>
      </div>

      {status === "error" && <p className="error-text">{error}</p>}

      {result && (
        <>
          <div className="overall-banner" data-passed={result.all_passed}>
            {result.all_passed ? "All checks passed" : "Some checks failed"}
          </div>

          <h3>Physics gates</h3>
          <div className="card-grid">
            {result.physics_gates.map((g) => (
              <div key={g.gate} className="card" data-passed={g.passed}>
                <div className="card-title">
                  Gate {g.gate} — {g.name}
                </div>
                <div className="card-detail">{g.detail}</div>
              </div>
            ))}
          </div>

          <h3>Unit tests</h3>
          <div className="card-grid">
            {result.unit_tests.map((t) => (
              <div key={t.name} className="card" data-passed={t.passed}>
                <div className="card-title">{t.name}</div>
              </div>
            ))}
          </div>

          <h3>Environment probe</h3>
          <div className="card-grid">
            <div className="card" data-passed={result.env_probe.passed}>
              <div className="card-title">GliderEnv reset() + step()</div>
              <div className="card-detail">
                obs shape [{result.env_probe.obs_shape.join(", ")}], action
                shape [{result.env_probe.action_shape.join(", ")}] —{" "}
                {result.env_probe.detail}
              </div>
            </div>
          </div>

          <button className="raw-toggle" onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? "Hide raw output" : "Show raw output"}
          </button>
          {showRaw && <pre className="raw-output">{result.raw_stdout}</pre>}
        </>
      )}
    </section>
  );
}

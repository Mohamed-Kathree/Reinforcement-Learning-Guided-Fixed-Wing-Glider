// parts.tsx -- small presentational pieces shared by the validation sections.
import type { ReactNode } from "react";
import { IconAlert, IconCheck, IconInfo, IconMinusCircle } from "../../icons";
import { successVerdict, VERDICT_LABEL, type StageResult } from "../../data/validation";

export function VerdictBadge({ stage }: { stage: StageResult }) {
  const v = successVerdict(stage);
  const cls = v === "significant" ? "badge-success" : v === "higher-not-significant" ? "badge-warning" : "badge";
  const Icon = v === "significant" ? IconCheck : v === "higher-not-significant" ? IconInfo : IconMinusCircle;
  return (
    <span className={`badge ${cls}`}>
      <Icon width={12} height={12} aria-hidden="true" />
      {VERDICT_LABEL[v]}
    </span>
  );
}

export function Meter({ label, value, color, strong }: { label: string; value: number; color: string; strong?: boolean }) {
  return (
    <div className="meter-row">
      <span style={strong ? { color: "var(--text)", fontWeight: 600 } : undefined}>{label}</span>
      <div className="meter-track" aria-hidden="true">
        <div className="meter-fill" style={{ width: `${Math.max(value * 100, value > 0 ? 1.5 : 0)}%`, background: color }} />
      </div>
      <strong>{Math.round(value * 100)}%</strong>
    </div>
  );
}

type Tone = "info" | "warning" | "danger" | "success";

export function Callout({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) {
  const Icon = tone === "warning" || tone === "danger" ? IconAlert : tone === "success" ? IconCheck : IconInfo;
  return (
    <div className={`callout callout-${tone}`} role={tone === "danger" ? "alert" : undefined}>
      <Icon className="callout-icon" width={18} height={18} aria-hidden="true" />
      <div>
        {title && <div className="callout-title">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function SectionHead({ id, kicker, title, children }: { id?: string; kicker?: string; title: string; children?: ReactNode }) {
  return (
    <div className="section-head">
      {kicker && <div className="section-kicker">{kicker}</div>}
      <h2 id={id}>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  );
}

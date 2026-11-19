// PanelBoundary.tsx -- one failing panel must not blank the whole page.
// (20 Nov 2026: a null value from the real aircraft crashed the telemetry panel
// and React unmounted the entire Ground Station page.)
import { Component, type ReactNode } from "react";

interface State { error: Error | null }

export default class PanelBoundary extends Component<{ name: string; children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(`[${this.props.name}]`, error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="callout callout-warning" role="alert" style={{ margin: "var(--space-4) 0" }}>
        <div>
          <div className="callout-title">The {this.props.name} panel hit an error</div>
          The rest of the page still works. Error: <code>{this.state.error.message}</code>{" "}
          <button className="btn-plain" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      </div>
    );
  }
}

// stageBandsPlugin.ts
// ====================
// V16 Phase B §B4 -- shades every chart's background by which curriculum
// stage was active at that x position, and marks stage-advance events with
// a vertical rule + hover label. One implementation, attached to every
// MetricTile's uPlot instance uniformly (§2 of the plan) -- this is what
// makes it immediately obvious whether a dip in a metric is a regression or
// just a stage advance, without cross-referencing a separate readout.
import type uPlot from "uplot";
import type { StageBand } from "./useMetricsStream";

const STAGE_COLORS = [
  "rgba(79, 140, 255, 0.05)",
  "rgba(76, 175, 80, 0.05)",
  "rgba(224, 169, 48, 0.06)",
  "rgba(244, 84, 60, 0.06)",
];
const RULE_COLOR = "rgba(255, 255, 255, 0.18)";

export function stageBandsPlugin(getBands: () => StageBand[]): uPlot.Plugin {
  return {
    hooks: {
      drawClear: [
        (u: uPlot) => {
          const bands = getBands();
          if (bands.length === 0) return;

          const ctx = u.ctx;
          const { top, height } = u.bbox;
          ctx.save();

          for (const band of bands) {
            const x0 = u.valToPos(band.startX, "x", true);
            const x1 = u.valToPos(band.endX, "x", true);
            ctx.fillStyle = STAGE_COLORS[band.stage % STAGE_COLORS.length];
            ctx.fillRect(x0, top, Math.max(x1 - x0, 1), height);
          }

          // Vertical rule at every advance (skip the very first band's
          // start -- that's the chart's left edge, not a transition).
          ctx.strokeStyle = RULE_COLOR;
          ctx.lineWidth = 1;
          for (let i = 1; i < bands.length; i++) {
            const x = Math.round(u.valToPos(bands[i].startX, "x", true)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(x, top);
            ctx.lineTo(x, top + height);
            ctx.stroke();
          }

          ctx.restore();
        },
      ],
    },
  };
}

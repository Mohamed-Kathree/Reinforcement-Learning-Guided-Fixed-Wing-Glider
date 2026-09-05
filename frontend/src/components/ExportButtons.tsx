// ExportButtons.tsx
// ==================
// V16 Phase D §D4 -- small "Export SVG" / "Export PNG" control, dropped
// into any chart that has a getSvg/getCanvas accessor. Both are optional so
// a canvas-only chart (uPlot) just omits the SVG button, and vice versa.
import { downloadCanvasPng, downloadSvg, downloadSvgAsPng } from "../exportChart";

interface ExportButtonsProps {
  filename: string;
  getSvg?: () => SVGSVGElement | null;
  getCanvas?: () => HTMLCanvasElement | null;
}

export default function ExportButtons({ filename, getSvg, getCanvas }: ExportButtonsProps) {
  function handleSvg() {
    const el = getSvg?.();
    if (el) downloadSvg(el, filename);
  }

  function handlePng() {
    const canvas = getCanvas?.();
    if (canvas) {
      downloadCanvasPng(canvas, filename);
      return;
    }
    const svg = getSvg?.();
    if (svg) downloadSvgAsPng(svg, filename);
  }

  return (
    <div className="export-buttons">
      {getSvg && (
        <button className="export-btn" onClick={handleSvg}>
          Export SVG
        </button>
      )}
      <button className="export-btn" onClick={handlePng}>
        Export PNG
      </button>
    </div>
  );
}

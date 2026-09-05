// exportChart.ts
// ===============
// V16 Phase D §D4 -- export any chart as a standalone SVG or PNG file, for
// report figures. Both charts stand alone: SVG export serialises the actual
// live element (already carrying its own axis labels/units/legend as real
// markup); PNG export rasterises a <canvas> that already drew its axis
// labels/units onto the bitmap itself (uPlot does this internally), so
// neither format loses that context once saved.
//
// Plain <a download> + Blob URL -- this is a local dev-server web app, not a
// published Artifact, so the sandboxed-download restriction that would
// otherwise block a same-page download doesn't apply here.
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Every chart's colours come from this app's CSS -- either a class
// (.scatter-axis, .outcome-crash, ...) or an inline var(--x) attribute --
// which only resolves while the element is attached to this page's DOM and
// its stylesheet. An exported file opened standalone (a real report
// figure, per the spec) has neither, so EVERY element's actually-computed
// fill/stroke/stroke-width/font is baked in as an explicit attribute here,
// unconditionally, regardless of how the original was styled -- otherwise
// anything styled via a CSS class (not just a literal var(...) attribute)
// would silently render with browser defaults (usually black, or invisible
// text) the moment it's opened outside this app.
function inlineResolvedColors(liveEl: Element, cloneEl: Element): void {
  if (liveEl instanceof SVGElement && cloneEl instanceof SVGElement) {
    const computed = getComputedStyle(liveEl);
    cloneEl.setAttribute("fill", computed.fill);
    cloneEl.setAttribute("stroke", computed.stroke);
    if (computed.strokeWidth) cloneEl.setAttribute("stroke-width", computed.strokeWidth);
    if (liveEl.tagName === "text" || liveEl.tagName === "tspan") {
      cloneEl.setAttribute("font-family", computed.fontFamily);
      cloneEl.setAttribute("font-size", computed.fontSize);
    }
  }
  const liveChildren = Array.from(liveEl.children);
  const cloneChildren = Array.from(cloneEl.children);
  for (let i = 0; i < liveChildren.length; i++) {
    inlineResolvedColors(liveChildren[i], cloneChildren[i]);
  }
}

export function downloadSvg(svgEl: SVGSVGElement, filename: string): void {
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  inlineResolvedColors(svgEl, clone);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const xml = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${xml}`], { type: "image/svg+xml" });
  triggerDownload(blob, filename.endsWith(".svg") ? filename : `${filename}.svg`);
}

export function downloadCanvasPng(canvasEl: HTMLCanvasElement, filename: string): void {
  canvasEl.toBlob((blob) => {
    if (!blob) return;
    triggerDownload(blob, filename.endsWith(".png") ? filename : `${filename}.png`);
  }, "image/png");
}

// For an SVG-native chart where a PNG is also wanted: rasterise by drawing
// the serialised SVG into an offscreen canvas sized to its viewBox/width.
export function downloadSvgAsPng(svgEl: SVGSVGElement, filename: string, scale = 2): void {
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  inlineResolvedColors(svgEl, clone);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const width = svgEl.viewBox?.baseVal?.width || svgEl.clientWidth || 400;
  const height = svgEl.viewBox?.baseVal?.height || svgEl.clientHeight || 300;
  const xml = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([xml], { type: "image/svg+xml" });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, width, height);
      downloadCanvasPng(canvas, filename);
    }
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

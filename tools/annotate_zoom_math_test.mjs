/**
 * Regression test for the video annotation tool's zoom/pan coordinate math.
 *
 * Mirrors the formulas in tools/video_annotate_server.py and asserts the
 * invariants that the 2026-08-17 bug violated (wheel keep-point drift caused
 * by double-subtracting / missing the pan offset). Run with:
 *     node tools/annotate_zoom_math_test.mjs
 */
const NAT_W = 1920;
const NAT_H = 1080;
const STAGE_LEFT = 28;
const STAGE_TOP = 64;
const STAGE_W = 1344;
const STAGE_H = 756;
const ZOOM_MIN = 1;
const ZOOM_MAX = 24;
const WHEEL_FACTOR = 1.3;

function clampZoom(z) {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
}

function clampPan(panX, panY, zoom) {
  const sw = STAGE_W;
  const sh = STAGE_H;
  return [
    Math.max(sw * (1 - zoom), Math.min(0, panX)),
    Math.max(sh * (1 - zoom), Math.min(0, panY)),
  ];
}

// toNat from the page: native = (client - rect.left) / (rect.width / NAT_W)
function toNat(clientX, clientY, panX, panY, zoom) {
  const rectLeft = STAGE_LEFT + panX;
  const rectTop = STAGE_TOP + panY;
  const rectW = STAGE_W * zoom;
  const rectH = STAGE_H * zoom;
  return [(clientX - rectLeft) / (rectW / NAT_W), (clientY - rectTop) / (rectH / NAT_H)];
}

// screen position of a native point (inverse of toNat, used by draw/canvas)
function screenOf(nx, ny, panX, panY, zoom) {
  const rectLeft = STAGE_LEFT + panX;
  const rectTop = STAGE_TOP + panY;
  return [rectLeft + nx * (STAGE_W * zoom / NAT_W), rectTop + ny * (STAGE_H * zoom / NAT_H)];
}

// wheel keep-point step as implemented in the page
function wheelStep(clientX, clientY, panX, panY, zoom, deltaY) {
  const rectLeft = STAGE_LEFT + panX;
  const rectTop = STAGE_TOP + panY;
  const s = (STAGE_W * zoom) / NAT_W;
  const nx = (clientX - rectLeft) / s;
  const ny = (clientY - rectTop) / s;
  const factor = deltaY < 0 ? WHEEL_FACTOR : 1 / WHEEL_FACTOR;
  const newZoom = clampZoom(zoom * factor);
  const ns = (STAGE_W * newZoom) / NAT_W;
  let newPanX = (clientX - STAGE_LEFT) - nx * ns;
  let newPanY = (clientY - STAGE_TOP) - ny * ns;
  [newPanX, newPanY] = clampPan(newPanX, newPanY, newZoom);
  return { panX: newPanX, panY: newPanY, zoom: newZoom };
}

let failures = 0;
function check(name, ok, detail) {
  if (!ok) failures += 1;
  console.log((ok ? "PASS" : "FAIL") + " | " + name + (detail ? " | " + detail : ""));
}

// invariant 1: toNat and screenOf are exact inverses at any zoom/pan
for (const zoom of [1, 1.3, 1.69, 2.197, 5, 24]) {
  for (const [px, py] of [[0, 0], [-200, -100], [-800, -400], [50, 30]]) {
    for (const [nx, ny] of [[0, 0], [960, 540], [100, 900], [1800, 100]]) {
      const [sx, sy] = screenOf(nx, ny, px, py, zoom);
      const [bx, by] = toNat(sx, sy, px, py, zoom);
      check(`toNat/screenOf inverse z=${zoom} p=(${px},${py}) n=(${nx},${ny})`,
        Math.abs(bx - nx) < 1e-9 && Math.abs(by - ny) < 1e-9);
    }
  }
}

// invariant 2a: wheel ZOOM-IN keeps the native point under the cursor fixed
for (const deltaY of [-100, -400]) {
  for (const zoom0 of [1, 1.3, 2.197]) {
    let panX = -150, panY = -80, zoom = zoom0;
    [panX, panY] = clampPan(panX, panY, zoom);
    const clientX = 700, clientY = 442;
    const [nx0, ny0] = toNat(clientX, clientY, panX, panY, zoom);
    for (let i = 0; i < 5; i++) {
      const r = wheelStep(clientX, clientY, panX, panY, zoom, deltaY);
      panX = r.panX; panY = r.panY; zoom = r.zoom;
    }
    const [sx, sy] = screenOf(nx0, ny0, panX, panY, zoom);
    check(`zoom-in keeps cursor point fixed d=${deltaY} z0=${zoom0}`,
      Math.abs(sx - clientX) < 1e-6 && Math.abs(sy - clientY) < 1e-6,
      `cursor(${clientX},${clientY}) -> (${sx.toFixed(3)},${sy.toFixed(3)})`);
  }
}

// invariant 2b: repeated wheel ZOOM-OUT returns to 1x with pan reset to 0
for (const zoom0 of [1.3, 2.197, 8]) {
  let panX = -150, panY = -80, zoom = zoom0;
  [panX, panY] = clampPan(panX, panY, zoom);
  for (let i = 0; i < 30; i++) {
    const r = wheelStep(700, 442, panX, panY, zoom, 300);
    panX = r.panX; panY = r.panY; zoom = r.zoom;
  }
  check(`zoom-out resets to 1x/pan0 z0=${zoom0}`, zoom === 1 && panX === 0 && panY === 0,
    `z=${zoom} p=(${panX.toFixed(3)},${panY.toFixed(3)})`);
}

// invariant 3: clampPan keeps the image covering the canvas (no blank edges)
for (const zoom of [1, 1.69, 8]) {
  const [px, py] = clampPan(-1e6, -1e6, zoom);
  check(`clampPan low bound z=${zoom}`, Math.abs(px - STAGE_W * (1 - zoom)) < 1e-9 && Math.abs(py - STAGE_H * (1 - zoom)) < 1e-9,
    `(${px.toFixed(1)},${py.toFixed(1)})`);
  const [px2, py2] = clampPan(1e6, 1e6, zoom);
  check(`clampPan high bound z=${zoom}`, px2 === 0 && py2 === 0);
}

console.log(failures === 0 ? "ALL PASS" : failures + " FAILURES");
process.exit(failures === 0 ? 0 : 1);

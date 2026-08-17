/**
 * UI regression test for the video annotation tool.
 *
 * Requires: annotation server running on http://127.0.0.1:8101 and
 * `npm i playwright` in this directory. Backs up the current annotations
 * file and restores it at the end (if the run crashes, restore manually
 * from the printed backup path).
 *
 * Run: node tools/annotate_ui_test.js
 */
const fs = require("fs");
const path = require("path");
let pw;
try {
  pw = require("playwright");
} catch (e) {
  const fallback = path.join(process.env.TEMP || require("os").tmpdir(), "opencode", "node_modules", "playwright");
  pw = require(fallback);
}
const { chromium } = pw;

const REPO = path.resolve(__dirname, "..");
const GT = path.join(REPO, "eval", "ball_gt", "demo4", "annotations.jsonl");
const BACKUP = GT + ".ui_test_backup";

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok });
  console.log((ok ? "PASS" : "FAIL") + " | " + name + (detail ? " | " + detail : ""));
}

(async () => {
  if (fs.existsSync(GT)) fs.copyFileSync(GT, BACKUP);
  console.log("backup -> " + BACKUP);

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:8101/annotate/demo4", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(4000);

  const gt = async () =>
    (await (await page.request.get("http://127.0.0.1:8101/api/gt/demo4")).json()).annotations;
  const seek = async (f) => {
    await page.evaluate((f) => { document.getElementById("vid").currentTime = f / 29.97; }, f);
    await page.waitForTimeout(900);
  };
  const natOf = async (cx, cy) =>
    page.evaluate(([cx, cy]) => {
      const r = document.getElementById("vid").getBoundingClientRect();
      return { x: (cx - r.left) / (r.width / 1920), y: (cy - r.top) / (r.height / 1080) };
    }, [cx, cy]);

  // 1. left click draw
  await seek(2600);
  const before = (await gt()).length;
  const box = await page.locator("#cv").boundingBox();
  const p1 = { x: box.x + box.width * 0.25, y: box.y + box.height * 0.3 };
  const n1 = await natOf(p1.x, p1.y);
  await page.mouse.click(p1.x, p1.y);
  await page.waitForTimeout(700);
  let anns = await gt();
  check("left-click creates annotation", anns.length === before + 1);
  const a2600 = anns.find((a) => a.frame === 2600);
  if (a2600) {
    const c = { x: (a2600.bbox[0] + a2600.bbox[2]) / 2, y: (a2600.bbox[1] + a2600.bbox[3]) / 2 };
    check("click position accurate", Math.abs(c.x - n1.x) < 4 && Math.abs(c.y - n1.y) < 4);
  } else check("click position accurate", false, "no annotation at 2600");

  // 2. drag rectangle
  const p2a = { x: box.x + box.width * 0.42, y: box.y + box.height * 0.42 };
  const p2b = { x: p2a.x + 120, y: p2a.y + 80 };
  await page.mouse.move(p2a.x, p2a.y);
  await page.mouse.down();
  await page.mouse.move(p2b.x, p2b.y, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(700);
  const rect = (await gt()).find((a) => a.frame === 2600);
  check("drag-draw makes rectangle", rect && rect.bbox[2] - rect.bbox[0] > 60);

  // 3. zoom keeps cursor point fixed
  await page.keyboard.press("z");
  await page.waitForTimeout(200);
  const mid = { x: box.x + box.width * 0.5, y: box.y + box.height * 0.5 };
  await page.mouse.move(mid.x, mid.y);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(300);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(300);
  const natMid = await natOf(mid.x, mid.y);
  const drift = await page.evaluate(([mx, my, nx, ny]) => {
    const r = document.getElementById("vid").getBoundingClientRect();
    return { x: r.left + nx * (r.width / 1920) - mx, y: r.top + ny * (r.height / 1080) - my };
  }, [mid.x, mid.y, natMid.x, natMid.y]);
  check("zoom-in keeps cursor point fixed", Math.abs(drift.x) < 2 && Math.abs(drift.y) < 2,
    "drift (" + drift.x.toFixed(2) + "," + drift.y.toFixed(2) + ")");
  const vw = await page.evaluate(() => document.getElementById("vid").getBoundingClientRect().width);
  check("video element zooms", vw > 1500, "w=" + vw.toFixed(0));

  // 4. right-drag pan, then zoomed click accuracy
  await page.mouse.move(mid.x, mid.y);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(mid.x + 90, mid.y + 50, { steps: 6 });
  await page.mouse.up({ button: "right" });
  await page.waitForTimeout(400);
  await seek(2700);
  const pz = { x: box.x + box.width * 0.3, y: box.y + box.height * 0.3 };
  const nz = await natOf(pz.x, pz.y);
  await page.mouse.click(pz.x, pz.y);
  await page.waitForTimeout(700);
  const a2700 = (await gt()).find((a) => a.frame === 2700);
  if (a2700) {
    const c = { x: (a2700.bbox[0] + a2700.bbox[2]) / 2, y: (a2700.bbox[1] + a2700.bbox[3]) / 2 };
    check("zoom+pan click accurate", Math.abs(c.x - nz.x) < 5 && Math.abs(c.y - nz.y) < 5);
  } else check("zoom+pan click accurate", false, "no annotation at 2700");

  // 5. interpolation display between keyframes
  await seek(2800);
  await page.mouse.click(mid.x, mid.y);
  await page.waitForTimeout(500);
  await seek(2850);
  await page.mouse.click(mid.x, mid.y);
  await page.waitForTimeout(500);
  anns = await gt();
  const a2800 = anns.find((a) => a.frame === 2800);
  const a2850 = anns.find((a) => a.frame === 2850);
  check("close keyframes join segment", !!a2800 && !!a2850 && a2800.seg === a2850.seg);
  const ibRes = await page.evaluate(async () => {
    document.getElementById("vid").currentTime = 2825 / 29.97;
    await new Promise((r) => setTimeout(r, 400));
    const f = Math.round(document.getElementById("vid").currentTime * 29.97);
    const b = interpBoxAt(f);
    return b ? b.map((v) => v.toFixed(1)) : null;
  });
  check("interp display non-null between keyframes", ibRes !== null && ibRes.length === 4, JSON.stringify(ibRes));

  // 6. Del removes keyframe, Backspace undo
  await seek(2850);
  const d0 = (await gt()).length;
  await page.keyboard.press("Delete");
  await page.waitForTimeout(600);
  const d1 = (await gt()).length;
  check("Del removes box", d1 === d0 - 1);
  await page.keyboard.press("Backspace");
  await page.waitForTimeout(600);
  check("Backspace removes a keyframe", (await gt()).length === d1 - 1);

  // 7. h hint toggle + arrow step + zoom reset
  const hk = await page.evaluate(() => document.getElementById("hintbox").checked);
  await page.keyboard.press("h");
  check("h toggles hint", hk === true && (await page.evaluate(() => document.getElementById("hintbox").checked)) === false);
  const t0 = await page.evaluate(() => document.getElementById("vid").currentTime);
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(300);
  const t1 = await page.evaluate(() => document.getElementById("vid").currentTime);
  check("arrow steps 1 frame", Math.abs((t1 - t0) * 29.97 - 1) < 0.6);
  await page.keyboard.press("z");
  await page.waitForTimeout(300);
  const vw2 = await page.evaluate(() => document.getElementById("vid").getBoundingClientRect().width);
  check("z off resets zoom", Math.abs(vw2 - 1344) <= 3, "w=" + vw2.toFixed(0));

  console.log("page errors:", JSON.stringify(errors));
  const failed = results.filter((r) => !r.ok).length;
  console.log("TOTAL: " + (results.length - failed) + "/" + results.length + " passed");

  await browser.close();

  if (fs.existsSync(BACKUP)) fs.copyFileSync(BACKUP, GT);
  fs.rmSync(BACKUP, { force: true });
  console.log("annotations restored");
  process.exit(failed === 0 ? 0 : 1);
})();

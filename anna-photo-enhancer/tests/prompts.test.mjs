import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const appJs = readFileSync(join(root, "bundle", "app.js"), "utf8");
const indexHtml = readFileSync(join(root, "bundle", "index.html"), "utf8");

test("bundle keeps three mode defs; UI ships only enhance + remove_bg", () => {
  assert.match(appJs, /enhance:/);
  assert.match(appJs, /remove_bg:/);
  assert.match(appJs, /remove_object:/);
  assert.match(appJs, /SHIPPED_MODES\s*=\s*\["enhance",\s*"remove_bg"\]/);
  assert.match(indexHtml, /data-mode="enhance"/);
  assert.match(indexHtml, /data-mode="remove_bg"/);
  assert.doesNotMatch(indexHtml, /data-mode="remove_object"/);
});

test("enhance prompt asks to preserve identity", () => {
  assert.match(appJs, /Keep the subject, composition, and identity unchanged/);
});

test("remove_bg prompt asks for transparent PNG", () => {
  assert.match(appJs, /transparent PNG/);
});

test("remove_object prompt helper retained for later unhide", () => {
  assert.match(appJs, /promptFor:\s*\(text\)\s*=>/);
  assert.match(appJs, /Remove the following from this photograph/);
});

test("upload uses image_input purpose", () => {
  assert.match(appJs, /purpose:\s*"image_input"/);
});

test("edit call uses image_url + prompt", () => {
  assert.match(appJs, /anna\.image\.edit\(\{[\s\S]*prompt[\s\S]*image_url/);
});

test("edit pins fal-ai/nano-banana-2 model", () => {
  assert.match(appJs, /EDIT_MODEL\s*=\s*"fal-ai\/nano-banana-2"/);
  assert.match(appJs, /anna\.image\.edit\(\{[\s\S]*model:\s*EDIT_MODEL/);
});

test("status maps 502 / APP_PROVIDER_ERROR to clear UX", () => {
  assert.match(appJs, /function formatHostError/);
  assert.match(appJs, /APP_PROVIDER_ERROR/);
  assert.match(appJs, /Platform image edit is temporarily unavailable/);
  assert.match(appJs, /function isHostEditUnavailable/);
  assert.match(appJs, /APP_NOT_GRANTED/);
  assert.match(appJs, /Enable grants for this app/);
});

test("enhance prefers declared local_enhance before optional host edit", () => {
  assert.match(appJs, /local_enhance/);
  assert.match(appJs, /fileToPngBase64/);
  assert.match(appJs, /invokeLocalTool/);
  assert.match(appJs, /function localToolCandidates/);
  assert.match(appJs, /bundled:photo-enhancer-local/);
  assert.match(appJs, /Done — enhanced locally\. You can download the result\./);
  // tool-dev id may exist as last-resort harness fallback only — never first candidate.
  const candidatesBlock = appJs.match(
    /function localToolCandidates\(\)[\s\S]*?^}/m
  )?.[0] || "";
  assert.match(candidatesBlock, /bundled:photo-enhancer-local/);
  assert.match(candidatesBlock, /tool-dev-photo-enhancer-local/);
  const bundledIdx = candidatesBlock.indexOf("bundled:photo-enhancer-local");
  const devIdx = candidatesBlock.indexOf("tool-dev-photo-enhancer-local");
  assert.ok(bundledIdx >= 0 && devIdx > bundledIdx, "tool-dev must be last resort");
});

test("remove_bg prefers local_remove_bg; remove_object not shipped", () => {
  assert.match(appJs, /local_remove_bg/);
  assert.match(appJs, /Done — background removed locally\. You can download the result\./);
  assert.match(appJs, /uv pip install \"rembg\[cpu\]\"/);
  assert.match(appJs, /not available in this release/);
});

test("Process button disabled while running", () => {
  assert.match(appJs, /runBtn\.disabled\s*=\s*true/);
  assert.match(appJs, /finally\s*\{[\s\S]*runBtn\.disabled/);
});

test("download affordance supports data URLs and host success copy", () => {
  assert.match(appJs, /function downloadResult/);
  assert.match(appJs, /a\.download\s*=/);
  assert.match(appJs, /You can download the result/);
  assert.match(appJs, /via Anna\. You can download the result/);
});

test("hero copy matches BG + enhance cut", () => {
  assert.match(indexHtml, /AI background removal and photo enhance/);
});

test("UI accepts HEIC/HEIF and converts before process", () => {
  assert.match(indexHtml, /image\/heic/);
  assert.match(indexHtml, /\.heic/);
  assert.match(indexHtml, /JPG \/ PNG \/ WEBP \/ HEIC/);
  assert.match(appJs, /function isAcceptedImageFile/);
  assert.match(appJs, /function isHeicFile/);
  assert.match(appJs, /function ensureProcessableImage/);
  assert.match(appJs, /function preloadHeic2Any/);
  assert.match(appJs, /HEIC_MAX_EDGE/);
  assert.match(appJs, /heic2any\.min\.js/);
  assert.match(appJs, /Please choose an image file \(JPG, PNG, WEBP, or HEIC\)\./);
  assert.match(appJs, /Converting HEIC to JPEG/);
});

test("enhance mode help does not mention local polish outage copy", () => {
  assert.doesNotMatch(appJs, /local polish if Anna photo-edit is down/);
  assert.match(appJs, /Improve clarity, color, and detail\./);
});

test("index loads anna-tool-ids before app module", () => {
  assert.match(indexHtml, /anna-tool-ids\.js/);
  const idsIdx = indexHtml.indexOf("anna-tool-ids.js");
  const appIdx = indexHtml.indexOf("app.js");
  assert.ok(idsIdx >= 0 && appIdx > idsIdx);
});

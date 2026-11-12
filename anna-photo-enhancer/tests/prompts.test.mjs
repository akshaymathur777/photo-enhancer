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

test("enhance falls back to local_enhance on host edit 502", () => {
  assert.match(appJs, /local_enhance/);
  assert.match(appJs, /fileToPngBase64/);
  assert.match(appJs, /invokeLocalTool/);
  assert.match(appJs, /Done — enhanced locally \(Anna photo-edit unavailable\)/);
  assert.match(appJs, /tool-dev-photo-enhancer-local/);
});

test("remove_bg may try local_remove_bg; remove_object not shipped", () => {
  assert.match(appJs, /local_remove_bg/);
  assert.match(appJs, /Done — background removed locally \(Anna photo-edit unavailable\)/);
  assert.match(appJs, /uv pip install \"rembg\[cpu\]\"/);
  assert.match(appJs, /not available in this release/);
  assert.match(appJs, /no fake inpaint/);
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

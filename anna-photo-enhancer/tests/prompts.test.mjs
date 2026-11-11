import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const appJs = readFileSync(join(root, "bundle", "app.js"), "utf8");

test("bundle defines three edit modes", () => {
  assert.match(appJs, /enhance:/);
  assert.match(appJs, /remove_bg:/);
  assert.match(appJs, /remove_object:/);
});

test("enhance prompt asks to preserve identity", () => {
  assert.match(appJs, /Keep the subject, composition, and identity unchanged/);
});

test("remove_bg prompt asks for transparent PNG", () => {
  assert.match(appJs, /transparent PNG/);
});

test("remove_object builds prompt from user text", () => {
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

test("remove_bg may try local_remove_bg; remove_object does not fake success", () => {
  assert.match(appJs, /local_remove_bg/);
  assert.match(appJs, /Done — background removed locally \(Anna photo-edit unavailable\)/);
  assert.match(appJs, /uv pip install \"rembg\[cpu\]\"/);
  assert.match(appJs, /Describe what to remove — for example/);
  assert.match(appJs, /Remove object needs Anna's photo-edit service/);
  assert.match(appJs, /Try Enhance or Remove background instead/);
  assert.match(appJs, /no fake inpaint/);
  assert.match(appJs, /Needs Anna photo-edit/);
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

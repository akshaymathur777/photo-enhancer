import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createRequire } from "node:module";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
const appJson = JSON.parse(readFileSync(join(root, "app.json"), "utf8"));

const require = createRequire(import.meta.url);
let mountBundle;
let HostApiError;
const cliTestCandidates = [
  () => require("@anna-ai/cli/test"),
  // npm --prefix ~/.local (cloud / unprivileged installs)
  () =>
    require(
      join(
        process.env.HOME || "",
        ".local/lib/node_modules/@anna-ai/cli/dist/test/index.js"
      )
    ),
  // Windows global npm
  () =>
    require(
      join(
        process.env.APPDATA || "",
        "npm/node_modules/@anna-ai/cli/dist/test/index.js"
      )
    ),
];
let lastErr;
for (const load of cliTestCandidates) {
  try {
    ({ mountBundle, HostApiError } = load());
    lastErr = null;
    break;
  } catch (e) {
    lastErr = e;
  }
}
if (lastErr) throw lastErr;

test("app.json declares bundled photo-enhancer executa", () => {
  assert.equal(appJson.slug, "reliclear");
  assert.ok(appJson.bundled_executas["photo-enhancer"]);
  assert.equal(appJson.bundled_executas["photo-enhancer"].path, "./executas/photo-enhancer");
});

test("manifest host_api declares image + upload surfaces", () => {
  const ha = manifest.ui.host_api;
  for (const m of ["generate", "edit", "resign"]) {
    assert.ok(ha.image.includes(m), `missing image.${m}`);
  }
  for (const m of ["inline", "negotiate", "confirm"]) {
    assert.ok(ha.upload.includes(m), `missing upload.${m}`);
  }
  assert.ok(ha.tools.includes("bundled:photo-enhancer"));
  assert.ok(
    ha.tools.includes("bundled:photo-enhancer-local"),
    "host_api.tools must allow local classical fallback"
  );
});

test("csp allows https images for R2 presigns", () => {
  assert.ok(manifest.ui.csp_overrides["img-src"].includes("https:"));
});

test("mountBundle allows storage.set; chat ACL depends on schema era", async () => {
  const h = await mountBundle({
    manifest,
    mocks: {
      "storage.set": async () => ({ ok: true }),
      "storage.get": async () => null,
      "window.set_title": async () => ({}),
      "window.ready": async () => ({}),
      // Schema 4 / dispatcher ≥0.33 retires ui.host_api ACL; mock so undeclared
      // chat is not a hard DENIED if the harness no longer gates on host_api.
      "chat.write_message": async () => ({ ok: true }),
    },
  });
  await h.runtime.storage.set("photo-enhancer:last", { mode: "enhance" });
  assert.equal(h.calls.lastOf("storage.set")?.outcome, "ok");

  try {
    await h.runtime.call("chat", "write_message", { text: "x" });
    // New harness: host_api retired → call may succeed via mock (not DENIED).
  } catch (err) {
    // Older harness: undeclared surface still DENIED.
    assert.ok(err instanceof HostApiError && err.code === "DENIED");
  }
});

test("tools.invoke for bundled handle is ACL-allowed (via required_executas fallback)", async () => {
  // deriveAcl strips "bundled:" prefix into allowedTools as "photo-enhancer"
  // and also adds required_executas tool_ids. Our required entry is bundled:photo-enhancer.
  const h = await mountBundle({
    manifest,
    mocks: {
      "tools.invoke": async (args) => ({
        success: true,
        data: { echo: args },
      }),
      "window.set_title": async () => ({}),
    },
  });

  // Bare tool-dev id used at runtime by local harness mapping
  const deniedOrOk = async (toolId) => {
    try {
      const out = await h.runtime.tools.invoke({
        tool_id: toolId,
        method: "ping",
        args: {},
      });
      return { ok: true, out };
    } catch (e) {
      return { ok: false, code: e.code, message: e.message };
    }
  };

  const bundled = await deniedOrOk("bundled:photo-enhancer");
  // ACL stores bare id after split — "photo-enhancer" — so full "bundled:..." may deny.
  const bare = await deniedOrOk("photo-enhancer");
  assert.ok(
    bundled.ok || bare.ok,
    `expected one tool id allowed; bundled=${JSON.stringify(bundled)} bare=${JSON.stringify(bare)}`
  );
});

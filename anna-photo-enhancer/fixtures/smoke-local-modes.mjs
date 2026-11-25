#!/usr/bin/env node
/**
 * Offline-ish smoke: local_enhance + local_remove_bg via anna-app executa,
 * plus CDP check that Remove Object is hidden on harness :5180.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE = resolve(ROOT, "fixtures/sample.png");
const LOCAL_DIR = resolve(ROOT, "executas/photo-enhancer-local");
const OUT = resolve(ROOT, ".anna/smoke-local-modes.json");
const HARNESS = process.argv[2] || "http://127.0.0.1:5180/";
const CDP_PORT = Number(process.env.CDP_PORT || 9223);

function b64() {
  return readFileSync(SAMPLE).toString("base64");
}

function invokeLocal(method) {
  const argsJson = JSON.stringify({ image_png_b64: b64() });
  const r = spawnSync(
    "anna-app",
    [
      "executa",
      "dev",
      "--dir",
      LOCAL_DIR,
      "--json",
      "--invoke",
      method,
      "--args",
      argsJson,
      "--no-image",
      "--storage",
      "off",
    ],
    { encoding: "utf8", cwd: ROOT, shell: true, timeout: 180000 }
  );
  const blob = (r.stdout || "") + (r.stderr || "");
  const lines = blob.split(/\r?\n/).filter((ln) => ln.trim().startsWith("{"));
  let parsed = null;
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      parsed = JSON.parse(lines[i]);
      break;
    } catch {}
  }
  return { code: r.status, parsed, preview: blob.slice(-400) };
}

async function getWsUrl() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`);
      if (res.ok) {
        const j = await res.json();
        if (j.webSocketDebuggerUrl) return j.webSocketDebuggerUrl;
      }
    } catch {}
    await sleep(250);
  }
  throw new Error("CDP not ready");
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      }
    };
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(payload));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout ${method}`));
        }
      }, 60000);
    });
  }
}

async function uiCheck() {
  const chrome = spawnSync(
    process.platform === "win32" ? "chrome" : "google-chrome",
    [],
    { encoding: "utf8" }
  );
  // Prefer Edge/Chrome paths on Windows via cmd start is flaky; use puppeteer-less CDP via edge
  const candidates = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  ].filter(Boolean);

  let browserPath = candidates.find((p) => {
    try {
      return !!readFileSync(p);
    } catch {
      return false;
    }
  });
  if (!browserPath) {
    return { skipped: true, reason: "no chrome/edge binary" };
  }

  const { spawn } = await import("node:child_process");
  const child = spawn(
    browserPath,
    [
      `--remote-debugging-port=${CDP_PORT}`,
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--user-data-dir=" + resolve(ROOT, ".anna/smoke-chrome"),
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] }
  );
  try {
    const wsUrl = await getWsUrl();
    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = () => rej(new Error("ws open failed"));
    });
    const cdp = new Cdp(ws);
    const { targetId } = await cdp.send("Target.createTarget", { url: HARNESS });
    const { sessionId } = await cdp.send("Target.attachToTarget", {
      targetId,
      flatten: true,
    });
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);
    for (let i = 0; i < 40; i++) {
      const { result } = await cdp.send(
        "Runtime.evaluate",
        {
          expression: "document.readyState",
          returnByValue: true,
        },
        sessionId
      );
      if (result?.value === "complete") break;
      await sleep(500);
    }
    await sleep(2000);
    const { result } = await cdp.send(
      "Runtime.evaluate",
      {
        expression: `({
          hero: document.querySelector('.hero')?.textContent || '',
          modes: [...document.querySelectorAll('.mode')].map(b => b.dataset.mode),
          hasRemoveObject: !!document.querySelector('[data-mode="remove_object"]'),
          hasEnhance: !!document.querySelector('[data-mode="enhance"]'),
          hasRemoveBg: !!document.querySelector('[data-mode="remove_bg"]'),
        })`,
        returnByValue: true,
      },
      sessionId
    );
    return result?.value || { err: "no value" };
  } finally {
    try {
      child.kill();
    } catch {}
  }
}

async function main() {
  mkdirSync(dirname(OUT), { recursive: true });
  const enhance = invokeLocal("local_enhance");
  const rembg = invokeLocal("local_remove_bg");
  let ui = null;
  try {
    ui = await uiCheck();
  } catch (e) {
    ui = { err: String(e?.message || e) };
  }

  const report = {
    at: new Date().toISOString(),
    enhance: {
      ok: !!(
        enhance.parsed?.success &&
        (enhance.parsed?.data?.image_png_b64 || enhance.parsed?.data?.image_b64)
      ),
      code: enhance.code,
      error: enhance.parsed?.error || null,
    },
    remove_bg: {
      ok: !!(
        rembg.parsed?.success &&
        (rembg.parsed?.data?.image_png_b64 || rembg.parsed?.data?.image_b64)
      ),
      code: rembg.code,
      error: rembg.parsed?.error || null,
      preview: rembg.preview,
    },
    ui,
  };
  writeFileSync(OUT, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  const pass =
    report.enhance.ok &&
    report.remove_bg.ok &&
    (report.ui?.skipped ||
      (report.ui?.hasEnhance &&
        report.ui?.hasRemoveBg &&
        report.ui?.hasRemoveObject === false));
  process.exit(pass ? 0 : 2);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

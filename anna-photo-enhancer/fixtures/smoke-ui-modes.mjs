#!/usr/bin/env node
import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as sleep } from "node:timers/promises";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const HARNESS = process.argv[2] || "http://127.0.0.1:5180/";
const CDP_PORT = 9223;
const candidates = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);
const browserPath = candidates.find((p) => existsSync(p));
if (!browserPath) {
  console.error("no browser");
  process.exit(1);
}

const userData = resolve(ROOT, ".anna/smoke-chrome-ui");
mkdirSync(userData, { recursive: true });
const child = spawn(
  browserPath,
  [
    `--remote-debugging-port=${CDP_PORT}`,
    "--headless=new",
    "--disable-gpu",
    `--user-data-dir=${userData}`,
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "pipe"] }
);

async function getWsUrl() {
  for (let i = 0; i < 40; i++) {
    try {
      const j = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).json();
      if (j.webSocketDebuggerUrl) return j.webSocketDebuggerUrl;
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
    ws.onmessage = (ev) => {
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
          reject(new Error(`timeout ${method}`));
        }
      }, 60000);
    });
  }
}

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
  await sleep(5000);
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression: `(() => {
        const docs = [document];
        const ifr = document.querySelector('iframe');
        try { if (ifr && ifr.contentDocument) docs.push(ifr.contentDocument); } catch (e) {}
        return docs.map((d) => ({
          title: d.title || '',
          hero: (d.querySelector('.hero') && d.querySelector('.hero').textContent) || '',
          modes: Array.from(d.querySelectorAll('.mode')).map((b) => b.dataset.mode),
          hasRemoveObject: !!d.querySelector('[data-mode="remove_object"]'),
          hasEnhance: !!d.querySelector('[data-mode="enhance"]'),
          hasRemoveBg: !!d.querySelector('[data-mode="remove_bg"]'),
        }));
      })()`,
      returnByValue: true,
    },
    sessionId
  );
  const rows = result?.value || [];
  console.log(JSON.stringify(rows, null, 2));
  const hit = rows.find((r) => r.hasEnhance || r.hasRemoveBg || (r.modes && r.modes.length));
  const ok =
    hit &&
    hit.hasEnhance &&
    hit.hasRemoveBg &&
    hit.hasRemoveObject === false &&
    !hit.modes.includes("remove_object");
  process.exit(ok ? 0 : 2);
} finally {
  try {
    child.kill();
  } catch {}
}

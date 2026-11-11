#!/usr/bin/env node
/**
 * CDP runner for fixtures/diag-edit-502.js against anna-app harness :5180.
 * Usage: node fixtures/run-diag-cdp.mjs [http://127.0.0.1:5180]
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const __dirname = dirname(fileURLToPath(import.meta.url));
const HARNESS = process.argv[2] || "http://127.0.0.1:5180/";
const CDP_PORT = Number(process.env.CDP_PORT || 9222);
const OUT = resolve(__dirname, "../.anna/diag-edit-502-result.json");
const PROBE = readFileSync(resolve(__dirname, "diag-edit-502.js"), "utf8");

function chromeArgs() {
  return [
    `--remote-debugging-port=${CDP_PORT}`,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--user-data-dir=/tmp/anna-diag-chrome",
    "about:blank",
  ];
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
  throw new Error("CDP not ready on port " + CDP_PORT);
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
      }, 600000);
    });
  }
}

async function main() {
  mkdirSync(dirname(OUT), { recursive: true });
  const chrome = spawn("google-chrome", chromeArgs(), {
    stdio: ["ignore", "ignore", "pipe"],
  });
  let chromeErr = "";
  chrome.stderr.on("data", (d) => {
    chromeErr += d.toString();
  });

  try {
    const wsUrl = await getWsUrl();
    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = () => rej(new Error("ws open failed: " + chromeErr.slice(0, 200)));
    });
    const cdp = new Cdp(ws);

    const { targetId } = await cdp.send("Target.createTarget", { url: HARNESS });
    const { sessionId } = await cdp.send("Target.attachToTarget", {
      targetId,
      flatten: true,
    });
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);

    // Wait for iframe + runtime
    let ready = false;
    for (let i = 0; i < 60; i++) {
      const { result } = await cdp.send(
        "Runtime.evaluate",
        {
          expression:
            "!!document.querySelector('iframe') && document.readyState === 'complete'",
          returnByValue: true,
        },
        sessionId
      );
      if (result?.value) {
        ready = true;
        break;
      }
      await sleep(1000);
    }
    if (!ready) throw new Error("harness iframe not ready");

    // Give app iframe a moment to connect AnnaAppRuntime
    await sleep(3000);

    const evalRes = await cdp.send(
      "Runtime.evaluate",
      {
        expression: PROBE,
        awaitPromise: true,
        returnByValue: true,
        timeout: 600000,
      },
      sessionId
    );

    if (evalRes.exceptionDetails) {
      const err = {
        err: "evaluate exception",
        details: evalRes.exceptionDetails,
      };
      writeFileSync(OUT, JSON.stringify(err, null, 2));
      console.error(JSON.stringify(err, null, 2));
      process.exit(2);
    }

    const value = evalRes.result?.value;
    writeFileSync(OUT, JSON.stringify(value, null, 2));
    console.log(JSON.stringify(value, null, 2));
    process.exit(value?.err ? 3 : 0);
  } finally {
    try {
      chrome.kill("SIGTERM");
    } catch {}
  }
}

main().catch((e) => {
  console.error(String(e?.stack || e));
  process.exit(1);
});

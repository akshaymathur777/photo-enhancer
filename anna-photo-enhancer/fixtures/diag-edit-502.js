/**
 * Differential diagnosis probe for image.edit CF 502.
 * Paste into the harness parent page CDP Runtime.evaluate (awaitPromise).
 * Expects the Reliclear app iframe to be connected (anna runtime ready).
 *
 * Matrix (same upload URL):
 *   C1 generate plain
 *   C2 generate + reference_image_urls
 *   E0 edit bare
 *   E1 edit modelPreferences nano-banana
 *   E2 edit model gpt-image-2
 *   E3 edit model nano-banana (current Reliclear pin)
 *   A1 editAsync / editAwait
 */
(async () => {
  const PROMPT =
    "Enhance this photograph: improve clarity, natural detail, color balance, and mild sharpening. Keep the subject, composition, and identity unchanged. Do not add or remove objects. Photorealistic output.";

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function summarizeErr(e) {
    const msg = String(e?.message || e || "");
    const code = String(e?.code || e?.name || "");
    const ray =
      (msg.match(/\b([a-f0-9]{16})\b/) || [])[1] ||
      null;
    const is502 = /502|Bad [Gg]ateway|<!DOCTYPE|<html[\s>]/i.test(msg);
    const isJson =
      /APP_|IMAGE_|NOT_GRANTED|INVALID|PROVIDER/i.test(msg) && !is502;
    return {
      ok: false,
      code,
      is502,
      isJson,
      ray,
      preview: msg.slice(0, 220).replace(/\s+/g, " "),
    };
  }

  async function run(label, fn) {
    const t0 = Date.now();
    try {
      const out = await fn();
      return {
        label,
        ok: true,
        ms: Date.now() - t0,
        model: out?.model || out?.result?.model || null,
        images: (out?.images || out?.result?.images || []).length,
        jobId: out?.jobId || null,
        state: out?.state || null,
      };
    } catch (e) {
      return { label, ms: Date.now() - t0, ...summarizeErr(e) };
    }
  }

  // Prefer iframe anna if exposed; otherwise drive via parent RPC is not available.
  // Inject a bridge: call Host API through the app iframe by evaluating there.
  const iframe = document.querySelector("iframe");
  if (!iframe?.contentWindow) return { err: "no iframe" };

  // Load sample from fixtures via fetch of bundle sample (may reload harness —
  // prefer data URL injection inside iframe).
  const sampleB64 =
    "iVBORw0KGgoAAAANSUhEUgAAAUAAAADwCAIAAAD+Tyo8AAAFEUlEQVR4nO3dP3LbRhjGYSajSofwIXwEFKlSqM6ZcpjULnAEH0e1R7FHoiWKBED82Xe/5+lpw/vtbxcc2eM/vvzz7wnI9OfRDwAsJ2AIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAIJmAI9nD0A7CO8fm/uR8ZHp+sfjoBl2h1+q+j6iwCrhXtrN9IzO0TcLt2i3bKA4i5TQJuzuHdXn8qJTdFwK1os9uPlNwUAR8vJd2Lj+1CPpaAjxSa7jkZH0vAx+gg3XMyPoqA99ZZuudkvD8B76fjdM/JeE8C3kORdM/JeB/+McPmCtb7qvKffR9u4A3Zvq7irbmBt6Jeq7EDN/D6pHtlWfzFj3W5gVemXuuzJwGvSb1WaWdeodch3QXL5XX6fm7gFajXuh1FwPdSr9U7kIDvot77WcN7CHg5O28tVnIxAS9kz63Lei4j4CXsti1Y1QUEPJt9th1rO5eA57HDtmaFZxHwDPbWPqzzdAK2q1qk4YkEPIn9tD9rPoWAIZiAb3MVHMXK3yTgG+yhY1n/6wR8jd3TAlO4QsAQTMCfcvC3wyw+I+DL7JjWmMhFAoZgAr7AYd8mc/lIwO/ZJS0znXcEDMEE/BsHfPvM6JyAIZiA3zjaU5jUKwFDMAH/4lDPYl4/CRiCCfiF4zzR+P//kFacgCGYgB3kwcbyl7CAIVj1gB3h6cbal3D1gCGagCFY6YCLv311Yyw8x9IBQzoBQ7C6AVd+7+rPWHWadQOGDggYggkYghUNuOxXpo6NJWdaNGDog4AhmIAhWMWAa35ZqmCsN9mKAUM3BAzBBAzBBAzBBAzBBAzBygVc8CcNpYzF5lsuYOiJgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCGYgCFYuYCHx6ejH4ENDcXmWy5g6ImAIZiAIZiAIZiAIZiAIVjFgKv9pKGOod5kKwYM3RAwBBMwBCsacMEvS90bSs60aMDQBwFDMAFDsLoB1/zK1Kuh6jTrBgwdEDAEKx1w2feuzgyF51g6YEgnYAhWPeDKb199GGpPsHrAEE3A1Y/waEP52QkYggn4hYM80VD++hUwZHMD/+I4z2JePwkYggn4jUM9hUm9EjAEE/BvHO3tM6NzAoZgAn7PAd8y03lHwBfYJW0yl48EDMEEfJnDvjUmcpGAP2XHtMSsPiNgCCbgaxz8LTCFKwR8g91zLOt/nYBvs4eOYuVvEjAEE/AkroL9WfMpBDyV/bQnqz2RgGewq/ZhnacT8Dz21tas8CwCns0O2461nUvAS9hnW7CqCwh4IbttXdZzGQEvZ8+txUou9rD8o6fT41/fT7X9ffpy9CP04PFUfSM9f/u67INuYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAgmYAj2cM+Hn799Xe9JgNncwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwBBMwHDK9QO6QBUp84VowAAAAABJRU5ErkJggg==";

  // Expose probe on iframe window by injecting a script that uses AnnaAppRuntime.
  const result = await iframe.contentWindow.eval(`(async () => {
    const PROMPT = ${JSON.stringify(PROMPT)};
    const sampleB64 = ${JSON.stringify(sampleB64)};
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    function summarizeErr(e) {
      const msg = String(e?.message || e || "");
      const code = String(e?.code || e?.name || "");
      const ray = (msg.match(/\\b([a-f0-9]{16})\\b/) || [])[1] || null;
      const is502 = /502|Bad [Gg]ateway|<!DOCTYPE|<html[\\s>]/i.test(msg);
      const isJson = /APP_|IMAGE_|NOT_GRANTED|INVALID|PROVIDER/i.test(msg) && !is502;
      return { ok: false, code, is502, isJson, ray, preview: msg.slice(0, 220).replace(/\\s+/g, " ") };
    }
    async function run(label, fn) {
      const t0 = Date.now();
      try {
        const out = await fn();
        return {
          label, ok: true, ms: Date.now() - t0,
          model: out?.model || out?.result?.model || null,
          images: (out?.images || out?.result?.images || []).length,
          jobId: out?.jobId || null,
          state: out?.state || out?.result?.state || null,
        };
      } catch (e) {
        return { label, ms: Date.now() - t0, ...summarizeErr(e) };
      }
    }

    let anna;
    if (window.__diagAnna) anna = window.__diagAnna;
    else {
      // AnnaAppRuntime.connect may already have been used by the app; try global reconnect.
      const RT = window.AnnaAppRuntime || parent.AnnaAppRuntime;
      if (!RT?.connect) return { err: "AnnaAppRuntime missing in iframe" };
      anna = await RT.connect();
      window.__diagAnna = anna;
    }

    const bin = Uint8Array.from(atob(sampleB64), (c) => c.charCodeAt(0));
    // Prefer upload.inline with base64
    const b64 = sampleB64;
    let uploaded;
    try {
      uploaded = await anna.upload.inline({
        filename: "sample.png",
        mime_type: "image/png",
        content_b64: b64,
        purpose: "image_input",
      });
    } catch (e) {
      try {
        uploaded = await anna.upload.inline({
          filename: "sample.png",
          mime: "image/png",
          bytes_b64: b64,
          purpose: "image_input",
        });
      } catch (e2) {
        return { err: "upload failed", detail: summarizeErr(e2) };
      }
    }
    const imageUrl = uploaded.download_url || uploaded.url;
    if (!imageUrl) return { err: "upload ok but no url", uploadedKeys: Object.keys(uploaded || {}) };

    const rows = [];
    rows.push(await run("C1_generate_plain", () =>
      anna.image.generate({ prompt: "A simple red circle on white background", n: 1 })
    ));
    rows.push(await run("C2_generate_reference", () =>
      anna.image.generate({ prompt: PROMPT, n: 1, reference_image_urls: [imageUrl] })
    ));
    rows.push(await run("E0_edit_bare", () =>
      anna.image.edit({ image_url: imageUrl, prompt: PROMPT, n: 1 })
    ));
    rows.push(await run("E1_edit_modelPreferences_nano", () =>
      anna.image.edit({
        image_url: imageUrl,
        prompt: PROMPT,
        n: 1,
        modelPreferences: { hints: [{ name: "fal-ai/nano-banana-2" }] },
      })
    ));
    rows.push(await run("E2_edit_model_gpt_image_2", () =>
      anna.image.edit({
        image_url: imageUrl,
        prompt: PROMPT,
        n: 1,
        model: "openai/gpt-image-2",
      })
    ));
    rows.push(await run("E3_edit_model_nano", () =>
      anna.image.edit({
        image_url: imageUrl,
        prompt: PROMPT,
        n: 1,
        model: "fal-ai/nano-banana-2",
      })
    ));

    // Async path
    let asyncRow;
    if (typeof anna.image.editAwait === "function") {
      asyncRow = await run("A1_editAwait_nano", () =>
        anna.image.editAwait({
          image_url: imageUrl,
          prompt: PROMPT,
          n: 1,
          model: "fal-ai/nano-banana-2",
        }, { timeoutMs: 180000 })
      );
    } else if (typeof anna.image.editAsync === "function") {
      asyncRow = await run("A1_editAsync_nano", async () => {
        const created = await anna.image.editAsync({
          image_url: imageUrl,
          prompt: PROMPT,
          n: 1,
          model: "fal-ai/nano-banana-2",
        });
        const jobId = created.jobId;
        const deadline = Date.now() + 180000;
        while (Date.now() < deadline) {
          const job = await anna.image.getJob({ jobId });
          if (["succeeded", "failed", "cancelled"].includes(job.state)) return job;
          await sleep(1500);
        }
        throw new Error("editAsync poll timeout jobId=" + jobId);
      });
    } else {
      asyncRow = { label: "A1_editAsync", ok: false, preview: "editAsync/editAwait not in runtime" };
    }
    rows.push(asyncRow);

    return {
      at: new Date().toISOString(),
      imageUrlHost: (() => { try { return new URL(imageUrl).host; } catch { return null; } })(),
      uploadOk: true,
      rows,
      anyEditOk: rows.some((r) => r.ok && /^E|^A/.test(r.label)),
      anyEditJsonErr: rows.some((r) => !r.ok && r.isJson && /^E|^A/.test(r.label)),
      allEdit502: rows.filter((r) => /^E|^A/.test(r.label)).every((r) => r.is502),
    };
  })()`);

  return result;
})()

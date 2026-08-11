import type { AiMode } from "./api";

export interface ClickPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export type SelectionMode = "smart" | "brush" | "auto-people" | "remove-others";
export type InpaintMode = "ai" | "quick";

export interface RemovalResult {
  blob: Blob;
  engine: string;
  scene: string;
  postfill: string;
}

async function parseRemovalResponse(res: Response): Promise<RemovalResult> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Remove failed (${res.status})`);
  }
  const blob = await res.blob();
  return {
    blob,
    engine: res.headers.get("X-Inpaint-Engine") ?? "unknown",
    scene: res.headers.get("X-Inpaint-Scene") ?? "generic",
    postfill: res.headers.get("X-Inpaint-Postfill") ?? "none",
  };
}

export async function segmentPreview(
  file: File,
  points: ClickPoint[]
): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("points", JSON.stringify(points.map((p) => [p.x, p.y])));
  form.append("labels", JSON.stringify(points.map((p) => p.label)));
  form.append("preview", "true");

  const res = await fetch("/api/segment", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Segment failed (${res.status})`);
  }
  return res.blob();
}

export async function autoSegmentPeople(file: File): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("preview", "true");

  const res = await fetch("/api/auto-segment-people", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `People detection failed (${res.status})`);
  }
  return res.blob();
}

export async function previewRemoveOthers(
  file: File,
  lockPoint?: { x: number; y: number } | null
): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("preview", "true");
  if (lockPoint) {
    form.append("lock_x", String(lockPoint.x));
    form.append("lock_y", String(lockPoint.y));
  }
  const res = await fetch("/api/auto-remove-others", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Preview failed (${res.status})`);
  }
  return res.blob();
}

export async function removeObjects(
  file: File,
  options: {
    points?: ClickPoint[];
    maskFile?: Blob;
    inpaintMode?: InpaintMode;
    edgeSoftness?: number;
    mode?: AiMode;
    lockPoint?: { x: number; y: number } | null;
  }
): Promise<RemovalResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode ?? "balanced");
  form.append("inpaint_mode", options.inpaintMode ?? "ai");
  form.append("edge_softness", String(options.edgeSoftness ?? 1.0));
  if (options.lockPoint) {
    form.append("lock_x", String(options.lockPoint.x));
    form.append("lock_y", String(options.lockPoint.y));
  }

  if (options.maskFile) {
    form.append("mask_file", options.maskFile, "mask.png");
  } else if (options.points?.length) {
    form.append("points", JSON.stringify(options.points.map((p) => [p.x, p.y])));
    form.append("labels", JSON.stringify(options.points.map((p) => p.label)));
  } else {
    throw new Error("Select or paint a region to remove");
  }

  const res = await fetch("/api/remove-objects", { method: "POST", body: form });
  return parseRemovalResponse(res);
}

export async function autoRemovePeople(
  file: File,
  inpaintMode: InpaintMode = "ai",
  edgeSoftness = 1.0,
  mode: AiMode = "balanced"
): Promise<RemovalResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  form.append("inpaint_mode", inpaintMode);
  form.append("edge_softness", String(edgeSoftness));

  const res = await fetch("/api/auto-remove-people", { method: "POST", body: form });
  return parseRemovalResponse(res);
}

export async function autoRemoveOthers(
  file: File,
  options: {
    inpaintMode?: InpaintMode;
    edgeSoftness?: number;
    mode?: AiMode;
    lockPoint?: { x: number; y: number } | null;
  } = {}
): Promise<RemovalResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode ?? "balanced");
  form.append("inpaint_mode", options.inpaintMode ?? "ai");
  form.append("edge_softness", String(options.edgeSoftness ?? 1.0));
  if (options.lockPoint) {
    form.append("lock_x", String(options.lockPoint.x));
    form.append("lock_y", String(options.lockPoint.y));
  }
  const res = await fetch("/api/auto-remove-others", { method: "POST", body: form });
  return parseRemovalResponse(res);
}

export async function warmRemovalModels(): Promise<Record<string, unknown>> {
  const res = await fetch("/api/warm-models", { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Warm failed (${res.status})`);
  }
  return res.json();
}

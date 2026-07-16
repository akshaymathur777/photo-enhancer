import type { AiMode } from "./api";

export interface ClickPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export type SelectionMode = "smart" | "brush" | "auto-people";
export type InpaintMode = "ai" | "quick";

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

export async function removeObjects(
  file: File,
  options: {
    points?: ClickPoint[];
    maskFile?: Blob;
    inpaintMode?: InpaintMode;
    edgeSoftness?: number;
    mode?: AiMode;
  }
): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode ?? "balanced");
  form.append("inpaint_mode", options.inpaintMode ?? "ai");
  form.append("edge_softness", String(options.edgeSoftness ?? 1.0));

  if (options.maskFile) {
    form.append("mask_file", options.maskFile, "mask.png");
  } else if (options.points?.length) {
    form.append("points", JSON.stringify(options.points.map((p) => [p.x, p.y])));
    form.append("labels", JSON.stringify(options.points.map((p) => p.label)));
  } else {
    throw new Error("Select or paint a region to remove");
  }

  const res = await fetch("/api/remove-objects", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Remove failed (${res.status})`);
  }
  return res.blob();
}

export async function autoRemovePeople(
  file: File,
  inpaintMode: InpaintMode = "ai",
  edgeSoftness = 1.0,
  mode: AiMode = "balanced"
): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  form.append("inpaint_mode", inpaintMode);
  form.append("edge_softness", String(edgeSoftness));

  const res = await fetch("/api/auto-remove-people", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Auto remove failed (${res.status})`);
  }
  return res.blob();
}

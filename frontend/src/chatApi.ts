export interface ChatMessageMeta {
  tool?: string | null;
  options?: Record<string, unknown> | null;
  clarify?: boolean;
  needs_clicks?: boolean;
  pending_tool?: string;
  pending_options?: Record<string, unknown>;
}

export interface ChatMessage {
  role: "user" | "assistant" | "error";
  content: string;
  meta?: ChatMessageMeta;
  imagePreview?: string;
  applied?: boolean;
  needsClicks?: boolean;
}

export type ChatBackend = "rules" | "ollama" | "openai";

export interface ChatStatus {
  ai_enabled: boolean;
  backend: ChatBackend;
  ollama_reachable: boolean;
  ollama_model_present: boolean;
  ollama_model: string | null;
  ollama_model_configured?: string | null;
  openai_available: boolean;
}

/** Hint shown when chat is on the rule-based "Basic" engine. */
export function basicChatHint(status: ChatStatus): string | null {
  if (status.backend !== "rules") return null;
  if (!status.ollama_reachable) {
    return "For Offline AI: install Ollama (ollama.com), keep it running, then run: ollama pull llama3.2:3b";
  }
  if (!status.ollama_model_present) {
    const configured = status.ollama_model_configured ?? "llama3.2:3b";
    return `Ollama is running but no chat model is ready. Run: ollama pull ${configured} — or add OLLAMA_MODEL=<your-model> to .env next to the app.`;
  }
  return null;
}

export interface ChatResponse {
  reply: string;
  image_base64: string | null;
  tool: string | null;
  needs_clicks: boolean;
  ai_enabled: boolean;
  backend?: ChatBackend;
  suggestions?: string[];
  clarify?: boolean;
  meta?: ChatMessageMeta;
}

export async function sendChatMessage(
  file: File,
  message: string,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("message", message);
  // Send slim history for backend context (role, content, meta)
  const slim = history
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role,
      content: m.content,
      ...(m.meta ? { meta: m.meta } : {}),
    }));
  form.append("history", JSON.stringify(slim));

  const res = await fetch("/api/chat", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Chat failed (${res.status})`);
  }
  return res.json();
}

export async function fetchChatStatus(): Promise<ChatStatus> {
  const fallback: ChatStatus = {
    ai_enabled: false,
    backend: "rules",
    ollama_reachable: false,
    ollama_model_present: false,
    ollama_model: null,
    openai_available: false,
  };
  try {
    const res = await fetch("/api/chat/status");
    if (!res.ok) return fallback;
    return { ...fallback, ...(await res.json()) };
  } catch {
    return fallback;
  }
}

export function base64ToBlob(dataUrl: string): Blob {
  const [header, b64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] ?? "image/jpeg";
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export const DEFAULT_SUGGESTIONS = [
  "Make it brighter",
  "Remove the background",
  "Auto enhance",
  "Upscale 2x",
];

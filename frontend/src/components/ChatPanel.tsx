import {
  Check,
  Cloud,
  Loader2,
  MessageSquare,
  MousePointerClick,
  RotateCcw,
  Send,
  ShieldCheck,
  Wand2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  base64ToBlob,
  basicChatHint,
  ChatMessage,
  ChatStatus,
  DEFAULT_SUGGESTIONS,
  fetchChatStatus,
  sendChatMessage,
} from "../chatApi";

interface ChatPanelProps {
  file: File;
  onApplyImage: (blob: Blob) => void;
  onNeedsObjectRemoval: () => void;
  onRevert: () => void;
  className?: string;
}

export function ChatPanel({
  file,
  onApplyImage,
  onNeedsObjectRemoval,
  onRevert,
  className = "",
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatStatus, setChatStatus] = useState<ChatStatus>({
    ai_enabled: false,
    backend: "rules",
    ollama_reachable: false,
    ollama_model_present: false,
    ollama_model: null,
    openai_available: false,
  });
  const [suggestions, setSuggestions] = useState<string[]>(DEFAULT_SUGGESTIONS);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { backend, ollama_model: ollamaModel } = chatStatus;
  const basicHint = basicChatHint(chatStatus);

  const applyChatStatus = (s: ChatStatus) => setChatStatus(s);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      fetchChatStatus().then((s) => {
        if (!cancelled) applyChatStatus(s);
      });
    };
    refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    const nextHistory = [...messages, userMsg];
    setMessages(nextHistory);
    setInput("");
    setLoading(true);

    try {
      const res = await sendChatMessage(file, trimmed, messages);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: res.reply,
        meta: res.meta,
        imagePreview: res.image_base64 ?? undefined,
        applied: !!res.image_base64,
        needsClicks: res.needs_clicks,
      };
      setMessages([...nextHistory, assistantMsg]);
      setSuggestions(res.suggestions?.length ? res.suggestions : DEFAULT_SUGGESTIONS);
      if (res.backend) {
        setChatStatus((prev) => ({ ...prev, backend: res.backend! }));
      }

      if (res.needs_clicks) {
        onNeedsObjectRemoval();
      } else if (res.image_base64) {
        onApplyImage(base64ToBlob(res.image_base64));
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Chat failed";
      setMessages([
        ...nextHistory,
        { role: "error", content: msg },
      ]);
      setSuggestions(["Try again", "Make it brighter", "Remove the background"]);
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(input);
    }
  };

  return (
    <div className={`flex flex-col h-full border-l border-surface-border bg-surface-raised/40 ${className}`}>
      <div className="px-4 py-3 border-b border-surface-border flex items-center gap-2 shrink-0">
        <MessageSquare className="w-4 h-4 text-accent" />
        <span className="text-sm font-semibold">AI Chat</span>
        {backend === "ollama" ? (
          <span
            title={ollamaModel ? `Local LLM via Ollama (${ollamaModel}) — runs fully offline` : "Local LLM via Ollama — runs fully offline"}
            className="ml-auto text-[10px] uppercase tracking-wide text-green-400/90 flex items-center gap-1 px-1.5 py-0.5 rounded bg-green-500/10 border border-green-500/20"
          >
            <ShieldCheck className="w-3 h-3" /> Offline AI
          </span>
        ) : backend === "openai" ? (
          <span
            title="Cloud LLM via OpenAI"
            className="ml-auto text-[10px] uppercase tracking-wide text-sky-400/90 flex items-center gap-1 px-1.5 py-0.5 rounded bg-sky-500/10 border border-sky-500/20"
          >
            <Cloud className="w-3 h-3" /> Cloud
          </span>
        ) : (
          <span
            title="Built-in rule-based understanding — runs fully offline"
            className="ml-auto text-[10px] uppercase tracking-wide text-amber-400/80 flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20"
          >
            <Wand2 className="w-3 h-3" /> Basic
          </span>
        )}
      </div>

      {basicHint && (
        <p className="px-4 py-2 text-[11px] leading-relaxed text-amber-300/90 bg-amber-500/5 border-b border-amber-500/20 shrink-0">
          {basicHint}
        </p>
      )}

      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-xs text-gray-500 leading-relaxed">
              Describe the edit you want — I&apos;ll remember our conversation and suggest next steps.
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex flex-col gap-1.5 max-w-[95%] ${
              m.role === "user" ? "ml-auto items-end" : "items-start"
            }`}
          >
            <div
              className={`text-sm rounded-lg px-3 py-2 ${
                m.role === "user"
                  ? "bg-accent/20 text-accent-hover"
                  : m.role === "error"
                    ? "bg-red-500/10 border border-red-500/30 text-red-300"
                    : "bg-surface border border-surface-border text-gray-300"
              } ${m.needsClicks ? "border-amber-500/40" : ""}`}
            >
              {m.needsClicks && (
                <div className="flex items-center gap-1.5 text-amber-400 text-xs mb-1.5 font-medium">
                  <MousePointerClick className="w-3.5 h-3.5" />
                  Click to select objects
                </div>
              )}
              {m.content}
            </div>

            {m.imagePreview && m.role === "assistant" && (
              <div className="flex items-center gap-2">
                <img
                  src={m.imagePreview}
                  alt="Applied edit preview"
                  className="w-14 h-14 rounded-md border border-surface-border object-cover"
                />
                {m.applied && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-green-400 flex items-center gap-1">
                      <Check className="w-3.5 h-3.5" /> Applied
                    </span>
                    <button
                      type="button"
                      onClick={onRevert}
                      className="text-xs text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
                    >
                      <RotateCcw className="w-3 h-3" /> Undo
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-start gap-2">
            <div className="bg-surface border border-surface-border rounded-lg px-3 py-2.5 flex items-center gap-2">
              <div className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce [animation-delay:0ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce [animation-delay:300ms]" />
              </div>
              <span className="text-xs text-gray-500">Thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {suggestions.length > 0 && (
        <div className="px-3 pb-2 flex flex-wrap gap-1.5 shrink-0">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              disabled={loading}
              onClick={() => void send(s)}
              className="text-xs px-2.5 py-1.5 rounded-full border border-surface-border text-gray-400 hover:text-white hover:border-accent/40 disabled:opacity-40 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        className="p-3 border-t border-surface-border shrink-0"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe an edit… (Enter to send)"
            disabled={loading}
            rows={1}
            className="flex-1 bg-surface border border-surface-border rounded-lg px-3 py-2 text-sm placeholder:text-gray-600 focus:outline-none focus:border-accent/50 resize-none min-h-[38px] max-h-24"
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 96)}px`;
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="p-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-40 transition-colors shrink-0"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        <p className="text-[10px] text-gray-600 mt-1.5">Enter to send · Shift+Enter for newline</p>
      </form>
    </div>
  );
}

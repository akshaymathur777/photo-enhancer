import {
  CheckCircle2,
  Circle,
  Download,
  HardDrive,
  Loader2,
  RefreshCw,
  Sparkles,
  X,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DownloadProgress,
  fetchDownloadProgress,
  fetchModelsStatus,
  GenerativeRemovalStatus,
  ModelsStatus,
  startDownloadAll,
  startDownloadModel,
} from "../api";
import { UpscaleLimitControl } from "./UpscaleLimitControl";
import { useUpscaleLimit } from "../hooks/useUpscaleLimit";

const ONBOARDING_KEY = "photo-enhancer-onboarding-dismissed";

function formatMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function errorHint(kind?: string | null): string | null {
  switch (kind) {
    case "network":
      return "Check your internet connection, VPN, or firewall. Corporate proxies may block HuggingFace or GitHub.";
    case "permission":
      return "Do not install to Program Files if your account cannot write there. Models save to %APPDATA%\\PhotoEnhancer\\models\\.";
    case "disk":
      return "Free at least 10 GB on the drive that holds your AppData folder, then retry.";
    case "ssl":
      return "Antivirus HTTPS scanning can break downloads. Temporarily allow PhotoEnhancer.exe or disable SSL inspection.";
    default:
      return null;
  }
}

function statusIcon(status: string) {
  if (status === "present") return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />;
  if (status === "downloading") return <Loader2 className="w-4 h-4 text-accent animate-spin shrink-0" />;
  if (status === "error") return <XCircle className="w-4 h-4 text-red-400 shrink-0" />;
  return <Circle className="w-4 h-4 text-gray-600 shrink-0" />;
}

function BestQualityRemovalCard({
  generative,
  canDownload,
  onDownload,
}: {
  generative: GenerativeRemovalStatus;
  canDownload: boolean;
  onDownload: (modelId: string) => void;
}) {
  const missing = generative.models.filter((m) => !m.present);
  const headline = !generative.cuda
    ? "Best Quality object removal is unavailable on this PC"
    : generative.ready
      ? "Best Quality object removal is ready"
      : missing.length === generative.models.length
        ? "Best Quality object removal is not enabled yet"
        : "Best Quality object removal is partly enabled";
  const detail = !generative.cuda
    ? "The generative erase needs an NVIDIA GPU (CUDA). Removals fall back to the classical LaMa / PatchMatch fill, which is still a good result on many scenes."
    : generative.ready
      ? "Water and sky scenes route to SDXL inpaint, indoor and structured scenes to PowerPaint erase."
      : "Without these weights, Best Quality falls back to the classical LaMa / PatchMatch fill — no mid-edit download is started. Balanced and Fast always use the classical fill on water and indoor scenes.";
  const tone = generative.ready
    ? "border-emerald-500/20 bg-emerald-500/10"
    : "border-amber-500/25 bg-amber-500/10";
  const headlineTone = generative.ready ? "text-emerald-300" : "text-amber-300";

  return (
    <div className={`rounded-lg border px-3 py-2.5 space-y-2 ${tone}`}>
      <div className="flex items-start gap-2">
        <Sparkles className={`w-4 h-4 shrink-0 mt-0.5 ${headlineTone}`} />
        <div className="space-y-1">
          <p className={`text-sm font-medium ${headlineTone}`}>{headline}</p>
          <p className="text-xs text-gray-400 leading-relaxed">{detail}</p>
        </div>
      </div>
      <ul className="space-y-1">
        {generative.models.map((m) => (
          <li key={m.id} className="flex items-center gap-2 text-xs">
            {m.present ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            ) : (
              <Circle className="w-3.5 h-3.5 text-gray-600 shrink-0" />
            )}
            <span className="text-gray-300 truncate">{m.name}</span>
            <span className="text-gray-500 shrink-0">
              {m.scenes} · ~{(m.size_estimate_mb / 1000).toFixed(1)} GB
            </span>
            {!m.present && canDownload && (
              <button
                type="button"
                onClick={() => onDownload(m.id)}
                className="ml-auto text-accent hover:text-accent-hover shrink-0"
              >
                Download
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

interface ModelDownloadPanelProps {
  mode: "onboarding" | "settings";
  onClose?: () => void;
}

export function ModelDownloadPanel({ mode, onClose }: ModelDownloadPanelProps) {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionError, setActionError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const { maxOutputMp, setMaxOutputMp, bounds, presets } = useUpscaleLimit();

  const refresh = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([fetchModelsStatus(), fetchDownloadProgress()]);
      setStatus(s);
      setProgress(p);
      setActionError(p.error ?? null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Could not reach server");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const isActive = progress?.status === "downloading" || status?.download_in_progress;
    if (isActive) {
      pollRef.current = setInterval(() => void refresh(), 800);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [progress?.status, status?.download_in_progress, refresh]);

  const handleDownloadAll = async () => {
    setActionError(null);
    try {
      await startDownloadAll();
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Download failed to start");
    }
  };

  const handleDownloadAsNeeded = () => {
    localStorage.setItem(ONBOARDING_KEY, "1");
    onClose?.();
  };

  const handleRetry = async () => {
    setActionError(null);
    setProgress((prev) => (prev ? { ...prev, status: "idle", error: null, error_kind: null } : prev));
    await handleDownloadAll();
  };

  const overallPercent =
    progress?.status === "downloading"
      ? progress.percent
      : status?.all_present
        ? 100
        : status
          ? (status.downloaded_count / status.total_count) * 100
          : 0;

  const isOnboarding = mode === "onboarding";

  return (
    <div
      className={`${
        isOnboarding
          ? "fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          : "flex flex-col h-full"
      }`}
      role={isOnboarding ? "dialog" : undefined}
      aria-modal={isOnboarding || undefined}
    >
      <div
        className={`bg-surface-raised border border-surface-border rounded-xl shadow-2xl ${
          isOnboarding ? "max-w-lg w-full max-h-[90vh] overflow-y-auto" : "flex flex-col flex-1 min-h-0"
        }`}
      >
        <div className="flex items-start justify-between gap-3 p-5 border-b border-surface-border">
          <div>
            <h2 className="text-lg font-semibold">
              {isOnboarding ? "Welcome to Photo Enhancer" : "AI Models"}
            </h2>
            <p className="text-sm text-gray-400 mt-1 leading-relaxed">
              {isOnboarding ? (
                <>
                  AI models download once to your computer (~{status?.total_estimate_mb ?? 6}–8 GB total).
                  After that, everything works offline — your photos never leave your device.
                </>
              ) : (
                <>
                  Models download once, then work offline. Stored in your local app data folder.
                  Object removal works without the large weights below — they only change what
                  Best Quality can reach.
                </>
              )}
            </p>
          </div>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-md text-gray-400 hover:text-white hover:bg-white/5"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <div className="p-5 space-y-4">
          {loading && !status ? (
            <div className="flex items-center justify-center py-8 text-gray-400">
              <Loader2 className="w-6 h-6 animate-spin mr-2" /> Checking models…
            </div>
          ) : (
            <>
              {/* Overall progress */}
              {(progress?.status === "downloading" || (status && !status.all_present)) && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>
                      {progress?.status === "downloading"
                        ? progress.message ?? progress.current_model_name ?? "Downloading…"
                        : `${status?.downloaded_count ?? 0} of ${status?.total_count ?? 0} models ready`}
                    </span>
                    <span>{Math.round(overallPercent)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-surface overflow-hidden">
                    <div
                      className="h-full bg-accent transition-all duration-300 rounded-full"
                      style={{ width: `${Math.min(100, overallPercent)}%` }}
                    />
                  </div>
                  {progress?.bytes_total ? (
                    <p className="text-[11px] text-gray-500">
                      {formatMb(progress.bytes_downloaded)} / {formatMb(progress.bytes_total)}
                    </p>
                  ) : null}
                </div>
              )}

              {status?.all_present && progress?.status !== "downloading" && (
                <div className="flex items-center gap-2 text-sm text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  All models downloaded — ready for offline use.
                </div>
              )}

              {progress?.status === "error" && (
                <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 px-3 py-2 text-sm space-y-2">
                  <p>{progress.error ?? "Download failed. Check your internet connection and try again."}</p>
                  {errorHint(progress.error_kind) && (
                    <p className="text-xs text-red-200/80">{errorHint(progress.error_kind)}</p>
                  )}
                  {status?.logs_dir && (
                    <p className="text-[11px] text-red-200/60">
                      Details: {status.logs_dir}\download.log
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={() => void handleRetry()}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-red-200 hover:text-white"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Retry download
                  </button>
                </div>
              )}

              {actionError && progress?.status !== "error" && (
                <div className="rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 px-3 py-2 text-sm">
                  {actionError}
                </div>
              )}

              {status?.generative_removal && (
                <BestQualityRemovalCard
                  generative={status.generative_removal}
                  canDownload={progress?.status !== "downloading" && !isOnboarding}
                  onDownload={(id) => void startDownloadModel(id).then(refresh)}
                />
              )}

              {/* Disk usage */}
              {status && (
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <HardDrive className="w-3.5 h-3.5" />
                  <span>
                    Disk used: {status.disk_used_mb} MB
                    {status.disk_used_mb === 0 && !status.all_present && " (no models yet)"}
                  </span>
                </div>
              )}

              {/* Per-model checklist */}
              {status && (
                <ul className="max-h-48 overflow-y-auto space-y-1 rounded-lg border border-surface-border bg-surface/50 p-2">
                  {status.models.map((m) => (
                    <li
                      key={m.id}
                      className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm hover:bg-white/5"
                    >
                      {statusIcon(m.status)}
                      <div className="flex-1 min-w-0">
                        <div className="truncate text-gray-200">{m.name}</div>
                        <div className="text-[10px] text-gray-500 truncate">
                          {m.feature} · ~{m.size_estimate_mb} MB
                        </div>
                      </div>
                      {m.status === "missing" && progress?.status !== "downloading" && !isOnboarding && (
                        <button
                          type="button"
                          onClick={() => void startDownloadModel(m.id).then(refresh)}
                          className="text-[10px] text-accent hover:text-accent-hover shrink-0"
                        >
                          Download
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>

        {!isOnboarding && presets.length > 0 && (
          <div className="px-5 pb-4 border-t border-surface-border pt-4">
            <h3 className="text-sm font-medium text-gray-300 mb-2">Upscale output limit</h3>
            <UpscaleLimitControl
              compact
              maxOutputMp={maxOutputMp}
              onChange={setMaxOutputMp}
              bounds={bounds}
              presets={presets}
            />
          </div>
        )}

        <div className="p-5 pt-0 flex flex-col sm:flex-row gap-2">
          {isOnboarding ? (
            <>
              <button
                type="button"
                onClick={() => void handleDownloadAll()}
                disabled={progress?.status === "downloading" || status?.all_present}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-40 text-sm font-semibold"
              >
                {progress?.status === "downloading" ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Downloading…</>
                ) : (
                  <><Download className="w-4 h-4" /> Download all now</>
                )}
              </button>
              <button
                type="button"
                onClick={handleDownloadAsNeeded}
                disabled={progress?.status === "downloading"}
                className="flex-1 py-2.5 rounded-lg border border-surface-border text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40"
              >
                Download as I go
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => void handleDownloadAll()}
              disabled={progress?.status === "downloading" || status?.all_present}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-40 text-sm font-semibold"
            >
              {progress?.status === "downloading" ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Downloading all models…</>
              ) : status?.all_present ? (
                <><CheckCircle2 className="w-4 h-4" /> All models downloaded</>
              ) : (
                <><Download className="w-4 h-4" /> Download all models</>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function shouldShowOnboarding(status: ModelsStatus | null): boolean {
  if (localStorage.getItem(ONBOARDING_KEY)) return false;
  if (!status) return false;
  return status.downloaded_count === 0;
}

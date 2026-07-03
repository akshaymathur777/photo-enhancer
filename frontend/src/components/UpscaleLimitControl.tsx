import { useEffect, useState } from "react";

import { UpscalePreset } from "../api";

interface UpscaleLimitControlProps {
  maxOutputMp: number;
  onChange: (mp: number) => void;
  bounds: { min: number; max: number; default: number };
  presets: UpscalePreset[];
  compact?: boolean;
}

export function UpscaleLimitControl({
  maxOutputMp,
  onChange,
  bounds,
  presets,
  compact = false,
}: UpscaleLimitControlProps) {
  const [customMode, setCustomMode] = useState(false);
  const matchedPreset = presets.find((p) => p.megapixels === maxOutputMp);
  const selected = customMode || !matchedPreset ? "custom" : matchedPreset.id;

  useEffect(() => {
    if (matchedPreset && !customMode) return;
    if (matchedPreset) setCustomMode(false);
  }, [maxOutputMp, matchedPreset, customMode]);

  return (
    <div className={`space-y-2 ${compact ? "" : "rounded-lg border border-surface-border bg-surface/40 p-3"}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm text-gray-400">Output limit</span>
        <span className="text-xs font-mono text-gray-300">{maxOutputMp} MP</span>
      </div>
      <select
        value={selected}
        onChange={(e) => {
          const id = e.target.value;
          if (id === "custom") {
            setCustomMode(true);
            return;
          }
          setCustomMode(false);
          const preset = presets.find((p) => p.id === id);
          if (preset) onChange(preset.megapixels);
        }}
        className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm"
      >
        {presets.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label} ({p.megapixels} MP)
          </option>
        ))}
        <option value="custom">Custom…</option>
      </select>
      {(customMode || !matchedPreset) && (
        <label className="block">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Custom limit</span>
            <span>
              {bounds.min}–{bounds.max} MP
            </span>
          </div>
          <input
            type="range"
            min={bounds.min}
            max={bounds.max}
            step={10}
            value={maxOutputMp}
            onChange={(e) => onChange(parseInt(e.target.value, 10))}
            className="w-full accent-accent"
          />
        </label>
      )}
      <p className="text-[11px] text-gray-500 leading-relaxed">
        Higher limits allow larger 4× outputs but use more RAM during processing.
        Default is {bounds.default} MP for typical 16 GB systems.
      </p>
    </div>
  );
}

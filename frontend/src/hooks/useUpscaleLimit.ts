import { useCallback, useEffect, useState } from "react";
import {
  DEFAULT_MAX_OUTPUT_MP,
  fetchSettings,
  getStoredMaxOutputMp,
  MAX_MAX_OUTPUT_MP_BOUND,
  MIN_MAX_OUTPUT_MP,
  saveMaxOutputMp,
  UpscalePreset,
} from "../api";

export function useUpscaleLimit() {
  const [maxOutputMp, setMaxOutputMpState] = useState(getStoredMaxOutputMp);
  const [presets, setPresets] = useState<UpscalePreset[]>([]);
  const [bounds, setBounds] = useState({
    min: MIN_MAX_OUTPUT_MP,
    max: MAX_MAX_OUTPUT_MP_BOUND,
    default: DEFAULT_MAX_OUTPUT_MP,
  });

  useEffect(() => {
    void fetchSettings()
      .then((s) => {
        setPresets(s.presets);
        setBounds({
          min: s.min_max_output_megapixels,
          max: s.max_max_output_megapixels,
          default: s.default_max_output_megapixels,
        });
        const stored = localStorage.getItem("photo-enhancer-max-output-mp");
        if (stored === null) {
          setMaxOutputMpState(s.max_output_megapixels);
        }
      })
      .catch(() => {
        /* server not ready */
      });
  }, []);

  const setMaxOutputMp = useCallback((mp: number) => {
    const clamped = Math.min(bounds.max, Math.max(bounds.min, Math.round(mp)));
    setMaxOutputMpState(clamped);
    void saveMaxOutputMp(clamped);
  }, [bounds]);

  return { maxOutputMp, setMaxOutputMp, bounds, presets };
}

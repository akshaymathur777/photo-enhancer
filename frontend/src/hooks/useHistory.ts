import { useCallback, useRef, useState } from "react";

export interface HistoryEntry {
  blob: Blob;
  url: string;
}

export function useHistory(maxSteps = 30) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [index, setIndex] = useState(-1);
  const indexRef = useRef(-1);
  const entriesRef = useRef<HistoryEntry[]>([]);

  const push = useCallback(
    (blob: Blob) => {
      const url = URL.createObjectURL(blob);
      const entry = { blob, url };
      setEntries((prev) => {
        const trimmed = prev.slice(0, indexRef.current + 1);
        const next = [...trimmed, entry];
        while (next.length > maxSteps) {
          const removed = next.shift()!;
          URL.revokeObjectURL(removed.url);
        }
        entriesRef.current = next;
        indexRef.current = next.length - 1;
        setIndex(indexRef.current);
        return next;
      });
      return url;
    },
    [maxSteps]
  );

  const reset = useCallback((blob: Blob) => {
    entriesRef.current.forEach((e) => URL.revokeObjectURL(e.url));
    const url = URL.createObjectURL(blob);
    const entry = { blob, url };
    entriesRef.current = [entry];
    indexRef.current = 0;
    setIndex(0);
    setEntries([entry]);
    return url;
  }, []);

  const canUndo = index > 0;
  const canRedo = index >= 0 && index < entries.length - 1;

  const undo = useCallback((): HistoryEntry | null => {
    if (indexRef.current <= 0) return null;
    indexRef.current -= 1;
    setIndex(indexRef.current);
    return entriesRef.current[indexRef.current] ?? null;
  }, []);

  const redo = useCallback((): HistoryEntry | null => {
    if (indexRef.current >= entriesRef.current.length - 1) return null;
    indexRef.current += 1;
    setIndex(indexRef.current);
    return entriesRef.current[indexRef.current] ?? null;
  }, []);

  return { push, reset, undo, redo, canUndo, canRedo, index, entries };
}

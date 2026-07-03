import { useCallback, useRef, useState } from "react";

interface CompareSliderProps {
  before: string;
  after: string;
}

export function CompareSlider({ before, after }: CompareSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState(50);
  const dragging = useRef(false);

  const update = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.max(2, Math.min(98, pct)));
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    update(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (dragging.current) update(e.clientX);
  };

  const onPointerUp = () => {
    dragging.current = false;
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-video max-h-[60vh] rounded-xl overflow-hidden select-none cursor-col-resize border border-surface-border"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <img src={after} alt="After" className="absolute inset-0 w-full h-full object-contain checkerboard" />
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${position}%` }}
      >
        <img
          src={before}
          alt="Before"
          className="absolute inset-0 w-full h-full object-contain"
          style={{ width: containerRef.current?.offsetWidth ?? "100%" }}
        />
      </div>
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg"
        style={{ left: `${position}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white shadow-md flex items-center justify-center">
          <div className="flex gap-0.5">
            <div className="w-0 h-0 border-y-[5px] border-y-transparent border-r-[6px] border-r-gray-600" />
            <div className="w-0 h-0 border-y-[5px] border-y-transparent border-l-[6px] border-l-gray-600" />
          </div>
        </div>
      </div>
      <span className="absolute top-3 left-3 text-xs font-medium bg-black/60 px-2 py-1 rounded">Before</span>
      <span className="absolute top-3 right-3 text-xs font-medium bg-black/60 px-2 py-1 rounded">After</span>
    </div>
  );
}

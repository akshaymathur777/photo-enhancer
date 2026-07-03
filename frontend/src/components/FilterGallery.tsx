import { StyleInfo } from "../api";

const STYLE_COLORS: Record<string, string> = {
  cinematic: "from-teal-700 to-orange-600",
  vintage: "from-amber-800 to-yellow-700",
  vivid: "from-pink-600 to-purple-600",
  portrait: "from-rose-400 to-pink-300",
  bw_dramatic: "from-gray-700 to-gray-900",
  golden_hour: "from-orange-500 to-yellow-400",
  cool_moody: "from-blue-800 to-indigo-900",
  faded: "from-gray-400 to-gray-300",
  noir: "from-gray-900 to-black",
  sepia: "from-amber-900 to-yellow-800",
  pastel: "from-pink-200 to-purple-200",
  arctic: "from-blue-200 to-white",
  sunset: "from-orange-600 to-pink-500",
  emerald: "from-green-700 to-emerald-500",
  rose_gold: "from-rose-400 to-amber-300",
  matte: "from-gray-500 to-gray-400",
  neon_pop: "from-fuchsia-500 to-cyan-400",
  velvia: "from-green-500 to-blue-600",
  twilight: "from-purple-700 to-blue-900",
};

interface FilterGalleryProps {
  styles: StyleInfo[];
  selected: string;
  onSelect: (id: string) => void;
}

export function FilterGallery({ styles, selected, onSelect }: FilterGalleryProps) {
  return (
    <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
      {styles.map((s) => {
        const gradient = STYLE_COLORS[s.id] ?? "from-accent/60 to-purple-600/60";
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onSelect(s.id)}
            className={`text-left rounded-lg border overflow-hidden transition-colors ${
              selected === s.id
                ? "border-accent ring-1 ring-accent/50"
                : "border-surface-border hover:border-accent/30"
            }`}
          >
            <div className={`h-10 bg-gradient-to-br ${gradient}`} />
            <div className="px-2 py-1.5">
              <div className="text-xs font-medium truncate">{s.name}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

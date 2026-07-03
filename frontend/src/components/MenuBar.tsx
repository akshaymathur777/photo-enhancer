import {
  FolderOpen,
  Redo2,
  Save,
  Sparkles,
} from "lucide-react";

interface MenuBarProps {
  fileName: string | null;
  isDirty: boolean;
  hasImage: boolean;
  onOpen: () => void;
  onSave: () => void;
  onSaveAs: () => void;
  onRevert: () => void;
}

export function MenuBar({
  fileName,
  isDirty,
  hasImage,
  onOpen,
  onSave,
  onSaveAs,
  onRevert,
}: MenuBarProps) {
  return (
    <header className="border-b border-surface-border bg-surface-raised shrink-0">
      <div className="flex items-center h-12 px-4 gap-1">
        <div className="flex items-center gap-2.5 pr-4 mr-2 border-r border-surface-border">
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center">
            <Sparkles className="w-4 h-4" />
          </div>
          <span className="font-semibold text-sm hidden sm:block">Photo Enhancer</span>
        </div>

        <button
          type="button"
          onClick={onOpen}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
        >
          <FolderOpen className="w-4 h-4" />
          Open
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={!hasImage}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 hover:text-white disabled:opacity-40 transition-colors"
        >
          <Save className="w-4 h-4" />
          Save
        </button>
        <button
          type="button"
          onClick={onSaveAs}
          disabled={!hasImage}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 hover:text-white disabled:opacity-40 transition-colors"
        >
          Save As…
        </button>
        <button
          type="button"
          onClick={onRevert}
          disabled={!isDirty}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 hover:text-white disabled:opacity-40 transition-colors"
        >
          <Redo2 className="w-4 h-4" />
          Revert
        </button>

        <div className="ml-auto text-xs text-gray-500 truncate max-w-[40%]">
          {fileName ? (
            <>
              {fileName}
              {isDirty && <span className="text-accent ml-1">• edited</span>}
            </>
          ) : (
            "No image open"
          )}
        </div>
      </div>
    </header>
  );
}

import React, { useEffect, useState } from "react";
import { Command } from "cmdk";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onTogglePanel: (panelId: string) => void;
  onExecuteCommand: (command: string) => void;
  activePanels: string[];
}

const PANEL_COMMANDS = [
  { id: "chats", label: "Open Chats Panel", icon: "💬" },
  { id: "browser", label: "Open Browser Panel", icon: "🌐" },
  { id: "search", label: "Open Search Panel", icon: "🔍" },
  { id: "parliament", label: "Open Parliament Panel", icon: "🏛️" },
  { id: "code", label: "Open Code Editor", icon: "💻" },
  { id: "agent-coding", label: "Open Agent Coding Panel", icon: "🧠" },
  { id: "voice", label: "Open Voice Panel", icon: "🎤" },
  { id: "youtube", label: "Open YouTube Panel", icon: "🎵" },
  { id: "tools", label: "Open Tools Panel", icon: "🛠️" },
];

const SLASH_COMMANDS = [
  { command: "/search", label: "Enhanced Search (Web + YouTube + AI)", icon: "🔍" },
  { command: "/browser", label: "Open Browser Panel", icon: "🌐" },
  { command: "/youtube", label: "Play YouTube Video", icon: "🎵" },
  { command: "/parliament", label: "Start Parliament Vote", icon: "🏛️" },
  { command: "/think", label: "Enable Thinking Mode", icon: "🧠" },
  { command: "/code", label: "Open Code Editor", icon: "💻" },
  { command: "/clear", label: "Clear Chat", icon: "🗑️" },
];

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onTogglePanel,
  onExecuteCommand,
  activePanels,
}) => {
  const [search, setSearch] = useState("");

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (!isOpen) {
          // Will be handled by parent
        }
      }
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-2xl mx-4">
        <Command
          className="rounded-lg border border-gray-700 bg-gray-900 shadow-2xl overflow-hidden"
          shouldFilter={true}
        >
          <div className="flex items-center border-b border-gray-700 px-4">
            <span className="text-gray-400 mr-2">🔍</span>
            <Command.Input
              value={search}
              onValueChange={setSearch}
              placeholder="Type a command or search..."
              className="w-full bg-transparent text-white placeholder-gray-500 py-4 outline-none"
            />
            <kbd className="hidden sm:inline-block px-2 py-1 text-xs font-semibold text-gray-400 bg-gray-800 border border-gray-700 rounded">
              ESC
            </kbd>
          </div>

          <Command.List className="max-h-96 overflow-y-auto p-2">
            <Command.Empty className="py-6 text-center text-sm text-gray-500">
              No results found.
            </Command.Empty>

            {/* Panel Commands */}
            <Command.Group
              heading="Panels"
              className="text-xs font-semibold text-gray-400 px-2 py-2"
            >
              {PANEL_COMMANDS.map((cmd) => {
                const isActive = activePanels.includes(cmd.id);
                return (
                  <Command.Item
                    key={cmd.id}
                    value={cmd.label}
                    onSelect={() => {
                      onTogglePanel(cmd.id);
                      onClose();
                    }}
                    className="flex items-center gap-3 px-3 py-2 rounded cursor-pointer hover:bg-gray-800 data-[selected=true]:bg-gray-800 transition-colors"
                  >
                    <span className="text-xl">{cmd.icon}</span>
                    <span className="flex-1 text-white text-sm">{cmd.label}</span>
                    {isActive && (
                      <span className="text-xs text-green-400 bg-green-400/20 px-2 py-0.5 rounded">
                        Active
                      </span>
                    )}
                  </Command.Item>
                );
              })}
            </Command.Group>

            <Command.Separator className="h-px bg-gray-700 my-2" />

            {/* Slash Commands */}
            <Command.Group
              heading="Commands"
              className="text-xs font-semibold text-gray-400 px-2 py-2"
            >
              {SLASH_COMMANDS.map((cmd) => (
                <Command.Item
                  key={cmd.command}
                  value={`${cmd.label} ${cmd.command}`}
                  onSelect={() => {
                    onExecuteCommand(cmd.command);
                    onClose();
                  }}
                  className="flex items-center gap-3 px-3 py-2 rounded cursor-pointer hover:bg-gray-800 data-[selected=true]:bg-gray-800 transition-colors"
                >
                  <span className="text-xl">{cmd.icon}</span>
                  <div className="flex-1">
                    <p className="text-white text-sm">{cmd.label}</p>
                    <p className="text-gray-500 text-xs">{cmd.command}</p>
                  </div>
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
};

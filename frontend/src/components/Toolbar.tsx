import React from "react";

interface ToolbarProps {
  activePanels: string[];
  onTogglePanel: (panelId: string) => void;
}

const panelButtons = [
  { id: "chats", label: "Chats", icon: "💬" },
  { id: "browser", label: "Browser", icon: "🌐" },
  { id: "search", label: "Search", icon: "🔍" },
  { id: "parliament", label: "Parliament", icon: "🏛️" },
  { id: "code", label: "Code", icon: "💻" },
  { id: "agent-coding", label: "Agent Coding", icon: "🤖" },
  { id: "voice", label: "Voice", icon: "🎤" },
  { id: "youtube", label: "YouTube", icon: "▶️" },
  { id: "media-player", label: "Media Player", icon: "🎬" },
  { id: "tools", label: "Tools", icon: "🔧" },
];

export const Toolbar: React.FC<ToolbarProps> = ({
  activePanels,
  onTogglePanel,
}) => {
  return (
    <div className="toolbar">
      {panelButtons.map((button) => (
        <button
          key={button.id}
          className={`toolbar-button ${activePanels.includes(button.id) ? "active" : ""}`}
          onClick={() => onTogglePanel(button.id)}
          title={button.label}
        >
          {button.icon && <span className="icon">{button.icon}</span>}
          <span className="label">{button.label}</span>
        </button>
      ))}
    </div>
  );
};

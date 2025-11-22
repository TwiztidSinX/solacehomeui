import React, { useState } from "react";

export type LeftPanelMode =
  | "chats"
  | "search"
  | "parliament"
  | "browser"
  | "closed";

interface Chat {
  id: string;
  name: string;
}

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

interface ParliamentMemberUpdate {
  key?: string;
  name?: string;
  provider?: "local" | "cloud";
  model?: string;
  prompt?: string;
  enabled?: boolean;
  status?: "idle" | "working" | "done";
}

interface LeftPanelProps {
  mode: LeftPanelMode;
  onClose?: () => void;
  panelWidth?: number;

  // Chat history mode props
  chats?: Chat[];
  activeChatId?: string | null;
  onNewChat?: () => void;
  onLoadChat?: (id: string) => void;
  onDeleteChat?: (id: string) => void;
  onRenameChat?: (id: string, newName: string) => void;

  // Search results mode props
  searchResults?: SearchResult[];
  searchQuery?: string;
  onSearchResultClick?: (url: string) => void;
  onSearchResultPreview?: (url: string) => void;

  // Parliament mode props
  parliamentRoles?: Array<{
    key: string;
    name: string;
    defaultModel?: string;
    provider: "local" | "cloud";
    model: string;
    prompt: string;
    enabled: boolean;
    status: "idle" | "working" | "done";
  }>;
  parliamentRoleOutputs?: Array<any>;
  onUpdateRole?: (key: string, updates: ParliamentMemberUpdate) => void;
  debugMode?: boolean;

  // Browser/iframe mode
  browserUrl?: string;
}

const LeftPanel: React.FC<LeftPanelProps> = ({
  mode,
  onClose,
  panelWidth,
  chats = [],
  activeChatId,
  onNewChat,
  onLoadChat,
  onDeleteChat,
  onRenameChat,
  searchResults = [],
  searchQuery = "",
  onSearchResultClick,
  onSearchResultPreview,
  parliamentRoles = [],
  parliamentRoleOutputs = [],
  onUpdateRole,
  browserUrl,
  debugMode = false,
}) => {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  const handleRename = (id: string, currentName: string) => {
    setRenamingId(id);
    setNewName(currentName);
  };

  const handleRenameSubmit = (id: string) => {
    if (newName.trim() && onRenameChat) {
      onRenameChat(id, newName.trim());
    }
    setRenamingId(null);
  };

  const getPanelTitle = () => {
    switch (mode) {
      case "chats":
        return "Previous Chats";
      case "search":
        return "Search Results";
      case "parliament":
        return "AI Parliament";
      case "browser":
        return "Web Assist";
      default:
        return "";
    }
  };

  const renderContent = () => {
    switch (mode) {
      case "chats":
        return (
          <>
            {onNewChat && (
              <button
                id="new-chat-btn"
                className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300 mb-4 custom-button"
                onClick={onNewChat}
              >
                + New Chat
              </button>
            )}
            <div className="flex-grow overflow-y-auto">
              <div id="chatList">
                {chats.map((chat) => (
                  <div key={chat.id} className="group">
                    {renamingId === chat.id ? (
                      <div className="flex items-center">
                        <div className="flex items-center">
                          <label
                            htmlFor={`rename-input-${chat.id}`}
                            className="sr-only"
                          >
                            Rename chat to:
                          </label>
                          <input
                            id={`rename-input-${chat.id}`}
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            onBlur={() => handleRenameSubmit(chat.id)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter")
                                handleRenameSubmit(chat.id);
                              if (e.key === "Escape") setRenamingId(null);
                            }}
                            className="w-full bg-gray-700 text-white p-1 rounded"
                            placeholder="Enter new chat name"
                            autoFocus
                            aria-label="Rename chat"
                          />
                        </div>
                      </div>
                    ) : (
                      <button
                        className={`w-full text-left rounded px-3 py-1 mb-1 flex justify-between items-center ${
                          activeChatId === chat.id
                            ? "bg-white/20"
                            : "bg-white/10 hover:bg-white/20"
                        } custom-button`}
                        onClick={() => onLoadChat && onLoadChat(chat.id)}
                      >
                        <span className="truncate flex-1">{chat.name}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRename(chat.id, chat.name);
                          }}
                          className="opacity-0 group-hover:opacity-100 ml-1"
                        >
                          ✏️
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteChat && onDeleteChat(chat.id);
                          }}
                          className="ml-2 text-red-400 hover:text-red-600 cursor-pointer opacity-0 group-hover:opacity-100 transition-all duration-200 text-xl leading-none hover:scale-125"
                        >
                          &times;
                        </button>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        );

      case "search":
        return (
          <div className="flex-grow overflow-y-auto">
            {searchQuery && (
              <div className="mb-4 p-3 bg-blue-500/20 rounded-lg">
                <p className="text-sm text-gray-300">Search query:</p>
                <p className="text-white font-semibold">{searchQuery}</p>
              </div>
            )}

            {searchResults.length > 0 ? (
              <div className="space-y-3">
                {searchResults.map((result, index) => (
                  <div
                    key={index}
                    className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
                  >
                    <h3
                      className="text-white font-semibold mb-1 line-clamp-2 cursor-pointer"
                      onClick={() =>
                        onSearchResultPreview &&
                        onSearchResultPreview(result.url)
                      }
                    >
                      {result.title}
                    </h3>
                    <p className="text-blue-400 text-xs mb-2 truncate">
                      {result.url}
                    </p>
                    <p className="text-gray-300 text-sm line-clamp-3">
                      {result.snippet}
                    </p>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={() =>
                          onSearchResultPreview &&
                          onSearchResultPreview(result.url)
                        }
                        className="px-2 py-1 text-xs rounded bg-blue-500/80 hover:bg-blue-600 text-white"
                      >
                        Preview
                      </button>
                      {onSearchResultClick && (
                        <button
                          onClick={() => onSearchResultClick(result.url)}
                          className="px-2 py-1 text-xs rounded bg-gray-600 hover:bg-gray-500 text-white"
                        >
                          Open
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <p>No search results</p>
              </div>
            )}
          </div>
        );

      case "parliament":
        return (
          <div className="flex-grow overflow-y-auto space-y-3">
            <div className="p-3 bg-purple-500/20 rounded-lg text-sm text-gray-100">
              <p className="font-semibold">AI Parliament</p>
              <p className="text-gray-300">
                Enable roles, pick provider/model, and review structured output
                expectations.
              </p>
            </div>
            <div className="space-y-2">
              {parliamentRoles.map((role) => (
                <div
                  key={role.key}
                  className="p-3 bg-white/10 rounded-lg space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={role.enabled}
                          onChange={(e) =>
                            onUpdateRole &&
                            onUpdateRole(role.key, {
                              enabled: e.target.checked,
                            })
                          }
                          className="h-4 w-4 accent-blue-500"
                        />
                        <p className="text-white font-semibold">{role.name}</p>
                      </div>
                      <p className="text-gray-400 text-xs mt-1">
                        Provider & model
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <select
                          value={role.provider}
                          onChange={(e) =>
                            onUpdateRole &&
                            onUpdateRole(role.key, {
                              provider: e.target.value as any,
                            })
                          }
                          className="bg-gray-700 text-white text-xs rounded px-2 py-1"
                        >
                          <option value="local">Local</option>
                          <option value="cloud">Cloud</option>
                        </select>
                        <input
                          type="text"
                          value={role.model}
                          onChange={(e) =>
                            onUpdateRole &&
                            onUpdateRole(role.key, { model: e.target.value })
                          }
                          placeholder={role.defaultModel || undefined}
                          className="bg-gray-800 text-white text-xs rounded px-2 py-1 w-48"
                        />
                      </div>
                    </div>
                    <div className="text-sm">
                      {role.status === "working" && (
                        <span className="text-blue-400">● Working</span>
                      )}
                      {role.status === "done" && (
                        <span className="text-green-400">● Done</span>
                      )}
                      {role.status === "idle" && (
                        <span className="text-gray-400">● Idle</span>
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-300 text-xs mb-1">System prompt</p>
                    <textarea
                      value={role.prompt}
                      onChange={(e) =>
                        onUpdateRole &&
                        onUpdateRole(role.key, { prompt: e.target.value })
                      }
                      className="w-full bg-gray-900 text-gray-100 text-xs rounded p-2 border border-white/10"
                      rows={3}
                    />
                  </div>
                  <div className="text-xs text-gray-300 bg-black/30 border border-white/10 rounded p-2">
                    <p className="text-gray-200 font-semibold mb-1">
                      Expected JSON shape:
                    </p>
                    <pre className="whitespace-pre-wrap text-gray-300">
                      {`{
  "analysis": "",
  "strengths": "",
  "weaknesses": "",
  "proposal": "",
  "confidence": 0-100
}`}
                    </pre>
                  </div>
                  {debugMode && parliamentRoleOutputs.length > 0 && (
                    <div className="text-xs text-gray-300 bg-black/30 border border-white/10 rounded p-2">
                      <p className="text-gray-200 font-semibold mb-1">
                        Last response
                      </p>
                      <pre className="whitespace-pre-wrap text-gray-300 max-h-32 overflow-auto">
                        {parliamentRoleOutputs.find((o) => o.key === role.key)
                          ?.response || "—"}
                      </pre>
                      <div className="flex justify-end mt-1">
                        <button
                          className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
                          onClick={() => {
                            const resp = parliamentRoleOutputs.find(
                              (o) => o.key === role.key,
                            )?.response;
                            if (resp) navigator.clipboard.writeText(resp);
                          }}
                        >
                          Copy
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (mode === "closed") {
    return null;
  }

  if (mode === "browser") {
    return (
      <div
        className="h-full bg-gray-800/90 backdrop-blur-sm flex flex-col panel-surface"
        style={panelWidth ? { width: panelWidth } : { width: 360 }}
      >
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <h2 className="text-lg font-semibold text-white">
            {getPanelTitle()}
          </h2>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded-md hover:bg-white/20 text-white text-xl leading-none"
            >
              &times;
            </button>
          )}
        </div>
        <div className="flex-1 min-h-0 bg-black/60">
          {browserUrl ? (
            <iframe
              title="Web view"
              src={browserUrl}
              className="w-full h-full border-0"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              <p>No page loaded</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-full bg-gray-800/80 backdrop-blur-sm p-4 flex flex-col transition-transform duration-300 translate-x-0 panel-surface`}
      style={panelWidth ? { width: panelWidth } : undefined}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">{getPanelTitle()}</h2>
        {onClose && mode !== "chats" && (
          <button
            onClick={onClose}
            className="p-2 rounded-md hover:bg-white/20 text-white text-xl leading-none"
          >
            &times;
          </button>
        )}
      </div>
      {renderContent()}
    </div>
  );
};

export default LeftPanel;

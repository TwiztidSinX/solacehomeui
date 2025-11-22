import React, { useRef, useEffect } from "react";
import { VoiceVisualizer } from "../VoiceVisualizer";
import { Socket } from "socket.io-client";
import MediaBrowser from "./MediaBrowser";
import MonacoCodeEditor from "./MonacoCodeEditor";
import FileBrowser from "./FileBrowser";
import { type OpenFile, type FileNode } from "../types/files";

export type RightPanelMode =
  | "voice"
  | "youtube"
  | "code"
  | "media"
  | "tools"
  | "closed";

interface RightPanelProps {
  mode: RightPanelMode;
  onClose: () => void;
  panelWidth?: number;

  // Voice mode props
  socket: Socket | null;
  isHandsFreeMode?: boolean;
  onHandsFreeModeChange?: (isHandsFree: boolean) => void;
  onVisualizerReady?: (visualizer: VoiceVisualizer) => void;

  // YouTube mode props
  youtubeVideoId?: string;
  youtubeUrl?: string;

  // Code editor mode props
  codeContent?: string;
  codeLanguage?: string;
  onCodeChange?: (code: string) => void;
  onCodeLanguageChange?: (language: string) => void;
  fileTree?: FileNode[];
  onRefreshFileTree?: () => void;
  onOpenFileFromTree?: (path: string) => void;
  openFiles?: OpenFile[];
  activeFilePath?: string | null;
  onSelectOpenFile?: (path: string) => void;
  onCloseFile?: (path: string) => void;
  onSaveActiveFile?: () => void;
  isLoadingFileTree?: boolean;
  onCreateFile?: (path: string) => void;
  onCreateFolder?: (path: string) => void;
  onRenameFile?: (oldPath: string, newPath: string) => void;
  onDeleteFile?: (path: string) => void;
  onChangeWorkspace?: () => void;
  toolQuery?: string;
  onToolQueryChange?: (value: string) => void;
  onRunToolSearch?: () => void;
  toolResults?: Array<{ title: string; url: string; snippet: string }>;
  toolLoading?: boolean;
  toolError?: string | null;
  toolList?: Array<{ name: string; description?: string; parameters?: any }>;
  selectedTool?: string;
  onSelectTool?: (value: string) => void;
  toolArgsText?: string;
  onToolArgsChange?: (value: string) => void;
  toolArgsForm?: Record<string, any>;
  onToolFieldChange?: (key: string, value: any) => void;
  onResetToolForm?: () => void;
  toolSchema?: any;
  onRunToolCall?: () => void;
  toolCallResult?: string | null;

  // Media browser props
  mediaBrowserQuery?: string;
  onPlayMedia?: (mediaId: string, mediaType: string) => void;
  novaSettings?: {
    mediaServerUrl?: string;
    mediaServerApiKey?: string;
  };
}

const RightPanel: React.FC<RightPanelProps> = ({
  mode,
  onClose,
  panelWidth,
  socket,
  isHandsFreeMode = false,
  onHandsFreeModeChange,
  onVisualizerReady,
  youtubeVideoId,
  youtubeUrl,
  codeContent = "",
  codeLanguage = "javascript",
  onCodeChange,
  onCodeLanguageChange,
  fileTree = [],
  onRefreshFileTree,
  onOpenFileFromTree,
  openFiles = [],
  activeFilePath = null,
  onSelectOpenFile,
  onCloseFile,
  onSaveActiveFile,
  isLoadingFileTree = false,
  onCreateFile,
  onCreateFolder,
  onRenameFile,
  onDeleteFile,
  onChangeWorkspace,
  toolQuery = "",
  onToolQueryChange,
  onRunToolSearch,
  toolResults = [],
  toolLoading = false,
  toolError = null,
  toolList = [],
  selectedTool = "",
  onSelectTool,
  toolArgsText = "{}",
  onToolArgsChange,
  toolArgsForm = {},
  onToolFieldChange,
  onResetToolForm,
  toolSchema,
  onRunToolCall,
  toolCallResult = null,
  mediaBrowserQuery = "",
  onPlayMedia,
  novaSettings,
}) => {
  // Voice mode refs
  const visualizerRef = useRef<VoiceVisualizer | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const logoRef = useRef<HTMLImageElement>(null);
  const dotsRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const statusLabelRef = useRef<HTMLParagraphElement>(null);

  const [isRecording, setIsRecording] = React.useState(false);

  const toolProperties = toolSchema?.parameters?.properties || {};
  const toolRequired: string[] = toolSchema?.parameters?.required || [];

  const renderToolField = (name: string, schema: any) => {
    const required = toolRequired.includes(name);
    const value = toolArgsForm ? toolArgsForm[name] : undefined;
    const label = (
      <div className="flex items-center justify-between text-xs text-gray-300">
        <span className="text-white font-semibold">
          {name}
          {required ? " *" : ""}
        </span>
        {schema?.type && <span className="text-gray-400">{schema.type}</span>}
      </div>
    );
    const description = schema?.description ? (
      <p className="text-[11px] text-gray-400 mb-1">{schema.description}</p>
    ) : null;
    const handleValue = (val: any) => {
      if (onToolFieldChange) onToolFieldChange(name, val);
    };

    if (schema?.enum) {
      return (
        <div>
          {label}
          {description}
          <select
            value={value ?? ""}
            onChange={(e) => handleValue(e.target.value)}
            className="w-full bg-gray-900 text-white text-sm rounded px-2 py-2 border border-white/10"
          >
            <option value="">-- choose --</option>
            {schema.enum.map((option: any) => (
              <option key={String(option)} value={option}>
                {String(option)}
              </option>
            ))}
          </select>
        </div>
      );
    }

    switch (schema?.type) {
      case "boolean":
        return (
          <label className="flex items-center gap-2 text-sm text-white">
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => handleValue(e.target.checked)}
              className="accent-blue-500 h-4 w-4"
            />
            <span>
              {name}
              {required ? " *" : ""}
            </span>
          </label>
        );
      case "integer":
      case "number":
        return (
          <div>
            {label}
            {description}
            <input
              type="number"
              value={value === undefined ? "" : value}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") {
                  handleValue("");
                  return;
                }
                const parsed =
                  schema.type === "integer"
                    ? parseInt(raw, 10)
                    : parseFloat(raw);
                if (!Number.isNaN(parsed)) {
                  handleValue(parsed);
                }
              }}
              className="w-full bg-gray-900 text-white text-sm rounded px-2 py-2 border border-white/10"
            />
          </div>
        );
      case "array":
        return (
          <div>
            {label}
            {description}
            <textarea
              value={Array.isArray(value) ? value.join("\n") : ""}
              onChange={(e) =>
                handleValue(
                  e.target.value
                    .split("\n")
                    .filter((line) => line.trim().length > 0),
                )
              }
              rows={3}
              className="w-full bg-gray-900 text-gray-100 text-xs rounded p-2 border border-white/10"
              placeholder="One item per line"
            />
          </div>
        );
      case "object":
        return (
          <div>
            {label}
            {description}
            <textarea
              value={JSON.stringify(value ?? {}, null, 2)}
              onChange={(e) => {
                try {
                  const parsed = e.target.value
                    ? JSON.parse(e.target.value)
                    : {};
                  handleValue(parsed);
                } catch {
                  // ignore parse errors for field-level edits
                }
              }}
              rows={4}
              className="w-full bg-gray-900 text-gray-100 text-xs rounded p-2 border border-white/10"
              placeholder="{ }"
            />
          </div>
        );
      default:
        return (
          <div>
            {label}
            {description}
            <input
              type="text"
              value={value ?? ""}
              onChange={(e) => handleValue(e.target.value)}
              className="w-full bg-gray-900 text-white text-sm rounded px-2 py-2 border border-white/10"
            />
          </div>
        );
    }
  };
  const [status, setStatus] = React.useState("Ready");

  // Voice mode setup
  useEffect(() => {
    if (
      mode === "voice" &&
      logoRef.current &&
      dotsRef.current &&
      canvasRef.current &&
      statusLabelRef.current
    ) {
      const visualizer = new VoiceVisualizer(
        logoRef.current,
        dotsRef.current,
        canvasRef.current,
        statusLabelRef.current,
      );
      visualizerRef.current = visualizer;
      if (onVisualizerReady) {
        onVisualizerReady(visualizer);
      }
    }

    return () => {
      visualizerRef.current?.stop();
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state === "recording"
      ) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [mode, onVisualizerReady]);

  const startRecording = async () => {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/wav",
        });
        if (socket) {
          socket.emit("transcribe", { audio: audioBlob });
          setStatus("Transcribing...");
        }
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus("Listening...");
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setStatus("Mic access denied");
    }
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const getPanelTitle = () => {
    switch (mode) {
      case "voice":
        return "Voice Mode";
      case "youtube":
        return "YouTube Player";
      case "code":
        return "Code Editor";
      case "media":
        return "Media Browser";
      case "tools":
        return "Tools";
      default:
        return "";
    }
  };

  const getYoutubeEmbedUrl = () => {
    if (youtubeVideoId) {
      return `https://www.youtube.com/embed/${youtubeVideoId}`;
    }
    if (youtubeUrl) {
      // Extract video ID from various YouTube URL formats
      const videoIdMatch = youtubeUrl.match(
        /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/,
      );
      if (videoIdMatch && videoIdMatch[1]) {
        return `https://www.youtube.com/embed/${videoIdMatch[1]}`;
      }
    }
    return "";
  };

  const renderContent = () => {
    switch (mode) {
      case "voice":
        return (
          <div className="flex-grow flex flex-col items-center justify-center space-y-4">
            <div
              id="visualizer-container"
              className="w-48 h-48 flex items-center justify-center"
            >
              <img
                id="voice-logo"
                ref={logoRef}
                src="/nova-logo.png"
                alt="Logo"
                className="w-full h-full object-contain"
              />
              <canvas
                id="waveform-canvas"
                ref={canvasRef}
                className="hidden w-full h-full"
              ></canvas>
              <div id="processing-dots" ref={dotsRef} className="hidden">
                <div></div>
                <div></div>
                <div></div>
              </div>
            </div>

            {onHandsFreeModeChange && (
              <button
                onClick={() => onHandsFreeModeChange(!isHandsFreeMode)}
                className={`mb-4 px-4 py-2 rounded-lg text-white font-semibold transition-colors duration-200 ${
                  isHandsFreeMode
                    ? "bg-blue-500 hover:bg-blue-600"
                    : "bg-gray-700 hover:bg-gray-600"
                }`}
              >
                Hands-Free: {isHandsFreeMode ? "On" : "Off"}
              </button>
            )}

            <button
              onClick={toggleRecording}
              className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors duration-300 ${
                isRecording
                  ? "bg-red-500 hover:bg-red-600"
                  : "bg-blue-500 hover:bg-blue-600"
              }`}
            >
              <svg
                className="w-10 h-10 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path>
              </svg>
            </button>

            <p
              id="voice-status-label"
              ref={statusLabelRef}
              className="text-lg text-gray-300 h-8"
            >
              {status}
            </p>
          </div>
        );

      case "youtube":
        const embedUrl = getYoutubeEmbedUrl();
        return (
          <div className="flex-grow flex flex-col">
            {embedUrl ? (
              <iframe
                src={embedUrl}
                className="w-full h-full rounded-lg"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title="YouTube video player"
              />
            ) : (
              <div className="flex-grow flex items-center justify-center text-gray-400">
                <p>No video loaded</p>
              </div>
            )}
          </div>
        );

      case "code":
        return (
          <div className="flex-grow flex flex-col h-full">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm text-gray-400">
                Language: {codeLanguage}
              </span>
              <div className="flex space-x-2">
                {onChangeWorkspace && (
                  <button
                    onClick={onChangeWorkspace}
                    className="px-3 py-1 text-sm bg-purple-600 hover:bg-purple-700 rounded-md transition-colors"
                  >
                    Change root
                  </button>
                )}
                <select
                  value={codeLanguage}
                  onChange={(e) => {
                    if (onCodeLanguageChange) {
                      onCodeLanguageChange(e.target.value);
                    }
                  }}
                  className="bg-gray-700 text-white text-sm rounded px-2 py-1"
                >
                  <option value="javascript">JavaScript</option>
                  <option value="typescript">TypeScript</option>
                  <option value="python">Python</option>
                  <option value="html">HTML</option>
                  <option value="css">CSS</option>
                  <option value="json">JSON</option>
                  <option value="markdown">Markdown</option>
                  <option value="sql">SQL</option>
                  <option value="shell">Shell</option>
                </select>
                <button
                  onClick={() =>
                    navigator.clipboard.writeText(codeContent || "")
                  }
                  className="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 rounded-md transition-colors"
                >
                  Copy
                </button>
                <button
                  onClick={() => onSaveActiveFile && onSaveActiveFile()}
                  className="px-3 py-1 text-sm bg-green-500 hover:bg-green-600 rounded-md transition-colors"
                >
                  Save
                </button>
                <button
                  onClick={() =>
                    activeFilePath && onCloseFile && onCloseFile(activeFilePath)
                  }
                  className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded-md transition-colors"
                >
                  Close file
                </button>
                <button
                  onClick={() => {
                    if (activeFilePath && onRenameFile) {
                      const newName = prompt(
                        "Rename to (relative path):",
                        activeFilePath,
                      );
                      if (newName && newName !== activeFilePath) {
                        onRenameFile(activeFilePath, newName);
                      }
                    }
                  }}
                  className="px-3 py-1 text-sm bg-purple-600 hover:bg-purple-700 rounded-md transition-colors"
                >
                  Rename
                </button>
                <button
                  onClick={() => {
                    if (activeFilePath && onDeleteFile) {
                      const confirmed = confirm(`Delete ${activeFilePath}?`);
                      if (confirmed) onDeleteFile(activeFilePath);
                    }
                  }}
                  className="px-3 py-1 text-sm bg-red-600 hover:bg-red-700 rounded-md transition-colors"
                >
                  Delete
                </button>
              </div>
            </div>
            <div className="flex flex-1 min-h-0 gap-3">
              <div className="w-64 shrink-0">
                <FileBrowser
                  tree={fileTree}
                  onRefresh={onRefreshFileTree || (() => {})}
                  onOpenFile={onOpenFileFromTree || (() => {})}
                  activeFilePath={activeFilePath}
                  isLoading={isLoadingFileTree}
                  onCreateFile={onCreateFile}
                  onCreateFolder={onCreateFolder}
                  onRename={onRenameFile}
                  onDelete={onDeleteFile}
                />
              </div>
              <div className="flex-1 flex flex-col min-w-0">
                <div className="flex items-center gap-2 mb-2 overflow-x-auto">
                  {openFiles.length === 0 ? (
                    <span className="text-gray-400 text-sm">No file open</span>
                  ) : (
                    openFiles.map((file) => (
                      <button
                        key={file.path}
                        onClick={() =>
                          onSelectOpenFile && onSelectOpenFile(file.path)
                        }
                        className={`px-2 py-1 rounded-md text-xs whitespace-nowrap ${
                          activeFilePath === file.path
                            ? "bg-blue-600 text-white"
                            : "bg-gray-700 text-gray-200 hover:bg-gray-600"
                        }`}
                      >
                        {file.path.split(/[/\\\\]/).pop()}
                      </button>
                    ))
                  )}
                </div>
                <div className="flex-grow h-96">
                  <MonacoCodeEditor
                    code={codeContent || ""}
                    language={codeLanguage}
                    onChange={onCodeChange || (() => {})}
                  />
                </div>
              </div>
            </div>
          </div>
        );

      case "media":
        if (onPlayMedia && novaSettings) {
          return (
            <MediaBrowser
              query={mediaBrowserQuery}
              onPlay={onPlayMedia}
              onClose={onClose}
              settings={novaSettings}
            />
          );
        }
        return (
          <div className="flex-grow flex items-center justify-center text-gray-400">
            <p>Media browser not configured</p>
          </div>
        );

      case "tools":
        return (
          <div className="flex-grow flex flex-col h-full">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="pill">Tool Runner</span>
                <span className="text-xs text-gray-400">
                  Search, pick, and execute
                </span>
              </div>
              <span className="text-[11px] text-gray-500">Live</span>
            </div>
            <div className="mb-3 flex items-center gap-2">
              <input
                type="text"
                value={toolQuery}
                onChange={(e) =>
                  onToolQueryChange && onToolQueryChange(e.target.value)
                }
                placeholder="Search (SearXNG tool)"
                className="flex-1 bg-gray-800 text-white text-sm rounded px-3 py-2 border border-white/10"
              />
              <button
                onClick={onRunToolSearch}
                className="px-3 py-2 text-sm bg-blue-500 hover:bg-blue-600 rounded-md transition-colors text-white"
                disabled={toolLoading}
              >
                {toolLoading ? "Searching..." : "Run"}
              </button>
            </div>
            {toolError && (
              <div className="mb-2 text-sm text-red-400">{toolError}</div>
            )}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
              <div className="flex flex-col gap-2">
                <div>
                  <label className="text-xs text-gray-300">Select tool</label>
                  <select
                    value={selectedTool}
                    onChange={(e) =>
                      onSelectTool && onSelectTool(e.target.value)
                    }
                    className="w-full bg-gray-800 text-white text-sm rounded px-2 py-2 border border-white/10"
                  >
                    <option value="">-- Choose a tool --</option>
                    {toolList.map((t) => (
                      <option key={t.name} value={t.name}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                  {selectedTool && (
                    <div className="mt-1 text-xs text-gray-400 space-y-1">
                      <p>
                        {toolList.find((t) => t.name === selectedTool)
                          ?.description || ""}
                      </p>
                      {toolSchema?.parameters && (
                        <pre className="bg-black/30 border border-white/10 rounded p-2 text-[11px] text-gray-200 whitespace-pre-wrap">
                          {JSON.stringify(toolSchema.parameters, null, 2)}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-300">
                      Arguments (form)
                    </label>
                    {onResetToolForm && (
                      <button
                        className="text-[11px] px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
                        onClick={onResetToolForm}
                      >
                        Reset
                      </button>
                    )}
                  </div>
                  {selectedTool && Object.keys(toolProperties).length > 0 ? (
                    <div className="space-y-2 bg-black/20 border border-white/10 rounded p-2 max-h-72 overflow-auto">
                      {Object.entries(toolProperties).map(([name, schema]) => (
                        <div
                          key={name}
                          className="p-2 rounded bg-white/5 border border-white/5"
                        >
                          {renderToolField(name, schema)}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-gray-400">
                      No parameters for this tool.
                    </p>
                  )}
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-xs text-gray-300">
                  Arguments (JSON)
                </label>
                <textarea
                  value={toolArgsText}
                  onChange={(e) =>
                    onToolArgsChange && onToolArgsChange(e.target.value)
                  }
                  rows={selectedTool ? 12 : 8}
                  className="w-full bg-gray-900 text-gray-100 text-xs rounded p-2 border border-white/10 flex-1"
                  placeholder='{"query": "example"}'
                />
                <div className="flex justify-between items-center">
                  <div className="text-[11px] text-gray-400">
                    {selectedTool
                      ? selectedTool
                      : "Select a tool to enable run"}
                  </div>
                  <button
                    onClick={onRunToolCall}
                    className="px-3 py-2 text-sm bg-purple-600 hover:bg-purple-500 rounded-md transition-colors text-white"
                  >
                    Run Tool
                  </button>
                </div>
                {toolCallResult && (
                  <div className="text-xs text-gray-200 bg-black/30 border border-white/10 rounded p-2 whitespace-pre-wrap">
                    {toolCallResult}
                  </div>
                )}
              </div>
            </div>
            <div className="flex-grow overflow-auto space-y-2 border-t border-white/10 pt-3">
              {toolResults.length === 0 && !toolLoading ? (
                <p className="text-gray-400 text-sm">No results</p>
              ) : (
                toolResults.map((r, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-white/10 rounded-lg space-y-1"
                  >
                    <div className="text-white font-semibold text-sm">
                      {r.title}
                    </div>
                    <div className="text-xs text-blue-300 truncate">
                      {r.url}
                    </div>
                    <div className="text-gray-300 text-sm line-clamp-3">
                      {r.snippet}
                    </div>
                    <div className="flex gap-2 pt-1">
                      <button
                        className="text-xs px-2 py-1 rounded bg-blue-500/80 hover:bg-blue-600 text-white"
                        onClick={() => window.open(r.url, "_blank")}
                      >
                        Open
                      </button>
                      <button
                        className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
                        onClick={() =>
                          navigator.clipboard.writeText(
                            `${r.title}\n${r.url}\n${r.snippet}`,
                          )
                        }
                      >
                        Copy
                      </button>
                      <button
                        className="text-xs px-2 py-1 rounded bg-purple-600 hover:bg-purple-500 text-white"
                        onClick={() =>
                          navigator.clipboard.writeText(JSON.stringify(r))
                        }
                      >
                        Copy JSON
                      </button>
                    </div>
                  </div>
                ))
              )}
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

  // Special handling for media browser (full overlay)
  if (mode === "media") {
    return renderContent();
  }

  return (
    <div
      className={`flex-shrink-0 h-full bg-gray-800/80 backdrop-blur-sm p-4 flex flex-col transform transition-transform duration-300 ease-in-out z-20 translate-x-0 panel-surface`}
      style={
        panelWidth
          ? { width: panelWidth }
          : { width: mode === "code" ? 950 : 384 }
      }
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">{getPanelTitle()}</h2>
        <button
          onClick={onClose}
          className="p-2 rounded-md hover:bg-white/20 text-white text-2xl leading-none"
        >
          &times;
        </button>
      </div>
      {renderContent()}
    </div>
  );
};

export default RightPanel;

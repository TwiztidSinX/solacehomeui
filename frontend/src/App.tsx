import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import io from "socket.io-client";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import ChatView from "./components/ChatView";
import MessageInput from "./components/MessageInput";
import SettingsPanel from "./components/SettingsPanel";
import MemoryGraph from "./components/MemoryGraph";
import { type Message } from "./types";
import { VoiceVisualizer } from "./VoiceVisualizer";
import ImageGenerator from "./components/ImageGenerator";
import AgentCodingPanel from "./components/AgentCodingPanel";
import { PanelManager } from "./components/PanelManager";
import { Toolbar } from "./components/Toolbar";
import { CommandPalette } from "./components/CommandPalette";
import { MiniPlayer } from "./components/MiniPlayer";
import toast, { Toaster } from "react-hot-toast";
import {
  PARLIAMENT_ROLES,
  type ChatSession,
  type ParliamentRoleConfig,
} from "./constants/parliament";
import { detectProviderFromModel } from "./utils/models";
import { detectLanguageFromPath } from "./utils/fileUtils";
import usePanels from "./hooks/usePanels";
import useTheme from "./hooks/useTheme";
import useCodePanel from "./hooks/useCodePanel";
import useTools from "./hooks/useTools";
import { type FileNode } from "./types/files";
import UsagePanel from "./components/UsagePanel";
import { invoke } from "@tauri-apps/api/core";
interface ModelConfig {
  [key: string]: any;
}
interface GraphData {
  nodes: any[];
  edges: any[];
}

const App: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    leftPanelMode,
    setLeftPanelMode,
    rightPanelMode,
    setRightPanelMode,
  } = usePanels();
  const { theme, handleThemeChange } = useTheme();
  const socketRef = useRef<any>(null);
  const streamingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const inThinkingMode = useRef(false);
  const tokenHistory = useRef("");
  const ttsBufferRef = useRef(""); // Buffer for sentence-based TTS
  const audioQueueRef = useRef<Blob[]>([]); // Queue for incoming audio chunks
  const isPlayingAudioRef = useRef(false); // Flag to prevent concurrent playback
  const lastParliamentUserMessage = useRef<string | null>(null);

  const [
    {
      codeContent,
      codeLanguage,
      fileTree,
      workspaceRoot,
      isLoadingFileTree,
      openFiles,
      activeFilePath,
    },
    {
      requestFileTree,
      handleFileContent,
      openFileFromTree,
      closeOpenFile,
      saveActiveFile,
      selectOpenFile,
      handleCodeEditorChange,
      createFile,
      createFolder,
      renamePath,
      deletePath,
      requestWorkspaceChange,
      openLocalFile,
      setWorkspaceRoot,
      setFileTree,
      setIsLoadingFileTree,
      setCodeContent,
      setCodeLanguage,
    },
  ] = useCodePanel(socketRef);

  const {
    toolQuery,
    setToolQuery,
    toolResults,
    toolLoading,
    toolError,
    toolList,
    selectedTool,
    setSelectedTool,
    toolArgsText,
    setToolArgsText,
    toolCallResult,
    runToolSearch,
    fetchToolList,
    runToolCall,
  } = useTools();

  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [browserUrl, setBrowserUrl] = useState("");
  const [youtubeVideoId, setYoutubeVideoId] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [miniPlayerVideoId, setMiniPlayerVideoId] = useState("");
  const [parliamentRoles, setParliamentRoles] =
    useState<ParliamentRoleConfig[]>(PARLIAMENT_ROLES);
  const [parliamentRoleOutputs, setParliamentRoleOutputs] = useState<any[]>([]);
  const [parliamentVoteResult, setParliamentVoteResult] = useState<any>(null);
  const [parliamentExpanded, setParliamentExpanded] = useState<
    Record<string, boolean>
  >({});
  const [parliamentPanelCollapsed, setParliamentPanelCollapsed] =
    useState<boolean>(false);
  const [showUsagePanel, setShowUsagePanel] = useState(false);
  const [activePanels, setActivePanels] = useState(["chat", "chats", "code"]);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const leftPanelIds = ["chats", "browser", "search", "parliament"];

  // Chat State
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [imageToSend, setImageToSend] = useState<string | null>(null);
  const [imageMimeToSend, setImageMimeToSend] = useState<string>("image/png");
  const [pendingAttachments, setPendingAttachments] = useState<
    { name: string; content: string }[]
  >([]);
  const [isAgentMode, setIsAgentMode] = useState<boolean>(false);
  const [messageInputText, setMessageInputText] = useState(""); // New state for the input
  const [isHandsFreeMode, setIsHandsFreeMode] = useState(false); // State for speech-to-speech
  const [showImageGenerator, setShowImageGenerator] = useState(false);
  const [imageGenerationPrompt, setImageGenerationPrompt] = useState("");
  const [mediaBrowserQuery, setMediaBrowserQuery] = useState("");

  // Settings State
  const [allConfigs, setAllConfigs] = useState<{ [key: string]: ModelConfig }>(
    {},
  );
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [currentBackend, setCurrentBackend] = useState("llama.cpp");
  const [selectedModel, setSelectedModel] = useState("");
  const [apiProvider, setApiProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [modelConfigOptions, setModelConfigOptions] = useState<ModelConfig>({});
  const [ollamaKvCache, setOllamaKvCache] = useState("f16");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [userName, setUserName] = useState("User");
  const [aiName, setAiName] = useState("Nova");
  const [userAvatar, setUserAvatar] = useState<string | null>(null);
  const [aiAvatar, setAiAvatar] = useState<string | null>(null);
  const [toolSettings, setToolSettings] = useState({
    n8nUrl: "",
    searXngUrl: "",
    serpApiApiKey: "",
  });
  const [novaSettings, setNovaSettings] = useState({
    searxngUrl: "",
    mediaServerUrl: "",
    mediaServerApiKey: "",
    imageGenUrl: "",
    aiName: "Nova", // Add aiName here
  });
  const [ttsSettings, setTtsSettings] = useState({
    type: "local", // 'local' or 'cloud'
    provider: "openai",
    url: "",
    apiKey: "",
    model: "",
    voice: "",
  });
  const [sttSettings, setSttSettings] = useState({
    type: "local", // 'local' or 'cloud'
    provider: "openai",
    url: "",
    apiKey: "",
    model: "",
  });
  const [debugMode, setDebugMode] = useState(false);
  const triggerBrowserSync = useCallback(() => {
    window.dispatchEvent(new CustomEvent("solace-browser-sync"));
  }, []);

  const handleNovaSettingsChange = (settings: any) => {
    setNovaSettings(settings);
  };

  const isTauriEnv =
    typeof window !== "undefined" &&
    Boolean(
      (window as any).__TAURI__ ||
        (window as any).__TAURI_IPC__ ||
        (window as any).__TAURI_METADATA__ ||
        (window as any).__TAURI_INTERNALS__ ||
        (navigator.userAgent || "").toLowerCase().includes("tauri"),
    );

  const toggleParliamentRole = (key: string) => {
    setParliamentExpanded((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const formatParliamentPayload = (payload: string) => {
    if (!payload) return "";
    try {
      const parsed = JSON.parse(payload);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return payload;
    }
  };

  const updateParliamentRole = useCallback(
    (key: string, updates: Partial<ParliamentRoleConfig>) => {
      setParliamentRoles((prev) =>
        prev.map((r) => (r.key === key ? { ...r, ...updates } : r)),
      );
    },
    [],
  );

  const runBrowserControl = useCallback(
    async (payload: any) => {
      if (!isTauriEnv) {
        return {
          success: false,
          error: "Browser control only available in Tauri (embedded webview).",
        };
      }

      const { action, selector = "", value = "", code = "" } = payload || {};
      const esc = (s: string) => s.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$");

      let script = "";
      switch (action) {
        case "click":
          script = `
            (() => {
              const el = document.querySelector(\`${esc(selector)}\`);
              if (!el) return;
              el.click();
            })();
          `;
          break;
        case "fill":
          script = `
            (() => {
              const el = document.querySelector(\`${esc(selector)}\`);
              if (!el) return;
              el.value = \`${esc(value)}\`;
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            })();
          `;
          break;
        case "scroll":
          script = `
            (() => {
              const el = document.querySelector(\`${esc(selector)}\`);
              if (!el) return;
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            })();
          `;
          break;
        case "exec":
          script = `
            (() => { ${code} })();
          `;
          break;
        default:
          return { success: false, error: `Unknown action: ${action}` };
      }

      try {
        await invoke("agent_execute_js", {
          label: "solace-browser",
          script,
        });
        return { success: true, message: "Dispatched to embedded webview." };
      } catch (err) {
        console.error("browser control failed", err);
        return {
          success: false,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    },
    [isTauriEnv],
  );

  const handleReloadOrchestrator = () => {
    socketRef.current.emit("reload_orchestrator", {});

    // Listen for response
    socketRef.current.once("orchestrator_reloaded", (data: any) => {
      if (data.success) {
        alert(`✅ Orchestrator reloaded: ${data.config.type} mode`);
      } else {
        alert(`❌ Failed to reload orchestrator: ${data.error}`);
      }
    });
  };
  const handleSaveNovaSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit("save_nova_settings", novaSettings);
  };

  useEffect(() => {
    if (
      rightPanelMode === "code" &&
      fileTree.length === 0 &&
      !isLoadingFileTree
    ) {
      requestFileTree(".");
    }
  }, [rightPanelMode, fileTree.length, isLoadingFileTree, requestFileTree]);

  // Command Palette keyboard listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen(prev => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Graph State
  const [graphData, setGraphData] = useState<GraphData | null>(null);

  // Voice Visualizer Ref
  const visualizerRef = useRef<VoiceVisualizer | null>(null);

  const addMessage = useCallback(
    (
      sender: string,
      message: string,
      type: Message["type"] = "ai",
      imageB64: string | null = null,
    ) => {
      const newMessage: Message = { sender, message, type, imageB64 };
      setMessages((prev) => [...prev, newMessage]);
    },
    [setMessages],
  ); // <--- Add setMessages to dependency array

  const applyConfigForModel = useCallback(
    (modelPath: string, configs: { [key: string]: ModelConfig }) => {
      console.log(`[DEBUG] Applying config for model: ${modelPath}`);
      if (modelPath && configs[modelPath]) {
        const config = configs[modelPath];
        const newName = config.aiName || "Nova";
        console.log(`[DEBUG] Found config. Setting aiName to: ${newName}`);
        setAiName(newName);
        setAiAvatar(config.aiAvatar || null);
        setSystemPrompt(config.system_prompt || "");
        setModelConfigOptions(config);
      } else {
        console.log(`[DEBUG] No config found. Resetting aiName to 'Nova'.`);
        // If no config exists for the selected model, reset to defaults.
        setAiName("Nova");
        setAiAvatar(null);
        setSystemPrompt("");
        setModelConfigOptions({});
      }
    },
    [],
  ); // Empty dependency array as it's a self-contained utility

  const handleModelChange = useCallback(
    (model: string) => {
      setSelectedModel(model);
      applyConfigForModel(model, allConfigs);
    },
    [allConfigs, applyConfigForModel],
  );

  const newChat = useCallback(() => {
    const modelName = selectedModel.split(/[/\\]/).pop() || "New Chat";
    const newName = `${currentBackend.toUpperCase()} - ${modelName}`;
    socketRef.current.emit("create_session", { name: newName });
  }, [currentBackend, selectedModel]);

  const loadChatById = useCallback(
    (id: string) => {
      if (id === activeChatId) return; // Don't reload the active chat
      setActiveChatId(id);
      setMessages([]); // Clear messages immediately for better UX
      socketRef.current.emit("get_session_messages", { session_id: id });
    },
    [activeChatId],
  );

  const deleteChat = useCallback((id: string) => {
    socketRef.current.emit("delete_session", { session_id: id });
  }, []);

  const renameChat = useCallback((id: string, newName: string) => {
    socketRef.current.emit("rename_session", {
      session_id: id,
      new_name: newName,
    });
  }, []);

  useEffect(() => {
    const storedName = localStorage.getItem("nova_user_name");
    if (storedName) setUserName(storedName);
    const storedAvatar = localStorage.getItem("nova_user_avatar");
    if (storedAvatar) setUserAvatar(storedAvatar);

    // The new logic will fetch sessions from the server via sockets.
    // This will be handled in the socket connection useEffect.
  }, []);

  useEffect(() => {
    socketRef.current = io("http://localhost:5000", {
      transports: ["websocket", "polling"],
      timeout: 60000, // 60 seconds
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
    const socket = socketRef.current;

    const handleConnect = () => {
      console.log("✅ Socket connected");
      socket.emit("get_sessions", { user_id: "default_user" });
    };
    const handleDisconnect = (reason: string) =>
      console.log(`❌ Socket disconnected: ${reason}`);
    const handleReconnect = (attempt: number) =>
      console.log(`🔄 Reconnecting (attempt ${attempt})`);

    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("reconnect", handleReconnect);
    socket.on('file_tree', (data: { tree: FileNode[], workspaceRoot?: string }) => {
      console.log('📁 File tree received:', data.tree.length, 'root nodes');
      setFileTree(data.tree);
      if (data.workspaceRoot) {
        setWorkspaceRoot(data.workspaceRoot);
      }
      setIsLoadingFileTree(false);
    });

    socket.on('workspace_root_changed', (data: { root: string }) => {
      console.log('📂 Workspace root changed to:', data.root);
      setWorkspaceRoot(data.root);
      toast.success(`Workspace changed to: ${data.root}`);
      requestFileTree('.');
    });

    socket.on('file_error', (data: { message: string }) => {
      console.error('❌ File operation error:', data.message);
      toast.error(data.message);
      setIsLoadingFileTree(false);
    });

    socket.on('file_saved', (data: { path: string }) => {
      console.log('💾 File saved:', data.path);
      toast.success(`Saved: ${data.path}`);
    });

    socket.on('file_content', (data: { path: string, content: string }) => {
      console.log('📄 File content received for:', data.path);
      handleFileContent(data);
    });
    const handleSessionsLoaded = (data: { sessions: ChatSession[] }) => {
      if (data.sessions && data.sessions.length > 0) {
        setChatSessions(data.sessions);
        const mostRecentSession = data.sessions[0];
        setActiveChatId(mostRecentSession._id);
        socket.emit("get_session_messages", {
          session_id: mostRecentSession._id,
        });
      } else {
        newChat();
      }
    };

    const handleSessionMessagesLoaded = (data: {
      session_id: string;
      messages: Message[];
    }) => {
      setMessages(data.messages);
    };

    const handleSessionCreated = (data: { session: ChatSession }) => {
      setChatSessions((prev) => [data.session, ...prev]);
      setActiveChatId(data.session._id);
      setMessages([]);
    };

    const handleSessionDeleted = (data: { session_id: string }) => {
      setChatSessions((prev) => prev.filter((s) => s._id !== data.session_id));
      if (activeChatId === data.session_id) {
        const remainingSessions = chatSessions.filter(
          (s) => s._id !== data.session_id,
        );
        if (remainingSessions.length > 0) {
          loadChatById(remainingSessions[0]._id);
        } else {
          newChat();
        }
      }
    };

    const handleSessionRenamed = (data: {
      session_id: string;
      new_name: string;
    }) => {
      setChatSessions((prev) =>
        prev.map((s) =>
          s._id === data.session_id ? { ...s, name: data.new_name } : s,
        ),
      );
    };

    socket.on("sessions_loaded", handleSessionsLoaded);
    socket.on("session_messages_loaded", handleSessionMessagesLoaded);
    socket.on("session_created", handleSessionCreated);
    socket.on("session_deleted", handleSessionDeleted);
    socket.on("session_renamed", handleSessionRenamed);

    // --- New Token Streaming Logic ---
    const MAX_HISTORY_LENGTH = 50;
    const THINKING_START_TOKENS = ["<think>", "<|thinking|>", "<｜thinking｜>"];
    const THINKING_END_TOKENS = [
      "</think>",
      "<|/thinking|>",
      "<｜/thinking｜>",
    ];

    const handleStreamStart = (data: { sender?: string }) => {
      const senderName = data?.sender || aiName; // Use sender from backend, fallback to state
      console.log(
        `[DEBUG] handleStreamStart: Creating new AI message bubble for sender: "${senderName}".`,
      );
      setIsStreaming(true);
      inThinkingMode.current = false;
      tokenHistory.current = "";
      setMessages((prev) => [
        ...prev,
        {
          sender: senderName, // Use the name from the backend
          message: "",
          type: "ai",
          thought: "",
          isThinking: false,
          imageB64: null,
        },
      ]);
    };

    const handleStream = (token: any) => {
      // Handle both string tokens (backward compat) and object tokens (new format)
      let processedToken: string;
      let incomingAiName: string | undefined;

      if (typeof token === "object" && token !== null) {
        // New format: {text: "...", aiName: "...", type: "..."}
        processedToken = String(token.text || "");
        incomingAiName = token.aiName;
      } else {
        // Old format: just a string
        processedToken = String(token);
        incomingAiName = undefined;
      }

      // If the token is literally '[Output]', ignore it.
      if (processedToken === "[Output]") {
        return;
      }

      // This function handles both the visual display of tokens and the TTS buffering.

      // 1. Update the visual message state (handles <think> tags)
      tokenHistory.current += processedToken;
      if (tokenHistory.current.length > MAX_HISTORY_LENGTH) {
        tokenHistory.current = tokenHistory.current.slice(-MAX_HISTORY_LENGTH);
      }

      const updateLastMessage = (
        updater: (lastMessage: Message) => Message,
      ) => {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const last = prev[prev.length - 1];

          // Match against either the aiName state OR the incoming aiName from token
          const expectedSender = incomingAiName || aiName;

          if (last.sender === expectedSender) {
            const updatedLast = updater({ ...last });
            return [...prev.slice(0, -1), updatedLast];
          }
          return prev;
        });
      };

      if (inThinkingMode.current) {
        for (const endTag of THINKING_END_TOKENS) {
          if (tokenHistory.current.trim().endsWith(endTag)) {
            inThinkingMode.current = false;
            const tagIndex = processedToken.lastIndexOf(endTag.charAt(0));
            const contentBeforeTag =
              tagIndex !== -1
                ? processedToken.slice(0, tagIndex)
                : processedToken;
            if (contentBeforeTag) {
              updateLastMessage((last) => {
                last.thought = (last.thought || "") + contentBeforeTag;
                return last;
              });
            }
            tokenHistory.current = "";
            return;
          }
        }
        updateLastMessage((last) => {
          last.thought = (last.thought || "") + processedToken;
          last.isThinking = true;
          return last;
        });
        return;
      }
      if (!inThinkingMode.current) {
        for (const startTag of THINKING_START_TOKENS) {
          if (tokenHistory.current.endsWith(startTag)) {
            inThinkingMode.current = true;
            const tagIndex = processedToken.lastIndexOf(startTag.charAt(0));
            const contentBeforeTag =
              tagIndex !== -1 ? processedToken.slice(0, tagIndex) : "";
            if (contentBeforeTag) {
              updateLastMessage((last) => {
                last.message += contentBeforeTag;
                return last;
              });
              if (isHandsFreeMode) ttsBufferRef.current += contentBeforeTag;
            }
            updateLastMessage((last) => {
              last.isThinking = true;
              return last;
            });
            tokenHistory.current = "";
            return;
          }
        }
        updateLastMessage((last) => {
          last.message += processedToken;
          last.isThinking = false;
          return last;
        });

        // 2. Handle TTS buffering if in hands-free mode and not thinking
        if (isHandsFreeMode && !inThinkingMode.current) {
          ttsBufferRef.current += processedToken;
          if (/[.!?]/.test(ttsBufferRef.current)) {
            if (socketRef.current)
              socketRef.current.emit("tts", { text: ttsBufferRef.current });
            ttsBufferRef.current = "";
          }
        }
      }
    };
    const handleStreamEnd = (data?: { aiName?: string; tokenMetrics?: any }) => {
      if (streamingTimeoutRef.current)
        clearTimeout(streamingTimeoutRef.current);
      setIsStreaming(false);
      inThinkingMode.current = false;

      // Flush any remaining text in the TTS buffer
      if (isHandsFreeMode && ttsBufferRef.current.trim()) {
        if (socketRef.current)
          socketRef.current.emit("tts", { text: ttsBufferRef.current });
        ttsBufferRef.current = "";
      }

      // Finalize the message state and save to DB
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const last = { ...prev[prev.length - 1] };
        if (last.sender === aiName) {
          last.isThinking = false;

          // Attach token metrics if available
          if (data?.tokenMetrics) {
            last.tokenMetrics = data.tokenMetrics;
          }

          if (last.message.trim() && activeChatId) {
            const messageToSave = {
              sender: aiName,
              message: last.message,
              type: "ai" as const,
              tokenMetrics: last.tokenMetrics,
            };
            if (socketRef.current) {
              socketRef.current.emit("save_message", {
                session_id: activeChatId,
                message: messageToSave,
              });
            }
          }
          return [...prev.slice(0, -1), last];
        }
        return prev;
      });
      tokenHistory.current = "";
    };
    const handleModelUnloaded = () =>
      addMessage("System", "Model unloaded.", "info");
    const handleError = (data: { message: string }) =>
      addMessage("System", `Error: ${data.message}`, "error");
    const handleConfigSaved = (data: { message: string }) =>
      addMessage("System", data.message, "info");
    const handleModels = (data: { backend: string; models: string[] }) => {
      setAvailableModels(data.models);
      if (data.models && data.models.length > 0) {
        handleModelChange(data.models[0]);
      }
    };
    const handleConfigs = (data: { [key: string]: ModelConfig }) => {
      setAllConfigs(data);
      // Re-apply config for the current model in case configs loaded after the model was selected.
      if (selectedModel) {
        applyConfigForModel(selectedModel, data);
      }
    };
    const handleGraphData = (data: GraphData) => setGraphData(data);
    // --- New Audio Playback Queue Logic ---
    const playNextInQueue = () => {
      if (audioQueueRef.current.length > 0 && !isPlayingAudioRef.current) {
        isPlayingAudioRef.current = true;
        const audioBlob = audioQueueRef.current.shift();
        if (audioBlob) {
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          audio.play();
          audio.onended = () => {
            isPlayingAudioRef.current = false;
            playNextInQueue(); // Play the next audio in the queue
          };
        }
      }
    };

    const handleVoiceStream = (data: { audio: ArrayBuffer }) => {
      const blob = new Blob([data.audio], { type: "audio/mpeg" });
      audioQueueRef.current.push(blob);
      playNextInQueue(); // Attempt to play immediately
      if (visualizerRef.current) visualizerRef.current.receiveAudio(data.audio);
    };

    const handleVoiceStreamEnd = () => {
      if (visualizerRef.current) visualizerRef.current.stop();
    };

    const handleTranscriptionResult = (data: { text: string }) => {
      if (isHandsFreeMode) {
        setMessageInputText(data.text); // Set the input text
        handleSendMessage(); // Call handleSendMessage without arguments
      } else {
        setMessageInputText(data.text);
      }
      if (visualizerRef.current)
        visualizerRef.current.startProcessing(data.text);
    };
    const handleLowConfidence = (data: {
      confidence?: number;
      reasoning?: string;
      query?: string;
    }) => {
      const confidenceText =
        typeof data.confidence === "number"
          ? `${data.confidence.toFixed(1)}%`
          : "unknown";
      const msg = `Low confidence (${confidenceText}). ${data.reasoning || "Consider a web search for more context."}`;
      addMessage(aiName, msg, "info");
      toast(msg, { icon: "💡" });
    };
    const handleCommandResponse = (data: {
      type: string;
      message: string;
      url?: string;
      embed_url?: string;
      video_id?: string;
      urls?: string[];
      sender?: string;
      prompt?: string;
      query?: string;
      image_url?: string;
    }) => {
      if (data.type === "image_generation") {
        setImageGenerationPrompt(data.prompt || "");
        setShowImageGenerator(true);
        return; // Don't add a message for this, it's a UI action
      } else if (data.type === "media_browser") {
        setMediaBrowserQuery(data.query || "");
        setRightPanelMode("media");
        return; // Don't add a message for this, it's a UI action
      } else if (data.type === "youtube_embed") {
        if (data.video_id) setYoutubeVideoId(data.video_id);
        if (data.url) setYoutubeUrl(data.url);
        setRightPanelMode("youtube");
        ensurePanelActive("youtube");
        return; // Panel-only update; no chat iframe
      } else if (data.type === "iframe") {
        if (data.url) {
          setBrowserUrl(data.url);
          setLeftPanelMode("browser");
          ensurePanelActive("browser");
          return; // Panel-only update; no chat iframe
        }
      }

      setMessages((prev) => {
        const lastMessage = prev[prev.length - 1];
        // Check if the last message is from the AI and is currently streaming or thinking
        if (
          lastMessage &&
          lastMessage.sender === aiName &&
          (isStreaming || lastMessage.isThinking)
        ) {
          const updatedLastMessage = { ...lastMessage };

          // Append or set the message content
          if (data.message) {
            updatedLastMessage.message =
              (updatedLastMessage.message || "") + data.message;
          }

          // Update specific command response fields
          if (data.type === "iframe") {
            updatedLastMessage.iframeUrl = data.url;
          } else if (data.type === "media_embed") {
            updatedLastMessage.iframeUrl = data.embed_url;
          } else if (data.type === "youtube_embed") {
            updatedLastMessage.youtubeVideoId = data.video_id;
          } else if (data.type === "image_gallery") {
            updatedLastMessage.imageGalleryUrls = data.urls;
          } else if (data.type === "image_generated") {
            updatedLastMessage.imageUrl = data.image_url;
          }
          updatedLastMessage.type = data.type === "error" ? "error" : "ai";

          return [...prev.slice(0, -1), updatedLastMessage];
        } else {
          // If not streaming or not an AI message, create a new message
          const newMessage: Message = {
            sender: data.sender || aiName,
            message: data.message,
            type: data.type === "error" ? "error" : "ai",
            imageB64: null,
            iframeUrl:
              data.type === "iframe"
                ? data.url
                : data.type === "media_embed"
                  ? data.embed_url
                  : undefined,
            youtubeVideoId:
              data.type === "youtube_embed" ? data.video_id : undefined,
            imageGalleryUrls:
              data.type === "image_gallery" ? data.urls : undefined,
            imageUrl:
              data.type === "image_generated" ? data.image_url : undefined,
          };
          return [...prev, newMessage];
        }
      });
    };

    const handleNovaSettingsLoaded = (data: any) => {
      setNovaSettings(data);
    };

    socket.on("stream_start", handleStreamStart);
    socket.on("stream", handleStream);
    socket.on("stream_end", handleStreamEnd);
    socket.on("model_unloaded", handleModelUnloaded);
    socket.on("error", handleError);
    socket.on("config_saved", handleConfigSaved);
    socket.on("models", handleModels);
    socket.on("configs", handleConfigs);
    socket.on("graph_data", handleGraphData);
    socket.on("voice_stream", handleVoiceStream);
    socket.on("voice_stream_end", handleVoiceStreamEnd);
    socket.on("transcription_result", handleTranscriptionResult);
    socket.on("command_response", handleCommandResponse);
    socket.on("low_confidence_warning", handleLowConfidence);
    socket.on("browser_control", async (data: any) => {
      const res = await runBrowserControl(data);
      socket.emit("browser_control_result", {
        action: data?.action,
        selector: data?.selector,
        value: data?.value,
        code: data?.code,
        ...res,
      });
    });
    socket.on("nova_settings_loaded", handleNovaSettingsLoaded);
    // Backend state sync
    socket.on("backend_set", (data: { backend: string }) => {
      console.log("✅ Backend set:", data.backend);
      setCurrentBackend(data.backend);
    });

    socket.on("model_loaded", (data: { model: string; backend: string }) => {
      console.log("✅ Model loaded:", data.model);
      toast.success(`Model loaded: ${data.model}`, {
        duration: 4000, // Show for 4 seconds
        icon: "🚀",
      });
    });

    // Settings save confirmations
    socket.on("tool_settings_saved", (data: { message: string }) => {
      console.log("✅", data.message);
      toast.success("Tool settings saved!");
    });

    socket.on("voice_settings_saved", (data: { message: string }) => {
      console.log("✅", data.message);
      toast.success("Voice settings saved successfully!");
    });

    socket.on("nova_settings_saved", (data: { message: string }) => {
      console.log("✅", data.message);
      toast.success("Nova settings saved!", { icon: "🤖" });
    });

    // Ollama server status
    socket.on("ollama_status", (data: { status: string; env?: any }) => {
      console.log("🔧 Ollama status:", data.status);
      if (data.status === "running") {
        toast.success("Ollama server is running", { icon: "🟢" });
      } else if (data.status === "stopped") {
        toast("Ollama server stopped", { icon: "🔴" });
      }
    });

    socket.on("auto_agent_status", (data: { enabled: boolean }) => {
      console.log("🤖 Auto agent:", data.enabled ? "enabled" : "disabled");
      toast(data.enabled ? "Agentic mode enabled" : "Agentic mode disabled", {
        icon: "🤖",
      });
    });

    // Search results listener
    socket.on("search_results", (data: { results: any[]; query: string }) => {
      console.log("dY\"? Search results received:", data);
      setSearchResults(data.results);
      setSearchQuery(data.query);
      setLeftPanelMode("search");
      // Add search panel to active panels if not already there
      setActivePanels((prev) =>
        prev.includes("search") ? prev : [...prev, "search"],
      );
    });

    // Parliament complete listener - forwards consensus to main model
    socket.on("parliament_complete", (data: { consensus_context: string; original_message: string; vote_result: any }) => {
      console.log("🏛️ Parliament complete! Voting result:", data.vote_result);
      console.log("📝 Consensus context:", data.consensus_context);
      console.log("🚀 Sending to main model...", { backend: currentBackend, provider: apiProvider });

      // Store vote result for display in Parliament panel
      setParliamentVoteResult(data.vote_result);

      // Show user that we're forwarding to main model
      toast(`Parliament consensus reached (${data.vote_result.votes} votes). Asking ${aiName}...`, { icon: "🏛️" });

      // Automatically send the consensus context to the main loaded model
      // This triggers a regular chat response with Parliament consensus as context
      socket.emit("chat", {
        text: data.consensus_context,
        history: messages.map(({ thought, ...rest }) => rest),  // Clean history without thoughts
        session_id: activeChatId,
        userName: userName,
        aiName: aiName,
        backend: currentBackend,
        provider: apiProvider,
        model_name: selectedModel,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      });
    });

    // File system listeners for code panel
    socket.on("file_tree", (data: { tree: FileNode[] }) => {
      setFileTree(data.tree || []);
      setIsLoadingFileTree(false);
      if (Array.isArray(data.tree) && data.tree.length > 0) {
        // Use top-level path as current workspace root hint
        setWorkspaceRoot(data.tree[0].path ? data.tree[0].path.split(/[\\\/]/).slice(0, 1).join("") : workspaceRoot);
      }
    });

    socket.on("file_content", (data: { path: string; content: string }) => {
      handleFileContent(data);
    });

    socket.on("file_saved", (data: { path: string }) => {
      toast.success(`Saved ${data.path}`, { icon: "✅" });
    });

    socket.on("file_error", (data: { message: string }) => {
      setIsLoadingFileTree(false);
      toast.error(data.message || "File action failed");
    });

    socket.on("workspace_root_changed", (data: { root: string }) => {
      setWorkspaceRoot(data.root || "");
      requestFileTree(".");
    });

    socket.on(
      "parliament_update",
      (data: { key?: string; status?: string; payload?: string }) => {
        if (!data.key) return;
        setParliamentRoles((prev) =>
          prev.map((r) =>
            r.key === data.key
              ? {
                  ...r,
                  status:
                    data.status === "done"
                      ? "done"
                      : (data.status as any) || r.status,
                }
              : r,
          ),
        );
      },
    );

    socket.on(
      "parliament_summary",
      (data: { roles: any[]; merged_prompt: string }) => {
        // Mark all as done
        setParliamentRoles((prev) =>
          prev.map((r) => (r.enabled ? { ...r, status: "done" } : r)),
        );
        setParliamentRoleOutputs(data?.roles || []);

        const merged = data?.merged_prompt || "";
        const lastUserMsg = lastParliamentUserMessage.current;
        if (merged && lastUserMsg && socketRef.current && activeChatId) {
          // Build synthetic history: prepend parliament context as system, then reuse prior history
          const historyForModel = messages.map(({ thought, ...rest }) => rest);
          const historyWithContext = [
            ...historyForModel,
            {
              role: "system",
              content: `[PARLIAMENT_CONTEXT]\n${merged}\n\nUse this merged analysis to answer the user's original question:`,
            },
          ];

          const payload: any = {
            text: lastUserMsg,
            history: historyWithContext,
            session_id: activeChatId,
            userName: userName,
            aiName: aiName,
            backend: currentBackend,
            provider: apiProvider,
            model_name: selectedModel, // Include model name for token tracking
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            mode: "parliament_synthesis",
          };

          if (isAgentMode) {
            socketRef.current.emit("agent_command", payload);
          } else {
            socketRef.current.emit("chat", payload);
          }
          setIsStreaming(true);
          if (debugMode) {
            addMessage("Parliament", merged, "info");
          }
        }
      },
    );

    return () => {

      socket.disconnect();
      socket.off('file_tree');
      socket.off('workspace_root_changed');
      socket.off('file_error');
      socket.off('file_saved');
      socket.off('file_content');
      socket.off("low_confidence_warning");
    };
  }, [handleFileContent]);

  const handleApiProviderChange = (provider: string) => {
    setApiProvider(provider);
    setAvailableModels([]);
    setSelectedModel("");
    socketRef.current.emit("set_backend", {
      backend: "api",
      provider: provider,
    });
  };

  // Slash command handler
  const getSearxUrl = () =>
    (toolSettings as any)?.searXngUrl ||
    (toolSettings as any)?.searxngUrl ||
    (novaSettings as any)?.searxngUrl ||
    (novaSettings as any)?.searXngUrl ||
    "";

  const searchYoutubeTop = useCallback(async (query: string) => {
    try {
      const res = await fetch(
        `http://localhost:8000/search?query=${encodeURIComponent(query)}&engine=videos`,
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const results = data.results || [];
      const youtubeHit = results.find(
        (r: any) =>
          typeof r.url === "string" &&
          (r.url.includes("youtube.com") || r.url.includes("youtu.be")),
      );
      if (!youtubeHit) throw new Error("No YouTube result found");
      const videoIdMatch = youtubeHit.url.match(
        /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/,
      );
      const videoId = videoIdMatch && videoIdMatch[1];
      if (!videoId) throw new Error("Could not extract video ID");
      return { title: youtubeHit.title || query, url: youtubeHit.url, videoId };
    } catch (e: any) {
      console.error("YouTube search failed", e);
      toast.error("YouTube search failed");
      return null;
    }
  }, []);

  const sendNowPlayingNotice = useCallback(
    (title: string, url: string) => {
      const text = `Now playing: ${title}\n${url}`;
      addMessage("System", text, "info");
      const historyForModel = messages.map(({ thought, ...rest }) => rest);
      const payload: any = {
        text,
        history: historyForModel,
        session_id: activeChatId,
        userName,
        aiName,
        backend: currentBackend,
        provider: apiProvider,
        model_name: selectedModel, // Include model name for token tracking
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };
      if (socketRef.current) {
        if (isAgentMode) {
          socketRef.current.emit("agent_command", payload);
        } else {
          socketRef.current.emit("chat", payload);
        }
      }
    },
    [
      addMessage,
      messages,
      activeChatId,
      userName,
      aiName,
      currentBackend,
      apiProvider,
      isAgentMode,
    ],
  );

  const ensurePanelActive = useCallback(
    (panelId: string) =>
      setActivePanels((prev) =>
        prev.includes(panelId) ? prev : [...prev, panelId],
      ),
    [],
  );

  const sendSlashToBackend = useCallback(
    (text: string) => {
      if (!socketRef.current) return;
      const historyForModel = messages.map(({ thought, ...rest }) => rest);
      const payload: any = {
        text,
        history: historyForModel,
        session_id: activeChatId,
        userName,
        aiName,
        backend: currentBackend,
        provider: apiProvider,
        model_name: selectedModel,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };
      addMessage(userName, text, "user");
      setIsStreaming(true);
      if (isAgentMode) {
        socketRef.current.emit("agent_command", payload);
      } else {
        socketRef.current.emit("chat", payload);
      }
    },
    [
      messages,
      activeChatId,
      userName,
      aiName,
      currentBackend,
      apiProvider,
      selectedModel,
      isAgentMode,
      addMessage,
    ],
  );

  const handleSlashCommand = async (command: string, args: string) => {
    switch (command) {
      case "youtube":
      case "yt": {
        const argText = args.trim();
        let urlToUse = "";
        let videoId = "";
        let title = argText;
        if (argText.startsWith("http")) {
          urlToUse = argText;
          const match = argText.match(
            /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/,
          );
          videoId = match && match[1] ? match[1] : "";
        } else if (argText) {
          toast("Searching YouTube?");
          const result = await searchYoutubeTop(argText);
          if (result) {
            urlToUse = result.url;
            videoId = result.videoId;
            title = result.title;
          }
        }
        if (!urlToUse || !videoId) {
          toast.error("No YouTube video found.");
          break;
        }
        setYoutubeUrl(urlToUse);
        setYoutubeVideoId(videoId);
        setRightPanelMode("youtube");
        ensurePanelActive("youtube");
        toast.success("YouTube video loaded");
        if (activeChatId) {
          sendNowPlayingNotice(title || urlToUse, urlToUse);
        }
        break;
      }

      case "browser":
      case "b": {
        if (args.trim()) {
          const query = args.trim();
          // If the user passed a URL or domain, browse it directly; otherwise fall back to search.
          const searxUrl = getSearxUrl();
          const base = searxUrl ? searxUrl.replace(/\/$/, "") : "";
          let targetUrl = "";
          if (/^https?:\/\//i.test(query)) {
            targetUrl = query;
          } else if (/^[\w.-]+\.[a-zA-Z]{2,}/.test(query)) {
            targetUrl = `https://${query}`;
          } else {
            targetUrl = searxUrl
              ? `${base}/?q=${encodeURIComponent(query)}`
              : `https://www.google.com/search?q=${encodeURIComponent(query)}`;
          }
          setBrowserUrl(targetUrl);
          setLeftPanelMode("browser");
          ensurePanelActive("browser");
          sendSlashToBackend(`/browser ${query}`);
          toast(`Browsing: ${query}`);
        } else {
          toast.error("Please provide a query: /browser <query>");
        }
        break;
      }

      case "search":
      case "s":
        if (args.trim()) {
          const query = args.trim();
          setSearchQuery(query);
          setLeftPanelMode("search");
          ensurePanelActive("search");
          setRightPanelMode("youtube");
          ensurePanelActive("youtube");
          setSearchResults([]);
          toast(`Searching: ${query}`);
          sendSlashToBackend(`/search ${query}`);
        } else {
          toast.error("Please provide a search query: /search <query>");
        }
        break;

      case "parliament":
      case "p":
        setLeftPanelMode("parliament");
        toast("AI Parliament activated");
        break;

      case "code":
      case "c":
        if (args.trim()) {
          setCodeContent(args.trim());
          setCodeLanguage("javascript");
        }
        setRightPanelMode("code");
        requestFileTree(".");
        toast("Code editor opened");
        break;

      case "media":
      case "m":
        setMediaBrowserQuery(args.trim());
        setRightPanelMode("media");
        break;
      case "tools":
      case "t":
        setRightPanelMode("tools");
        fetchToolList();
        if (args.trim()) {
          setToolQuery(args.trim());
          runToolSearch();
        }
        toast("Tools panel opened");
        break;

      case "voice":
      case "v":
        setRightPanelMode("voice");
        toast("Voice mode activated");
        break;

                    case "help":
      case "?":
        const helpMessage = `dY\"s Available Slash Commands:\n\n/youtube or /yt <url|query> - Load YouTube video in right panel\n/browser or /b <query> - Open SearXNG/browser panel for a query\n/search or /s <query> - Open search + YouTube panels and enrich the model\n/parliament or /p - Activate AI Parliament mode\n/code or /c [code] - Open code editor\n/media or /m [query] - Open media browser\n/voice or /v - Activate voice mode\n/close [left|right] - Close panels\n/help or /? - Show this help message`;

        toast(helpMessage, {
          duration: 10000,
          icon: "i",
          style: {
            maxWidth: "500px",
            whiteSpace: "pre-line",
          },
        });
        break;

      default:
        toast.error(
          `❌ Unknown command: /${command}. Type /help for available commands.`,
        );
    }
  };

  const handleSendMessage = async () => {
    const currentMessage = messageInputText; // Use the state variable

    // Check for slash commands
    if (currentMessage.trim().startsWith("/")) {
      const parts = currentMessage.trim().split(" ");
      const command = parts[0].substring(1).toLowerCase();
      const args = parts.slice(1).join(" ");

      await handleSlashCommand(command, args);
      setMessageInputText(""); // Clear the input
      return; // Don't send to backend
    }

    if (currentMessage.trim() || imageToSend) {
      // Check if this is the first message in the session to trigger rename
      // This check must happen *before* the new message is added to the state.
      if (messages.length === 0 && currentMessage.trim()) {
        socketRef.current.emit("summarize_and_rename", {
          session_id: activeChatId,
          text: currentMessage,
          user_id: "default_user", // Assuming a default user for now
        });
      }

      // Add user's message to chat immediately for display
      addMessage(userName, currentMessage, "user", imageToSend); // Re-add this line

      // Create a clean version of the history that doesn't include the model's thought process.
      const historyForModel = messages.map(({ thought, ...rest }) => rest);

      console.log(
        `[DEBUG] handleSendMessage: Current aiName from state is "${aiName}". Sending this to backend.`,
      );

      const enabledRoles = parliamentRoles.filter((r) => r.enabled);
      const isParliamentActive = enabledRoles.length > 0 && activePanels.includes("parliament");
      
      if (isParliamentActive) {
        // Parliament mode: Send request to Parliament, DON'T send regular chat yet
        socketRef.current.emit("parliament_request", {
          message: currentMessage,
          roles: enabledRoles.map((r) => ({
            key: r.key,
            name: r.name,
            model: r.model,
            provider: detectProviderFromModel(r.model, r.provider),
            prompt: r.prompt,
          })),
          session_id: activeChatId,
        });
        setParliamentRoles((prev) =>
          prev.map((r) => (r.enabled ? { ...r, status: "working" } : r)),
        );
        lastParliamentUserMessage.current = currentMessage;
        setParliamentRoleOutputs([]);
        setIsStreaming(true); // Set streaming state since we're waiting for Parliament
      } else {
        // Normal mode: Send regular chat message
        const messagePayload: any = {
          text: currentMessage,
          history: historyForModel, // Use the cleaned history
          session_id: activeChatId,
          userName: userName,
          aiName: aiName, // Use the model-specific aiName from state
          backend: currentBackend, // Use currentBackend state
          provider: apiProvider, // Use apiProvider state
          model_name: selectedModel, // Include model name for token tracking
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        };

        if (imageToSend) {
          messagePayload.image_base_64 = imageToSend;
          messagePayload.image_mime_type = imageMimeToSend || "image/png";
          setImageToSend(null);
          setImageMimeToSend("image/png");
        }

        if (pendingAttachments.length > 0) {
          messagePayload.attachments = pendingAttachments;
          setPendingAttachments([]);
        }

        if (isAgentMode) {
          socketRef.current.emit("agent_command", messagePayload);
        } else {
          socketRef.current.emit("chat", messagePayload);
        }
      }
      
      setMessageInputText(""); // Clear the input after sending
    }
  };

  const handleStop = () => {
    socketRef.current.emit("stop");
    setIsStreaming(false);
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    if (tab === "graph") {
      socketRef.current.emit("get_graph_data");
    }
  };

  const handleBackendChange = (backend: string) => {
    setCurrentBackend(backend);
    setSelectedModel("");
    setAvailableModels([]);
    socketRef.current.emit("set_backend", { backend });
  };
  // Add handler functions for image generation and media playback
  const handleGenerateImage = (
    prompt: string,
    model: string,
    settings: any,
  ) => {
    // Send request to your image generation API
    if (socketRef.current) {
      socketRef.current.emit("generate_image", {
        prompt,
        model,
        settings,
        session_id: activeChatId,
      });
    }
  };

  const handlePlayMedia = (mediaId: string, mediaType: string) => {
    // Send request to your media server API
    if (socketRef.current) {
      socketRef.current.emit("play_media", {
        mediaId,
        mediaType,
        session_id: activeChatId,
      });
      setRightPanelMode("closed");
    }
  };
  const handleLoadModel = () => {
    if (!selectedModel) {
      addMessage("System", "No model selected", "error");
      return;
    }

    // Determine thinking mode based on the thinking level.
    const thinkingLevel = modelConfigOptions.thinking_level || "none";
    const thinkingMode = thinkingLevel !== "none";

    const payload: any = {
      model_path: selectedModel,
      backend: currentBackend,
      system_prompt: systemPrompt,
      ...modelConfigOptions, // Spread all options from the state
      thinking_mode: thinkingMode, // Send the derived boolean
      thinking_level: thinkingLevel, // Send the selected level
    };

    if (currentBackend === "api") {
      payload.provider = apiProvider;
    }

    console.log("Loading model with payload:", payload);
    socketRef.current.emit("load_model", payload);
  };

  const handleUnloadModel = () => socketRef.current.emit("unload_model");
  const handleSaveConfig = () => {
    if (!selectedModel) return;
    const payload = {
      model_path: selectedModel,
      system_prompt: systemPrompt,
      ...modelConfigOptions,
      aiName: aiName,
      aiAvatar: aiAvatar,
    };
    socketRef.current.emit("save_config", payload);
  };

  const handleUserAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setUserAvatar(base64);
        localStorage.setItem("nova_user_avatar", base64);
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  const handleAiAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setAiAvatar(base64);
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      Array.from(files).forEach((file) => {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (event) => {
          const base64 = (event.target?.result as string)?.split(",")[1];
          setImageToSend(base64);
          setImageMimeToSend(file.type || "image/png");
        };
        reader.readAsDataURL(file);
        return;
      }

      // Handle text/code file drop: open in code panel
      const textLike =
        file.type.startsWith("text/") ||
        /\.(ts|tsx|js|jsx|json|md|py|html|css|sql|sh)$/i.test(file.name);
      if (textLike) {
        const reader = new FileReader();
        reader.onload = (event) => {
          const content = event.target?.result as string;
          const language = detectLanguageFromPath(file.name);
          const pseudoPath = file.name;
          setRightPanelMode("code");
          openLocalFile(pseudoPath, content, language);
          setPendingAttachments((prev) => {
            const filtered = prev.filter((f) => f.name !== file.name);
            return [...filtered, { name: file.name, content }];
          });
          toast.success(`Attached ${file.name} for context`);
        };
        reader.readAsText(file);
        return;
      }
      });
    }
  };

  const handleSaveToolSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit("save_tool_settings", toolSettings);
  };

  const handleSaveVoiceSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit("save_voice_settings", {
      tts: ttsSettings,
      stt: sttSettings,
    });
  };

  const handleTogglePanel = (panelId: string) => {
    const leftModes: Record<string, typeof leftPanelMode> = {
      chats: "chats",
      browser: "browser",
      search: "search",
      parliament: "parliament",
    };
    const rightModes: Record<string, typeof rightPanelMode> = {
      code: "code",
      voice: "voice",
      youtube: "youtube",
      tools: "tools",
    };

    if (leftModes[panelId]) {
      setLeftPanelMode(leftModes[panelId]);
    }

    if (rightModes[panelId]) {
      setRightPanelMode(rightModes[panelId]);
    }

    setActivePanels((prev) =>
      prev.includes(panelId)
        ? prev.filter((id) => id !== panelId)
        : [...prev, panelId],
    );
  };

  const handleExecuteCommand = (command: string) => {
    // Set the command in the message input and trigger processing
    setMessageInputText(command);
    // The command will be processed by the existing slash command handler
  };

  const hideLeftPanel = () => {
    setLeftPanelMode("closed");
    setActivePanels((prev) =>
      prev.filter((id) => !leftPanelIds.includes(id)),
    );
  };

  const visiblePanelIds = useMemo(() => {
    const ids = new Set<string>([...activePanels, "chat"]);
    return Array.from(ids);
  }, [activePanels]);
  const panels = [
    {
      id: "chat",
      title: "Chat",
      component: (
        <div
          className="flex h-full w-full flex-col"
          style={{ backgroundColor: theme.backgroundColor }}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <header className="flex-shrink-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-between p-2 z-10">
            <button
              onClick={() => handleTogglePanel("chats")}
              className="p-2 rounded-md hover:bg-gray-700 focus:outline-none text-white icon-button"
              aria-label="Toggle left panel"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 6h16M4 12h16M4 18h16"
                ></path>
              </svg>
            </button>
            <div className="flex items-center space-x-6">
              <button
                onClick={() => handleTabChange("chat")}
                className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === "chat" ? "bg-blue-600 text-white" : "bg-gray-700 hover:bg-gray-600"} custom-button`}
              >
                Chat
              </button>
              <button
                onClick={() => handleTabChange("graph")}
                className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === "graph" ? "bg-blue-600 text-white" : "bg-gray-700 hover:bg-gray-600"} custom-button`}
              >
                Memory Graph
              </button>
              <button
                onClick={() => handleTabChange("settings")}
                className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === "settings" ? "bg-blue-600 text-white" : "bg-gray-700 hover:bg-gray-600"} custom-button`}
              >
                Settings
              </button>
            </div>
            <button
              onClick={() => setShowUsagePanel(true)}
              className="px-3 py-2 rounded-lg text-sm bg-gray-800 hover:bg-gray-700 text-white border border-white/10 transition-colors"
            >
              Usage
            </button>
            <button
              onClick={() => handleTogglePanel("parliament")}
              className="p-2 rounded-md hover:bg-gray-700 focus:outline-none text-white icon-button"
              aria-label="Toggle parliament panel"
              title="AI Parliament"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 9l9-6 9 6-9 6-9-6z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 15l9 6 9-6"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 9l9 6 9-6"
                />
              </svg>
            </button>
            <button
              onClick={() => handleTogglePanel("voice")}
              className="p-2 rounded-md hover:bg-gray-700 focus:outline-none text-white icon-button"
              aria-label="Toggle right panel"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                ></path>
              </svg>
            </button>
          </header>
          <main
            className={`flex-1 flex flex-col min-h-0 relative ${activeTab !== "chat" ? "items-center" : ""}`}
          >
            {activeTab === "chat" && (
              <div className="flex-1 flex flex-col min-h-0 p-4 relative">
                {activePanels.includes("parliament") &&
                  parliamentRoles.some((r) => r.enabled) && (
                    <div className="mb-3 bg-gray-900/70 border border-white/10 rounded-lg p-3">
                      <div className="flex items-center justify-between text-xs text-gray-200 mb-2">
                        <div className="flex items-center gap-2">
                          <span>Parliament Live</span>
                          <span className="px-2 py-1 rounded-full text-[11px] bg-white/10 border border-white/10">
                            Live
                          </span>
                          <span className="text-[11px] text-gray-400">
                            {
                              parliamentRoles.filter(
                                (r) => r.enabled && r.status === "done",
                              ).length
                            }
                            /
                            {
                              parliamentRoles.filter((r) => r.enabled).length
                            }{" "}
                            done
                          </span>
                        </div>
                        <button
                          onClick={() =>
                            setParliamentPanelCollapsed((prev) => !prev)
                          }
                          className="px-2 py-1 rounded border border-white/10 text-[11px] hover:bg-white/5 transition"
                        >
                          {parliamentPanelCollapsed ? "Expand" : "Minimize"}
                        </button>
                      </div>
                      {!parliamentPanelCollapsed && (
                        <div className="flex flex-wrap gap-2">
                          {parliamentRoles
                            .filter((r) => r.enabled)
                            .map((role) => {
                              const payload =
                                parliamentRoleOutputs.find(
                                  (o) => o.key === role.key,
                                )?.response || "";
                              const statusColor =
                                role.status === "done"
                                  ? "bg-green-500"
                                  : role.status === "working"
                                    ? "bg-blue-400"
                                    : "bg-gray-500";
                              const isExpanded = parliamentExpanded[role.key];
                              const formattedPayload = formatParliamentPayload(
                                payload,
                              );
                              const statusLabel =
                                role.status === "working"
                                  ? "Streaming… (click to view)"
                                  : role.status === "done"
                                    ? "Done – click to view output"
                                    : "Waiting to start";
                              return (
                                <button
                                  key={role.key}
                                  onClick={() => toggleParliamentRole(role.key)}
                                  className="min-w-[220px] bg-black/40 border border-white/10 rounded-md p-2 text-left hover:bg-white/5 transition"
                                >
                                  <div className="flex items-center gap-2">
                                    <span
                                      className={`h-2 w-2 rounded-full ${statusColor}`}
                                    />
                                    <span className="text-sm text-white truncate flex-1">
                                      {role.name}
                                    </span>
                                    <span className="text-[10px] text-gray-400 truncate">
                                      {role.model}
                                    </span>
                                    <span className="text-[11px] text-gray-300">
                                      {isExpanded ? "Hide" : "Show"}
                                    </span>
                                  </div>
                                  {isExpanded ? (
                                    <pre className="text-[11px] text-gray-200 bg-gray-900/70 border border-white/5 rounded mt-2 p-2 max-h-48 overflow-auto whitespace-pre-wrap">
                                      {formattedPayload ||
                                        (role.status === "working"
                                          ? "Streaming..."
                                          : "Waiting...")}
                                    </pre>
                                  ) : (
                                    <p className="text-[11px] text-gray-200 mt-1">
                                      {statusLabel}
                                    </p>
                                  )}
                                </button>
                              );
                            })}
                        </div>
                      )}
                    </div>
                  )}
                <ChatView
                  messages={messages}
                  userAvatar={userAvatar}
                  aiAvatar={aiAvatar}
                />
                <div className="absolute bottom-4 left-4 right-4 z-10">
                  <MessageInput
                    onSendMessage={handleSendMessage}
                    onStop={handleStop}
                    isStreaming={isStreaming}
                    image={imageToSend}
                    setImage={(img) => {
                      setImageToSend(img);
                      if (!img) setImageMimeToSend("image/png");
                    }}
                    isAgentMode={isAgentMode}
                    onAgentModeChange={setIsAgentMode}
                    message={messageInputText}
                    onMessageChange={setMessageInputText}
                  />
                </div>
              </div>
            )}
            {activeTab === "graph" && <MemoryGraph data={graphData} />}
            {activeTab === "settings" && (
              <div className="flex justify-center w-full h-full p-8">
                <div className="w-full max-w-6xl">
                  <SettingsPanel
                    allConfigs={allConfigs}
                    availableModels={availableModels || []}
                    currentBackend={currentBackend}
                    selectedModel={selectedModel}
                    apiProvider={apiProvider}
                    apiKey={apiKey}
                    modelConfigOptions={modelConfigOptions}
                    systemPrompt={systemPrompt}
                    userName={userName}
                    aiName={aiName}
                    ollamaKvCache={ollamaKvCache}
                    onBackendChange={handleBackendChange}
                    onApiProviderChange={handleApiProviderChange}
                    onApiKeyChange={setApiKey}
                    onSaveApiKey={() =>
                      socketRef.current.emit("set_backend", {
                        backend: "api",
                        provider: apiProvider,
                        api_key: apiKey,
                      })
                    }
                    onClearApiKey={() => {
                      setApiKey("");
                      socketRef.current.emit("set_backend", {
                        backend: "api",
                        provider: apiProvider,
                        api_key: "",
                      });
                    }}
                    onModelChange={handleModelChange}
                    onModelConfigChange={(key: string, value: any) =>
                      setModelConfigOptions((prev) => ({
                        ...prev,
                        [key]: value,
                      }))
                    }
                    onSystemPromptChange={setSystemPrompt}
                    onLoadModel={handleLoadModel}
                    onUnloadModel={handleUnloadModel}
                    onSaveConfig={handleSaveConfig}
                    onUserNameChange={setUserName}
                    onAiNameChange={setAiName}
                    onUserAvatarChange={handleUserAvatarChange}
                    onAiAvatarChange={handleAiAvatarChange}
                    theme={theme}
                    onThemeChange={handleThemeChange}
                    toolSettings={toolSettings}
                    onToolSettingsChange={setToolSettings}
                    ttsSettings={ttsSettings}
                    onTtsSettingsChange={setTtsSettings}
                    sttSettings={sttSettings}
                    onSttSettingsChange={setSttSettings}
                    onOllamaKvCacheChange={setOllamaKvCache}
                    onStopOllama={() =>
                      socketRef.current.emit("manage_ollama", {
                        action: "stop",
                      })
                    }
                    onRestartOllama={() =>
                      socketRef.current.emit("manage_ollama", {
                        action: "restart",
                        env: { OLLAMA_KV_CACHE_TYPE: ollamaKvCache },
                      })
                    }
                    onSaveToolSettings={handleSaveToolSettings}
                    onSaveVoiceSettings={handleSaveVoiceSettings}
                    novaSettings={novaSettings}
                    onNovaSettingsChange={handleNovaSettingsChange}
                    onSaveNovaSettings={handleSaveNovaSettings}
                    onReloadOrchestrator={handleReloadOrchestrator}
                    debugMode={debugMode}
                    onDebugModeChange={setDebugMode}
                  />
                </div>
              </div>
            )}
          </main>
        </div>
      ),
    },
    {
      id: "chats",
      title: "Chats",
      component: (
        <LeftPanel
          mode="chats"
          onClose={hideLeftPanel}
          // Chat mode props
          chats={chatSessions.map((c) => ({ id: c._id, name: c.name }))}
          activeChatId={activeChatId}
          onNewChat={newChat}
          onLoadChat={loadChatById}
          onDeleteChat={deleteChat}
          onRenameChat={renameChat}
          // Search mode props
          searchResults={searchResults}
          searchQuery={searchQuery}
          onSearchResultClick={(url) => {
            setBrowserUrl(url);
            setLeftPanelMode("browser");
            ensurePanelActive("browser");
            toast.success("Opening in browser panel");
          }}
          onSearchResultPreview={(url) => {
            setBrowserUrl(url);
            setLeftPanelMode("browser");
          }}
          browserUrl={browserUrl}
          // Parliament mode props
          parliamentRoles={parliamentRoles}
          parliamentRoleOutputs={parliamentRoleOutputs}
          onUpdateRole={updateParliamentRole}
          debugMode={debugMode}
        />
      ),
    },
    {
      id: "search",
      title: "Search",
      component: (
        <LeftPanel
          mode="search"
          onClose={() => handleTogglePanel("search")}
          searchResults={searchResults}
          searchQuery={searchQuery}
          onSearchResultClick={(url) => {
            setBrowserUrl(url);
            setLeftPanelMode("browser");
            ensurePanelActive("browser");
            toast.success("Opening in browser panel");
          }}
          onSearchResultPreview={(url) => {
            setBrowserUrl(url);
            setLeftPanelMode("browser");
          }}
          browserUrl={browserUrl}
        />
      ),
    },
    {
      id: "browser",
      title: "Browser",
      component: (
        <LeftPanel
          mode="browser"
          onClose={() => handleTogglePanel("browser")}
          browserUrl={browserUrl}
          searchResults={searchResults}
          searchQuery={searchQuery}
        />
      ),
    },
    {
      id: "parliament",
      title: "Parliament",
      component: (
        <LeftPanel
          mode="parliament"
          onClose={() => handleTogglePanel("parliament")}
          parliamentRoles={parliamentRoles}
          parliamentRoleOutputs={parliamentRoleOutputs}
          parliamentVoteResult={parliamentVoteResult}
          onUpdateRole={updateParliamentRole}
          debugMode={debugMode}
        />
      ),
    },
    {
      id: "code",
      title: "Code",
      component: (
        <RightPanel
          mode="code"
          onClose={() => handleTogglePanel("code")}
          // Voice mode props
          socket={socketRef.current}
          isHandsFreeMode={isHandsFreeMode}
          onHandsFreeModeChange={setIsHandsFreeMode}
          onVisualizerReady={(visualizer) => {
            visualizerRef.current = visualizer;
          }}
          // YouTube mode props
          youtubeVideoId={youtubeVideoId}
          youtubeUrl={youtubeUrl}
          // Code mode props
          codeContent={codeContent}
          codeLanguage={codeLanguage}
          onCodeChange={handleCodeEditorChange}
          onCodeLanguageChange={setCodeLanguage}
          fileTree={fileTree}
          workspaceRoot={workspaceRoot}
          onRefreshFileTree={() => requestFileTree(".")}
          onOpenFileFromTree={openFileFromTree}
          openFiles={openFiles}
          activeFilePath={activeFilePath}
          onSelectOpenFile={selectOpenFile}
          onCloseFile={closeOpenFile}
          onSaveActiveFile={saveActiveFile}
          isLoadingFileTree={isLoadingFileTree}
          onCreateFile={createFile}
          onCreateFolder={createFolder}
          onRenameFile={renamePath}
          onDeleteFile={deletePath}
          onChangeWorkspace={requestWorkspaceChange}
          // Tools panel props
          toolQuery={toolQuery}
          onToolQueryChange={setToolQuery}
          onRunToolSearch={runToolSearch}
          toolResults={toolResults}
          toolLoading={toolLoading}
          toolError={toolError}
          toolList={toolList}
          selectedTool={selectedTool}
          onSelectTool={setSelectedTool}
          toolArgsText={toolArgsText}
          onToolArgsChange={setToolArgsText}
          onRunToolCall={runToolCall}
          toolCallResult={toolCallResult}
          // Media browser props
          mediaBrowserQuery={mediaBrowserQuery}
          onPlayMedia={handlePlayMedia}
          novaSettings={novaSettings}
        />
      ),
    },
    {
      id: "agent-coding",
      title: "Agent Coding",
      component: (
        <AgentCodingPanel
          socket={socketRef.current}
          workspaceRoot={workspaceRoot}
          onFileOpen={(path) => {
            openFileFromTree(path);
            setRightPanelMode("code");
          }}
        />
      ),
    },
    {
      id: "voice",
      title: "Voice",
      component: (
        <RightPanel
          mode="voice"
          onClose={() => handleTogglePanel("voice")}
          socket={socketRef.current}
          isHandsFreeMode={isHandsFreeMode}
          onHandsFreeModeChange={setIsHandsFreeMode}
          onVisualizerReady={(visualizer) => {
            visualizerRef.current = visualizer;
          }}
          youtubeVideoId={youtubeVideoId}
          youtubeUrl={youtubeUrl}
          codeContent={codeContent}
          codeLanguage={codeLanguage}
          onCodeChange={handleCodeEditorChange}
          onCodeLanguageChange={setCodeLanguage}
          fileTree={fileTree}
          workspaceRoot={workspaceRoot}
          onRefreshFileTree={() => requestFileTree(".")}
          onOpenFileFromTree={openFileFromTree}
          openFiles={openFiles}
          activeFilePath={activeFilePath}
          onSelectOpenFile={selectOpenFile}
          onCloseFile={closeOpenFile}
          onSaveActiveFile={saveActiveFile}
          isLoadingFileTree={isLoadingFileTree}
          onCreateFile={createFile}
          onCreateFolder={createFolder}
          onRenameFile={renamePath}
          onDeleteFile={deletePath}
          onChangeWorkspace={requestWorkspaceChange}
          toolQuery={toolQuery}
          onToolQueryChange={setToolQuery}
          onRunToolSearch={runToolSearch}
          toolResults={toolResults}
          toolLoading={toolLoading}
          toolError={toolError}
          toolList={toolList}
          selectedTool={selectedTool}
          onSelectTool={setSelectedTool}
          toolArgsText={toolArgsText}
          onToolArgsChange={setToolArgsText}
          onRunToolCall={runToolCall}
          toolCallResult={toolCallResult}
          mediaBrowserQuery={mediaBrowserQuery}
          onPlayMedia={handlePlayMedia}
          novaSettings={novaSettings}
        />
      ),
    },
    {
      id: "youtube",
      title: "YouTube",
      component: (
        <RightPanel
          mode="youtube"
          onClose={() => handleTogglePanel("youtube")}
          socket={socketRef.current}
          isHandsFreeMode={isHandsFreeMode}
          onHandsFreeModeChange={setIsHandsFreeMode}
          youtubeVideoId={youtubeVideoId}
          youtubeUrl={youtubeUrl}
          onPopOutVideo={(videoId) => setMiniPlayerVideoId(videoId)}
          codeContent={codeContent}
          codeLanguage={codeLanguage}
          onCodeChange={handleCodeEditorChange}
          onCodeLanguageChange={setCodeLanguage}
          fileTree={fileTree}
          workspaceRoot={workspaceRoot}
          onRefreshFileTree={() => requestFileTree(".")}
          onOpenFileFromTree={openFileFromTree}
          openFiles={openFiles}
          activeFilePath={activeFilePath}
          onSelectOpenFile={selectOpenFile}
          onCloseFile={closeOpenFile}
          onSaveActiveFile={saveActiveFile}
          isLoadingFileTree={isLoadingFileTree}
          onCreateFile={createFile}
          onCreateFolder={createFolder}
          onRenameFile={renamePath}
          onDeleteFile={deletePath}
          onChangeWorkspace={requestWorkspaceChange}
          toolQuery={toolQuery}
          onToolQueryChange={setToolQuery}
          onRunToolSearch={runToolSearch}
          toolResults={toolResults}
          toolLoading={toolLoading}
          toolError={toolError}
          toolList={toolList}
          selectedTool={selectedTool}
          onSelectTool={setSelectedTool}
          toolArgsText={toolArgsText}
          onToolArgsChange={setToolArgsText}
          onRunToolCall={runToolCall}
          toolCallResult={toolCallResult}
          mediaBrowserQuery={mediaBrowserQuery}
          onPlayMedia={handlePlayMedia}
          novaSettings={novaSettings}
        />
      ),
    },
    {
      id: "tools",
      title: "Tools",
      component: (
        <RightPanel
          mode="tools"
          onClose={() => handleTogglePanel("tools")}
          socket={socketRef.current}
          isHandsFreeMode={isHandsFreeMode}
          onHandsFreeModeChange={setIsHandsFreeMode}
          youtubeVideoId={youtubeVideoId}
          youtubeUrl={youtubeUrl}
          codeContent={codeContent}
          codeLanguage={codeLanguage}
          onCodeChange={handleCodeEditorChange}
          onCodeLanguageChange={setCodeLanguage}
          fileTree={fileTree}
          workspaceRoot={workspaceRoot}
          onRefreshFileTree={() => requestFileTree(".")}
          onOpenFileFromTree={openFileFromTree}
          openFiles={openFiles}
          activeFilePath={activeFilePath}
          onSelectOpenFile={selectOpenFile}
          onCloseFile={closeOpenFile}
          onSaveActiveFile={saveActiveFile}
          isLoadingFileTree={isLoadingFileTree}
          onCreateFile={createFile}
          onCreateFolder={createFolder}
          onRenameFile={renamePath}
          onDeleteFile={deletePath}
          onChangeWorkspace={requestWorkspaceChange}
          toolQuery={toolQuery}
          onToolQueryChange={setToolQuery}
          onRunToolSearch={runToolSearch}
          toolResults={toolResults}
          toolLoading={toolLoading}
          toolError={toolError}
          toolList={toolList}
          selectedTool={selectedTool}
          onSelectTool={setSelectedTool}
          toolArgsText={toolArgsText}
          onToolArgsChange={setToolArgsText}
          onRunToolCall={runToolCall}
          toolCallResult={toolCallResult}
          mediaBrowserQuery={mediaBrowserQuery}
          onPlayMedia={handlePlayMedia}
          novaSettings={novaSettings}
        />
      ),
    },
  ].filter((panel) => visiblePanelIds.includes(panel.id));

  return (
    <div
      className="h-screen w-screen overflow-hidden"
      style={{ backgroundColor: theme.backgroundColor }}
    >
      <Toaster position="top-right" />
      <Toolbar activePanels={activePanels} onTogglePanel={handleTogglePanel} />
      <div className="grid-wrapper">
        <PanelManager
          panels={panels}
          onPanelMoveOrResize={(id) => {
            if (id === "browser") triggerBrowserSync();
          }}
        />
      </div>
      <UsagePanel
        isOpen={showUsagePanel}
        onClose={() => setShowUsagePanel(false)}
      />
      {showImageGenerator && (
        <ImageGenerator
          prompt={imageGenerationPrompt}
          onGenerate={handleGenerateImage}
          onClose={() => setShowImageGenerator(false)}
        />
      )}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onTogglePanel={handleTogglePanel}
        onExecuteCommand={handleExecuteCommand}
        activePanels={activePanels}
      />
      {miniPlayerVideoId && (
        <MiniPlayer
          videoId={miniPlayerVideoId}
          onClose={() => setMiniPlayerVideoId("")}
        />
      )}
    </div>
  );
};

export default App;









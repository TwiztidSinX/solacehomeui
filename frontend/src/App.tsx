import React, { useState, useEffect, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import MessageInput from './components/MessageInput';
import SettingsPanel from './components/SettingsPanel';
import VoicePanel from './components/VoicePanel';
import MemoryGraph from './components/MemoryGraph';
import { type Message } from './types';
import { VoiceVisualizer } from './VoiceVisualizer';
import ImageGenerator from './components/ImageGenerator';
import MediaBrowser from './components/MediaBrowser';

// Type definitions
interface ChatSession {
  _id: string;
  name: string;
  messages: Message[];
}
interface ModelConfig { [key: string]: any; }
interface GraphData {
  nodes: any[];
  edges: any[];
}

const App: React.FC = () => {
  // UI State
  const [activeTab, setActiveTab] = useState('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isVoicePanelOpen, setIsVoicePanelOpen] = useState(false);

  // Socket and Timeout Refs
  const socketRef = useRef<any>(null);
  const streamingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const inThinkingMode = useRef(false);
  const tokenHistory = useRef('');
  const ttsBufferRef = useRef(''); // Buffer for sentence-based TTS
  const audioQueueRef = useRef<Blob[]>([]); // Queue for incoming audio chunks
  const isPlayingAudioRef = useRef(false); // Flag to prevent concurrent playback

  // Chat State
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [imageToSend, setImageToSend] = useState<string | null>(null);
  const [isAgentMode, setIsAgentMode] = useState<boolean>(false);
  const [isOrchestratorMode, setIsOrchestratorMode] = useState<boolean>(false); // New state
  const [messageInputText, setMessageInputText] = useState(''); // New state for the input
  const [isHandsFreeMode, setIsHandsFreeMode] = useState(false); // State for speech-to-speech
  const [showImageGenerator, setShowImageGenerator] = useState(false);
  const [showMediaBrowser, setShowMediaBrowser] = useState(false);
  const [imageGenerationPrompt, setImageGenerationPrompt] = useState('');
  const [mediaBrowserQuery, setMediaBrowserQuery] = useState('');
  
  // Settings State
  const [allConfigs, setAllConfigs] = useState<{ [key: string]: ModelConfig }>({});
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [currentBackend, setCurrentBackend] = useState('llama.cpp');
  const [selectedModel, setSelectedModel] = useState('');
  const [apiProvider, setApiProvider] = useState('openai');
  const [apiKey, setApiKey] = useState('');
  const [modelConfigOptions, setModelConfigOptions] = useState<ModelConfig>({});
  const [ollamaKvCache, setOllamaKvCache] = useState('f16');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [userName, setUserName] = useState('User');
  const [aiName, setAiName] = useState('Nova');
  const [userAvatar, setUserAvatar] = useState<string | null>(null);
  const [aiAvatar, setAiAvatar] = useState<string | null>(null);
  const [toolSettings, setToolSettings] = useState({
    n8nUrl: '',
    searXngUrl: '',
    serpApiApiKey: '',
  });
  const [novaSettings, setNovaSettings] = useState({
    searxngUrl: '',
    mediaServerUrl: '',
    mediaServerApiKey: '',
    imageGenUrl: '',
    aiName: 'Nova', // Add aiName here
  });
  const [ttsSettings, setTtsSettings] = useState({
    type: 'local', // 'local' or 'cloud'
    provider: 'openai',
    url: '',
    apiKey: '',
    model: '',
    voice: '',
  });
  const [sttSettings, setSttSettings] = useState({
    type: 'local', // 'local' or 'cloud'
    provider: 'openai',
    url: '',
    apiKey: '',
    model: '',
  });
  const [debugMode, setDebugMode] = useState(false);
  const [theme, setTheme] = useState({
    primaryColor: '#4a90e2',
    backgroundColor: '#1a1a2e',
    chatBackgroundColor: '#22223b',
    chatInputBackgroundColor: '#333355',
    userMessageColor: '#4a236b',
    aiMessageColor: '#333355',
    textColor: '#e0e0e0',
  });

  const handleThemeChange = (property: string, value: string) => {
    setTheme(prevTheme => ({ ...prevTheme, [property]: value }));
  };

  const handleNovaSettingsChange = (settings: any) => {
    setNovaSettings(settings);
  };

  const handleSaveNovaSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit('save_nova_settings', novaSettings);
  };

  useEffect(() => {
    const savedTheme = localStorage.getItem('nova_theme');
    if (savedTheme) {
      setTheme(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('nova_theme', JSON.stringify(theme));
    Object.keys(theme).forEach(key => {
      document.documentElement.style.setProperty(`--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`, theme[key as keyof typeof theme]);
    });
  }, [theme]);

  // Graph State
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  
  // Voice Visualizer Ref
  const visualizerRef = useRef<VoiceVisualizer | null>(null);

  const addMessage = useCallback((sender: string, message: string, type: Message['type'] = 'ai', imageB64: string | null = null) => {
    const newMessage: Message = { sender, message, type, imageB64 };
    setMessages(prev => [...prev, newMessage]);
  }, [setMessages]); // <--- Add setMessages to dependency array

  const handleModelChange = useCallback((model: string) => {
    setSelectedModel(model);
    setAllConfigs(prevConfigs => {
      const config = prevConfigs[model] || {};
      setSystemPrompt(config.system_prompt || '');
      setAiName(config.aiName || 'Nova');
      setAiAvatar(config.aiAvatar || null);
      setModelConfigOptions(config);
      return prevConfigs; // Return the unchanged state
    });
  }, []);

  const newChat = useCallback(() => {
    const modelName = selectedModel.split(/[/\\]/).pop() || 'New Chat';
    const newName = `${currentBackend.toUpperCase()} - ${modelName}`;
    socketRef.current.emit('create_session', { name: newName });
  }, [currentBackend, selectedModel]);

  const loadChatById = useCallback((id: string) => {
    if (id === activeChatId) return; // Don't reload the active chat
    setActiveChatId(id);
    setMessages([]); // Clear messages immediately for better UX
    socketRef.current.emit('get_session_messages', { session_id: id });
  }, [activeChatId]);

  const deleteChat = useCallback((id: string) => {
    socketRef.current.emit('delete_session', { session_id: id });
  }, []);

  const renameChat = useCallback((id: string, newName: string) => {
    socketRef.current.emit('rename_session', { session_id: id, new_name: newName });
  }, []);

  useEffect(() => {
    const storedName = localStorage.getItem('nova_user_name');
    if (storedName) setUserName(storedName);
    const storedAvatar = localStorage.getItem('nova_user_avatar');
    if (storedAvatar) setUserAvatar(storedAvatar);

    // The new logic will fetch sessions from the server via sockets.
    // This will be handled in the socket connection useEffect.
  }, []);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket', 'polling'],
      timeout: 60000, // 60 seconds
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
    const socket = socketRef.current;

    const handleConnect = () => {
      console.log('✅ Socket connected');
      socket.emit('get_sessions', { user_id: 'default_user' });
    };
    const handleDisconnect = (reason: string) => console.log(`❌ Socket disconnected: ${reason}`);
    const handleReconnect = (attempt: number) => console.log(`🔄 Reconnecting (attempt ${attempt})`);

    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('reconnect', handleReconnect);

    const handleSessionsLoaded = (data: { sessions: ChatSession[] }) => {
      if (data.sessions && data.sessions.length > 0) {
        setChatSessions(data.sessions);
        const mostRecentSession = data.sessions[0];
        setActiveChatId(mostRecentSession._id);
        socket.emit('get_session_messages', { session_id: mostRecentSession._id });
      } else {
        newChat();
      }
    };

    const handleSessionMessagesLoaded = (data: { session_id: string, messages: Message[] }) => {
      setMessages(data.messages);
    };

    const handleSessionCreated = (data: { session: ChatSession }) => {
      setChatSessions(prev => [data.session, ...prev]);
      setActiveChatId(data.session._id);
      setMessages([]);
    };
    
    const handleSessionDeleted = (data: { session_id: string }) => {
      setChatSessions(prev => prev.filter(s => s._id !== data.session_id));
      if (activeChatId === data.session_id) {
        const remainingSessions = chatSessions.filter(s => s._id !== data.session_id);
        if (remainingSessions.length > 0) {
          loadChatById(remainingSessions[0]._id);
        } else {
          newChat();
        }
      }
    };

    const handleSessionRenamed = (data: { session_id: string, new_name: string }) => {
      setChatSessions(prev => prev.map(s => s._id === data.session_id ? { ...s, name: data.new_name } : s));
    };

    socket.on('sessions_loaded', handleSessionsLoaded);
    socket.on('session_messages_loaded', handleSessionMessagesLoaded);
    socket.on('session_created', handleSessionCreated);
    socket.on('session_deleted', handleSessionDeleted);
    socket.on('session_renamed', handleSessionRenamed);

    // --- New Token Streaming Logic ---
    const MAX_HISTORY_LENGTH = 50;
    const THINKING_START_TOKENS = ['<think>', '<|thinking|>', '<｜thinking｜>'];
    const THINKING_END_TOKENS = ['</think>', '<|/thinking|>', '<｜/thinking｜>'];

    const handleStreamStart = () => {
      setIsStreaming(true);
      inThinkingMode.current = false;
      tokenHistory.current = '';
      setMessages(prev => [...prev, { 
        sender: aiName,  // Use aiName state instead of hardcoded "Nova"
        message: '', 
        type: 'ai', 
        thought: '', 
        isThinking: false, 
        imageB64: null 
      }]);
    };

    const handleStream = (token: string) => {
      // This function handles both the visual display of tokens and the TTS buffering.
      
      // 1. Update the visual message state (handles <think> tags)
      tokenHistory.current += token;
      if (tokenHistory.current.length > MAX_HISTORY_LENGTH) {
        tokenHistory.current = tokenHistory.current.slice(-MAX_HISTORY_LENGTH);
      }
      const updateLastMessage = (updater: (lastMessage: Message) => Message) => {
        setMessages(prev => {
          if (prev.length === 0) return prev;
          const last = prev[prev.length - 1];
          if (last.sender === aiName) {  // Use aiName instead of hardcoded "Nova"
            const updatedLast = updater({ ...last });
            return [...prev.slice(0, -1), updatedLast];
          }
          return prev;
        });
      };
      if (inThinkingMode.current) {
        for (const endTag of THINKING_END_TOKENS) {
          if (tokenHistory.current.endsWith(endTag)) {
            inThinkingMode.current = false;
            const tagIndex = token.lastIndexOf(endTag.charAt(0));
            const contentBeforeTag = tagIndex !== -1 ? token.slice(0, tagIndex) : token;
            if (contentBeforeTag) {
              updateLastMessage(last => {
                last.thought = (last.thought || '') + contentBeforeTag;
                return last;
              });
            }
            tokenHistory.current = '';
            return;
          }
        }
        updateLastMessage(last => {
          last.thought = (last.thought || '') + token;
          last.isThinking = true;
          return last;
        });
        return;
      }
      if (!inThinkingMode.current) {
        for (const startTag of THINKING_START_TOKENS) {
          if (tokenHistory.current.endsWith(startTag)) {
            inThinkingMode.current = true;
            const tagIndex = token.lastIndexOf(startTag.charAt(0));
            const contentBeforeTag = tagIndex !== -1 ? token.slice(0, tagIndex) : '';
            if (contentBeforeTag) {
               updateLastMessage(last => {
                  last.message += contentBeforeTag;
                  return last;
               });
               if (isHandsFreeMode) ttsBufferRef.current += contentBeforeTag;
            }
            updateLastMessage(last => {
              last.isThinking = true;
              return last;
            });
            tokenHistory.current = '';
            return;
          }
        }
        updateLastMessage(last => {
          last.message += token;
          last.isThinking = false;
          return last;
        });
        
        // 2. Handle TTS buffering if in hands-free mode and not thinking
        if (isHandsFreeMode && !inThinkingMode.current) {
            ttsBufferRef.current += token;
            if (/[.!?]/.test(ttsBufferRef.current)) {
                if (socketRef.current) socketRef.current.emit('tts', { text: ttsBufferRef.current });
                ttsBufferRef.current = '';
            }
        }
      }
    };

    const handleStreamEnd = () => {
      if (streamingTimeoutRef.current) clearTimeout(streamingTimeoutRef.current);
      setIsStreaming(false);
      inThinkingMode.current = false;

      // Flush any remaining text in the TTS buffer
      if (isHandsFreeMode && ttsBufferRef.current.trim()) {
          if (socketRef.current) socketRef.current.emit('tts', { text: ttsBufferRef.current });
          ttsBufferRef.current = '';
      }

      // Finalize the message state and save to DB
      setMessages(prev => {
        if (prev.length === 0) return prev;
        const last = { ...prev[prev.length - 1] };
        if (last.sender === aiName) {
          last.isThinking = false;
          if (last.message.trim() && activeChatId) {
            const messageToSave = {
              sender: aiName,
              message: last.message,
              type: 'ai' as 'ai',
            };
            if (socketRef.current) {
              socketRef.current.emit('save_message', {
                session_id: activeChatId,
                message: messageToSave
              });
            }
          }
          return [...prev.slice(0, -1), last];
        }
        return prev;
      });
      tokenHistory.current = '';
    };

    const handleModelUnloaded = () => addMessage('System', 'Model unloaded.', 'info');
    const handleError = (data: { message: string }) => addMessage('System', `Error: ${data.message}`, 'error');
    const handleConfigSaved = (data: { message: string }) => addMessage('System', data.message, 'info');
    const handleModels = (data: { backend: string, models: string[] }) => {
      setAvailableModels(data.models);
      if (data.models && data.models.length > 0) {
        handleModelChange(data.models[0]);
      }
    };
    const handleConfigs = (data: { [key: string]: ModelConfig }) => setAllConfigs(data);
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
      const blob = new Blob([data.audio], { type: 'audio/mpeg' });
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
      if (visualizerRef.current) visualizerRef.current.startProcessing(data.text);
    };
    const handleCommandResponse = (data: { 
        type: string, 
        message: string, 
        url?: string, 
        embed_url?: string, 
        video_id?: string, 
        urls?: string[], 
        sender?: string,
        prompt?: string,
        query?: string,
        image_url?: string
    }) => {
        if (data.type === 'image_generation') {
            setImageGenerationPrompt(data.prompt || '');
            setShowImageGenerator(true);
        } else if (data.type === 'media_browser') {
            setMediaBrowserQuery(data.query || '');
            setShowMediaBrowser(true);
        }
        
        const newMessage: Message = {
            sender: data.sender || 'Nova',
            message: data.message,
            type: data.type === 'error' ? 'error' : 'ai',
            imageB64: null,
            iframeUrl: data.type === 'iframe' ? data.url : data.type === 'media_embed' ? data.embed_url : undefined,
            youtubeVideoId: data.type === 'youtube_embed' ? data.video_id : undefined,
            imageGalleryUrls: data.type === 'image_gallery' ? data.urls : undefined,
            imageUrl: data.type === 'image_generated' ? data.image_url : undefined,
        };
        setMessages(prev => [...prev, newMessage]);
    };

    const handleNovaSettingsLoaded = (data: any) => {
      setNovaSettings(data);
    };

    socket.on('stream_start', handleStreamStart);
    socket.on('stream', handleStream);
    socket.on('stream_end', handleStreamEnd);
    socket.on('model_unloaded', handleModelUnloaded);
    socket.on('error', handleError);
    socket.on('config_saved', handleConfigSaved);
    socket.on('models', handleModels);
    socket.on('configs', handleConfigs);
    socket.on('graph_data', handleGraphData);
    socket.on('voice_stream', handleVoiceStream);
    socket.on('voice_stream_end', handleVoiceStreamEnd);
    socket.on('transcription_result', handleTranscriptionResult);
    socket.on('command_response', handleCommandResponse);
    socket.on('nova_settings_loaded', handleNovaSettingsLoaded);

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleApiProviderChange = (provider: string) => {
    setApiProvider(provider);
    setAvailableModels([]);
    setSelectedModel('');
    socketRef.current.emit('set_backend', { backend: 'api', provider: provider });
  };
  const handleSendMessage = () => {
    const currentMessage = messageInputText; // Use the state variable
    if (currentMessage.trim() || imageToSend) {
      // Add user's message to chat immediately for display
      addMessage(userName, currentMessage, 'user', imageToSend); // Re-add this line

      const currentChatSession = chatSessions.find(session => session._id === activeChatId); // Define here
      const messagePayload: any = {
        text: currentMessage,
        history: currentChatSession?.messages || [],
        session_id: currentChatSession?._id,
        userName: userName,
        aiName: novaSettings.aiName,
        backend: currentBackend, // Use currentBackend state
        provider: apiProvider, // Use apiProvider state
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      };

      if (imageToSend) {
        messagePayload.image_base_64 = imageToSend;
        setImageToSend(null);
      }

      if (isAgentMode) {
        socketRef.current.emit('agent_command', messagePayload);
      } else if (isOrchestratorMode) {
        socketRef.current.emit('chat', { ...messagePayload, isOrchestratorMode: true });
      } else {
        socketRef.current.emit('chat', messagePayload);
      }
      setMessageInputText(''); // Clear the input after sending
    }
  };

  const handleStop = () => {
    socketRef.current.emit('stop');
    setIsStreaming(false);
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    if (tab === 'graph') {
      socketRef.current.emit('get_graph_data');
    }
  };

  const handleBackendChange = (backend: string) => {
    setCurrentBackend(backend);
    setSelectedModel('');
    setAvailableModels([]);
    socketRef.current.emit('set_backend', { backend });
  };
  // Add handler functions for image generation and media playback
  const handleGenerateImage = (prompt: string, model: string, settings: any) => {
      // Send request to your image generation API
      if (socketRef.current) {
          socketRef.current.emit('generate_image', {
              prompt,
              model,
              settings,
              session_id: activeChatId
          });
      }
  };

  const handlePlayMedia = (mediaId: string, mediaType: string) => {
      // Send request to your media server API
      if (socketRef.current) {
          socketRef.current.emit('play_media', {
              mediaId,
              mediaType,
              session_id: activeChatId
          });
          setShowMediaBrowser(false);
      }
  };
  const handleLoadModel = () => {
    if (!selectedModel) {
      addMessage('System', 'No model selected', 'error');
      return;
    }

    // Determine thinking mode based on the thinking level.
    const thinkingLevel = modelConfigOptions.thinking_level || 'none';
    const thinkingMode = thinkingLevel !== 'none';

    const payload: any = {
      model_path: selectedModel,
      backend: currentBackend,
      system_prompt: systemPrompt,
      ...modelConfigOptions, // Spread all options from the state
      thinking_mode: thinkingMode, // Send the derived boolean
      thinking_level: thinkingLevel, // Send the selected level
    };

    if (currentBackend === 'api') {
      payload.provider = apiProvider;
    }
    
    console.log('Loading model with payload:', payload);
    socketRef.current.emit('load_model', payload);
  };

  const handleUnloadModel = () => socketRef.current.emit('unload_model');
  const handleSaveConfig = () => {
    if (!selectedModel) return;
    const payload = {
      model_path: selectedModel,
      system_prompt: systemPrompt,
      ...modelConfigOptions,
      aiName: aiName,
      aiAvatar: aiAvatar
    };
    socketRef.current.emit('save_config', payload);
  };

  const handleUserAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setUserAvatar(base64);
        localStorage.setItem('nova_user_avatar', base64);
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
    if (files && files[0] && files[0].type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = (event.target?.result as string)?.split(',')[1];
        setImageToSend(base64);
      };
      reader.readAsDataURL(files[0]);
    }
  };

  const handleSaveToolSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit('save_tool_settings', toolSettings);
  };

  const handleSaveVoiceSettings = () => {
    if (!socketRef.current) return;
    socketRef.current.emit('save_voice_settings', {
      tts: ttsSettings,
      stt: sttSettings,
    });
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden" style={{ backgroundColor: theme.backgroundColor }} onDragOver={handleDragOver} onDrop={handleDrop}>
      {/* Sidebar */}
      <Sidebar 
        chats={chatSessions.map(c => ({id: c._id, name: c.name}))} 
        activeChatId={activeChatId} 
        onNewChat={newChat} 
        onLoadChat={loadChatById} 
        onDeleteChat={deleteChat} 
        onRenameChat={renameChat}
        isOpen={isSidebarOpen}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex-shrink-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-between p-2 z-10">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
            className="p-2 rounded-md hover:bg-gray-700 focus:outline-none text-white icon-button"
            aria-label="Open chat history"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </button>
          <div className="flex items-center space-x-6">
            <button onClick={() => handleTabChange('chat')} className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === 'chat' ? 'bg-blue-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} custom-button`}>Chat</button>
            <button onClick={() => handleTabChange('graph')} className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === 'graph' ? 'bg-blue-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} custom-button`}>Memory Graph</button>
            <button onClick={() => handleTabChange('settings')} className={`tab-button px-4 py-2 mx-2 rounded-lg transition-colors duration-300 ${activeTab === 'settings' ? 'bg-blue-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} custom-button`}>Settings</button>
          </div>
          <button 
            onClick={() => setIsVoicePanelOpen(!isVoicePanelOpen)} 
            className="p-2 rounded-md hover:bg-gray-700 focus:outline-none text-white icon-button"
            aria-label="Open voice panel"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
            </svg>
          </button>
        </header>

        {/* Tab Content Area */}
        <main className={`flex-1 flex flex-col min-h-0 relative ${activeTab !== 'chat' ? 'items-center' : ''}`}>
          {activeTab === 'chat' && (
            <div className="flex-1 flex flex-col min-h-0 p-4 relative">
              <ChatView messages={messages} userAvatar={userAvatar} aiAvatar={aiAvatar} />
              <div className="absolute bottom-4 left-4 right-4 z-10">
                <MessageInput 
                  onSendMessage={handleSendMessage} 
                  onStop={handleStop} 
                  isStreaming={isStreaming} 
                  image={imageToSend}
                  setImage={setImageToSend}
                            isAgentMode={isAgentMode}
                            onAgentModeChange={setIsAgentMode}
                            isOrchestratorMode={isOrchestratorMode} // New prop
                            onOrchestratorModeChange={setIsOrchestratorMode} // New prop
                  message={messageInputText} // Pass the state down
                  onMessageChange={setMessageInputText} // Pass the setter down
                />
              </div>
            </div>
          )}
          {activeTab === 'graph' && <MemoryGraph data={graphData} />}
          {activeTab === 'settings' && (
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
                    socketRef.current.emit('set_backend', {
                      backend: 'api',
                      provider: apiProvider,
                      api_key: apiKey,
                    })
                  }
                  onClearApiKey={() => {
                    setApiKey('');
                    socketRef.current.emit('set_backend', {
                      backend: 'api',
                      provider: apiProvider,
                      api_key: '',
                    });
                  }}
                  onModelChange={handleModelChange}
                  onModelConfigChange={(key: string, value: any) =>
                    setModelConfigOptions((prev) => ({ ...prev, [key]: value }))
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
                    socketRef.current.emit('manage_ollama', { action: 'stop' })
                  }
                  onRestartOllama={() =>
                    socketRef.current.emit('manage_ollama', {
                      action: 'restart',
                      env: { OLLAMA_KV_CACHE_TYPE: ollamaKvCache },
                    })
                  }
                  onSaveToolSettings={handleSaveToolSettings}
                  onSaveVoiceSettings={handleSaveVoiceSettings}
                  novaSettings={novaSettings}
                  onNovaSettingsChange={handleNovaSettingsChange}
                  onSaveNovaSettings={handleSaveNovaSettings}
                  debugMode={debugMode}
                  onDebugModeChange={setDebugMode}
                />
              </div>
            </div>
          )}
        </main>
        
        {/* Message Input Area */}
        
      </div>

      {/* Voice Panel */}
      <div className={`flex-shrink-0 h-full w-80 bg-gray-800/80 backdrop-blur-sm p-4 flex flex-col transform transition-transform duration-300 ease-in-out z-20 ${isVoicePanelOpen ? 'translate-x-0' : 'translate-x-full'}`} style={{ backgroundColor: 'var(--primary-color)' }}>
        <VoicePanel 
          onClose={() => setIsVoicePanelOpen(false)} 
          socket={socketRef.current}
          isHandsFreeMode={isHandsFreeMode}
          onHandsFreeModeChange={setIsHandsFreeMode}
        />
      </div>

      {/* Image Generator Modal */}
      {showImageGenerator && (
        <ImageGenerator
          prompt={imageGenerationPrompt}
          onGenerate={handleGenerateImage}
          onClose={() => setShowImageGenerator(false)}
        />
      )}

      {/* Media Browser Modal */}
      {showMediaBrowser && (
        <MediaBrowser
          query={mediaBrowserQuery}
          onPlay={handlePlayMedia}
          onClose={() => setShowMediaBrowser(false)}
          settings={novaSettings} // Pass novaSettings
        />
      )}
    </div>
  );
};


export default App;
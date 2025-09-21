import React from 'react';
import ThemeEditor from './ThemeEditor';
import ToolSettings from './ToolSettings';
import TtsPanel from './TtsPanel';
import SttPanel from './SttPanel';
import NovaCustomizations from './NovaCustomizations'; // Import the new component

// A generic interface for any model configuration
interface ModelConfig {
  [key: string]: any;
}

interface SettingsPanelProps {
  // State
  allConfigs: { [key: string]: ModelConfig };
  availableModels: string[];
  currentBackend: string;
  selectedModel: string;
  apiProvider: string;
  apiKey: string;
  modelConfigOptions: ModelConfig;
  systemPrompt: string;
  userName: string;
  aiName: string;
  ollamaKvCache: string;
  theme: { [key: string]: string };
  toolSettings: any;
  ttsSettings: any;
  sttSettings: any;
  novaSettings: any;

  // Callbacks
  onBackendChange: (backend: string) => void;
  onApiProviderChange: (provider: string) => void;
  onApiKeyChange: (key: string) => void;
  onSaveApiKey: () => void;
  onClearApiKey: () => void;
  onModelChange: (model: string) => void;
  onModelConfigChange: (key: string, value: any) => void;
  onSystemPromptChange: (prompt: string) => void;
  onLoadModel: () => void;
  onUnloadModel: () => void;
  onSaveConfig: () => void;
  onUserNameChange: (name: string) => void;
  onAiNameChange: (name: string) => void;
  onUserAvatarChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onAiAvatarChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onOllamaKvCacheChange: (value: string) => void;
  onStopOllama: () => void;
  onRestartOllama: () => void;
  onThemeChange: (property: string, value: string) => void;
  onToolSettingsChange: (settings: any) => void;
  onTtsSettingsChange: (settings: any) => void;
  onSttSettingsChange: (settings: any) => void;
  onSaveToolSettings: () => void;
  onSaveVoiceSettings: () => void;
  onNovaSettingsChange: (settings: any) => void;
  onSaveNovaSettings: () => void;
  debugMode: boolean;
  onDebugModeChange: (enabled: boolean) => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = (props) => {
  const {
    availableModels,
    currentBackend,
    selectedModel,
    apiProvider,
    apiKey,
    modelConfigOptions,
    systemPrompt,
    userName,
    aiName,
    onBackendChange,
    onApiProviderChange,
    onApiKeyChange,
    onSaveApiKey,
    onClearApiKey,
    onModelChange,
    onModelConfigChange,
    onSystemPromptChange,
    onLoadModel,
    onUnloadModel,
    onSaveConfig,
    onUserNameChange,
    onAiNameChange,
    onUserAvatarChange,
    onAiAvatarChange,
    theme,
    onThemeChange,
    ollamaKvCache,
    onOllamaKvCacheChange,
    onStopOllama,
    onRestartOllama,
    toolSettings,
    onToolSettingsChange,
    ttsSettings,
    onTtsSettingsChange,
    sttSettings,
    onSttSettingsChange,
    onSaveToolSettings,
    onSaveVoiceSettings,
    novaSettings,
    onNovaSettingsChange,
    onSaveNovaSettings,
    debugMode,
    onDebugModeChange,
  } = props;

  const renderBackendOptions = () => {
    const contextTokenOptions = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

    switch (currentBackend) {
      case 'llama.cpp':
        return (
          <>
            <label className="block mt-2">Context Tokens</label>
            <select
              value={modelConfigOptions.context_tokens || 8192}
              onChange={(e) =>
                onModelConfigChange('context_tokens', parseInt(e.target.value))
              }
              className="w-full p-2 rounded inputfield-background"
            >
              {contextTokenOptions.map((val) => (
                <option key={val} value={val}>
                  {val}
                </option>
              ))}
            </select>

            <label className="block mt-2">Temperature</label>
            <input
              type="number"
              step="0.01"
              value={modelConfigOptions.temperature || ''}
              onChange={(e) =>
                onModelConfigChange('temperature', parseFloat(e.target.value))
              }
              className="w-full p-2 rounded inputfield-background"
              placeholder="0.7"
            />

            <label className="block mt-2">GPU Layers</label>
            <input
              type="number"
              value={modelConfigOptions.gpuLayers || ''}
              onChange={(e) =>
                onModelConfigChange('gpuLayers', parseInt(e.target.value))
              }
              className="w-full p-2 rounded inputfield-background"
              placeholder="35"
            />

            <label className="block mt-2">Thinking Level</label>
            <select
              value={modelConfigOptions.thinking_level || 'none'}
              onChange={(e) =>
                onModelConfigChange('thinking_level', e.target.value)
              }
              className="w-full p-2 rounded inputfield-background"
            >
              <option value="none">None</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>

            <label className="block mt-2">KV Cache Quantization</label>
            <select
              value={modelConfigOptions.kv_cache_quant || 'fp16'}
              onChange={(e) =>
                onModelConfigChange('kv_cache_quant', e.target.value)
              }
              className="w-full p-2 rounded inputfield-background"
            >
              <option value="fp16">FP16</option>
              <option value="int8">INT8</option>
              <option value="int4">INT4</option>
            </select>
          </>
        );

      case 'ollama':
      case 'safetensors':
      case 'api':
        return (
          <>
            {currentBackend !== 'api' && (
              <>
                <label className="block mt-2">Context Tokens</label>
                <select
                  value={modelConfigOptions.context_tokens || 8192}
                  onChange={(e) =>
                    onModelConfigChange(
                      'context_tokens',
                      parseInt(e.target.value),
                    )
                  }
                  className="w-full p-2 rounded inputfield-background"
                >
                  {contextTokenOptions.map((val) => (
                    <option key={val} value={val}>
                      {val}
                    </option>
                  ))}
                </select>
              </>
            )}

            <label className="block mt-2">Temperature</label>
            <input
              type="number"
              step="0.01"
              value={modelConfigOptions.temperature || ''}
              onChange={(e) =>
                onModelConfigChange('temperature', parseFloat(e.target.value))
              }
              className="w-full p-2 rounded inputfield-background"
              placeholder="0.7"
            />

            {currentBackend === 'api' && (
              <>
                <label className="block mt-2">Max Tokens</label>
                <input
                  type="number"
                  value={modelConfigOptions.max_tokens || ''}
                  onChange={(e) =>
                    onModelConfigChange(
                      'max_tokens',
                      parseInt(e.target.value),
                    )
                  }
                  className="w-full p-2 rounded inputfield-background"
                  placeholder="4096"
                />
              </>
            )}

            {currentBackend === 'safetensors' && (
              <>
                <label className="block mt-2">Use Flash Attention</label>
                <input
                  type="checkbox"
                  checked={modelConfigOptions.use_flash_attention || false}
                  onChange={(e) =>
                    onModelConfigChange('use_flash_attention', e.target.checked)
                  }
                  className="form-checkbox h-5 w-5 text-blue-500"
                />

                <label className="block mt-2">KV Cache Quantization</label>
                <select
                  value={modelConfigOptions.kv_cache_quant || 'fp16'}
                  onChange={(e) =>
                    onModelConfigChange('kv_cache_quant', e.target.value)
                  }
                  className="w-full p-2 rounded inputfield-background"
                >
                  <option value="fp16">FP16</option>
                  <option value="int8">INT8</option>
                  <option value="int4">INT4</option>
                </select>

                <label className="block mt-2">Quantization</label>
                <select
                  value={modelConfigOptions.quantization || 'none'}
                  onChange={(e) =>
                    onModelConfigChange('quantization', e.target.value)
                  }
                  className="w-full p-2 rounded inputfield-background"
                >
                  <option value="none">None</option>
                  <option value="4bit">4-bit</option>
                  <option value="8bit">8-bit</option>
                </select>
              </>
            )}
            
            {(currentBackend === 'ollama' || currentBackend === 'safetensors') && (
                 <>
                    <label className="block mt-2">Thinking Level</label>
                    <select
                      value={modelConfigOptions.thinking_level || 'none'}
                      onChange={(e) =>
                        onModelConfigChange('thinking_level', e.target.value)
                      }
                      className="w-full p-2 rounded inputfield-background"
                    >
                      <option value="none">None</option>
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                 </>
            )}

            {currentBackend === 'ollama' && (
              <>
                <div className="setting-section mt-4">
                  <h3 className="text-lg font-semibold mb-2">Ollama Server</h3>
                  <div className="mb-4">
                    <label
                      htmlFor="ollamaKvCache"
                      className="block text-sm font-medium mb-1"
                    >
                      KV Cache Quantization
                    </label>
                    <select
                      id="ollamaKvCache"
                      value={ollamaKvCache}
                      onChange={(e) => onOllamaKvCacheChange(e.target.value)}
                      className="w-full p-2 rounded bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="f16">f16</option>
                      <option value="q8_0">q8_0</option>
                      <option value="q4_0">q4_0</option>
                    </select>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={onStopOllama}
                      className="btn-danger flex-1"
                    >
                      Stop Ollama
                    </button>
                    <button
                      onClick={onRestartOllama}
                      className="btn-primary flex-1"
                    >
                      Apply & Restart Ollama
                    </button>
                  </div>
                </div>
              </>
            )}
          </>
        );

      default:
        return null;
    }
  };

  return (
    <div className="tab-content flex-1 p-4 space-y-4 overflow-y-auto h-full max-h-[calc(100vh-6rem)]">
      <div className="w-full max-w-[144rem] mx-auto">
        <h2 className="text-2xl font-bold text-center mb-4">Settings</h2>

        <div className="flex flex-wrap -mx-2">
          <div className="w-full md:w-1/2 px-2">
            <ThemeEditor theme={theme} onThemeChange={onThemeChange} />
          </div>
          <div className="w-full md:w-1/2 px-2">
            <div
              className="bg-gray-800/50 p-4 rounded-lg"
              style={{ backgroundColor: 'var(--primary-color)' }}
            >
              <h3 className="text-xl mb-2">User & AI Profile</h3>
              <div className="space-y-4">
                <div>
                  <label className="block mb-1">User Name</label>
                  <input
                    type="text"
                    id="user-name-input"
                    className="w-full p-2 rounded inputfield-background"
                    placeholder="Enter your name"
                    value={userName}
                    onChange={(e) => onUserNameChange(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block mb-1">User Avatar</label>
                  <label className="p-2 rounded custom-button cursor-pointer inline-block text-center">
                    Choose File
                    <input
                      type="file"
                      id="user-avatar-input"
                      accept="image/*"
                      className="hidden"
                      onChange={onUserAvatarChange}
                    />
                  </label>
                </div>
                <div>
                  <label className="block mb-1">AI Name</label>
                  <input
                    type="text"
                    id="ai-name-input"
                    className="w-full p-2 rounded inputfield-background"
                    placeholder="Enter AI's name"
                    value={aiName}
                    onChange={(e) => onAiNameChange(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block mb-1">AI Avatar</label>
                  <label className="p-2 rounded custom-button cursor-pointer inline-block text-center">
                    Choose File
                    <input
                      type="file"
                      id="ai-avatar-input"
                      accept="image/*"
                      className="hidden"
                      onChange={onAiAvatarChange}
                    />
                  </label>
                </div>
                <div className="flex items-center justify-between">
                  <label htmlFor="debug-mode-toggle" className="block mb-1">Debug Mode</label>
                  <input
                    type="checkbox"
                    id="debug-mode-toggle"
                    checked={debugMode}
                    onChange={(e) => onDebugModeChange(e.target.checked)}
                    className="form-checkbox h-5 w-5 text-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap -mx-2 mt-4">
          <div className="w-full md:w-1/2 px-2">
            <ToolSettings settings={toolSettings} onChange={onToolSettingsChange} onSave={onSaveToolSettings} />
          </div>
          <div className="w-full md:w-1/2 px-2">
            <TtsPanel settings={ttsSettings} onChange={onTtsSettingsChange} onSave={onSaveVoiceSettings} />
            <div className="mt-4">
              <SttPanel settings={sttSettings} onChange={onSttSettingsChange} onSave={onSaveVoiceSettings} />
            </div>
          </div>
        </div>

        {/* Nova Customizations Section */}
        <div className="mt-4">
          <NovaCustomizations settings={novaSettings} onChange={onNovaSettingsChange} onSave={onSaveNovaSettings} />
        </div>

        <div className="bg-gray-800/50 p-4 rounded-lg mt-4" style={{ backgroundColor: 'var(--primary-color)' }}>
          <h3 className="text-xl mb-2 text-center">Backend Selection</h3>
          <div className="flex items-center justify-center space-x-4">
            {['llama.cpp', 'ollama', 'api', 'safetensors'].map((backend) => (
              <label key={backend} className="inline-flex items-center">
                <input
                  type="radio"
                  name="backend"
                  value={backend}
                  checked={currentBackend === backend}
                  onChange={() => onBackendChange(backend)}
                  className="form-radio h-5 w-5 text-blue-500"
                />
                <span className="ml-2">{backend}</span>
              </label>
            ))}
          </div>
        </div>

        {currentBackend === 'api' && (
          <div className="bg-gray-800/50 p-4 rounded-lg mt-4 max-w-2xl mx-auto" style={{ backgroundColor: 'var(--primary-color)' }}>
            <h3 className="text-xl mb-2">API Provider</h3>
            <select
              value={apiProvider}
              onChange={(e) => onApiProviderChange(e.target.value)}
              className="w-full p-2 rounded inputfield-background mb-2"
            >
              {[ 
                'openai',
                'google',
                'anthropic',
                'meta',
                'xai',
                'qwen',
                'deepseek',
                'perplexity',
                'openrouter',
              ].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <h3 className="text-xl mb-2 mt-4">API Key</h3>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
              className="w-full p-2 rounded inputfield-background"
              placeholder="Enter your API Key"
            />
            <div className="mt-2 flex gap-2">
              <button
                onClick={onSaveApiKey}
                className="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded custom-button"
              >
                Save Key
              </button>
              <button
                onClick={onClearApiKey}
                className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded custom-button"
              >
                Clear Key
              </button>
            </div>
          </div>
        )}

        <div className="bg-gray-800/50 p-4 rounded-lg mt-4 max-w-2xl mx-auto" style={{ backgroundColor: 'var(--primary-color)' }}>
          <h3 className="text-xl mb-2 text-center">LLM Model Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => onModelChange(e.target.value)}
                className="w-full p-2 rounded inputfield-background"
              >
                {availableModels.map((model) => (
                  <option key={model} value={model}>
                    {model.split(/[/\\]/).pop()}
                  </option>
                ))}
              </select>
            </div>

            <div id="model-config-options">{renderBackendOptions()}</div>

            <div>
              <label className="block mb-1">System Prompt</label>
              <textarea
                value={systemPrompt}
                onChange={(e) => onSystemPromptChange(e.target.value)}
                className="w-full p-2 rounded inputfield-background h-24"
                placeholder="e.g., You are a helpful assistant..."
              ></textarea>
            </div>
          </div>

          <div className="flex gap-2 mt-4 justify-center">
            <button
              onClick={onLoadModel}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded custom-button"
            >
              Load Model
            </button>
            <button
              onClick={onUnloadModel}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded custom-button"
            >
              Unload Model
            </button>
            <button
              onClick={onSaveConfig}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded custom-button"
            >
              Save Config
            </button>
          </div>
        </div>

        <div className="bg-gray-800/50 p-4 rounded-lg mt-4 text-center" style={{ backgroundColor: 'var(--primary-color)' }}>
            <h3 className="text-xl mb-2">Credits</h3>
            <p className="text-gray-300">A special thank you to Gemini-2.5-Pro and GeminiCLI. Without the coding help from this model, I would not have made it even half as far as I did.</p>
        </div>

      </div>
    </div>
  );
};

export default SettingsPanel;

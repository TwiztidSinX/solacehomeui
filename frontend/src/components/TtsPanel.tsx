import React from 'react';

interface TtsPanelProps {
  settings: {
    type: 'local' | 'cloud';
    provider: string;
    url: string;
    apiKey: string;
    model: string;
    voice: string;
  };
  onChange: (newSettings: any) => void;
  onSave: () => void;
}

const TtsPanel: React.FC<TtsPanelProps> = ({ settings, onChange, onSave }) => {
  const handleTypeChange = (type: 'local' | 'cloud') => {
    onChange({ ...settings, type });
  };

  const handleChange = (field: string, value: string) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <div className="bg-gray-800/50 p-4 rounded-lg" style={{ backgroundColor: 'var(--primary-color)' }}>
      <h3 className="text-xl mb-2">Text-to-Speech Settings</h3>
      <div className="flex items-center space-x-4 mb-4">
        <button onClick={() => handleTypeChange('local')} className={`px-4 py-2 rounded ${settings.type === 'local' ? 'bg-blue-600' : 'bg-gray-600'}`}>Local</button>
        <button onClick={() => handleTypeChange('cloud')} className={`px-4 py-2 rounded ${settings.type === 'cloud' ? 'bg-blue-600' : 'bg-gray-600'}`}>Cloud</button>
      </div>
      <div className="space-y-4">
        {settings.type === 'cloud' && (
          <div>
            <label className="block mb-1">Provider</label>
            <select value={settings.provider} onChange={(e) => handleChange('provider', e.target.value)} className="w-full p-2 rounded inputfield-background">
              <option value="openai">OpenAI</option>
              <option value="google">Google</option>
            </select>
          </div>
        )}
        <div>
          <label className="block mb-1">URL</label>
          <input type="text" value={settings.url} onChange={(e) => handleChange('url', e.target.value)} className="w-full p-2 rounded inputfield-background" />
        </div>
        {settings.type === 'cloud' && (
          <div>
            <label className="block mb-1">API Key</label>
            <input type="password" value={settings.apiKey} onChange={(e) => handleChange('apiKey', e.target.value)} className="w-full p-2 rounded inputfield-background" />
          </div>
        )}
        <div>
          <label className="block mb-1">Model</label>
          <input type="text" value={settings.model} onChange={(e) => handleChange('model', e.target.value)} className="w-full p-2 rounded inputfield-background" />
        </div>
        <div>
          <label className="block mb-1">Voice</label>
          <input type="text" value={settings.voice} onChange={(e) => handleChange('voice', e.target.value)} className="w-full p-2 rounded inputfield-background" />
        </div>
      </div>
      <button onClick={onSave} className="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded custom-button">
        Save TTS Settings
      </button>
    </div>
  );
};

export default TtsPanel;
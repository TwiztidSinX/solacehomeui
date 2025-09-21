import React from 'react';

interface ToolSettingsProps {
  settings: {
    n8nUrl: string;
    searXngUrl: string;
    serpApiApiKey: string;
  };
  onChange: (newSettings: any) => void;
  onSave: () => void;
}

const ToolSettings: React.FC<ToolSettingsProps> = ({ settings, onChange, onSave }) => {
  const handleChange = (field: string, value: string) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <div className="bg-gray-800/50 p-4 rounded-lg" style={{ backgroundColor: 'var(--primary-color)' }}>
      <h3 className="text-xl mb-2">Tool Servers</h3>
      <div className="space-y-4">
        <div>
          <label className="block mb-1">n8n URL</label>
          <input 
            type="text" 
            value={settings.n8nUrl}
            onChange={(e) => handleChange('n8nUrl', e.target.value)}
            className="w-full p-2 rounded inputfield-background" 
          />
        </div>
        <div>
          <label className="block mb-1">SearXNG URL</label>
          <input 
            type="text" 
            value={settings.searXngUrl}
            onChange={(e) => handleChange('searXngUrl', e.target.value)}
            className="w-full p-2 rounded inputfield-background" 
            placeholder="http://localhost:8181" 
          />
        </div>
        <div>
          <label className="block mb-1">SerpAPI API Key</label>
          <input 
            type="password" 
            value={settings.serpApiApiKey}
            onChange={(e) => handleChange('serpApiApiKey', e.target.value)}
            className="w-full p-2 rounded inputfield-background" 
          />
        </div>
      </div>
      <button onClick={onSave} className="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded custom-button">
        Save Tool Settings
      </button>
    </div>
  );
};

export default ToolSettings;
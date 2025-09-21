import React from 'react';

interface NovaCustomizationsProps {
  settings: {
    searxngUrl: string;
    mediaServerUrl: string;
    mediaServerApiKey: string;
    imageGenUrl: string;
  };
  onChange: (newSettings: any) => void;
  onSave: () => void;
}

const NovaCustomizations: React.FC<NovaCustomizationsProps> = ({ settings, onChange, onSave }) => {
  const handleChange = (field: string, value: string) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <div className="p-4 bg-gray-800 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-white border-b border-gray-600 pb-2">Nova Customizations</h2>
      
      {/* SearXNG Settings */}
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-3 text-gray-300">Search Provider</h3>
        <div className="flex flex-col space-y-2">
          <label htmlFor="searxng-url" className="text-gray-400">SearXNG Instance URL</label>
          <input
            id="searxng-url"
            type="text"
            placeholder="https://searx.example.com"
            className="p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={settings.searxngUrl}
            onChange={(e) => handleChange('searxngUrl', e.target.value)}
          />
        </div>
      </div>

      {/* Media Server Settings */}
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-3 text-gray-300">Media Server</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col space-y-2">
            <label htmlFor="media-server-url" className="text-gray-400">Jellyfin/Plex/Emby URL</label>
            <input
              id="media-server-url"
              type="text"
              placeholder="http://192.168.1.100:8096"
              className="p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={settings.mediaServerUrl}
              onChange={(e) => handleChange('mediaServerUrl', e.target.value)}
            />
          </div>
          <div className="flex flex-col space-y-2">
            <label htmlFor="media-server-key" className="text-gray-400">API Key</label>
            <input
              id="media-server-key"
              type="password"
              placeholder="Enter API Key"
              className="p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={settings.mediaServerApiKey}
              onChange={(e) => handleChange('mediaServerApiKey', e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Image Generation Settings */}
      <div>
        <h3 className="text-xl font-semibold mb-3 text-gray-300">Image Generation</h3>
        <div className="flex flex-col space-y-2">
          <label htmlFor="sd-url" className="text-gray-400">A1111/ComfyUI URL</label>
          <input
            id="sd-url"
            type="text"
            placeholder="http://127.0.0.1:7860"
            className="p-2 rounded bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={settings.imageGenUrl}
            onChange={(e) => handleChange('imageGenUrl', e.target.value)}
          />
        </div>
      </div>

      <div className="mt-6 text-right">
        <button
          onClick={onSave}
          className="px-6 py-2 rounded bg-blue-600 text-white font-bold hover:bg-blue-700 transition-colors duration-300"
        >
          Save Settings
        </button>
      </div>

    </div>
  );
};

export default NovaCustomizations;

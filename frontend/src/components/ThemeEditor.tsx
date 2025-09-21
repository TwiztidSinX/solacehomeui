import React from 'react';

interface ThemeEditorProps {
  theme: { [key: string]: string };
  onThemeChange: (property: string, value: string) => void;
}

const ThemeEditor: React.FC<ThemeEditorProps> = ({ theme, onThemeChange }) => {
  return (
    <div className="p-4 rounded-lg" style={{ backgroundColor: 'var(--primary-color)' }}>
      <h3 className="text-xl mb-2">Theme Editor</h3>
      <div className="grid grid-cols-1 gap-4">
        {Object.keys(theme).map((key) => (
          <div key={key} className="flex items-center">
            <label className="capitalize w-1/2">{key.replace(/([A-Z])/g, ' $1')}</label>
            <input
              type="color"
              value={theme[key]}
              onChange={(e) => onThemeChange(key, e.target.value)}
              className="w-16 h-8 p-1"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default ThemeEditor;

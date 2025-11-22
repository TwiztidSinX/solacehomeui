import React from 'react';
import MonacoEditor from '@monaco-editor/react';

interface MonacoCodeEditorProps {
  code: string;
  language: string;
  onChange: (value: string) => void;
  theme?: string;
}

const MonacoCodeEditor: React.FC<MonacoCodeEditorProps> = ({
  code,
  language,
  onChange,
  theme = 'vs-dark'
}) => {
  // Determine language based on file extension or content
  const getLanguage = () => {
    if (language) return language;
    return 'plaintext';
  };

  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      onChange(value);
    }
  };

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 h-full">
        <MonacoEditor
          height="100%"
          language={getLanguage()}
          value={code}
          theme={theme}
          onChange={handleEditorChange}
          options={{
            minimap: { enabled: true },
            fontSize: 14,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            'semanticHighlighting.enabled': true,
            wordWrap: 'on',
            tabSize: 2,
            insertSpaces: true,
            autoIndent: 'advanced',
            formatOnPaste: true,
            formatOnType: true,
            quickSuggestions: true,
            parameterHints: { enabled: true },
            suggestOnTriggerCharacters: true,
            acceptSuggestionOnCommitCharacter: true,
            acceptSuggestionOnEnter: 'on',
            snippetSuggestions: 'top',
            wordBasedSuggestions: 'off',
            selectionHighlight: true,
            occurrencesHighlight: 'singleFile',
            folding: true,
            lineNumbers: 'on',
            renderLineHighlight: 'all',
            scrollbar: {
              vertical: 'auto',
              horizontal: 'auto',
            },
            overviewRulerLanes: 2,
            overviewRulerBorder: false,
          }}
        />
      </div>
    </div>
  );
};

export default MonacoCodeEditor;

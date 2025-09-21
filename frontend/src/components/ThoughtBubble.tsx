import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import LoadingSpinner from './LoadingSpinner';

interface ThoughtBubbleProps {
  thought: string | undefined;
  isThinking: boolean | undefined;
}

const ThoughtBubble: React.FC<ThoughtBubbleProps> = ({ thought, isThinking }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Automatically expand the thought bubble when the first thought content arrives
  useEffect(() => {
    if (thought && !isExpanded) {
      setIsExpanded(true);
    }
  }, [thought]);


  if (!isThinking && !thought?.trim()) {
    return null;
  }

  return (
    <div className="bg-gray-800/60 rounded-lg mb-2 border border-gray-700/80 shadow-inner">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full text-left p-2 text-sm text-gray-300 hover:bg-gray-700/50 focus:outline-none rounded-t-lg flex items-center"
      >
        {isExpanded ? '▼ Hide Thought' : '► Show Thought'}
        {isThinking && <LoadingSpinner text="Thinking..." />}
      </button>
      {isExpanded && thought?.trim() && (
        <div className="p-3 border-t border-gray-700/80">
          <div className="prose prose-sm prose-invert max-w-none">
            <ReactMarkdown>{thought}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
};

export default ThoughtBubble;

import React, { useEffect, useRef } from 'react';
import { VoiceVisualizer } from '../VoiceVisualizer';

interface VoicePanelProps {
  onClose: () => void;
}

const VoicePanel: React.FC<VoicePanelProps> = ({ onClose }) => {
  const visualizerRef = useRef<VoiceVisualizer | null>(null);

  useEffect(() => {
    visualizerRef.current = new VoiceVisualizer();
    visualizerRef.current.startListening();

    return () => {
      visualizerRef.current?.stop();
    };
  }, []);

  return (
    <>
      <button id="voice-panel-close" className="self-end p-2 rounded-md hover:bg-white/20" onClick={onClose}>&times;</button>
      <h2 className="text-xl font-bold mb-4">Voice Mode</h2>
      <div className="flex-grow flex flex-col items-center justify-center space-y-4">
        <div id="visualizer-container" className="w-48 h-48 flex items-center justify-center">
          <img id="voice-logo" src="/nova-logo.png" alt="Logo" className="w-full h-full object-contain"/>

          <canvas id="waveform-canvas" className="hidden w-full h-full"></canvas>
        </div>
        <div id="processing-dots" className="hidden">
          <div></div><div></div><div></div>
        </div>
        <p id="voice-status-label" className="text-lg text-gray-300 h-8">Ready</p>
      </div>
    </>
  );
};

export default VoicePanel;
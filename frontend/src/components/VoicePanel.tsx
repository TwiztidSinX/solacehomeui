import React, { useState, useEffect, useRef } from 'react';
import { VoiceVisualizer } from '../VoiceVisualizer';
import { Socket } from 'socket.io-client'; // Assuming you pass the socket instance as a prop

interface VoicePanelProps {
  onClose: () => void;
  socket: Socket | null;
  isHandsFreeMode: boolean;
  onHandsFreeModeChange: (isHandsFree: boolean) => void;
}

const VoicePanel: React.FC<VoicePanelProps> = ({ onClose, socket, isHandsFreeMode, onHandsFreeModeChange }) => {
  const visualizerRef = useRef<VoiceVisualizer | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Refs for the DOM elements
  const logoRef = useRef<HTMLImageElement>(null);
  const dotsRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const statusLabelRef = useRef<HTMLParagraphElement>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('Ready');

  useEffect(() => {
    // Ensure all refs are attached to DOM elements before creating the visualizer
    if (logoRef.current && dotsRef.current && canvasRef.current && statusLabelRef.current) {
      visualizerRef.current = new VoiceVisualizer(
        logoRef.current,
        dotsRef.current,
        canvasRef.current,
        statusLabelRef.current
      );
    }

    return () => {
      visualizerRef.current?.stop();
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  const startRecording = async () => {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        if (socket) {
          socket.emit('transcribe', { audio: audioBlob });
          setStatus('Transcribing...');
        }
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus('Listening...');
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setStatus('Mic access denied');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <>
      <button id="voice-panel-close" className="self-end p-2 rounded-md hover:bg-white/20" onClick={onClose}>&times;</button>
      <h2 className="text-xl font-bold mb-4">Voice Mode</h2>
      <div className="flex-grow flex flex-col items-center justify-center space-y-4">
        <div id="visualizer-container" className="w-48 h-48 flex items-center justify-center">
          <img id="voice-logo" ref={logoRef} src="/nova-logo.png" alt="Logo" className="w-full h-full object-contain"/>
          <canvas id="waveform-canvas" ref={canvasRef} className="hidden w-full h-full"></canvas>
          <div id="processing-dots" ref={dotsRef} className="hidden">
            <div></div><div></div><div></div>
          </div>
        </div>
        
        {/* Hands-Free Button */}
        <button
          onClick={() => onHandsFreeModeChange(!isHandsFreeMode)}
          className={`mb-4 px-4 py-2 rounded-lg text-white font-semibold transition-colors duration-200 ${isHandsFreeMode ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
        >
          Hands-Free: {isHandsFreeMode ? 'On' : 'Off'}
        </button>

        {/* Mic Button */}
        <button 
          onClick={toggleRecording}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors duration-300 ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'}`}
        >
          <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path>
          </svg>
        </button>

        <p id="voice-status-label" ref={statusLabelRef} className="text-lg text-gray-300 h-8">{status}</p>
      </div>
    </>
  );
};

export default VoicePanel;
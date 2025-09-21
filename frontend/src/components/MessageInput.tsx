import React, { useState, useRef, useEffect } from 'react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  image: string | null;
  setImage: (image: string | null) => void;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, onStop, isStreaming, image, setImage }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSend = () => {
    if (message.trim() || image) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleImagePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          const reader = new FileReader();
          reader.onload = (event) => {
            const base64 = (event.target?.result as string)?.split(',')[1];
            setImage(base64);
          };
          reader.readAsDataURL(file);
        }
      }
    }
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 pt-0 flex-shrink-0">
      <div id="drop-zone" className="glass rounded-xl p-4" style={{ backgroundColor: 'var(--chat-input-background-color)' }}>
        {image && (
          <div id="image-preview-container" className="relative w-24 h-24 mb-2">
            <img id="image-preview" src={`data:image/png;base64,${image}`} className="w-full h-full object-cover rounded-md" />
            <button 
              id="remove-image-btn" 
              className="absolute top-0 right-0 bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center -mt-2 -mr-2 text-lg font-bold"
              onClick={() => setImage(null)}
            >
              &times;
            </button>
          </div>
        )}
        <div className="flex items-end">
          <textarea
            ref={textareaRef}
            id="user-input"
            className="flex-grow bg-transparent border-none focus:outline-none text-white placeholder-gray-300 resize-none max-h-32 overflow-y-auto"
            placeholder="Type your message or drop/paste an image..."
            rows={1}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handleImagePaste}
            style={{ minHeight: '44px' }}
          />
          {isStreaming ? (
            <button id="stop-button" className="ml-4 bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button id="send-button" className="ml-4 bg-blue-500 hover:bg-blue-600 text-white font-bold p-2 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0" onClick={handleSend}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageInput;
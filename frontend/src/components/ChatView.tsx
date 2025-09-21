import React, { useEffect, useRef } from 'react';
import { type Message as MessageType } from '../types';
import Message from './Message';

interface ChatViewProps {
  messages: MessageType[];
  userAvatar: string | null;
  aiAvatar: string | null;
}

const ChatView: React.FC<ChatViewProps> = ({ messages, userAvatar, aiAvatar }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div id="chat-container" className="flex-1 space-y-4 overflow-y-auto p-4 rounded-xl m-4" style={{ backgroundColor: 'var(--chat-background-color)', paddingBottom: '150px' }}>
      {messages.map((msg, index) => (
        <Message 
          key={index} 
          {...msg} // Spread all message properties as props
          avatar={msg.type === 'user' ? userAvatar : aiAvatar}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatView;

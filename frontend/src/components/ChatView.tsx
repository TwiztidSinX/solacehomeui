import React, { useEffect, useRef, useState } from 'react';
import { type Message as MessageType } from '../types';
import Message from './Message';

interface ChatViewProps {
  messages: MessageType[];
  userAvatar: string | null;
  aiAvatar: string | null;
}

const ChatView: React.FC<ChatViewProps> = ({ messages, userAvatar, aiAvatar }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll only if user is at bottom
  useEffect(() => {
    if (!autoScroll) return;

    const container = containerRef.current;
    if (!container) return;

    container.scrollTo({
      top: container.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, autoScroll]);

  // Detect if user scrolled up
  const handleScroll = () => {
    const container = containerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setAutoScroll(isNearBottom);
  };

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      id="chat-container"
      className="flex-1 space-y-4 overflow-y-auto p-4 rounded-xl m-4 panel-surface"
      style={{ backgroundColor: 'var(--chat-background-color)', paddingBottom: '150px' }}
    >
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

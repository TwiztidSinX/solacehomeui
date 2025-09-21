import React, { useEffect } from 'react';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';
import ThoughtBubble from './ThoughtBubble';

interface MessageProps {
  sender: string;
  message: string;
  type?: 'ai' | 'user' | 'system' | 'error' | 'info';
  imageB64?: string | null;
  avatar: string | null;
  isThinking?: boolean;
  thought?: string;
  iframeUrl?: string;
  youtubeVideoId?: string;
  imageGalleryUrls?: string[];
}

const formatMarkdown = (content: string) => {
  // Bold/Italic
  content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Code blocks — FIXED: Triple backtick support
  content = content.replace(/```([a-z]*)\n([\s\S]*?)\n```/g, (_, lang, code) => {
    const languageClass = lang ? `language-${lang}` : '';
    // Use highlightAuto for language detection, or the specified lang
    const highlightedCode = lang ? hljs.highlight(code, { language: lang }).value : hljs.highlightAuto(code).value;
    return `<pre><code class="hljs ${languageClass}">${highlightedCode}</code></pre>`;
  });

  // Inline code
  content = content.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // Bullet points — FIXED: Group into single <ul>
  content = content.replace(/^-\s+(.+)$/gm, '<li>$1</li>');
  content = content.replace(/(<li>.*?<\/li>\s*)+/gs, (matchedItems) => {
    return `<ul class="list-disc pl-5">${matchedItems.trim()}</ul>`;
  });

  return content;
};


const Message: React.FC<MessageProps> = ({
  sender,
  message,
  type = 'ai',
  imageB64,
  avatar,
  isThinking,
  thought,
  iframeUrl,
  youtubeVideoId,
  imageGalleryUrls,
}) => {
  useEffect(() => {
    // This re-highlights blocks when message content changes (e.g., streaming)
    document.querySelectorAll('pre code:not(.hljs)').forEach((block) => {
      hljs.highlightElement(block as HTMLElement);
    });
  }, [message]);

  const isUser = type === 'user';

  const messageClass = 'p-4 rounded-lg glass max-w-6xl w-full'; // Expanded width for embeds
  const messageStyle = {
    backgroundColor:
      isUser
        ? 'var(--user-message-color)'
        : 'var(--ai-message-color)',
  };

  const renderMessageContent = () => {
    if (iframeUrl) {
      return (
        <div className="w-full">
          <div className="bg-gray-900/80 text-white p-2 rounded-t-lg flex justify-between items-center">
            <span className="text-sm truncate">{iframeUrl}</span>
            <a href={iframeUrl} target="_blank" rel="noopener noreferrer" className="ml-4 px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 rounded">
              Open in New Tab
            </a>
          </div>
          <div className="resize overflow-auto border border-gray-600 rounded-b-lg" style={{ minWidth: '560px' }}>
            <iframe
              src={iframeUrl}
              className="w-full h-96"
              title="Search Result"
            />
          </div>
        </div>
      );
    }
    if (youtubeVideoId) {
      return (
        <iframe
          width="560"
          height="315"
          src={`https://www.youtube.com/embed/${youtubeVideoId}`}
          title="YouTube video player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="rounded-lg"
        ></iframe>
      );
    }
    if (imageGalleryUrls) {
      return (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {imageGalleryUrls.map((url, index) => (
            <img key={index} src={url} alt={`Search result ${index + 1}`} className="rounded-lg object-cover h-40 w-full" />
          ))}
        </div>
      )
    }
    return (
      <>
        <div
          className="message-content"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(message) }}
        />
        {imageB64 && (
          <img
            src={`data:image/png;base64,${imageB64}`}
            className="mt-2 rounded-lg max-w-xs"
            alt="User upload"
          />
        )}
      </>
    );
  };

  const messageBody = (
    <div className={messageClass} style={messageStyle}>
        <p className="font-bold">{sender}</p>
        {(isThinking || thought) && (
          <ThoughtBubble isThinking={isThinking} thought={thought} />
        )}
        {renderMessageContent()}
      </div>
  );

  const avatarImage = (
     <img
        src={avatar || (isUser ? '/default-user.png' : '/nova-logo.png')}
        alt="Avatar"
        className="w-8 h-8 rounded-full object-cover flex-shrink-0"
      />
  );

  return (
    <div className={`w-full flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className="flex items-start gap-4 max-w-full">
        {isUser ? (
            <>
                {messageBody}
                {avatarImage}
            </>
        ) : (
            <>
                {avatarImage}
                {messageBody}
            </>
        )}
      </div>
    </div>
  );
};

export default Message;

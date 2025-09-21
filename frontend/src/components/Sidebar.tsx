import React, { useState } from 'react';

interface Chat {
  id: string; // Changed to string for MongoDB _id
  name: string;
}

interface SidebarProps {
  chats: Chat[];
  activeChatId: string | null; // Changed to string
  onNewChat: () => void;
  onLoadChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
  onRenameChat: (id: string, newName: string) => void;
  isOpen: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  chats,
  activeChatId,
  onNewChat,
  onLoadChat,
  onDeleteChat,
  onRenameChat,
  isOpen,
}) => {
  const [renamingId, setRenamingId] = useState<string | null>(null); // Changed to string
  const [newName, setNewName] = useState('');

  const handleRename = (id: string, currentName: string) => {
    setRenamingId(id);
    setNewName(currentName);
  };

  const handleRenameSubmit = (id: string) => {
    if (newName.trim()) {
      onRenameChat(id, newName.trim());
    }
    setRenamingId(null);
  };

  return (
    <div
      id="sidebar"
      className={`h-full w-64 bg-gray-800/80 backdrop-blur-sm p-4 flex flex-col transition-transform duration-300 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}
    >
      <button
        id="new-chat-btn"
        className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300 mb-4 custom-button"
        onClick={onNewChat}
      >
        + New Chat
      </button>
      <div className="flex-grow overflow-y-auto">
        <h2 className="text-lg font-semibold mb-2">Previous Chats</h2>
        <div id="chatList">
          {chats.map((chat) => (
            <div key={chat.id} className="group">
              {renamingId === chat.id ? (
                <div className="flex items-center">
                  <div className="flex items-center">
                    <label htmlFor={`rename-input-${chat.id}`} className="sr-only">
                      Rename chat to:
                    </label>
                    <input
                      id={`rename-input-${chat.id}`}
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      onBlur={() => handleRenameSubmit(chat.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRenameSubmit(chat.id);
                        if (e.key === 'Escape') setRenamingId(null);
                      }}
                      className="w-full bg-gray-700 text-white p-1 rounded"
                      placeholder="Enter new chat name"
                      autoFocus
                      aria-label="Rename chat"
                    />
                  </div>
                </div>
              ) : (
                <button
                  className={`w-full text-left rounded px-3 py-1 mb-1 flex justify-between items-center ${
                    activeChatId === chat.id ? 'bg-white/20' : 'bg-white/10 hover:bg-white/20'
                  } custom-button`}
                  onClick={() => onLoadChat(chat.id)}
                >
                  <span className="truncate flex-1">{chat.name}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRename(chat.id, chat.name);
                    }}
                    className="opacity-0 group-hover:opacity-100 ml-1"
                  >
                    ✏️
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="ml-2 text-red-400 hover:text-red-600 cursor-pointer opacity-0 group-hover:opacity-100 transition-all duration-200 text-xl leading-none hover:scale-125"
                  >
                    &times;
                  </button>
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
import React, { useState } from "react";
import { Rnd } from "react-rnd";

interface MiniPlayerProps {
  videoId: string;
  onClose: () => void;
}

export const MiniPlayer: React.FC<MiniPlayerProps> = ({ videoId, onClose }) => {
  const [isMinimized, setIsMinimized] = useState(false);
  const [size, setSize] = useState({ width: 400, height: 225 });
  const [position, setPosition] = useState({ x: 100, y: 100 });

  if (!videoId) return null;

  const minimizedHeight = 48;

  return (
    <Rnd
      size={isMinimized ? { width: size.width, height: minimizedHeight } : size}
      position={position}
      onDragStop={(_e, d) => setPosition({ x: d.x, y: d.y })}
      onResizeStop={(_e, _direction, ref, _delta, position) => {
        setSize({
          width: parseInt(ref.style.width),
          height: parseInt(ref.style.height),
        });
        setPosition(position);
      }}
      minWidth={280}
      minHeight={isMinimized ? minimizedHeight : 180}
      maxWidth={800}
      maxHeight={600}
      bounds="window"
      className="mini-player-container"
      style={{ zIndex: 9999 }}
      enableResizing={!isMinimized}
    >
      <div className="h-full flex flex-col bg-gray-900 rounded-lg shadow-2xl border border-gray-700 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between bg-gray-800 px-3 py-2 cursor-move border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-red-500 text-sm">🎵</span>
            <span className="text-white text-sm font-semibold truncate">
              {isMinimized ? "YouTube Player" : "Mini Player"}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="p-1 hover:bg-gray-700 rounded transition-colors"
              title={isMinimized ? "Maximize" : "Minimize"}
            >
              <span className="text-gray-400 text-xs">
                {isMinimized ? "□" : "−"}
              </span>
            </button>
            <button
              onClick={onClose}
              className="p-1 hover:bg-red-600/20 rounded transition-colors"
              title="Close"
            >
              <span className="text-gray-400 hover:text-red-400 text-xs">×</span>
            </button>
          </div>
        </div>

        {/* Player Content */}
        {!isMinimized && (
          <div className="flex-1 bg-black">
            <iframe
              src={`https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&modestbranding=1`}
              title="YouTube Mini Player"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="w-full h-full border-0"
            />
          </div>
        )}
      </div>
    </Rnd>
  );
};

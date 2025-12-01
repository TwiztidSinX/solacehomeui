import React, { useState, useMemo } from "react";
import { type FileNode } from "../types/files";

interface FileBrowserProps {
  tree: FileNode[];
  onRefresh: () => void;
  onOpenFile: (path: string) => void;
  activeFilePath?: string | null;
  isLoading?: boolean;
  onCreateFile?: (path: string) => void;
  onCreateFolder?: (path: string) => void;
  onRename?: (oldPath: string, newPath: string) => void;
  onDelete?: (path: string) => void;
  workspaceRoot?: string;
}

const FileBrowser: React.FC<FileBrowserProps> = ({
  tree,
  onRefresh,
  onOpenFile,
  activeFilePath,
  isLoading = false,
  onCreateFile,
  onCreateFolder,
  onRename,
  onDelete,
  workspaceRoot,
}) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggleExpanded = (path: string) => {
    setExpanded((prev) => ({ ...prev, [path]: !prev[path] }));
  };

  // ✅ FIXED: sortedTree must be declared BEFORE useEffect uses it
  const sortedTree = useMemo(() => {
    const sortNodes = (nodes: FileNode[]): FileNode[] =>
      nodes
        .slice()
        .sort((a, b) => {
          if (a.type === b.type) return a.name.localeCompare(b.name);
          return a.type === "dir" ? -1 : 1;
        })
        .map((node) =>
          node.type === "dir" && node.children
            ? { ...node, children: sortNodes(node.children) }
            : node,
        );
    return sortNodes(tree);
  }, [tree]);

  React.useEffect(() => {
    if (sortedTree.length > 0) {
      // Auto-expand ALL root-level directories
      const newExpanded: Record<string, boolean> = {};
      sortedTree.forEach((node) => {
        if (node.type === "dir") {
          newExpanded[node.path] = true;
        }
      });
      setExpanded((prev) => ({ ...prev, ...newExpanded }));
    }
  }, [sortedTree]);

  const renderNode = (node: FileNode, depth: number) => {
    const isDir = node.type === "dir";
    const isOpen = !!expanded[node.path];
    const isActive = activeFilePath === node.path;

    return (
      <div key={node.path} className="group">
        <div className="flex items-center">
          <button
            className={`flex-1 flex items-center text-left px-2 py-1 rounded transition-colors ${
              isActive
                ? "bg-blue-600 text-white"
                : "hover:bg-white/10 text-gray-200"
            }`}
            style={{ paddingLeft: depth * 12 + 8 }}
            onClick={() => {
              if (isDir) {
                toggleExpanded(node.path);
              } else {
                onOpenFile(node.path);
              }
            }}
            onDoubleClick={() => {
              if (!isDir) onOpenFile(node.path);
            }}
          >
            <span className="mr-2">
              {isDir ? (isOpen ? "📂" : "📁") : "📄"}
            </span>
            <span className="truncate">{node.name}</span>
          </button>
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity pr-1">
            {onRename && (
              <button
                className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white"
                onClick={() => {
                  const newName = prompt(
                    "Rename to (relative path):",
                    node.path,
                  );
                  if (newName && newName !== node.path)
                    onRename(node.path, newName);
                }}
              >
                Rename
              </button>
            )}
            {onDelete && (
              <button
                className="text-xs px-2 py-1 rounded bg-red-600 hover:bg-red-700 text-white"
                onClick={() => {
                  if (confirm(`Delete ${node.path}?`)) onDelete(node.path);
                }}
              >
                Del
              </button>
            )}
          </div>
        </div>
        {isDir && isOpen && node.children && (
          <div className="ml-2">
            {node.children.map((child) => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-900/60 border border-white/10 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
        <div className="flex flex-col">
          <span className="text-sm text-white font-semibold">Files</span>
          {workspaceRoot && (
            <span className="text-[10px] text-gray-400 truncate max-w-[220px]">
              {workspaceRoot}
            </span>
          )}
        </div>
        <button
          onClick={onRefresh}
          className="text-xs px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-white"
        >
          ↻ Refresh
        </button>
      </div>
      {(onCreateFile || onCreateFolder) && (
        <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10">
          {onCreateFile && (
            <button
              className="text-xs px-2 py-1 rounded bg-blue-500/80 hover:bg-blue-600 text-white"
              onClick={() => {
                const name = prompt(
                  "New file name (relative to root)",
                  "new-file.txt",
                );
                if (name) onCreateFile(name);
              }}
            >
              + File
            </button>
          )}
          {onCreateFolder && (
            <button
              className="text-xs px-2 py-1 rounded bg-green-500/80 hover:bg-green-600 text-white"
              onClick={() => {
                const name = prompt(
                  "New folder name (relative to root)",
                  "new-folder",
                );
                if (name) onCreateFolder(name);
              }}
            >
              + Folder
            </button>
          )}
        </div>
      )}
      <div className="flex-1 overflow-auto p-2 text-sm font-mono">
        {isLoading ? (
          <p className="text-gray-400">Loading file tree...</p>
        ) : sortedTree.length === 0 ? (
          <p className="text-gray-400">No files found</p>
        ) : (
          sortedTree.map((node) => renderNode(node, 0))
        )}
      </div>
    </div>
  );
};

export default FileBrowser;

import { useCallback, useState } from "react";
import toast from "react-hot-toast";
import { type FileNode, type OpenFile } from "../types/files";
import { detectLanguageFromPath } from "../utils/fileUtils";

export interface CodePanelState {
  codeContent: string;
  codeLanguage: string;
  fileTree: FileNode[];
  workspaceRoot: string;
  isLoadingFileTree: boolean;
  openFiles: OpenFile[];
  activeFilePath: string | null;
}

export interface CodePanelActions {
  requestFileTree: (path?: string) => void;
  handleFileContent: (data: { path: string; content: string }) => void;
  openFileFromTree: (path: string) => void;
  closeOpenFile: (path: string) => void;
  saveActiveFile: () => void;
  selectOpenFile: (path: string) => void;
  handleCodeEditorChange: (value: string) => void;
  createFile: (path: string) => void;
  createFolder: (path: string) => void;
  renamePath: (oldPath: string, newPath: string) => void;
  deletePath: (path: string) => void;
  requestWorkspaceChange: () => void;
  openLocalFile: (path: string, content: string, language?: string) => void;
  setWorkspaceRoot: (root: string) => void;
  setFileTree: React.Dispatch<React.SetStateAction<FileNode[]>>;
  setIsLoadingFileTree: React.Dispatch<React.SetStateAction<boolean>>;
  setCodeContent: React.Dispatch<React.SetStateAction<string>>;
  setCodeLanguage: React.Dispatch<React.SetStateAction<string>>;
}

export const useCodePanel = (
  socketRef: React.MutableRefObject<any>,
): [CodePanelState, CodePanelActions] => {
  const [codeContent, setCodeContent] = useState("");
  const [codeLanguage, setCodeLanguage] = useState("javascript");
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [workspaceRoot, setWorkspaceRoot] = useState<string>("");
  const [isLoadingFileTree, setIsLoadingFileTree] = useState(false);
  const [openFiles, setOpenFiles] = useState<OpenFile[]>([]);
  const [activeFilePath, setActiveFilePath] = useState<string | null>(null);

  const requestFileTree = useCallback(
    (path = ".") => {
      if (!socketRef.current) return;
      setIsLoadingFileTree(true);
      socketRef.current.emit("list_files", { path });
    },
    [socketRef],
  );

  const handleFileContent = useCallback((data: { path: string; content: string }) => {
    const language = detectLanguageFromPath(data.path);
    setOpenFiles((prev) => {
      const existing = prev.find((f) => f.path === data.path);
      if (existing) {
        return prev.map((f) =>
          f.path === data.path ? { ...f, content: data.content, language } : f,
        );
      }
      return [...prev, { path: data.path, content: data.content, language }];
    });
    setActiveFilePath(data.path);
    setCodeContent(data.content);
    setCodeLanguage(language);
  }, []);

  const openFileFromTree = useCallback(
    (path: string) => {
      setActiveFilePath(path);
      const existing = openFiles.find((f) => f.path === path);
      if (existing) {
        setCodeContent(existing.content);
        setCodeLanguage(existing.language);
        return;
      }
      if (socketRef.current) {
        socketRef.current.emit("read_file", { path });
      } else {
        toast.error("Socket not connected; cannot load file");
      }
    },
    [openFiles, socketRef],
  );

  const closeOpenFile = useCallback(
    (path: string) => {
      setOpenFiles((prev) => {
        const remaining = prev.filter((f) => f.path !== path);
        if (activeFilePath === path) {
          if (remaining.length > 0) {
            const next = remaining[0];
            setActiveFilePath(next.path);
            setCodeContent(next.content);
            setCodeLanguage(next.language);
          } else {
            setActiveFilePath(null);
            setCodeContent("");
            setCodeLanguage("javascript");
          }
        }
        return remaining;
      });
    },
    [activeFilePath],
  );

  const saveActiveFile = useCallback(() => {
    if (!activeFilePath) {
      toast.error("No file selected to save");
      return;
    }
    if (!socketRef.current) {
      toast.error("Socket not connected; cannot save file");
      return;
    }
    socketRef.current.emit("save_file", {
      path: activeFilePath,
      content: codeContent,
    });
    toast("Saving file...", { icon: "dY'_" });
  }, [activeFilePath, codeContent, socketRef]);

  const selectOpenFile = useCallback(
    (path: string) => {
      const target = openFiles.find((f) => f.path === path);
      if (target) {
        setActiveFilePath(path);
        setCodeContent(target.content);
        setCodeLanguage(target.language);
      }
    },
    [openFiles],
  );

  const handleCodeEditorChange = useCallback(
    (value: string) => {
      setCodeContent(value);
      if (activeFilePath) {
        setOpenFiles((prev) =>
          prev.map((f) =>
            f.path === activeFilePath ? { ...f, content: value } : f,
          ),
        );
      }
    },
    [activeFilePath],
  );

  const openLocalFile = useCallback(
    (path: string, content: string, language?: string) => {
      const lang = language || detectLanguageFromPath(path);
      setOpenFiles((prev) => {
        const existing = prev.find((f) => f.path === path);
        if (existing) {
          return prev.map((f) =>
            f.path === path ? { ...f, content, language: lang } : f,
          );
        }
        return [...prev, { path, content, language: lang }];
      });
      setActiveFilePath(path);
      setCodeContent(content);
      setCodeLanguage(lang);
    },
    [],
  );

  const createFile = useCallback(
    (path: string) => {
      if (!socketRef.current) return;
      socketRef.current.emit("create_file", { path });
    },
    [socketRef],
  );

  const createFolder = useCallback(
    (path: string) => {
      if (!socketRef.current) return;
      socketRef.current.emit("create_folder", { path });
    },
    [socketRef],
  );

  const renamePath = useCallback(
    (oldPath: string, newPath: string) => {
      if (!socketRef.current) return;
      socketRef.current.emit("rename_path", {
        old_path: oldPath,
        new_path: newPath,
      });
      setOpenFiles((prev) =>
        prev.map((f) => (f.path === oldPath ? { ...f, path: newPath } : f)),
      );
      if (activeFilePath === oldPath) {
        setActiveFilePath(newPath);
      }
    },
    [activeFilePath, socketRef],
  );

  const deletePath = useCallback(
    (path: string) => {
      if (!socketRef.current) return;
      socketRef.current.emit("delete_path", { path });
      setOpenFiles((prev) => prev.filter((f) => f.path !== path));
      if (activeFilePath === path) {
        setActiveFilePath(null);
        setCodeContent("");
      }
    },
    [activeFilePath, socketRef],
  );

  const requestWorkspaceChange = useCallback(() => {
    const next = prompt(
      "Enter new workspace absolute path (directory):",
      workspaceRoot || "",
    );
    if (next && socketRef.current) {
      socketRef.current.emit("set_workspace_root", { path: next });
    }
  }, [socketRef, workspaceRoot]);

  return [
    {
      codeContent,
      codeLanguage,
      fileTree,
      workspaceRoot,
      isLoadingFileTree,
      openFiles,
      activeFilePath,
    },
    {
      requestFileTree,
      handleFileContent,
      openFileFromTree,
      closeOpenFile,
      saveActiveFile,
      selectOpenFile,
      handleCodeEditorChange,
      createFile,
      createFolder,
      renamePath,
      deletePath,
      requestWorkspaceChange,
      openLocalFile,
      setWorkspaceRoot,
      setFileTree,
      setIsLoadingFileTree,
      setCodeContent,
      setCodeLanguage,
    },
  ];
};

export default useCodePanel;

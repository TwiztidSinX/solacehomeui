export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'dir';
  children?: FileNode[];
}

export interface OpenFile {
  path: string;
  content: string;
  language: string;
}

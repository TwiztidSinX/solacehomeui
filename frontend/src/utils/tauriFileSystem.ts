/**
 * Tauri File System Utilities
 *
 * Provides file system operations using Tauri's native APIs
 * Falls back to browser APIs when not running in Tauri
 */

import { exists, readTextFile, writeTextFile, mkdir, readDir, BaseDirectory } from '@tauri-apps/plugin-fs';
import { open, save } from '@tauri-apps/plugin-dialog';

/**
 * Check if running in Tauri environment
 */
export function isTauri(): boolean {
  return '__TAURI__' in window;
}

/**
 * Read a text file from the file system
 */
export async function readFile(path: string): Promise<string> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    const content = await readTextFile(path);
    return content;
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
}

/**
 * Write text content to a file
 */
export async function writeFile(path: string, content: string): Promise<void> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    await writeTextFile(path, content);
  } catch (error) {
    console.error('Error writing file:', error);
    throw error;
  }
}

/**
 * Check if a file or directory exists
 */
export async function fileExists(path: string): Promise<boolean> {
  if (!isTauri()) {
    return false;
  }

  try {
    return await exists(path);
  } catch (error) {
    console.error('Error checking file existence:', error);
    return false;
  }
}

/**
 * Create a directory
 */
export async function createDirectory(path: string): Promise<void> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    await mkdir(path, { recursive: true });
  } catch (error) {
    console.error('Error creating directory:', error);
    throw error;
  }
}

/**
 * Read directory contents
 */
export async function readDirectory(path: string): Promise<string[]> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    const entries = await readDir(path);
    return entries.map(entry => entry.name);
  } catch (error) {
    console.error('Error reading directory:', error);
    throw error;
  }
}

/**
 * Open a file picker dialog
 */
export async function openFilePicker(options?: {
  multiple?: boolean;
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
}): Promise<string | string[] | null> {
  if (!isTauri()) {
    throw new Error('File picker requires Tauri');
  }

  try {
    const result = await open({
      multiple: options?.multiple ?? false,
      directory: options?.directory ?? false,
      filters: options?.filters,
    });
    return result;
  } catch (error) {
    console.error('Error opening file picker:', error);
    return null;
  }
}

/**
 * Open a save file dialog
 */
export async function saveFilePicker(options?: {
  defaultPath?: string;
  filters?: { name: string; extensions: string[] }[];
}): Promise<string | null> {
  if (!isTauri()) {
    throw new Error('Save file dialog requires Tauri');
  }

  try {
    const result = await save({
      defaultPath: options?.defaultPath,
      filters: options?.filters,
    });
    return result;
  } catch (error) {
    console.error('Error opening save dialog:', error);
    return null;
  }
}

/**
 * Read a file from app-specific directories
 */
export async function readAppFile(filename: string, dir: BaseDirectory = BaseDirectory.AppData): Promise<string> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    const content = await readTextFile(filename, { baseDir: dir });
    return content;
  } catch (error) {
    console.error('Error reading app file:', error);
    throw error;
  }
}

/**
 * Write a file to app-specific directories
 */
export async function writeAppFile(
  filename: string,
  content: string,
  dir: BaseDirectory = BaseDirectory.AppData
): Promise<void> {
  if (!isTauri()) {
    throw new Error('File system operations require Tauri');
  }

  try {
    await writeTextFile(filename, content, { baseDir: dir });
  } catch (error) {
    console.error('Error writing app file:', error);
    throw error;
  }
}

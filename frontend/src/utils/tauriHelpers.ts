/**
 * Tauri Helper Utilities
 *
 * Provides helper functions for Tauri features like notifications, shell commands, and window operations
 */

import { sendNotification, isPermissionGranted, requestPermission } from '@tauri-apps/plugin-notification';
import { Command } from '@tauri-apps/plugin-shell';
import { getCurrentWindow } from '@tauri-apps/api/window';

/**
 * Check if running in Tauri environment
 */
export function isTauri(): boolean {
  return '__TAURI__' in window;
}

/**
 * Send a desktop notification
 */
export async function notify(title: string, body?: string): Promise<void> {
  if (!isTauri()) {
    console.warn('Notifications require Tauri');
    return;
  }

  try {
    let permissionGranted = await isPermissionGranted();
    if (!permissionGranted) {
      const permission = await requestPermission();
      permissionGranted = permission === 'granted';
    }

    if (permissionGranted) {
      sendNotification({ title, body });
    }
  } catch (error) {
    console.error('Error sending notification:', error);
  }
}

/**
 * Execute a shell command
 */
export async function executeCommand(
  program: string,
  args?: string[]
): Promise<{ stdout: string; stderr: string; code: number }> {
  if (!isTauri()) {
    throw new Error('Shell commands require Tauri');
  }

  try {
    const command = Command.create(program, args);
    const output = await command.execute();
    return {
      stdout: output.stdout,
      stderr: output.stderr,
      code: output.code ?? 0, // Default to 0 if null
    };
  } catch (error) {
    console.error('Error executing command:', error);
    throw error;
  }
}

/**
 * Window operations
 */
export const windowOps = {
  /**
   * Minimize the window
   */
  minimize: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.minimize();
  },

  /**
   * Maximize the window
   */
  maximize: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.maximize();
  },

  /**
   * Unmaximize the window
   */
  unmaximize: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.unmaximize();
  },

  /**
   * Toggle maximize state
   */
  toggleMaximize: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.toggleMaximize();
  },

  /**
   * Close the window
   */
  close: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.close();
  },

  /**
   * Hide the window
   */
  hide: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.hide();
  },

  /**
   * Show the window
   */
  show: async () => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.show();
  },

  /**
   * Set window title
   */
  setTitle: async (title: string) => {
    if (!isTauri()) return;
    const window = getCurrentWindow();
    await window.setTitle(title);
  },

  /**
   * Check if window is maximized
   */
  isMaximized: async (): Promise<boolean> => {
    if (!isTauri()) return false;
    const window = getCurrentWindow();
    return await window.isMaximized();
  },

  /**
   * Check if window is focused
   */
  isFocused: async (): Promise<boolean> => {
    if (!isTauri()) return false;
    const window = getCurrentWindow();
    return await window.isFocused();
  },
};

/**
 * Get platform information
 */
export async function getPlatform(): Promise<string> {
  if (!isTauri()) {
    return 'web';
  }

  // Use the platform API when available
  return navigator.platform || 'unknown';
}

/**
 * Check if app is running on Windows
 */
export function isWindows(): boolean {
  return navigator.platform.toLowerCase().includes('win');
}

/**
 * Check if app is running on macOS
 */
export function isMacOS(): boolean {
  return navigator.platform.toLowerCase().includes('mac');
}

/**
 * Check if app is running on Linux
 */
export function isLinux(): boolean {
  return navigator.platform.toLowerCase().includes('linux');
}

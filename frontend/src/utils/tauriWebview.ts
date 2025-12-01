/**
 * Tauri Webview Utilities
 *
 * Opens external URLs in native webview windows (no iframe blocking!)
 */

import { invoke } from '@tauri-apps/api/core';
import { isTauri } from './tauriHelpers';

/**
 * Open a URL in a new native webview window
 * This bypasses iframe restrictions and allows full navigation
 */
export async function openWebview(url: string, options?: {
  label?: string;
  title?: string;
}): Promise<void> {
  if (!isTauri()) {
    // Fallback for web: open in new tab
    window.open(url, '_blank');
    return;
  }

  try {
    const label = options?.label || `webview-${Date.now()}`;
    const title = options?.title || 'Browser';

    await invoke('open_webview', {
      url,
      label,
      title,
    });
  } catch (error) {
    console.error('Failed to open webview:', error);
    throw error;
  }
}

/**
 * Close a webview window by label
 */
export async function closeWebview(label: string): Promise<void> {
  if (!isTauri()) {
    return;
  }

  try {
    await invoke('close_webview', { label });
  } catch (error) {
    console.error('Failed to close webview:', error);
  }
}

/**
 * Open a search results page in webview
 */
export async function openSearchWebview(query: string, engine: 'google' | 'searxng' = 'google'): Promise<void> {
  let url: string;

  if (engine === 'searxng') {
    url = `http://localhost:8080/search?q=${encodeURIComponent(query)}`;
  } else {
    url = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
  }

  await openWebview(url, {
    label: 'search-results',
    title: `Search: ${query}`,
  });
}

/**
 * Open a URL in the browser webview
 */
export async function openBrowserWebview(url: string): Promise<void> {
  // Ensure URL has protocol
  let fullUrl = url;
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    fullUrl = `https://${url}`;
  }

  await openWebview(fullUrl, {
    label: 'browser',
    title: 'Browser',
  });
}

/**
 * Open YouTube in webview
 */
export async function openYouTubeWebview(searchOrUrl: string): Promise<void> {
  let url: string;

  // Check if it's already a YouTube URL
  if (searchOrUrl.includes('youtube.com') || searchOrUrl.includes('youtu.be')) {
    url = searchOrUrl;
  } else {
    // Treat as search query
    url = `https://www.youtube.com/results?search_query=${encodeURIComponent(searchOrUrl)}`;
  }

  await openWebview(url, {
    label: 'youtube',
    title: 'YouTube',
  });
}

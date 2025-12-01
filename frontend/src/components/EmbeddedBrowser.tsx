/**
 * Embedded Browser Component
 *
 * Uses Tauri's WebView to embed a real Chromium browser inside the panel
 * NO iframe restrictions - full website access for agentic control
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';

interface EmbeddedBrowserProps {
  url: string;
  label?: string;
}

export default function EmbeddedBrowser({ url, label = 'embedded-browser' }: EmbeddedBrowserProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const currentWindowRef = useRef(getCurrentWindow());

  const computeBounds = useCallback(async () => {
    if (!containerRef.current) return null;
    const rect = containerRef.current.getBoundingClientRect();
    let offsetX = 0;
    let offsetY = 0;
    let scale = 1;
    try {
      const win = currentWindowRef.current;
      const pos = await win.outerPosition();
      const factor = await win.scaleFactor();
      offsetX = pos.x;
      offsetY = pos.y;
      scale = factor;
    } catch (e) {
      console.error("Tauri API call failed during computeBounds:", e);
    }
    return {
      x: Math.round(rect.left * scale + offsetX),
      y: Math.round(rect.top * scale + offsetY),
      width: Math.round(rect.width * scale),
      height: Math.round(rect.height * scale),
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    const createWebview = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const bounds = await computeBounds();
        if (!bounds) return;

        // Create the embedded webview
        await invoke('create_embedded_webview', {
          label,
          url,
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          height: bounds.height,
        });

        if (mounted) {
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Failed to create embedded webview:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : String(err));
          setIsLoading(false);
        }
      }
    };

    createWebview();

    // Cleanup: remove webview when component unmounts
    return () => {
      mounted = false;
      invoke('remove_embedded_webview', { label }).catch(console.error);
    };
  }, [url, label]);

  // Handle window resize
  useEffect(() => {
    const runUpdate = async () => {
      const bounds = await computeBounds();
      if (!bounds) return;
      invoke('update_embedded_webview', {
        label,
        url,
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
      }).catch((err) => {
        console.error('Failed to update webview, recreating:', err);
        invoke('remove_embedded_webview', { label }).finally(() => {
          invoke('create_embedded_webview', {
            label,
            url,
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: bounds.height,
          }).catch(console.error);
        });
      });
    };

    const handler = () => {
      runUpdate();
    };

    window.addEventListener('resize', handler);
    window.addEventListener('solace-browser-sync', handler as EventListener);

    return () => {
      window.removeEventListener('resize', handler);
      window.removeEventListener('solace-browser-sync', handler as EventListener);
    };
  }, [url, label, computeBounds]);

  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-red-500/10 rounded-lg">
        <div className="text-center p-4">
          <p className="text-red-400 font-semibold mb-2">Browser Error</p>
          <p className="text-sm text-gray-300">{error}</p>
          <p className="text-xs text-gray-400 mt-2">
            Falling back to iframe...
          </p>
          <iframe
            src={url}
            className="w-full h-[400px] mt-4 rounded border border-gray-700"
            title="Fallback browser"
          />
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-full rounded-lg overflow-hidden bg-black/20 relative"
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-2"></div>
            <p className="text-white text-sm">Loading browser...</p>
          </div>
        </div>
      )}
      {/* The webview will be rendered here by Tauri */}
    </div>
  );
}

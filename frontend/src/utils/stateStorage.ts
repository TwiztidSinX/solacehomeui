/**
 * State Persistence Layer for SolaceOS
 * Saves and restores UI state across page refreshes
 */

export interface SolaceOSState {
  selectedBackend: string;
  selectedModel: string;
  apiProvider: string;
  rightPanelMode: string;
  leftPanelMode: string;
  activePanels: string[];
  mediaPlaybackUrl?: string;
  mediaTitle?: string;
  mediaDescription?: string;
  youtubeVideoId?: string;
  youtubeUrl?: string;
}

const STATE_KEY = 'solaceos_ui_state';
const STATE_VERSION = 1;

/**
 * Save UI state to localStorage
 * Debounce this in the caller to avoid excessive writes
 */
export function saveUIState(state: Partial<SolaceOSState>): void {
  try {
    const existing = loadUIState();
    const updated = { 
      ...existing, 
      ...state, 
      version: STATE_VERSION,
      lastUpdated: new Date().toISOString()
    };
    localStorage.setItem(STATE_KEY, JSON.stringify(updated));
    console.log('✅ UI state saved:', Object.keys(state));
  } catch (error) {
    console.warn('⚠️ Failed to save UI state:', error);
  }
}

/**
 * Load UI state from localStorage
 * Returns null if no state exists or version mismatch
 */
export function loadUIState(): SolaceOSState | null {
  try {
    const stored = localStorage.getItem(STATE_KEY);
    if (!stored) {
      console.log('ℹ️ No saved UI state found');
      return null;
    }
    
    const parsed = JSON.parse(stored);
    if (parsed.version !== STATE_VERSION) {
      console.warn('⚠️ UI state version mismatch, clearing outdated state');
      localStorage.removeItem(STATE_KEY);
      return null;
    }
    
    console.log('✅ UI state loaded:', parsed);
    return parsed;
  } catch (error) {
    console.warn('⚠️ Failed to load UI state:', error);
    return null;
  }
}

/**
 * Clear saved UI state
 */
export function clearUIState(): void {
  try {
    localStorage.removeItem(STATE_KEY);
    console.log('✅ UI state cleared');
  } catch (error) {
    console.warn('⚠️ Failed to clear UI state:', error);
  }
}

/**
 * Validate that a selected model is compatible with the current backend
 */
export function validateModelForBackend(
  selectedModel: string,
  availableModels: string[]
): { valid: boolean; suggestedModel?: string } {
  if (!selectedModel || !availableModels || availableModels.length === 0) {
    return { valid: false };
  }

  const isValid = availableModels.includes(selectedModel);
  
  if (!isValid && availableModels.length > 0) {
    return {
      valid: false,
      suggestedModel: availableModels[0]
    };
  }

  return { valid: isValid };
}

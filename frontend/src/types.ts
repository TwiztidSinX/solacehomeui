export interface TokenMetrics {
  inputTokens: number;
  outputTokens: number;
  timeToFirstToken?: number; // milliseconds
  tokensPerSecond?: number;
  estimatedCost?: number;
  model?: string;
  timestamp?: Date;
}

export interface Message {
  sender: string;
  message: string;
  type?: 'ai' | 'user' | 'system' | 'error' | 'info';
  imageB64?: string | null;
  isThinking?: boolean;
  thought?: string;
  iframeUrl?: string;
  youtubeVideoId?: string;
  imageGalleryUrls?: string[];
  imageUrl?: string;
  tokenMetrics?: TokenMetrics;
}

export interface UsageByDay {
  date: string;
  totalTokens: number;
  totalCost: number;
}

export interface UsageByModel {
  model: string;
  provider?: string;
  totalTokens: number;
  totalCost: number;
  messages: number;
}

export interface UsageSummary {
  from: string;
  to: string;
  totalCost: number;
  totalTokens: number;
  byDay: UsageByDay[];
  byModel: UsageByModel[];
}

export interface ModelUsageStats {
  model: string;
  provider?: string;
  days: number;
  totalTokens: number;
  totalCost: number;
  messages: number;
  avgTokensPerMessage: number;
  avgCostPerMessage: number;
  byDay: UsageByDay[];
}

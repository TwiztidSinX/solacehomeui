/**
 * Token Pricing and Cost Calculation Utilities
 * Prices are per million tokens (USD)
 */

export interface ModelPricing {
  input: number;  // Cost per 1M input tokens
  output: number; // Cost per 1M output tokens
}

export const MODEL_PRICING: Record<string, ModelPricing> = {
  // Anthropic Claude Models
  'claude-sonnet-4-5': { input: 3.00, output: 15.00 },
  'claude-sonnet-4': { input: 3.00, output: 15.00 },
  'claude-3-5-sonnet': { input: 3.00, output: 15.00 },
  'claude-3-opus': { input: 15.00, output: 75.00 },
  'claude-3-haiku': { input: 0.25, output: 1.25 },
  'claude-3-sonnet': { input: 3.00, output: 15.00 },

  // OpenAI GPT Models
  'gpt-5': { input: 1.25, output: 10.00 },
  'gpt-5-mini': { input: 0.25, output: 2.00 },
  'gpt-5-nano': { input: 0.05, output: 0.40 },
  'gpt-5-chat-latest': { input: 1.25, output: 10.00 },
  'gpt-5.1': { input: 1.25, output: 10.00 },
  'gpt-4o': { input: 2.50, output: 10.00 },
  'gpt-4-turbo': { input: 10.00, output: 30.00 },
  'gpt-4': { input: 30.00, output: 60.00 },
  'gpt-3.5-turbo': { input: 0.50, output: 1.50 },
  'o1': { input: 15.00, output: 60.00 },
  'o1-mini': { input: 3.00, output: 12.00 },

  // DeepSeek Models
  'deepseek-r1': { input: 0.55, output: 2.19 },
  'deepseek-v3': { input: 0.27, output: 1.10 },
  'deepseek-chat': { input: 0.14, output: 0.28 },

  // Google Gemini Models
  'gemini-2.0-flash': { input: 0.00, output: 0.00 }, // Free tier
  'gemini-1.5-pro': { input: 1.25, output: 5.00 },
  'gemini-1.5-flash': { input: 0.075, output: 0.30 },

  // xAI Grok Models
  'grok-2': { input: 2.00, output: 10.00 },
  'grok-2-vision': { input: 2.00, output: 10.00 },

  // Meta Llama Models (via various providers)
  'llama-3.3-70b': { input: 0.35, output: 0.40 },
  'llama-3.1-405b': { input: 2.70, output: 2.70 },
  'llama-3.1-70b': { input: 0.35, output: 0.40 },

  // Mistral Models
  'mistral-large': { input: 2.00, output: 6.00 },
  'mistral-medium': { input: 0.70, output: 2.00 },
  'mistral-small': { input: 0.20, output: 0.60 },

  // Default fallback for unknown models
  'default': { input: 1.00, output: 3.00 },
};

/**
 * Calculate the estimated cost for a given token usage
 * @param inputTokens Number of input tokens
 * @param outputTokens Number of output tokens
 * @param modelName Name of the model used
 * @returns Estimated cost in USD
 */
export function calculateCost(
  inputTokens: number,
  outputTokens: number,
  modelName?: string
): number {
  // Normalize model name to match pricing keys
  const normalizedModel = normalizeModelName(modelName);
  const pricing = MODEL_PRICING[normalizedModel] || MODEL_PRICING['default'];

  const inputCost = (inputTokens / 1_000_000) * pricing.input;
  const outputCost = (outputTokens / 1_000_000) * pricing.output;

  return inputCost + outputCost;
}

/**
 * Normalize model name to match pricing keys
 * Handles various naming conventions from different providers
 */
function normalizeModelName(modelName?: string): string {
  if (!modelName) return 'default';

  const lower = modelName.toLowerCase();

  // Direct matches
  if (MODEL_PRICING[lower]) return lower;

  // Partial matches for common patterns
  if (lower.includes('gpt-5.1') || lower.includes('gpt5.1') || lower.includes('gpt-5_1')) return 'gpt-5.1';
  if (lower.includes('gpt-5-mini')) return 'gpt-5-mini';
  if (lower.includes('gpt-5-nano')) return 'gpt-5-nano';
  if (lower.includes('gpt-5-chat')) return 'gpt-5-chat-latest';
  if (lower.includes('claude') && lower.includes('sonnet-4')) return 'claude-sonnet-4';
  if (lower.includes('claude') && lower.includes('opus')) return 'claude-3-opus';
  if (lower.includes('claude') && lower.includes('haiku')) return 'claude-3-haiku';
  if (lower.includes('gpt-5')) return 'gpt-5';
  if (lower.includes('gpt-4o')) return 'gpt-4o';
  if (lower.includes('gpt-4-turbo')) return 'gpt-4-turbo';
  if (lower.includes('gpt-4')) return 'gpt-4';
  if (lower.includes('gpt-3.5')) return 'gpt-3.5-turbo';
  if (lower.includes('deepseek-r1')) return 'deepseek-r1';
  if (lower.includes('deepseek-v3')) return 'deepseek-v3';
  if (lower.includes('deepseek')) return 'deepseek-chat';
  if (lower.includes('gemini-2.0')) return 'gemini-2.0-flash';
  if (lower.includes('gemini') && lower.includes('pro')) return 'gemini-1.5-pro';
  if (lower.includes('gemini') && lower.includes('flash')) return 'gemini-1.5-flash';
  if (lower.includes('grok-2')) return 'grok-2';
  if (lower.includes('grok')) return 'grok-2';
  if (lower.includes('llama') && lower.includes('405')) return 'llama-3.1-405b';
  if (lower.includes('llama') && lower.includes('70')) return 'llama-3.3-70b';
  if (lower.includes('mistral-large')) return 'mistral-large';
  if (lower.includes('mistral-medium')) return 'mistral-medium';
  if (lower.includes('mistral-small')) return 'mistral-small';

  return 'default';
}

/**
 * Format cost as a human-readable string
 * @param cost Cost in USD
 * @returns Formatted string (e.g., "$0.003" or "$12.50")
 */
export function formatCost(cost: number): string {
  if (cost === 0) return 'Free';
  if (cost < 0.001) return '<$0.001';
  if (cost < 1) return `$${cost.toFixed(3)}`;
  return `$${cost.toFixed(2)}`;
}

/**
 * Format tokens as a human-readable string with commas
 * @param tokens Number of tokens
 * @returns Formatted string (e.g., "1,234" or "1.2K")
 */
export function formatTokens(tokens: number): string {
  if (tokens < 1000) return tokens.toString();
  if (tokens < 10000) return tokens.toLocaleString();
  if (tokens < 1_000_000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${(tokens / 1_000_000).toFixed(2)}M`;
}

/**
 * Format duration in milliseconds to human-readable string
 * @param ms Duration in milliseconds
 * @returns Formatted string (e.g., "0.8s" or "1.2s")
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Format tokens per second rate
 * @param rate Tokens per second
 * @returns Formatted string (e.g., "45.2 tok/s")
 */
export function formatTokenRate(rate: number): string {
  return `${rate.toFixed(1)} tok/s`;
}

/**
 * Get pricing information for a specific model
 * @param modelName Name of the model
 * @returns Pricing information or null if not found
 */
export function getModelPricing(modelName?: string): ModelPricing | null {
  if (!modelName) return null;
  const normalized = normalizeModelName(modelName);
  return MODEL_PRICING[normalized] || MODEL_PRICING['default'];
}

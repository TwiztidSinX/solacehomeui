import React, { useState } from 'react';
import { TokenMetrics } from '../types';
import {
  formatTokens,
  formatCost,
  formatDuration,
  formatTokenRate,
} from '../utils/tokenPricing';

interface TokenBadgeProps {
  metrics: TokenMetrics;
  className?: string;
}

const TokenBadge: React.FC<TokenBadgeProps> = ({ metrics, className = '' }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const totalTokens = metrics.inputTokens + metrics.outputTokens;

  // Calculate response time - use timeToFirstToken if available
  const responseTime = metrics.timeToFirstToken;

  // Build compact summary text
  const summaryParts: string[] = [];
  summaryParts.push(`${formatTokens(totalTokens)} tokens`);

  if (responseTime !== undefined) {
    summaryParts.push(formatDuration(responseTime));
  }

  if (metrics.estimatedCost !== undefined) {
    summaryParts.push(formatCost(metrics.estimatedCost));
  }

  const summaryText = summaryParts.join(' · ');

  return (
    <div className={`token-badge-container ${className}`}>
      {/* Compact Badge */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="token-badge flex items-center gap-2 px-3 py-1.5 rounded-full text-xs
                   bg-gray-800/60 hover:bg-gray-700/70 border border-gray-600/40
                   transition-all duration-200 backdrop-blur-sm group"
        title="Click for detailed token metrics"
      >
        <span className="text-gray-300 group-hover:text-gray-100">
          {summaryText}
        </span>
        <svg
          className={`w-3 h-3 text-gray-400 transition-transform duration-200 ${
            isExpanded ? 'rotate-180' : ''
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded Details Panel */}
      {isExpanded && (
        <div className="token-details mt-2 p-3 rounded-lg bg-gray-800/80 border border-gray-600/40
                        backdrop-blur-sm animate-fadeIn">
          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Input Tokens */}
            <div className="flex flex-col">
              <span className="text-gray-400 mb-1">Input Tokens</span>
              <span className="text-gray-100 font-mono">
                {formatTokens(metrics.inputTokens)}
              </span>
            </div>

            {/* Output Tokens */}
            <div className="flex flex-col">
              <span className="text-gray-400 mb-1">Output Tokens</span>
              <span className="text-gray-100 font-mono">
                {formatTokens(metrics.outputTokens)}
              </span>
            </div>

            {/* Time to First Token */}
            {metrics.timeToFirstToken !== undefined && (
              <div className="flex flex-col">
                <span className="text-gray-400 mb-1">TTFT</span>
                <span className="text-gray-100 font-mono">
                  {formatDuration(metrics.timeToFirstToken)}
                </span>
              </div>
            )}

            {/* Tokens per Second */}
            {metrics.tokensPerSecond !== undefined && (
              <div className="flex flex-col">
                <span className="text-gray-400 mb-1">Speed</span>
                <span className="text-gray-100 font-mono">
                  {formatTokenRate(metrics.tokensPerSecond)}
                </span>
              </div>
            )}

            {/* Estimated Cost */}
            {metrics.estimatedCost !== undefined && (
              <div className="flex flex-col">
                <span className="text-gray-400 mb-1">Estimated Cost</span>
                <span className="text-green-400 font-mono font-semibold">
                  {formatCost(metrics.estimatedCost)}
                </span>
              </div>
            )}

            {/* Model Name */}
            {metrics.model && (
              <div className="flex flex-col">
                <span className="text-gray-400 mb-1">Model</span>
                <span className="text-gray-100 font-mono text-[10px] truncate">
                  {metrics.model}
                </span>
              </div>
            )}
          </div>

          {/* Total Summary Row */}
          <div className="mt-3 pt-3 border-t border-gray-600/40">
            <div className="flex justify-between items-center text-xs">
              <span className="text-gray-400">Total Tokens</span>
              <span className="text-gray-100 font-mono font-semibold">
                {formatTokens(totalTokens)}
              </span>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
        }

        .token-badge {
          cursor: pointer;
          user-select: none;
        }

        .token-badge:active {
          transform: scale(0.98);
        }
      `}</style>
    </div>
  );
};

export default TokenBadge;

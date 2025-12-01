import React from "react";
import useModelUsage from "../hooks/useModelUsage";
import useModelPricing from "../hooks/useModelPricing";
import { formatCost, formatTokens } from "../utils/tokenPricing";
import { detectProviderFromModel } from "../utils/models";

interface ModelUsagePreviewProps {
  modelName?: string;
  days?: number;
  contextTokens?: number;
  providerHint?: string;
}

const ModelUsagePreview: React.FC<ModelUsagePreviewProps> = ({
  modelName,
  days = 7,
  contextTokens,
  providerHint,
}) => {
  const { data: usage, loading: usageLoading, error: usageError } =
    useModelUsage(modelName, days);
  const { data: pricing, loading: pricingLoading, error: pricingError } =
    useModelPricing(modelName);

  if (!modelName) return null;

  const provider =
    usage?.provider ||
    providerHint ||
    detectProviderFromModel(modelName, "cloud");

  const inputCost = pricing?.pricing?.input;
  const outputCost = pricing?.pricing?.output;

  return (
    <div className="mt-3 rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="flex justify-between items-center mb-2">
        <div>
          <p className="text-sm font-semibold text-white">
            {modelName.split(/[/\\\\]/).pop()}
          </p>
          <p className="text-[11px] text-gray-400">
            Provider: {provider || "unknown"}
          </p>
        </div>
        {(usageLoading || pricingLoading) && (
          <span className="text-[11px] text-gray-400 animate-pulse">
            Loading...
          </span>
        )}
      </div>

      {pricingError && (
        <p className="text-xs text-red-400 mb-1">{pricingError}</p>
      )}

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-gray-900/60 border border-white/5 rounded-lg p-3">
          <p className="text-gray-400 mb-1">Input / 1M</p>
          <p className="text-white font-semibold">
            {inputCost !== undefined ? `$${inputCost.toFixed(3)}` : "N/A"}
          </p>
        </div>
        <div className="bg-gray-900/60 border border-white/5 rounded-lg p-3">
          <p className="text-gray-400 mb-1">Output / 1M</p>
          <p className="text-white font-semibold">
            {outputCost !== undefined ? `$${outputCost.toFixed(3)}` : "N/A"}
          </p>
        </div>
        {contextTokens ? (
          <div className="bg-gray-900/60 border border-white/5 rounded-lg p-3 col-span-2">
            <p className="text-gray-400 mb-1">Context Window</p>
            <p className="text-white font-semibold">
              {formatTokens(contextTokens)} tokens
            </p>
          </div>
        ) : null}
      </div>

      <div className="mt-4 bg-gray-900/60 border border-white/5 rounded-lg p-3">
        <div className="flex justify-between items-center mb-2 text-xs text-gray-300">
          <span>Last {days} days</span>
          {usageError && <span className="text-red-400">{usageError}</span>}
        </div>
        {usage ? (
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <p className="text-gray-400 mb-1">Messages</p>
              <p className="text-white font-semibold">{usage.messages}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Tokens</p>
              <p className="text-white font-semibold">
                {formatTokens(usage.totalTokens)}
              </p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Cost</p>
              <p className="text-emerald-300 font-semibold">
                {formatCost(usage.totalCost)}
              </p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Avg / msg</p>
              <p className="text-white font-semibold">
                {formatCost(usage.avgCostPerMessage)}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-xs text-gray-500">
            {usageLoading ? "Loading usage…" : "No usage yet for this model."}
          </p>
        )}
      </div>
    </div>
  );
};

export default ModelUsagePreview;

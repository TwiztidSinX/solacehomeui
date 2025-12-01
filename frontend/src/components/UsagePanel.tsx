import React, { useMemo, useState } from "react";
import useUsageStats from "../hooks/useUsageStats";
import { formatCost, formatTokens } from "../utils/tokenPricing";

interface UsagePanelProps {
  isOpen: boolean;
  onClose: () => void;
  defaultDays?: number;
}

const UsagePanel: React.FC<UsagePanelProps> = ({
  isOpen,
  onClose,
  defaultDays = 7,
}) => {
  const [days, setDays] = useState(defaultDays);
  const { data, loading, error } = useUsageStats(days);

  const maxTokensByDay = useMemo(() => {
    if (!data?.byDay?.length) return 1;
    return Math.max(...data.byDay.map((d) => d.totalTokens), 1);
  }, [data]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="w-full max-w-5xl bg-gray-900/95 border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-lg font-semibold text-white">Usage Analytics</h2>
            <p className="text-xs text-gray-400">
              Rolling window of your token spend and throughput.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value, 10) || 7)}
              className="bg-gray-800 text-white text-sm rounded px-3 py-2 border border-white/10"
            >
              <option value={7}>Last 7 days</option>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
            </select>
            <button
              onClick={onClose}
              className="text-gray-300 hover:text-white bg-white/5 hover:bg-white/10 rounded-full p-2"
              aria-label="Close usage analytics"
            >
              X
            </button>
          </div>
        </div>

        <div className="p-6 space-y-4">
          {error && (
            <div className="text-red-400 text-sm bg-red-900/30 border border-red-700/50 rounded px-3 py-2">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-gray-400 mb-1">Total Tokens</p>
              <p className="text-2xl font-semibold text-white">
                {data ? formatTokens(data.totalTokens) : "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-gray-400 mb-1">Total Cost</p>
              <p className="text-2xl font-semibold text-emerald-300">
                {data ? formatCost(data.totalCost) : "N/A"}
              </p>
            </div>
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-gray-400 mb-1">Window</p>
              <p className="text-2xl font-semibold text-white">{`${days}d`}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-white">
                  Daily Tokens
                </h3>
                {loading && (
                  <span className="text-xs text-gray-400 animate-pulse">
                    Loading...
                  </span>
                )}
              </div>
              <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                {data?.byDay?.length ? (
                  data.byDay.map((entry) => {
                    const width =
                      maxTokensByDay > 0
                        ? Math.max(
                            8,
                            Math.min(
                              100,
                              (entry.totalTokens / maxTokensByDay) * 100,
                            ),
                          )
                        : 0;
                    return (
                      <div key={entry.date}>
                        <div className="flex justify-between text-xs text-gray-300 mb-1">
                          <span className="text-gray-400">{entry.date}</span>
                          <span>{formatTokens(entry.totalTokens)}</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${width}%` }}
                          />
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-xs text-gray-500">
                    {loading ? "Fetching usage..." : "No usage in this window."}
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-white">
                  Usage by Model
                </h3>
              </div>
              <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
                {data?.byModel?.length ? (
                  data.byModel.map((model) => (
                    <div
                      key={`${model.provider || "unknown"}-${model.model}`}
                      className="p-3 rounded-lg bg-gray-800/80 border border-white/5"
                    >
                      <div className="flex justify-between text-sm text-white">
                        <span className="truncate mr-2">
                          {model.model}
                          {model.provider
                            ? ` (${model.provider})`
                            : ""}
                        </span>
                        <span className="text-gray-300">
                          {formatTokens(model.totalTokens)}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs text-gray-400 mt-1">
                        <span>{model.messages} msgs</span>
                        <span>{formatCost(model.totalCost)}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-gray-500">
                    {loading ? "Loading..." : "No model breakdown yet."}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UsagePanel;

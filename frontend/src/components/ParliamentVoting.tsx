import React from "react";

interface VoteResult {
  winning_answer: string;
  winning_model: string;
  confidence: number;
  votes: number;
  total_clusters: number;
  cluster_details: Array<{
    models: string[];
    size: number;
    avg_confidence: number;
  }>;
}

interface ParliamentVotingProps {
  voteResult: VoteResult | null;
  onRetryVote?: () => void;
}

export const ParliamentVoting: React.FC<ParliamentVotingProps> = ({
  voteResult,
  onRetryVote,
}) => {
  if (!voteResult) {
    return (
      <div className="p-4 bg-purple-500/20 rounded-lg text-center text-gray-400">
        <p>No voting data yet. Send a Parliament request to see results.</p>
      </div>
    );
  }

  const { winning_model, confidence, votes, total_clusters, cluster_details } =
    voteResult;

  // Calculate total models
  const totalModels = cluster_details.reduce(
    (sum, cluster) => sum + cluster.size,
    0
  );

  // Sort clusters by size (largest first)
  const sortedClusters = [...cluster_details].sort((a, b) => b.size - a.size);
  const majorityCluster = sortedClusters[0];

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="p-3 bg-gradient-to-r from-purple-600/30 to-blue-600/30 rounded-lg border border-purple-500/30">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-white font-semibold flex items-center gap-2">
            🏛️ Parliament Vote Complete
          </h3>
          {onRetryVote && (
            <button
              onClick={onRetryVote}
              className="px-3 py-1 text-xs rounded bg-purple-600 hover:bg-purple-700 text-white transition-colors"
            >
              Retry Vote
            </button>
          )}
        </div>
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div>
            <p className="text-gray-400 text-xs">Consensus</p>
            <p className="text-white font-semibold">
              {votes}/{totalModels} models
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Clusters</p>
            <p className="text-white font-semibold">{total_clusters}</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Confidence</p>
            <p className="text-white font-semibold">
              {(confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* Winner Banner */}
      <div className="p-3 bg-green-600/20 rounded-lg border border-green-500/30">
        <p className="text-green-400 text-xs font-semibold mb-1">
          🏆 WINNING MODEL
        </p>
        <p className="text-white font-semibold">{winning_model}</p>
        <div className="mt-2 bg-gray-900/50 rounded h-2 overflow-hidden">
          <div
            className="bg-green-500 h-full transition-all duration-500"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Cluster Breakdown */}
      <div className="space-y-2">
        <p className="text-gray-300 text-sm font-semibold">Vote Distribution</p>

        {sortedClusters.map((cluster, index) => {
          const isMajority = cluster === majorityCluster;
          const percentage = ((cluster.size / totalModels) * 100).toFixed(0);

          return (
            <div
              key={index}
              className={`p-3 rounded-lg border ${
                isMajority
                  ? "bg-green-600/20 border-green-500/30"
                  : "bg-gray-700/30 border-gray-600/30"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs font-semibold px-2 py-0.5 rounded ${
                      isMajority
                        ? "bg-green-500/30 text-green-300"
                        : "bg-gray-600/30 text-gray-300"
                    }`}
                  >
                    {isMajority ? "MAJORITY" : `CLUSTER ${index + 1}`}
                  </span>
                  <span className="text-white font-semibold">
                    {cluster.size} {cluster.size === 1 ? "vote" : "votes"}
                  </span>
                </div>
                <span className="text-gray-400 text-sm">{percentage}%</span>
              </div>

              {/* Progress bar */}
              <div className="mb-2 bg-gray-900/50 rounded h-2 overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    isMajority ? "bg-green-500" : "bg-gray-500"
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>

              {/* Models in cluster */}
              <div className="flex flex-wrap gap-1">
                {cluster.models.map((model, i) => (
                  <span
                    key={i}
                    className={`px-2 py-0.5 rounded text-xs ${
                      isMajority
                        ? "bg-green-500/20 text-green-200"
                        : "bg-gray-600/20 text-gray-300"
                    }`}
                  >
                    {model}
                  </span>
                ))}
              </div>

              {/* Average confidence */}
              <div className="mt-2 text-xs text-gray-400">
                Avg confidence: {(cluster.avg_confidence * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>

      {/* Consensus Quality Indicator */}
      <div
        className={`p-2 rounded text-xs text-center ${
          votes >= 4
            ? "bg-green-600/20 text-green-300"
            : votes >= 2
              ? "bg-yellow-600/20 text-yellow-300"
              : "bg-red-600/20 text-red-300"
        }`}
      >
        {votes >= 4
          ? "✓ Strong consensus reached"
          : votes >= 2
            ? "⚠ Weak consensus - results may vary"
            : "⚠ No clear consensus - high disagreement"}
      </div>
    </div>
  );
};

import { useCallback, useEffect, useState } from "react";
import { type ModelUsageStats } from "../types";

export const useModelUsage = (modelName?: string, days: number = 7) => {
  const [data, setData] = useState<ModelUsageStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUsage = useCallback(
    async (overrideModel?: string) => {
      const model = overrideModel ?? modelName;
      if (!model) {
        setData(null);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/usage/model/${encodeURIComponent(model)}?days=${days}`,
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const json = await res.json();
        setData(json as ModelUsageStats);
      } catch (e: any) {
        console.error("Failed to load model usage", e);
        setError(e?.message || "Failed to load model usage");
      } finally {
        setLoading(false);
      }
    },
    [modelName, days],
  );

  useEffect(() => {
    fetchUsage();
  }, [fetchUsage]);

  return { data, loading, error, refresh: fetchUsage };
};

export default useModelUsage;

import { useCallback, useEffect, useState } from "react";
import { type UsageSummary } from "../types";

export const useUsageStats = (days: number = 7) => {
  const [data, setData] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(
    async (overrideDays?: number) => {
      const windowDays = overrideDays ?? days;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/usage/summary?days=${windowDays}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const json = await res.json();
        setData(json as UsageSummary);
      } catch (e: any) {
        console.error("Failed to load usage summary", e);
        setError(e?.message || "Failed to load usage summary");
      } finally {
        setLoading(false);
      }
    },
    [days],
  );

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return { data, loading, error, refresh: fetchStats };
};

export default useUsageStats;

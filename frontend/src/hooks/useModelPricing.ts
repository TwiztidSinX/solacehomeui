import { useCallback, useEffect, useState } from "react";

interface ModelPricingResponse {
  model: string;
  pricing: { input?: number; output?: number };
}

export const useModelPricing = (modelName?: string) => {
  const [data, setData] = useState<ModelPricingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPricing = useCallback(
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
          `/api/models/pricing/${encodeURIComponent(model)}`,
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const json = (await res.json()) as ModelPricingResponse;
        setData(json);
      } catch (e: any) {
        console.error("Failed to load pricing", e);
        setError(e?.message || "Failed to load pricing");
      } finally {
        setLoading(false);
      }
    },
    [modelName],
  );

  useEffect(() => {
    fetchPricing();
  }, [fetchPricing]);

  return { data, loading, error, refresh: fetchPricing };
};

export default useModelPricing;

import { type ProviderType } from "../constants/parliament";

export const detectProviderFromModel = (
  model: string,
  fallback: ProviderType | string = "cloud",
) => {
  const m = (model || "").toLowerCase();
  if (fallback !== "cloud" && fallback !== "local") return fallback as string;
  if (m.includes("deepseek")) return "deepseek";
  if (m.includes("qwen")) return "qwen";
  if (m.includes("gpt") || m.includes("openai")) return "openai";
  if (m.includes("claude") || m.includes("anthropic")) return "anthropic";
  if (m.includes("gemini")) return "google";
  if (m.includes("grok") || m.includes("xai")) return "xai";
  return fallback as string;
};

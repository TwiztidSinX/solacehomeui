import { type Message } from "../types";

export type ProviderType = "local" | "cloud";

export interface ParliamentRoleConfig {
  key: string;
  name: string;
  defaultModel: string;
  provider: ProviderType;
  model: string;
  prompt: string;
  enabled: boolean;
  status: "idle" | "working" | "done";
}

export const PARLIAMENT_ROLES: ParliamentRoleConfig[] = [
  {
    key: "analyst",
    name: "Analyst - GPT-5.1",
    defaultModel: "gpt-5.1",
    provider: "cloud",
    model: "gpt-5.1",
    enabled: true,
    status: "idle",
    prompt: `You are the Analyst. Deliver crisp analysis of the user's request.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}. Stay concise.`,
  },
  {
    key: "researcher",
    name: "Researcher - DeepSeek",
    defaultModel: "deepseek-chat",
    provider: "cloud",
    model: "deepseek-chat",
    enabled: true,
    status: "idle",
    prompt: `You are the Researcher. Validate facts and surface missing info.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}. Cite sources when possible.`,
  },
  {
    key: "specialist",
    name: "Specialist - Qwen3-Max",
    defaultModel: "qwen3-max",
    provider: "cloud",
    model: "qwen3-max",
    enabled: true,
    status: "idle",
    prompt: `You are the Specialist. Provide technical depth and edge cases.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}.`,
  },
  {
    key: "philosopher",
    name: "Philosopher - Claude Sonnet 4.5",
    defaultModel: "claude-sonnet-4-5-20250929",
    provider: "cloud",
    model: "claude-sonnet-4-5-20250929",
    enabled: false,
    status: "idle",
    prompt: `You are the Philosopher. Clarify principles, ethics, and ambiguity.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}.`,
  },
  {
    key: "synthesizer",
    name: "Synthesizer - Gemini 3.0",
    defaultModel: "gemini-3.0",
    provider: "cloud",
    model: "gemini-3.0",
    enabled: false,
    status: "idle",
    prompt: `You are the Synthesizer. Combine insights and recommend a plan.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}.`,
  },
  {
    key: "maverick",
    name: "Maverick - Grok 4.1",
    defaultModel: "grok-4-1-fast-reasoning",
    provider: "cloud",
    model: "grok-4-1-fast-reasoning",
    enabled: false,
    status: "idle",
    prompt: `You are the Maverick. Offer bold alternatives and risks.
Return JSON: {"analysis":"","strengths":"","weaknesses":"","proposal":"","confidence":0-100}.`,
  },
];

export interface ChatSession {
  _id: string;
  name: string;
  messages: Message[];
}

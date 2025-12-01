"""
Token Pricing and Cost Calculation Utilities for Backend
Prices are per million tokens (USD)
Last Updated: November 2025
"""

import time
import tiktoken
from typing import Optional, Dict, Any

# Model pricing per million tokens
MODEL_PRICING = {
    # ==================== ANTHROPIC CLAUDE MODELS ====================
    'claude-sonnet-4-5': {'input': 3.00, 'output': 15.00},
    'claude-sonnet-4': {'input': 3.00, 'output': 15.00},
    'claude-opus-4.1': {'input': 15.00, 'output': 75.00},
    'claude-opus-4': {'input': 15.00, 'output': 75.00},
    'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
    'claude-3-opus': {'input': 15.00, 'output': 75.00},
    'claude-3-5-haiku': {'input': 0.80, 'output': 4.00},
    'claude-3-haiku': {'input': 0.25, 'output': 1.25},
    'claude-3-sonnet': {'input': 3.00, 'output': 15.00},

    # ==================== OPENAI GPT MODELS ====================
    # GPT-5.1 Series
    'gpt-5.1': {'input': 1.25, 'output': 10.00},

    # GPT-5 Series (Released Aug 2025)
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00},
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'gpt-5-chat-latest': {'input': 1.25, 'output': 10.00},
    
    # GPT-4.1 Series
    'gpt-4.1': {'input': 2.10, 'output': 8.40},
    'gpt-4.1-mini': {'input': 0.42, 'output': 1.68},
    'gpt-4.1-nano': {'input': 0.105, 'output': 0.42},
    
    # GPT-4 Series
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
    'gpt-4': {'input': 30.00, 'output': 60.00},
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
    
    # O-Series (Reasoning Models)
    'o3': {'input': 2.10, 'output': 8.40},
    'o3-mini': {'input': 1.155, 'output': 4.62},
    'o3-pro': {'input': 21.00, 'output': 84.00},
    'o4-mini': {'input': 1.155, 'output': 4.62},
    'o1': {'input': 15.00, 'output': 60.00},
    'o1-mini': {'input': 3.00, 'output': 12.00},

    # ==================== DEEPSEEK MODELS ====================
    # Note: DeepSeek API only has 2 models (updated Sept 2025)
    'deepseek-chat': {'input': 0.56, 'output': 1.68},
    'deepseek-reasoner': {'input': 0.56, 'output': 1.68},  # Also known as deepseek-r1

    # ==================== GOOGLE GEMINI MODELS ====================
    # Gemini 3.0 Series
    'gemini-3-pro': {'input': 2.00, 'output': 12.00},  # Up to 200K context
    'gemini-3-pro-long': {'input': 4.00, 'output': 18.00},  # Over 200K context
    
    # Gemini 2.5 Series
    'gemini-2.5-pro': {'input': 2.00, 'output': 10.00},
    'gemini-2.5-flash': {'input': 0.10, 'output': 2.00},
    'gemini-2.5-flash-lite': {'input': 0.02, 'output': 0.50},
    
    # Gemini 2.0 Series
    'gemini-2.0-flash': {'input': 0.20, 'output': 0.80},
    'gemini-2.0-flash-lite': {'input': 0.02, 'output': 0.08},
    
    # Gemini 1.5 Series (Legacy)
    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
    'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},

    # ==================== XAI GROK MODELS ====================
    # Grok 4.1 Series (Released Nov 2025 - SUPER CHEAP!)
    'grok-4.1-fast-reasoning': {'input': 0.20, 'output': 0.50},
    'grok-4.1-fast-non-reasoning': {'input': 0.20, 'output': 0.50},
    'grok-4.1': {'input': 0.20, 'output': 0.50},
    
    # Grok 4 Series
    'grok-4': {'input': 3.00, 'output': 15.00},
    'grok-4-fast': {'input': 3.00, 'output': 15.00},
    'grok-code-fast-1': {'input': 3.00, 'output': 15.00},
    
    # Grok 3 Series
    'grok-3': {'input': 2.00, 'output': 10.00},
    'grok-3-mini': {'input': 0.50, 'output': 2.00},
    
    # Grok 2 Series
    'grok-2': {'input': 2.00, 'output': 10.00},
    'grok-2-vision': {'input': 2.00, 'output': 10.00},

    # ==================== QWEN/ALIBABA CLOUD MODELS ====================
    # Qwen 3 Max Series
    'qwen-max': {'input': 1.68, 'output': 6.72},
    'qwen3-max': {'input': 1.68, 'output': 6.72},
    'qwen3-max-preview': {'input': 1.68, 'output': 6.72},
    
    # Qwen Plus/Turbo
    'qwen-plus': {'input': 0.42, 'output': 1.26},
    'qwen3-plus': {'input': 0.42, 'output': 1.26},
    'qwen-turbo': {'input': 0.0525, 'output': 0.21},
    'qwen3-turbo': {'input': 0.0525, 'output': 0.21},
    
    # Qwen 3 Thinking/Reasoning
    'qwen3-235b-a22b-thinking': {'input': 0.2415, 'output': 2.415},
    'qwen3-235b-a22b-instruct': {'input': 0.2415, 'output': 2.415},
    
    # Qwen Coder Models
    'qwen3-coder-plus': {'input': 0.40, 'output': 1.20},
    'qwen3-coder-flash': {'input': 0.10, 'output': 0.30},
    
    # Qwen 2.5 Series (Open Source variants)
    'qwen2.5-72b': {'input': 0.35, 'output': 0.40},
    'qwen2.5-32b': {'input': 0.20, 'output': 0.30},
    'qwen2.5-14b': {'input': 0.10, 'output': 0.20},
    'qwen2.5-7b': {'input': 0.05, 'output': 0.10},

    # ==================== META LLAMA MODELS ====================
    'llama-3.3-70b': {'input': 0.35, 'output': 0.40},
    'llama-3.1-405b': {'input': 2.70, 'output': 2.70},
    'llama-3.1-70b': {'input': 0.35, 'output': 0.40},
    'llama-3.1-8b': {'input': 0.10, 'output': 0.10},

    # ==================== MISTRAL MODELS ====================
    'mistral-large': {'input': 2.00, 'output': 6.00},
    'mistral-medium': {'input': 0.70, 'output': 2.00},
    'mistral-small': {'input': 0.20, 'output': 0.60},

    # ==================== PERPLEXITY MODELS ====================
    'perplexity-sonar-pro': {'input': 3.00, 'output': 15.00},
    'perplexity-sonar': {'input': 1.00, 'output': 1.00},

    # Default fallback
    'default': {'input': 1.00, 'output': 3.00},
}


class TokenTracker:
    """Track token usage and performance metrics during streaming"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = None
        self.first_token_time = None
        self.token_times = []

    def start(self):
        """Mark the start of generation"""
        self.start_time = time.time()

    def add_input_tokens(self, count: int):
        """Add input token count"""
        self.input_tokens += count

    def add_output_token(self, token_text: str = ""):
        """Add an output token and track timing"""
        self.output_tokens += 1

        # Record time to first token
        if self.first_token_time is None and self.start_time is not None:
            self.first_token_time = time.time()

        # Track token times for throughput calculation
        if self.start_time is not None:
            self.token_times.append(time.time())

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return all metrics"""
        metrics = {
            'inputTokens': self.input_tokens,
            'outputTokens': self.output_tokens,
            'model': self.model_name,
        }

        # Calculate time to first token (TTFT)
        if self.first_token_time is not None and self.start_time is not None:
            ttft_seconds = self.first_token_time - self.start_time
            metrics['timeToFirstToken'] = int(ttft_seconds * 1000)  # Convert to milliseconds

        # Calculate tokens per second
        if self.token_times and len(self.token_times) > 1:
            total_time = self.token_times[-1] - self.token_times[0]
            if total_time > 0:
                metrics['tokensPerSecond'] = round(len(self.token_times) / total_time, 1)

        # Calculate estimated cost
        cost = calculate_cost(self.input_tokens, self.output_tokens, self.model_name)
        if cost > 0:
            metrics['estimatedCost'] = round(cost, 6)

        return metrics


def normalize_model_name(model_name: Optional[str]) -> str:
    """Normalize model name to match pricing keys"""
    if not model_name:
        return 'default'

    lower = model_name.lower()

    # Direct match
    if lower in MODEL_PRICING:
        return lower

    # ==================== ANTHROPIC CLAUDE ====================
    if 'claude' in lower:
        if 'opus-4.1' in lower or 'opus-4-1' in lower:
            return 'claude-opus-4.1'
        if 'opus-4' in lower:
            return 'claude-opus-4'
        if 'sonnet-4-5' in lower or 'sonnet-4.5' in lower:
            return 'claude-sonnet-4-5'
        if 'sonnet-4' in lower:
            return 'claude-sonnet-4'
        if '3-5-haiku' in lower or '3.5-haiku' in lower:
            return 'claude-3-5-haiku'
        if 'opus' in lower:
            return 'claude-3-opus'
        if 'haiku' in lower:
            return 'claude-3-haiku'
        if 'sonnet' in lower:
            return 'claude-3-sonnet'

    # ==================== OPENAI GPT ====================
    if 'gpt' in lower or 'o1' in lower or 'o3' in lower or 'o4' in lower:
        # GPT-5.1 Series
        if 'gpt-5.1' in lower or 'gpt5.1' in lower or 'gpt-5_1' in lower:
            return 'gpt-5.1'

        # GPT-5 Series
        if 'gpt-5-nano' in lower or 'gpt5-nano' in lower:
            return 'gpt-5-nano'
        if 'gpt-5-mini' in lower or 'gpt5-mini' in lower:
            return 'gpt-5-mini'
        if 'gpt-5-chat' in lower or 'gpt5-chat' in lower:
            return 'gpt-5-chat-latest'
        if 'gpt-5' in lower or 'gpt5' in lower:
            return 'gpt-5'
        
        # GPT-4.1 Series
        if 'gpt-4.1-nano' in lower or 'gpt-4-1-nano' in lower:
            return 'gpt-4.1-nano'
        if 'gpt-4.1-mini' in lower or 'gpt-4-1-mini' in lower:
            return 'gpt-4.1-mini'
        if 'gpt-4.1' in lower or 'gpt-4-1' in lower:
            return 'gpt-4.1'
        
        # GPT-4 Series
        if 'gpt-4o-mini' in lower:
            return 'gpt-4o-mini'
        if 'gpt-4o' in lower:
            return 'gpt-4o'
        if 'gpt-4-turbo' in lower:
            return 'gpt-4-turbo'
        if 'gpt-4' in lower:
            return 'gpt-4'
        if 'gpt-3.5' in lower or '3.5' in lower:
            return 'gpt-3.5-turbo'
        
        # O-Series
        if 'o4-mini' in lower:
            return 'o4-mini'
        if 'o3-pro' in lower:
            return 'o3-pro'
        if 'o3-mini' in lower:
            return 'o3-mini'
        if 'o3' in lower:
            return 'o3'
        if 'o1-mini' in lower:
            return 'o1-mini'
        if 'o1' in lower:
            return 'o1'

    # ==================== DEEPSEEK ====================
    if 'deepseek' in lower:
        if 'reason' in lower or 'r1' in lower:
            return 'deepseek-reasoner'
        return 'deepseek-chat'

    # ==================== GOOGLE GEMINI ====================
    if 'gemini' in lower:
        # Gemini 3.0
        if '3.0' in lower or '3-0' in lower or 'gemini-3' in lower:
            if 'long' in lower or '200k' in lower:
                return 'gemini-3-pro-long'
            return 'gemini-3-pro'
        
        # Gemini 2.5
        if '2.5' in lower or '2-5' in lower:
            if 'lite' in lower:
                return 'gemini-2.5-flash-lite'
            if 'flash' in lower:
                return 'gemini-2.5-flash'
            if 'pro' in lower:
                return 'gemini-2.5-pro'
        
        # Gemini 2.0
        if '2.0' in lower or '2-0' in lower:
            if 'lite' in lower:
                return 'gemini-2.0-flash-lite'
            return 'gemini-2.0-flash'
        
        # Gemini 1.5
        if 'pro' in lower:
            return 'gemini-1.5-pro'
        if 'flash' in lower:
            return 'gemini-1.5-flash'

    # ==================== XAI GROK ====================
    if 'grok' in lower:
        # Grok 4.1
        if '4.1' in lower or '4-1' in lower:
            if 'non-reason' in lower:
                return 'grok-4.1-fast-non-reasoning'
            if 'reason' in lower:
                return 'grok-4.1-fast-reasoning'
            return 'grok-4.1'
        
        # Grok 4
        if '4' in lower:
            if 'code' in lower:
                return 'grok-code-fast-1'
            if 'fast' in lower:
                return 'grok-4-fast'
            return 'grok-4'
        
        # Grok 3
        if '3' in lower:
            if 'mini' in lower:
                return 'grok-3-mini'
            return 'grok-3'
        
        # Grok 2
        if 'vision' in lower:
            return 'grok-2-vision'
        return 'grok-2'

    # ==================== QWEN ====================
    if 'qwen' in lower:
        # Qwen 3
        if 'max' in lower or '235b' in lower:
            if 'preview' in lower:
                return 'qwen3-max-preview'
            return 'qwen-max'
        if 'plus' in lower:
            return 'qwen-plus'
        if 'turbo' in lower:
            return 'qwen-turbo'
        if 'thinking' in lower or 'reason' in lower:
            return 'qwen3-235b-a22b-thinking'
        if 'coder-plus' in lower:
            return 'qwen3-coder-plus'
        if 'coder-flash' in lower or 'coder' in lower:
            return 'qwen3-coder-flash'
        
        # Qwen 2.5
        if '2.5' in lower or '2-5' in lower:
            if '72b' in lower:
                return 'qwen2.5-72b'
            if '32b' in lower:
                return 'qwen2.5-32b'
            if '14b' in lower:
                return 'qwen2.5-14b'
            if '7b' in lower:
                return 'qwen2.5-7b'

    # ==================== LLAMA ====================
    if 'llama' in lower:
        if '405' in lower or '405b' in lower:
            return 'llama-3.1-405b'
        if '3.3' in lower and '70' in lower:
            return 'llama-3.3-70b'
        if '70' in lower or '70b' in lower:
            return 'llama-3.1-70b'
        if '8' in lower or '8b' in lower:
            return 'llama-3.1-8b'

    # ==================== MISTRAL ====================
    if 'mistral' in lower:
        if 'large' in lower:
            return 'mistral-large'
        if 'medium' in lower:
            return 'mistral-medium'
        if 'small' in lower:
            return 'mistral-small'

    # ==================== PERPLEXITY ====================
    if 'perplexity' in lower or 'sonar' in lower:
        if 'pro' in lower:
            return 'perplexity-sonar-pro'
        return 'perplexity-sonar'

    return 'default'


def calculate_cost(input_tokens: int, output_tokens: int, model_name: Optional[str] = None) -> float:
    """Calculate estimated cost for token usage"""
    normalized = normalize_model_name(model_name)
    pricing = MODEL_PRICING.get(normalized, MODEL_PRICING['default'])

    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']

    return input_cost + output_cost


def count_tokens_simple(text: str) -> int:
    """
    Simple token counting based on word splitting.
    More accurate than character count, but less accurate than proper tokenization.
    Rule of thumb: ~1.3 tokens per word for English text
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)


def count_tokens_tiktoken(text: str, model_name: Optional[str] = None) -> int:
    """
    Count tokens using tiktoken library (for OpenAI-compatible models).
    Falls back to simple counting if tiktoken is not available.
    """
    try:
        # Try to get the encoding for the model
        if model_name:
            # Normalize model name for tiktoken
            if 'gpt-4' in model_name.lower() or 'gpt-5' in model_name.lower():
                encoding = tiktoken.encoding_for_model('gpt-4')
            elif 'gpt-3.5' in model_name.lower():
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            else:
                encoding = tiktoken.get_encoding('cl100k_base')
        else:
            encoding = tiktoken.get_encoding('cl100k_base')

        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple counting
        return count_tokens_simple(text)


def _coerce_content_to_text(content) -> str:
    """
    Extract plain text from content that may be a string or a multimodal list.
    """
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(p for p in parts if p)
    if isinstance(content, str):
        return content
    return ""


def estimate_input_tokens(conversation_history: list, user_input: str, model_name: Optional[str] = None) -> int:
    """
    Estimate the total input tokens from conversation history and current input.
    """
    total_text = ""

    # Add conversation history
    for msg in conversation_history:
        if isinstance(msg, dict):
            total_text += _coerce_content_to_text(msg.get('content', '')) + "\n"
        elif isinstance(msg, str):
            total_text += msg + "\n"

    # Add current user input
    total_text += _coerce_content_to_text(user_input)

    # Use tiktoken if available, otherwise simple counting
    try:
        return count_tokens_tiktoken(total_text, model_name)
    except Exception:
        return count_tokens_simple(total_text)

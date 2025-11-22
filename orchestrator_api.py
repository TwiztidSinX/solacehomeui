"""
API Orchestrator Wrapper for SolaceHomeUI
Provides llama.cpp-compatible interface for API models (OpenAI, DeepSeek, Qwen)
"""

import os
import json
from typing import List, Dict, Any, Optional, Iterator
import requests


class APIOrchestrator:
    """
    Wrapper that makes API models (OpenAI, DeepSeek, Qwen) compatible
    with the llama.cpp interface used by the agentic orchestrator.
    """
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        """
        Initialize API orchestrator.
        
        Args:
            provider: "openai", "deepseek", or "qwen"
            model: Model name (e.g., "gpt-4o-mini", "deepseek-chat")
            api_key: Optional API key (will try environment if not provided)
        """
        self.provider = provider.lower()
        self.model = model
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.environ.get(env_var)
            if not self.api_key:
                raise ValueError(f"No API key provided and {env_var} not set")
        
        # Set API endpoint based on provider
        self.base_url = self._get_base_url()

        print(f"✅ API Orchestrator initialized: {provider}/{model}")
    
    def _get_base_url(self) -> str:
        """Get the API base URL for the provider."""
        urls = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
        return urls.get(self.provider, "https://api.openai.com/v1")
    
    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        llama.cpp-compatible completion interface.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            stream: Whether to stream (not implemented for orchestrator)
            
        Returns:
            Dict with 'choices' containing generated text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build request
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        # OpenAI's newer models expect max_completion_tokens instead of max_tokens
        if self.provider == "openai":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        
        if stop:
            payload["stop"] = stop
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                body = ""
                try:
                    body = response.text[:500]
                except Exception:
                    pass
                msg = f"API request failed ({self.provider}/{self.model}): {e}; body={body}"
                print(f"❌ {msg}")
                raise requests.exceptions.RequestException(msg) from e

            data = response.json()
            
            # Convert to llama.cpp format
            return {
                "choices": [
                    {
                        "text": data["choices"][0]["message"]["content"],
                        "finish_reason": data["choices"][0].get("finish_reason", "stop")
                    }
                ]
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            raise
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Make the object callable like llama.cpp models."""
        return self.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=stream
        )
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        llama.cpp-style chat completion interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            stream: Whether to stream (not implemented)
            
        Returns:
            Dict with 'choices' containing the response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if self.provider == "openai":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        
        if stop:
            payload["stop"] = stop
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                body = ""
                try:
                    body = response.text[:500]
                except Exception:
                    pass
                msg = f"API request failed ({self.provider}/{self.model}): {e}; body={body}"
                print(f"❌ {msg}")
                raise requests.exceptions.RequestException(msg) from e

            data = response.json()
            
            # Convert to llama.cpp format (with nested 'message' structure)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": data["choices"][0]["message"]["content"]
                        },
                        "finish_reason": data["choices"][0].get("finish_reason", "stop")
                    }
                ]
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            raise

class OpenAIOrchestrator(APIOrchestrator):
    """Convenience wrapper for OpenAI models."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__("openai", model, api_key)


class DeepSeekOrchestrator(APIOrchestrator):
    """Convenience wrapper for DeepSeek models."""
    
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        super().__init__("deepseek", model, api_key)


class QwenOrchestrator(APIOrchestrator):
    """Convenience wrapper for Qwen models."""
    
    def __init__(self, model: str = "qwen-max", api_key: Optional[str] = None):
        super().__init__("qwen", model, api_key)


def _normalize_model(provider: str, model: str) -> str:
    """Map friendly/legacy model names to provider-supported ones."""
    p = (provider or "").lower()
    m = model or ""
    if p == "deepseek":
        # Align to the public chat model name
        if "deepseek" in m and "chat" not in m:
            return "deepseek-chat"
    return m


def create_api_orchestrator(provider: str, model: str, api_key: Optional[str] = None) -> APIOrchestrator:
    """
    Factory function to create the appropriate API orchestrator.
    
    Args:
        provider: "openai", "deepseek", or "qwen"
        model: Model name
        api_key: Optional API key
        
    Returns:
        APIOrchestrator instance
    """
    provider = provider.lower()
    
    model = _normalize_model(provider, model)

    if provider == "openai":
        return OpenAIOrchestrator(model, api_key)
    elif provider == "deepseek":
        return DeepSeekOrchestrator(model, api_key)
    elif provider == "qwen":
        return QwenOrchestrator(model, api_key)
    else:
        # Generic API orchestrator
        return APIOrchestrator(provider, model, api_key)


if __name__ == "__main__":
    print("API Orchestrator module loaded.")
    print("Available providers: openai, deepseek, qwen")

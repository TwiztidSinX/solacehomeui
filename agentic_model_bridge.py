"""
Agentic Model Bridge
====================

This module bridges the existing model_loader.py with the new model_client_wrapper.py
to provide tool-enabled model clients for the Agentic Coding system.

This keeps the existing chat system completely untouched while enabling proper
tool calling for agentic workflows.
"""

import os
from model_loader import load_model
from model_client_wrapper import create_model_client


def get_model_client_for_agentic_coding(
    model_path: str,
    backend: str = "llama.cpp",
    provider: str = None,
    **load_kwargs
):
    """
    Load a model and wrap it for agentic coding with tool support.
    
    This function:
    1. Loads the model using the existing load_model() function
    2. Wraps it in ModelClientWrapper to enable tool calling
    3. Returns a client that works with the Agentic Coder
    
    Args:
        model_path: Path or identifier for the model
        backend: Backend type ("llama.cpp", "ollama", "api", "safetensors")
        provider: API provider if backend is "api"
        **load_kwargs: Additional arguments passed to load_model()
        
    Returns:
        ModelClientWrapper instance with tool calling support
        
    Example:
        >>> client = get_model_client_for_agentic_coding(
        ...     model_path="gpt-4o",
        ...     backend="api",
        ...     provider="openai"
        ... )
        >>> response = client.chat(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     tools=[...tool_schemas...]
        ... )
    """
    print(f"🔧 Loading model for agentic coding: {model_path}")
    print(f"   Backend: {backend}, Provider: {provider}")
    
    # Load the model using existing system
    model_instance = load_model(
        model_path=model_path,
        backend=backend,
        provider=provider,
        **load_kwargs
    )
    
    if model_instance is None:
        raise ValueError(f"Failed to load model: {model_path}")
    
    # Determine actual provider for wrapper
    # This maps backend/provider combinations to the wrapper's expected format
    wrapper_provider = _determine_wrapper_provider(backend, provider, model_path)
    
    print(f"✅ Model loaded, wrapping for tool support...")
    
    # Wrap in ModelClientWrapper
    wrapped_client = create_model_client(
        model_instance=model_instance,
        provider=wrapper_provider,
        model_path=model_path,
        backend=backend
    )
    
    print(f"✅ Model client ready for agentic coding!")
    return wrapped_client


def _determine_wrapper_provider(backend: str, provider: str, model_path: str) -> str:
    """
    Determine the provider string that ModelClientWrapper expects.
    
    This handles the mapping between model_loader's backend/provider system
    and the wrapper's provider system.
    """
    # If backend is API, use the provider directly
    if backend == "api":
        if provider:
            return provider.lower()
        
        # Try to infer from model path
        model_lower = model_path.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "qwen" in model_lower:
            return "qwen"
        elif "grok" in model_lower:
            return "xai"
        else:
            return "openai"  # Default fallback
    
    # For non-API backends, use the backend as the provider
    return backend


def create_api_model_client(provider: str, model_name: str, api_key: str = None):
    """
    Shortcut function to create an API model client directly.
    
    This is useful when you already know the provider and model name
    and don't want to go through load_model().
    
    Args:
        provider: API provider ("openai", "anthropic", "google", etc.)
        model_name: Model identifier
        api_key: Optional API key (will use environment variable if not provided)
        
    Returns:
        ModelClientWrapper instance
        
    Example:
        >>> client = create_api_model_client("openai", "gpt-4o")
        >>> response = client.chat(messages=[...], tools=[...])
    """
    provider = provider.lower()
    
    # Get API key from environment if not provided
    if not api_key:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"API key not provided and {env_var} not set")
    
    # Create the appropriate client based on provider
    if provider == "openai":
        from openai import OpenAI
        model_instance = OpenAI(api_key=api_key)
    
    elif provider == "anthropic":
        import anthropic
        model_instance = anthropic.Anthropic(api_key=api_key)
    
    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model_name)
    
    elif provider in ["deepseek", "qwen", "xai"]:
        # These use OpenAI-compatible APIs
        from openai import OpenAI
        
        base_urls = {
            "deepseek": "https://api.deepseek.com/v1",
            "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "xai": "https://api.x.ai/v1"
        }
        
        model_instance = OpenAI(
            api_key=api_key,
            base_url=base_urls[provider]
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Wrap in ModelClientWrapper
    return create_model_client(
        model_instance=model_instance,
        provider=provider,
        model_path=model_name,
        backend="api"
    )


if __name__ == "__main__":
    print("Agentic Model Bridge module loaded.")
    print("Use get_model_client_for_agentic_coding() to create tool-enabled model clients.")

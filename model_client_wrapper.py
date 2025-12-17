"""
Unified Model Client Wrapper for SolaceOS
==========================================

This module provides a consistent interface for tool-enabled chat across all model types:
- API models (OpenAI, Anthropic, Gemini, DeepSeek, Qwen, xAI, etc.)
- Local models (llama.cpp, Ollama)
- Transformers models

The wrapper abstracts away provider differences and enables tool calling for the
Agentic Coding system and other tool-enabled features.
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatCompletionResponse:
    """Standardized response from any model."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"
    raw_response: Any = None  # Store original response for debugging


class ModelClientWrapper:
    """
    Unified wrapper for all model types that provides a consistent interface
    for tool-enabled chat completions.
    
    This allows the Agentic Coder and other systems to work with any model
    without knowing the underlying provider details.
    """
    
    def __init__(
        self,
        model_instance: Any,
        provider: str,
        model_path: str,
        backend: str,
        stream_function: Optional[Callable] = None
    ):
        """
        Initialize the model client wrapper.
        
        Args:
            model_instance: The loaded model object (Llama, API client, etc.)
            provider: Provider name ("openai", "anthropic", "local", etc.)
            model_path: Path or identifier for the model
            backend: Backend type ("api", "llama.cpp", "ollama", "transformers")
            stream_function: Optional streaming function from model_loader
        """
        self.model_instance = model_instance
        self.provider = provider.lower()
        self.model_path = model_path
        self.backend = backend.lower()
        self.stream_function = stream_function
        
        print(f"✅ ModelClientWrapper initialized: {provider}/{backend}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None
    ) -> ChatCompletionResponse:
        """
        Unified chat interface with optional tool calling.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool schemas (OpenAI format)
            tool_choice: Optional tool choice ("auto", "none", or specific tool)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            ChatCompletionResponse with content and/or tool calls
        """
        # Route to the appropriate provider implementation
        if self.provider in ["openai", "gpt"]:
            return self._chat_openai(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.provider == "google":
            return self._chat_google(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.provider == "deepseek":
            return self._chat_deepseek(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.provider == "qwen":
            return self._chat_qwen(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.provider in ["xai", "grok"]:
            return self._chat_xai(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.backend == "llama.cpp":
            return self._chat_llamacpp(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.backend == "ollama":
            return self._chat_ollama(messages, tools, tool_choice, temperature, max_tokens, stop)
        elif self.backend == "transformers":
            return self._chat_transformers(messages, tools, tool_choice, temperature, max_tokens, stop)
        else:
            raise ValueError(f"Unsupported provider/backend: {self.provider}/{self.backend}")
    
    def create_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Simple completion interface (no tools, no chat format).
        
        This is for backward compatibility with systems that just need text generation.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Convert to single-message chat format
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return response.content or ""
    
    # =========================================================================
    # PROVIDER-SPECIFIC IMPLEMENTATIONS
    # =========================================================================
    
    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """OpenAI API chat completion with tool calling."""
        try:
            # Build request payload
            payload = {
                "model": self.model_path,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                payload["stop"] = stop
            
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
            
            # Make API call
            # self.model_instance should be an OpenAI client
            response = self.model_instance.chat.completions.create(**payload)
            
            # Parse response
            message = response.choices[0].message
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
            
            return ChatCompletionResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ OpenAI chat error: {e}")
            raise
    
    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """Anthropic API chat completion with tool calling."""
        try:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = None
            if tools:
                anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            # Separate system message from conversation
            system_message = None
            conversation_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    conversation_messages.append(msg)
            
            # Build request
            request_kwargs = {
                "model": self.model_path,
                "messages": conversation_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if system_message:
                request_kwargs["system"] = system_message
            
            if anthropic_tools:
                request_kwargs["tools"] = anthropic_tools
            
            if stop:
                request_kwargs["stop_sequences"] = stop
            
            # Make API call
            # self.model_instance should be an Anthropic client
            response = self.model_instance.messages.create(**request_kwargs)
            
            # Parse response
            content = None
            tool_calls = None
            
            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    ))
            
            return ChatCompletionResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason,
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ Anthropic chat error: {e}")
            raise
    
    def _chat_google(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """Google Gemini API chat completion with tool calling."""
        try:
            # Google uses a different format - convert messages
            google_messages = self._convert_messages_to_google(messages)
            
            # Convert tools to Google format if provided
            google_tools = None
            if tools:
                google_tools = self._convert_tools_to_google(tools)
            
            # Build generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
            
            if stop:
                generation_config["stop_sequences"] = stop
            
            # Make API call
            # self.model_instance should be a Gemini model
            if google_tools:
                response = self.model_instance.generate_content(
                    google_messages,
                    generation_config=generation_config,
                    tools=google_tools
                )
            else:
                response = self.model_instance.generate_content(
                    google_messages,
                    generation_config=generation_config
                )
            
            # Parse response
            content = None
            tool_calls = None
            
            if hasattr(response, 'text'):
                content = response.text
            
            # Check for function calls
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            if tool_calls is None:
                                tool_calls = []
                            
                            # Google doesn't provide IDs for function calls
                            import uuid
                            tool_calls.append(ToolCall(
                                id=str(uuid.uuid4())[:8],
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args)
                            ))
            
            return ChatCompletionResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason="stop",
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ Google chat error: {e}")
            raise
    
    def _chat_deepseek(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """DeepSeek API chat completion (OpenAI-compatible)."""
        # DeepSeek uses OpenAI-compatible API, so we can reuse that logic
        return self._chat_openai_compatible(
            messages, tools, tool_choice, temperature, max_tokens, stop
        )
    
    def _chat_qwen(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """Qwen API chat completion (OpenAI-compatible)."""
        # Qwen uses OpenAI-compatible API
        return self._chat_openai_compatible(
            messages, tools, tool_choice, temperature, max_tokens, stop
        )
    
    def _chat_xai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """xAI Grok API chat completion (OpenAI-compatible)."""
        # xAI/Grok uses OpenAI-compatible API
        return self._chat_openai_compatible(
            messages, tools, tool_choice, temperature, max_tokens, stop
        )
    
    def _chat_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """
        Generic handler for OpenAI-compatible APIs (DeepSeek, Qwen, xAI, etc.).
        """
        try:
            # Build request
            payload = {
                "model": self.model_path,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                payload["stop"] = stop
            
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
            
            # Make API call using the instance's chat.completions.create
            response = self.model_instance.chat.completions.create(**payload)
            
            # Parse response (same as OpenAI)
            message = response.choices[0].message
            
            tool_calls = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
            
            return ChatCompletionResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ OpenAI-compatible API chat error ({self.provider}): {e}")
            raise
    
    def _chat_llamacpp(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """llama.cpp chat completion with tool calling."""
        try:
            # llama.cpp supports tool calling via create_chat_completion
            kwargs = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                kwargs["stop"] = stop
            
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            
            # Call llama.cpp
            response = self.model_instance.create_chat_completion(**kwargs)
            
            # Parse response
            message = response['choices'][0]['message']
            
            tool_calls = None
            if 'tool_calls' in message and message['tool_calls']:
                tool_calls = []
                for tc in message['tool_calls']:
                    tool_calls.append(ToolCall(
                        id=tc['id'],
                        name=tc['function']['name'],
                        arguments=json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments']
                    ))
            
            return ChatCompletionResponse(
                content=message.get('content'),
                tool_calls=tool_calls,
                finish_reason=response['choices'][0].get('finish_reason', 'stop'),
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ llama.cpp chat error: {e}")
            # Fallback to non-tool mode if tool calling not supported
            if tools:
                print("⚠️ Tool calling failed, falling back to regular completion")
                return self._chat_llamacpp(messages, None, None, temperature, max_tokens, stop)
            raise
    
    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """Ollama chat completion with tool calling."""
        try:
            # Ollama supports tool calling in newer versions
            options = {
                "temperature": temperature,
                "num_predict": max_tokens
            }
            
            if stop:
                options["stop"] = stop
            
            kwargs = {
                "model": self.model_path,
                "messages": messages,
                "options": options
            }
            
            if tools:
                kwargs["tools"] = tools
            
            # Call Ollama
            response = self.model_instance.chat(**kwargs)
            
            # Parse response
            content = response.get('message', {}).get('content')
            
            tool_calls = None
            if 'message' in response and 'tool_calls' in response['message']:
                tool_calls = []
                for tc in response['message']['tool_calls']:
                    tool_calls.append(ToolCall(
                        id=tc.get('id', 'ollama_tool_call'),
                        name=tc['function']['name'],
                        arguments=tc['function']['arguments']
                    ))
            
            return ChatCompletionResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason="stop",
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ Ollama chat error: {e}")
            # Fallback to non-tool mode
            if tools:
                print("⚠️ Tool calling failed, falling back to regular completion")
                return self._chat_ollama(messages, None, None, temperature, max_tokens, stop)
            raise
    
    def _chat_transformers(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> ChatCompletionResponse:
        """Transformers (local) chat completion."""
        # Transformers models typically don't support native tool calling
        # We'll need to do tool calling via prompt engineering
        
        if tools:
            print("⚠️ Transformers backend: Tool calling via prompt engineering not yet implemented")
            print("⚠️ Proceeding without tools")
        
        try:
            # Build prompt from messages
            # This is a simplified version - real implementation would use proper chat template
            prompt = self._messages_to_prompt(messages)
            
            # Generate
            # self.model_instance should be (model, tokenizer) tuple
            model, tokenizer = self.model_instance
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return ChatCompletionResponse(
                content=generated_text,
                tool_calls=None,
                finish_reason="stop",
                raw_response=outputs
            )
            
        except Exception as e:
            print(f"❌ Transformers chat error: {e}")
            raise
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _convert_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get('type') == 'function':
                func = tool['function']
                anthropic_tools.append({
                    "name": func['name'],
                    "description": func.get('description', ''),
                    "input_schema": func.get('parameters', {})
                })
        
        return anthropic_tools
    
    def _convert_tools_to_google(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Google format."""
        import google.generativeai as genai
        
        google_tools = []
        
        for tool in tools:
            if tool.get('type') == 'function':
                func = tool['function']
                
                # Google uses FunctionDeclaration
                function_declaration = genai.protos.FunctionDeclaration(
                    name=func['name'],
                    description=func.get('description', ''),
                    parameters=func.get('parameters', {})
                )
                
                google_tools.append(genai.protos.Tool(
                    function_declarations=[function_declaration]
                ))
        
        return google_tools
    
    def _convert_messages_to_google(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Google format."""
        google_messages = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # Google uses "user" and "model" roles
            if role == 'system':
                # Prepend system message to first user message
                if google_messages and google_messages[-1]['role'] == 'user':
                    google_messages[-1]['parts'][0] = f"{content}\n\n{google_messages[-1]['parts'][0]}"
                else:
                    google_messages.append({
                        'role': 'user',
                        'parts': [content]
                    })
            elif role == 'user':
                google_messages.append({
                    'role': 'user',
                    'parts': [content]
                })
            elif role == 'assistant':
                google_messages.append({
                    'role': 'model',
                    'parts': [content]
                })
        
        return google_messages
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a simple prompt for models without chat support."""
        parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                parts.append(f"System: {content}")
            elif role == 'user':
                parts.append(f"User: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def create_model_client(
    model_instance: Any,
    provider: str,
    model_path: str,
    backend: str,
    stream_function: Optional[Callable] = None
) -> ModelClientWrapper:
    """
    Factory function to create a ModelClientWrapper.
    
    Args:
        model_instance: The loaded model object
        provider: Provider name
        model_path: Model identifier
        backend: Backend type
        stream_function: Optional streaming function
        
    Returns:
        ModelClientWrapper instance
    """
    return ModelClientWrapper(
        model_instance=model_instance,
        provider=provider,
        model_path=model_path,
        backend=backend,
        stream_function=stream_function
    )


if __name__ == "__main__":
    print("Model Client Wrapper module loaded.")
    print("This provides unified tool-enabled chat across all model types.")

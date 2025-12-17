"""
Model Streaming Functions with ReAct Tool Calling Support

This module contains all streaming functions for different model backends,
plus a universal ReAct wrapper that adds tool calling capabilities.
"""

import os
import json
import re
import gc
import time
from datetime import datetime
from threading import Thread
from typing import Iterator, Dict, Any, Callable, List
import xml.etree.ElementTree as ET

# Import dependencies
import openai
from openai import OpenAI as OpenAIClient
import anthropic
import google.generativeai as genai
import ollama
import torch
from transformers import TextIteratorStreamer
from PIL import Image
import io
import base64

# Import from other modules
from upgraded_memory_manager import get_context_for_model
from tools import dispatch_tool  # For executing tool calls


# ============================================================================
# REACT TOOL CALLING WRAPPER
# ============================================================================

def parse_tool_call_xml(xml_string: str) -> Dict[str, Any]:
    """
    Parse <tool_call>...</tool_call> XML to extract tool name and arguments.
    
    Expected format:
    <tool_call>
    {"name": "web_search", "arguments": {"query": "Paris weather"}}
    </tool_call>
    
    Or:
    <tool_call>
    <name>web_search</name>
    <arguments>{"query": "Paris weather"}</arguments>
    </tool_call>
    """
    try:
        # Try JSON format first (most common)
        json_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', xml_string, re.DOTALL)
        if json_match:
            tool_data = json.loads(json_match.group(1))
            return {
                'name': tool_data.get('name'),
                'arguments': tool_data.get('arguments', {})
            }
        
        # Try XML format
        root = ET.fromstring(xml_string)
        name = root.find('name')
        arguments = root.find('arguments')
        
        if name is not None:
            args = {}
            if arguments is not None:
                # Arguments might be JSON string or nested XML
                try:
                    args = json.loads(arguments.text)
                except:
                    # Parse as nested XML
                    for child in arguments:
                        args[child.tag] = child.text
            
            return {
                'name': name.text,
                'arguments': args
            }
    except Exception as e:
        print(f"⚠️ Failed to parse tool call XML: {e}")
        print(f"   XML: {xml_string}")
    
    return None


def execute_tool_call(tool_call_xml: str) -> str:
    """Execute a tool call from XML and return the result."""
    tool_data = parse_tool_call_xml(tool_call_xml)
    
    if not tool_data:
        return "Error: Failed to parse tool call"
    
    tool_name = tool_data['name']
    tool_args = tool_data['arguments']
    
    print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
    
    try:
        result = dispatch_tool(tool_name, tool_args)
        print(f"✅ Tool result: {str(result)[:200]}...")
        return str(result)
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


def stream_with_react_tool_calling(
    streamer_func: Callable,
    *args,
    max_tool_iterations: int = 5,
    **kwargs
) -> Iterator[Dict[str, Any]]:
    """
    Universal ReAct wrapper for any streaming function.
    
    Monitors the stream for <tool_call> tags, executes tools,
    and feeds results back to the model for continued reasoning.
    
    Args:
        streamer_func: The underlying streaming function to wrap
        *args: Positional arguments for the streamer
        max_tool_iterations: Maximum number of tool calling loops (default 5)
        **kwargs: Keyword arguments for the streamer
    
    Yields:
        Dict with 'type' and 'token' keys
    """
    conversation_history = kwargs.get('conversation_history', [])
    
    for iteration in range(max_tool_iterations):
        buffer = ""
        in_tool_call = False
        tool_call_buffer = ""
        tool_call_found = False
        
        print(f"🔄 ReAct iteration {iteration + 1}/{max_tool_iterations}")
        
        # Stream from the underlying provider
        for chunk in streamer_func(*args, **kwargs):
            chunk_type = chunk.get('type', 'reply')
            token = chunk.get('token', '')
            
            # Pass through errors immediately
            if chunk_type == 'error':
                yield chunk
                continue
            
            # Accumulate tokens in buffer to detect tool calls
            buffer += token
            
            # Check for <tool_call> tag
            if '<tool_call>' in buffer and not in_tool_call:
                in_tool_call = True
                
                # Emit everything before <tool_call>
                parts = buffer.split('<tool_call>', 1)
                before_tool = parts[0]
                if before_tool.strip():
                    yield {'type': 'reply', 'token': before_tool}
                
                # Start accumulating tool call
                tool_call_buffer = '<tool_call>'
                buffer = parts[1] if len(parts) > 1 else ''
                continue
            
            if in_tool_call:
                # Accumulate tool call content
                tool_call_buffer += token
                
                # Check for closing tag
                if '</tool_call>' in tool_call_buffer:
                    tool_call_found = True
                    
                    # Extract complete tool call
                    parts = tool_call_buffer.split('</tool_call>', 1)
                    complete_tool_call = parts[0] + '</tool_call>'
                    remaining = parts[1] if len(parts) > 1 else ''
                    
                    # Emit the tool call to user (so they see what's happening)
                    yield {'type': 'tool_call_start', 'token': complete_tool_call}
                    
                    # Execute the tool
                    tool_result = execute_tool_call(complete_tool_call)
                    
                    # Emit tool result
                    yield {'type': 'tool_result', 'token': f"\n[Tool result: {tool_result[:100]}...]\n"}
                    
                    # Add to conversation history
                    conversation_history.append({
                        'role': 'assistant',
                        'content': complete_tool_call
                    })
                    conversation_history.append({
                        'role': 'user',
                        'content': f"<tool_result>{tool_result}</tool_result>"
                    })
                    
                    # Update kwargs for next iteration
                    kwargs['conversation_history'] = conversation_history
                    
                    # Reset buffer with remaining content
                    buffer = remaining
                    in_tool_call = False
                    tool_call_buffer = ""
                    
                    # Break to call model again with tool result
                    break
            else:
                # Normal streaming - emit token
                yield {'type': 'reply', 'token': token}
        
        # If no tool call found in this iteration, we're done
        if not tool_call_found:
            print("✅ No more tool calls, streaming complete")
            break
    
    # Max iterations reached
    if iteration >= max_tool_iterations - 1 and tool_call_found:
        yield {'type': 'error', 'token': '\n[Max tool calling iterations reached]\n'}


# ============================================================================
# STREAMING FUNCTIONS (All backends)
# ============================================================================

def stream_openai_compatible(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                           base_url, api_key_env, extra_headers=None, image_data=None, timezone='UTC',
                           model_states=None):
    """Universal streamer for OpenAI-compatible APIs (DeepSeek, XAI, Qwen, etc)"""
    try:
        from model_loader import _coerce_image_data, manage_context_window
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.") if model_states else "You are a helpful AI assistant."
        model_name = model_instance or model_id_str

        if not model_name:
            yield {'type': 'error', 'token': f"[STREAM ERROR ({api_key_env}): missing model name]"}
            return

        now = datetime.now()
        
        # Normalize history
        normalized_history = []
        try:
            for entry in (conversation_history or []):
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    role = entry.get('role')
                    content = entry.get('content', '')
                else:
                    role = 'user' if (str(entry.get('type', '')).lower() == 'user') else 'assistant'
                    content = entry.get('message', '')
                if role and (content or entry.get('imageB64')):
                    normalized_history.append({'role': role, 'content': content})
        except Exception:
            normalized_history = []

        normalized_image = _coerce_image_data(image_data)
        if normalized_image and normalized_history:
            for msg in reversed(normalized_history):
                if msg.get('role') == 'user':
                    if isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:{normalized_image['media_type']};base64,{normalized_image['data']}"}}
                        ]
                    break

        if user_input:
            if not normalized_history or normalized_history[-1].get('content') != user_input or normalized_history[-1].get('role') != 'user':
                normalized_history.append({"role": "user", "content": user_input})

        truncated_history = manage_context_window(normalized_history, 8192 - 2048)
        combined_system = (
            f"{system_prompt}\n\n"
            f"Relevant Memories:\n{memory_context}\n\n"
            f"[SYSTEM TIME] User's timezone: {timezone}. Current time: {now.strftime('%H:%M')}, date: {now.strftime('%m/%d/%Y')}."
        )
        messages = [
            {"role": "system", "content": combined_system},
            *truncated_history,
        ]

        api_key = os.getenv(api_key_env)
        if not api_key:
            yield {'type': 'error', 'token': f"[ERROR: {api_key_env} not set]"}
            return

        client = OpenAIClient(
            api_key=api_key, 
            base_url=base_url,
            default_headers=extra_headers if extra_headers else None
        )

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
            if chunk.choices[0].delta.content:
                yield {'type': 'reply', 'token': str(chunk.choices[0].delta.content)}

    except Exception as e:
        yield {'type': 'error', 'token': f"[STREAM ERROR ({api_key_env}): {str(e)}]"}


def stream_deepseek(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                    image_data=None, timezone='UTC', model_states=None):
    """DeepSeek API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        image_data=image_data,
        timezone=timezone,
        model_states=model_states
    )


def stream_xai(model_instance, model_id_str, user_input, conversation_history, should_stop, 
               image_data=None, model_states=None):
    """XAI/Grok API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        image_data=image_data,
        model_states=model_states
    )


def stream_qwen(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                image_data=None, model_states=None):
    """Qwen API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="QWEN_API_KEY",
        image_data=image_data,
        model_states=model_states
    )


def stream_meta(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                image_data=None, model_states=None):
    """Meta/Groq API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY", 
        image_data=image_data,
        model_states=model_states
    )


def stream_perplexity(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                      image_data=None, model_states=None):
    """Perplexity API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
        image_data=image_data,
        model_states=model_states
    )


def stream_openrouter(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                      image_data=None, model_states=None):
    """OpenRouter API streaming"""
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_headers={"HTTP-Referer": "NovaAI", "X-Title": "NovaAI"},
        image_data=image_data,
        model_states=model_states
    )


def stream_openai(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                  image_data=None, timezone='UTC', model_states=None):
    """OpenAI API streaming"""
    try:
        from model_loader import _coerce_image_data, manage_context_window
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.") if model_states else "You are a helpful AI assistant."

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        normalized_history = []
        try:
            for entry in (conversation_history or []):
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    role = entry.get('role')
                    content = entry.get('content', '')
                else:
                    role = 'user' if (str(entry.get('type', '')).lower() == 'user') else 'assistant'
                    content = entry.get('message', '')
                if role and (content or entry.get('imageB64')):
                    normalized_history.append({'role': role, 'content': content})
        except Exception:
            normalized_history = []

        normalized_image = _coerce_image_data(image_data)
        if normalized_image and normalized_history:
            for msg in reversed(normalized_history):
                if msg.get('role') == 'user':
                    if isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:{normalized_image['media_type']};base64,{normalized_image['data']}"}}
                        ]
                    break
        
        if user_input:
            if not normalized_history or normalized_history[-1].get('content') != user_input or normalized_history[-1].get('role') != 'user':
                normalized_history.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": time_message},
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Relevant Memories:\n{memory_context}"},
            *manage_context_window(normalized_history, 128000 - 4096),
        ]

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        stream = client.chat.completions.create(
            model=model_instance,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
        
            delta = chunk.choices[0].delta.content
            if delta:
                yield {'type': 'reply', 'token': str(delta)}
    except Exception as e:
        yield {'type': 'error', 'token': f"[STREAM ERROR (OpenAI): {str(e)}]"}


def stream_anthropic(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                     image_data=None, timezone='UTC', model_states=None):
    """Anthropic/Claude API streaming"""
    try:
        from model_loader import _coerce_image_data, manage_context_window
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.") if model_states else "You are a helpful AI assistant."
        model_name = model_id_str or model_instance

        if not model_name:
            yield {'type': 'error', 'token': "[STREAM ERROR (Anthropic): missing model name]"}
            return

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = []
        for msg in manage_context_window(conversation_history, 200000 - 4096):
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        
        normalized_image = _coerce_image_data(image_data)
        if normalized_image:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": normalized_image.get("media_type", "image/jpeg"),
                            "data": normalized_image["data"]
                        }
                    },
                    {
                        "type": "text",
                        "text": user_input
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_input})
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        with client.messages.stream(
            model=model_name,
            max_tokens=4096,
            system=f"{system_prompt_with_time}\nRelevant Memories:\n{memory_context}",
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                if should_stop():
                    yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                    break
                yield {'type': 'reply', 'token': str(text)}

    except Exception as e:
        yield {'type': 'error', 'token': f"[STREAM ERROR (Anthropic): {str(e)}]"}


def stream_google(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                  image_data=None, timezone='UTC', model_states=None):
    """Google/Gemini API streaming"""
    try:
        from model_loader import _coerce_image_data, manage_context_window, build_gemini_prompt
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."
        
        conversation_history.insert(0, {"role": "system", "content": time_message})
        conversation_history = manage_context_window(conversation_history, 32768 - 2048)

        normalized_image = _coerce_image_data(image_data)
        if normalized_image:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    if 'image' not in msg:
                         msg['image'] = []
                    msg['image'].append(normalized_image["data"])
                    break

        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.") if model_states else "You are a helpful AI assistant."
        messages = build_gemini_prompt(user_input, conversation_history, memory_context, normalized_image["data"] if normalized_image else None, system_prompt)

        stream = model_instance.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
            if chunk.parts and hasattr(chunk.parts[0], 'text'):
                yield {'type': 'reply', 'token': str(chunk.parts[0].text)}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- GOOGLE STREAM ERROR --- \n{error_details}\n--------------------------")
        yield {'type': 'error', 'token': f"[STREAM ERROR (Google): {repr(e)} - Check logs for details]"}


def stream_ollama(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                  image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium', 
                  model_states=None):
    """Ollama local model streaming"""
    try:
        from model_loader import manage_context_window, clean_and_normalize_history
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        model_state = model_states.get(model_id_str, {}) if model_states else {}
        system_prompt = model_state.get("system_prompt", "You are a helpful AI assistant.")
        thinking_style = model_state.get("thinking_style", "none")

        if thinking_style == 'advanced':
            thinking_instructions = {
                "low": "Provide a direct answer with minimal reasoning. Avoid verbose chains of thought.",
                "medium": "Briefly outline key reasoning steps, then give the final answer.",
                "high": "Think step by step. Provide a short summary of the reasoning, then the final answer."
            }
            if thinking_level in thinking_instructions:
                system_prompt += f"\n\n--- Reasoning Instructions ---\n{thinking_instructions[thinking_level]}"
        
        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        messages = manage_context_window(messages, 8192 - 1024)

        force_think_prepend = ('gpt-oss' in str(model_id_str).lower()) or ('gpt_oss' in str(model_id_str).lower())

        # Add image to the last user message if provided
        normalized_image = None
        if image_data:
            from model_loader import _coerce_image_data
            normalized_image = _coerce_image_data(image_data)
            
            if normalized_image:
                # Find the last user message and add image
                for msg in reversed(messages):
                    if msg['role'] == 'user':
                        # Ollama expects images as base64 data in the content
                        if isinstance(msg['content'], str):
                            msg['images'] = [normalized_image['data']]
                        break

        if debug_mode:
            print("\n--- OLLAMA PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("---------------------------\n")

        # Enable thinking for Qwen3-Thinking models
        enable_thinking = 'thinking' in model_id_str.lower() or 'qwen3' in model_id_str.lower()
        
        stream = ollama._client.chat(
            model=model_instance,
            messages=messages,
            stream=True,
            options={"think": enable_thinking} if enable_thinking else {}
        )

        thinking_started = False
        
        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            # Ollama returns thinking and content separately
            thinking = chunk['message'].get('thinking', '')
            content = chunk['message'].get('content', '')
            
            # Emit thinking first (if present)
            if thinking:
                if not thinking_started:
                    # Open thinking block on first thinking chunk
                    yield {'type': 'reply', 'token': '<think>'}
                    thinking_started = True
                yield {'type': 'thought', 'token': thinking}
            
            # Close thinking block before content (if we were thinking)
            if content and thinking_started:
                yield {'type': 'reply', 'token': '</think>'}
                thinking_started = False
            
            # Emit content
            if content:
                yield {'type': 'reply', 'token': content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (Ollama): {str(e)}]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stream_llamacpp(model_instance, model_id_str, user_input, conversation_history, should_stop,
                    image_data=None, timezone='UTC', tools=None, tool_outputs=None, debug_mode=False, 
                    thinking_level='medium', model_states=None):
    """llama.cpp local model streaming"""
    try:
        from model_loader import _coerce_image_data, manage_context_window, clean_and_normalize_history, BASE_DIR
        
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        model_state = model_states.get(model_id_str, {}) if model_states else {}
        system_prompt = model_state.get("system_prompt", "You are a helpful AI assistant.")
        thinking_style = model_state.get("thinking_style", "none")
        force_think_prepend = False

        if thinking_style in ['simple', 'advanced']:
            force_think_prepend = True

        if thinking_style == 'advanced':
            thinking_instructions = {
                "low": "Provide a direct answer with minimal reasoning. Avoid verbose chains of thought.",
                "medium": "Briefly outline key reasoning steps, then give the final answer.",
                "high": "Think step by step. Provide a short summary of the reasoning, then the final answer."
            }
            if thinking_level in thinking_instructions:
                system_prompt += f"\n\n--- Reasoning Instructions ---\n{thinking_instructions[thinking_level]}"

        config_path = os.path.join(BASE_DIR, "config.json")
        with open(config_path, "r", encoding="utf-8-sig") as f:
            all_configs = json.load(f)
        config = all_configs.get(model_id_str, {})
        
        now = datetime.now()
        time_message = f"Current Time: {now.strftime('%H:%M')} | Current Date: {now.strftime('%m/%d/%Y')} | User Timezone: {timezone}"
        full_system_prompt = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        
        messages.insert(0, {"role": "system", "content": full_system_prompt})

        if debug_mode:
            print("\n--- EXACT PROMPT SENT TO MODEL (llama.cpp) ---")
            print(json.dumps(messages, indent=2))
            print("==============================================")

        if tool_outputs:
            for tool_output in tool_outputs:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_output['output'])
                })

        total_context_tokens = config.get('context_tokens', 8192)
        max_tokens_for_history = int(total_context_tokens * 0.4)
        messages = manage_context_window(messages, max_tokens_for_history)

        normalized_image = _coerce_image_data(image_data)
        if normalized_image:
            for msg in reversed(messages):
                if msg['role'] == 'user' and isinstance(msg['content'], str):
                    msg['content'] = [
                        {"type": "text", "text": msg['content']},
                        {"type": "image_url", "image_url": {"url": f"data:{normalized_image['media_type']};base64,{normalized_image['data']}"}}
                    ]
                    break

        request_params = {
            "messages": messages,
            "max_tokens": config.get("max_tokens", 4096),
            "temperature": config.get("temperature", 0.7) or 0.7,
            "stream": True
        }
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        stream = model_instance.create_chat_completion(**request_params)
        
        if not hasattr(stream, '__iter__') or hasattr(stream, 'choices'):
            full_response = stream
            try:
                if force_think_prepend:
                    yield {'type': 'reply', 'token': '<think>'}

                # Handle both dict and object response formats
                if isinstance(full_response, dict):
                    # Dict format (newer llama-cpp-python)
                    content = full_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    tool_calls = full_response.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
                else:
                    # Object format (older llama-cpp-python)
                    try:
                        content = full_response.choices[0].message.get("content", "")
                    except AttributeError:
                        # Try accessing as dict
                        content = full_response['choices'][0]['message'].get('content', '')
                    
                    try:
                        tool_calls = full_response.choices[0].message.get("tool_calls", [])
                    except AttributeError:
                        tool_calls = full_response['choices'][0]['message'].get('tool_calls', [])

                # No Apriel cleanup — let the GGUF template define everything
                if content:
                    yield {'type': 'reply', 'token': content}

                for tool_call in tool_calls:
                    try:
                        arguments = json.loads(tool_call['function']['arguments'])
                    except json.JSONDecodeError:
                        arguments = {"error": "Failed to decode arguments", "raw": tool_call['function']['arguments']}

                    yield {
                        'type': 'tool_call',
                        'tool_call': {
                            'id': tool_call.get('id', 'tool_call_id'),
                            'name': tool_call['function']['name'],
                            'arguments': arguments
                        }
                    }

            except (AttributeError, IndexError, KeyError, TypeError) as e:
                # Debug output to help diagnose the issue
                print(f"🔴 ERROR: Response structure issue: {e}")
                print(f"🔴 Response type: {type(full_response)}")
                print(f"🔴 Response content: {str(full_response)[:500]}")
                yield {'type': 'error', 'token': f"[Chat Format Error: {str(e)}. Check console for details.]"}
            return
        
        if force_think_prepend:
            yield {'type': 'reply', 'token': '<think>'}

        active_tool_calls = {}
        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if not delta: continue

            if delta.get("content"):
                token = str(delta["content"])
                yield {'type': 'reply', 'token': token}

            if delta.get("tool_calls"):
                for tool_call_chunk in delta["tool_calls"]:
                    index = tool_call_chunk.get("index")
                    if index is None: continue

                    if index not in active_tool_calls:
                        active_tool_calls[index] = {"id": "", "function": {"name": "", "arguments": ""}, "type": "function"}
                    
                    if tool_call_chunk.get("id"):
                        active_tool_calls[index]['id'] = tool_call_chunk["id"]
                    if tool_call_chunk.get("function", {}).get("name"):
                        active_tool_calls[index]['function']['name'] = tool_call_chunk["function"]["name"]
                    if tool_call_chunk.get("function", {}).get("arguments"):
                        active_tool_calls[index]['function']['arguments'] += tool_call_chunk["function"]["arguments"]
        
        for index, tool_call in active_tool_calls.items():
            try:
                arguments = json.loads(tool_call['function']['arguments'])
            except json.JSONDecodeError:
                arguments = {"error": "Failed to decode arguments", "raw": tool_call['function']['arguments']}

            yield {
                'type': 'tool_call', 
                'tool_call': {
                    'id': tool_call['id'],
                    'name': tool_call['function']['name'],
                    'arguments': arguments
                }
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (llama.cpp): {str(e)}]"}


def stream_safetensors(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                       image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium',
                       model_states=None):
    """SafeTensors (HuggingFace transformers) local model streaming"""
    try:
        from model_loader import _coerce_image_data, manage_context_window, clean_and_normalize_history
        from qwen_omni_utils import process_mm_info
        from queue import Empty
        
        if len(model_instance) == 4:
            model, tokenizer, _, processor = model_instance
        elif len(model_instance) == 3:
            model, tokenizer, _ = model_instance
            processor = None
        else:
            raise ValueError("Invalid model_instance tuple size")

        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        model_state = model_states.get(model_id_str, {}) if model_states else {}
        system_prompt = model_state.get("system_prompt", "You are a helpful AI assistant.")
        thinking_style = model_state.get("thinking_style", "none")
        force_think_prepend = False

        if thinking_style in ['simple', 'advanced']:
            force_think_prepend = True
        
        if thinking_style == 'advanced':
            thinking_instructions = {
                "low": "Provide a direct answer with minimal reasoning. Avoid verbose chains of thought.",
                "medium": "Briefly outline key reasoning steps, then give the final answer.",
                "high": "Think step by step. Provide a short summary of the reasoning, then the final answer."
            }
            if thinking_level in thinking_instructions:
                system_prompt += f"\n\n--- Reasoning Instructions ---\n{thinking_instructions[thinking_level]}"
        
        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."

        now = datetime.now()
        time_message = f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: {timezone})"
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        model_context_window = model_state.get('context_tokens', 8192)
        messages = manage_context_window(messages, model_context_window - 1024)

        if debug_mode:
            print("\n--- SafeTensors PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("--------------------------------\n")

        normalized_image = _coerce_image_data(image_data)
        if normalized_image and processor:
            try:
                for msg in reversed(messages):
                    if msg['role'] == 'user':
                        if isinstance(msg['content'], str):
                            msg['content'] = [{"type": "text", "text": msg['content']}]
                        image_bytes = base64.b64decode(normalized_image["data"])
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        msg['content'].append({"type": "image", "image": pil_image})
                        break
                
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                _, images, _ = process_mm_info(messages, use_audio_in_video=False)
                inputs = processor(text=text_prompt, images=images, return_tensors="pt").to(model.device)

            except Exception as e:
                yield {'type': 'error', 'token': f"[ERROR processing image: {e}]"}
                return
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

        print(f"🔧 DEBUG: Setting up SafeTensors generation for {model_id_str}")
        print(f"🔧 DEBUG: Input shape: {inputs['input_ids'].shape}")

        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs.get('attention_mask'),
            "streamer": streamer,
            "max_new_tokens": 4096,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        print(f"🔧 DEBUG: Generation kwargs keys: {list(generation_kwargs.keys())}")

        print(f"🔧 DEBUG: Starting generation thread...")
        thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        print(f"🔧 DEBUG: Generation thread started, waiting for tokens...")

        if force_think_prepend:
            yield {'type': 'reply', 'token': '<think>'}

        token_count = 0
        try:
            while True:
                if should_stop():
                    print(f"🔧 DEBUG: Stop signal received after {token_count} tokens")
                    yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                    break

                try:
                    token = streamer.text_queue.get(timeout=0.05)
                except Empty:
                    if not thread.is_alive():
                        break
                    continue

                if token is None:
                    break

                token_count += 1
                if token_count == 1:
                    print(f"🔧 DEBUG: First token received!")
                if token:
                    yield {'type': 'reply', 'token': str(token)}

        except StopIteration:
            print(f"🔧 DEBUG: Streamer stopped normally after {token_count} tokens")
        except Exception as stream_error:
            print(f"⚠️ Streaming error after {token_count} tokens: {stream_error}")
            import traceback
            traceback.print_exc()
            yield {'type': 'error', 'token': f"[Streaming interrupted: {str(stream_error)}]"}

        print(f"🔧 DEBUG: Streaming complete. Total tokens: {token_count}")

        thread.join(timeout=300)
        if thread.is_alive():
            print("⚠️ Generation thread did not complete in time")
            yield {'type': 'error', 'token': "\n[Generation timeout - thread still running]"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (SafeTensors): {str(e)} ]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stream_vllm(model_instance, model_id_str, user_input, conversation_history, should_stop,
                image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium',
                model_states=None):
    """vLLM local model streaming with high-performance inference"""
    try:
        from model_loader import _coerce_image_data, manage_context_window, clean_and_normalize_history

        memory_context = get_context_for_model(user_input, model_id=model_id_str)

        model_state = model_states.get(model_id_str, {}) if model_states else {}
        system_prompt = model_state.get("system_prompt", "You are a helpful AI assistant.")
        thinking_style = model_state.get("thinking_style", "none")
        force_think_prepend = False

        if thinking_style in ['simple', 'advanced']:
            force_think_prepend = True

        if thinking_style == 'advanced':
            thinking_instructions = {
                "low": "Provide a direct answer with minimal reasoning. Avoid verbose chains of thought.",
                "medium": "Briefly outline key reasoning steps, then give the final answer.",
                "high": "Think step by step. Provide a short summary of the reasoning, then the final answer."
            }
            if thinking_level in thinking_instructions:
                system_prompt += f"\n\n--- Reasoning Instructions ---\n{thinking_instructions[thinking_level]}"

        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."

        now = datetime.now()
        time_message = f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: {timezone})"
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        model_context_window = model_state.get('context_tokens', 8192)
        messages = manage_context_window(messages, model_context_window - 1024)

        if debug_mode:
            print("\n--- vLLM PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("-------------------------\n")

        # vLLM uses the model's tokenizer to format messages
        try:
            tokenizer = model_instance.get_tokenizer()
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback to simple concatenation if chat template fails
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant: "

        # vLLM SamplingParams
        try:
            from vllm import SamplingParams
        except ImportError:
            yield {'type': 'error', 'token': "[ERROR: vLLM not installed]"}
            return

        temperature = model_state.get('temperature', 0.7) or 0.7
        max_tokens = model_state.get('max_tokens', 4096) or 4096

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            stream=True
        )

        print(f"🚀 Starting vLLM generation...")

        if force_think_prepend:
            yield {'type': 'reply', 'token': '<think>'}

        # vLLM streaming
        outputs = model_instance.generate([prompt], sampling_params, use_tqdm=False)

        # Stream the output
        for output in outputs:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            # vLLM returns RequestOutput objects
            for completion_output in output.outputs:
                text = completion_output.text

                # Stream token by token
                if text:
                    yield {'type': 'reply', 'token': text}

        print(f"✅ vLLM streaming complete")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (vLLM): {str(e)} ]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stream_llamacpp_server(model_instance, model_id_str, user_input, conversation_history, should_stop,
                           image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium',
                           model_states=None):
    """llama.cpp server streaming via OpenAI-compatible API"""
    try:
        from model_loader import _coerce_image_data, manage_context_window, clean_and_normalize_history, llamacpp_server_host
        import requests

        memory_context = get_context_for_model(user_input, model_id=model_id_str)

        model_state = model_states.get(model_id_str, {}) if model_states else {}
        system_prompt = model_state.get("system_prompt", "You are a helpful AI assistant.")
        thinking_style = model_state.get("thinking_style", "none")
        force_think_prepend = False

        if thinking_style in ['simple', 'advanced']:
            force_think_prepend = True

        if thinking_style == 'advanced':
            thinking_instructions = {
                "low": "Provide a direct answer with minimal reasoning. Avoid verbose chains of thought.",
                "medium": "Briefly outline key reasoning steps, then give the final answer.",
                "high": "Think step by step. Provide a short summary of the reasoning, then the final answer."
            }
            if thinking_level in thinking_instructions:
                system_prompt += f"\n\n--- Reasoning Instructions ---\n{thinking_instructions[thinking_level]}"

        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."

        now = datetime.now()
        time_message = f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: {timezone})"
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        model_context_window = model_state.get('context_tokens', 8192)
        messages = manage_context_window(messages, model_context_window - 1024)

        if debug_mode:
            print("\n--- llama.cpp server PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("--------------------------------------\n")

        # Handle images (llama.cpp server supports images in OpenAI format)
        normalized_image = _coerce_image_data(image_data)
        if normalized_image:
            for msg in reversed(messages):
                if msg['role'] == 'user' and isinstance(msg['content'], str):
                    msg['content'] = [
                        {"type": "text", "text": msg['content']},
                        {"type": "image_url", "image_url": {"url": f"data:{normalized_image['media_type']};base64,{normalized_image['data']}"}}
                    ]
                    break

        temperature = model_state.get('temperature', 0.7) or 0.7
        max_tokens = model_state.get('max_tokens', 4096) or 4096

        # Call llama.cpp server's OpenAI-compatible endpoint
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        print(f"🚀 Streaming from llama.cpp server at {llamacpp_server_host}")

        if force_think_prepend:
            yield {'type': 'reply', 'token': '<think>'}

        # Stream from llama.cpp server
        response = requests.post(
            f"{llamacpp_server_host}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        # Parse SSE stream
        for line in response.iter_lines():
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            if not line:
                continue

            line = line.decode('utf-8')

            # Skip comments and empty lines
            if not line.startswith('data: '):
                continue

            # Remove 'data: ' prefix
            data_str = line[6:]

            # Check for [DONE] signal
            if data_str.strip() == '[DONE]':
                break

            try:
                data = json.loads(data_str)

                # Extract token from OpenAI-compatible response
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')

                    if content:
                        yield {'type': 'reply', 'token': content}

            except json.JSONDecodeError:
                # Skip malformed JSON
                continue

        print(f"✅ llama.cpp server streaming complete")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (llama.cpp server): {str(e)} ]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
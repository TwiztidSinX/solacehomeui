import json, os, time, gc, re
import subprocess
import signal
from llama_cpp import Llama, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16
from llama_cpp.llama_chat_format import Jinja2ChatFormatter, get_chat_completion_handler
import ollama
ollama._client = ollama.Client(host='http://127.0.0.1:11434')
import google.generativeai as genai
from threading import Thread, Lock
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai
import anthropic
from groq import Groq  # For Meta Llama models
import httpx  # For general API calls
from openai import OpenAI as OpenAIClient
from upgraded_memory_manager import get_context_for_model
import requests
import base64
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("✅ vLLM loaded successfully")
except ImportError as e:
    VLLM_AVAILABLE = False
    if "vllm._C" in str(e):
        print("⚠️ vLLM installed but CUDA extensions not found.")
        print("   vLLM requires proper CUDA environment and may need to be built from source.")
        print("   For now, use 'transformers' backend instead for HuggingFace models.")
    else:
        print(f"⚠️ vLLM not installed: {e}")
        print("   Install with: pip install vllm")
except Exception as e:
    VLLM_AVAILABLE = False
    print(f"⚠️ vLLM import failed: {e}")
from model_streaming import (
    stream_with_react_tool_calling,  # ReAct wrapper
    stream_openai_compatible,
    stream_deepseek,
    stream_xai,
    stream_qwen,
    stream_meta,
    stream_perplexity,
    stream_openrouter,
    stream_openai,
    stream_anthropic,
    stream_google,
    stream_ollama,
    stream_llamacpp,
    stream_safetensors
)

def _coerce_image_data(image_data, default_media_type: str = "image/png"):
    """
    Normalize image payloads to a dict with base64 data and media_type.
    Accepts either a raw base64 string or a dict containing 'data'/'base64'.
    """
    if not image_data:
        return None
    if isinstance(image_data, dict):
        data = image_data.get("data") or image_data.get("base64") or image_data.get("image_base64")
        media_type = image_data.get("media_type") or image_data.get("mime_type") or image_data.get("type") or default_media_type
    else:
        data = image_data
        media_type = default_media_type
    if not data:
        return None
    return {"data": data, "media_type": media_type}
KNOWN_OUTER_WRAPPERS = [
    (r'^\[INST\](.*)\[/INST\]$', re.DOTALL),                # [INST] ... [/INST]
    (r'^\s*<\|im_start\|>(.*)<\|im_end\|>\s*$', re.DOTALL), # <|im_start|> ... <|im_end|>
    (r'^\s*<<SYS>>(.*)<</SYS>>\s*$', re.DOTALL),           # <<SYS>> ... <</SYS>>
    (r'^\s*<start_of_turn>(.*)<end_of_turn>\s*$', re.DOTALL), # <start_of_turn> ... <end_of_turn>
]

def strip_outer_template_wrappers(text: str) -> str:
    """If text is wrapped entirely by a known outer chat template, return inner content."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    for pattern, flags in KNOWN_OUTER_WRAPPERS:
        m = re.match(pattern, s, flags)
        if m:
            inner = m.group(1).strip()
            # avoid returning empty accidentally
            if inner:
                return inner
    return text

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Custom Chat Templates ---
LFM2_TEMPLATE = """<|startoftext|>{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}<|im_start|>assistant
"""

APRIEL_TEMPLATE = """{%- set available_tools_string, thought_instructions, add_tool_id, tool_output_format = '', '', true, "default" -%}

{%- if tools is not none and tools|length > 0 -%}
    {%- set available_tools_string -%}
You are provided with function signatures within <available_tools></available_tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about the arguments. You should infer the argument values from previous user responses and the system message. Here are the available tools:
<available_tools>
{% for tool in tools %}
{{ tool|string }}
{% endfor %}
</available_tools>
{%- endset -%}
{%- endif -%}
{%- if tool_output_format is none or tool_output_format == "default" -%}
{%- set tool_output_instructions -%}
Return all function calls as a list of json objects within <tool_call></tool_call> XML tags. Each json object should contain a function name and arguments as follows:
<tool_calls>[{"name": <function-name-1>, "arguments": <args-dict-1>}, {"name": <function-name-2>, "arguments": <args-dict-2>},...]</tool_calls>
{%- endset -%}
{%- elif tool_output_format == "yaml" -%}
{%- set tool_output_instructions -%}
Return all function calls as a list of yaml objects within <tool_call></tool_call> XML tags. Each yaml object should contain a function name and arguments as follows:
<tool_calls>
- name: <function-name-1>
  arguments: <args-dict-1>
- name: <function-name-2>
  arguments: <args-dict-2>
...
</tool_calls>
{%- endset -%}
{%- endif -%}
{%- if add_thoughts -%}
{%- set thought_instructions -%}
Prior to generating the function calls, you should generate the reasoning for why you're calling the function. Please generate these reasoning thoughts between <thinking> and </thinking> XML tags.
{%- endset -%}
{%- endif -%}
{{- bos_token -}}
{%- set reasoning_prompt='You are a thoughtful and systematic AI assistant built by ServiceNow Language Models (SLAM) lab. Before providing an answer, analyze the problem carefully and present your reasoning step by step. After explaining your thought process, provide the final solution in the following format: [BEGIN FINAL RESPONSE] ... [END FINAL RESPONSE].' -%}
{%- if messages[0]['role'] != 'system' and tools is not none and tools|length > 0 -%}
    {{- '<|system|>\n' + reasoning_prompt + available_tools_string + "\n" + tool_output_instructions + '\n<|end|>\n' -}}
{%- endif -%}
{%- if messages|selectattr('role', 'equalto', 'system')|list|length == 0 -%}
{{- '<|system|>\n' + reasoning_prompt + '\n<|end|>\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|user|>\n' }}
        {%- if message['content'] is not string %}
            {%- for chunk in message['content'] %}
                {%- if chunk['type'] == 'text' %}
                    {{- chunk['text'] }}
                {%- elif chunk['type'] == 'image' or chunk['type'] == 'image_url'%}
                    {{- '[IMG]' }}
                {%- else %}
                    {{- raise_exception('Unrecognized content type!') }}
                {%- endif %}
            {%- endfor %}
        {%- else %}
            {{- message['content'] }}
        {%- endif %}
        {{- '\n<|end|>\n' }}
    {%- elif message['role'] == 'content' -%}
        {%- if message['content'] is not string %}
            {{- '<|content|>\n' + message['content'][0]['text'] + '\n<|end|>\n' -}}
        {%- else %}
            {{- '<|content|>\n' + message['content'] + '\n<|end|>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'system' -%}
        {%- if message['content'] is not none and message['content']|length > 0 %}
            {%- if message['content'] is string %}
                {%- set system_message = message['content'] %}
            {%- else %}
                {%- set system_message = message['content'][0]['text'] %}
            {%- endif -%}
        {%- else %}
            {%- set system_message = '' %}\n        {%- endif -%}\n        {%- if tools is not none and tools|length > 0 -%}\n            {{- '<|system|>\n' + reasoning_prompt + system_message + '\n' + available_tools_string + '\n<|end|>\n' -}}\n        {%- else -%}\n            {{- '<|system|>\n' + reasoning_prompt + system_message + '\n<|end|>\n' -}}\n        {%- endif -%}\n    {%- elif message['role'] == 'assistant' -%}\n        {%- if loop.last -%}\n            {%- set add_tool_id = false -%}\n        {%- endif -%}\n        {{- '<|assistant|>\n' -}}\n        {%- if message['content'] is not none and message['content']|length > 0 -%}\n            {%- if message['content'] is not string and message['content'][0]['text'] is not none %}\n                {{- message['content'][0]['text'] }}\n            {%- else %}\n                {{- message['content'] -}}\n            {%- endif -%}\n        {%- elif message['chosen'] is not none and message['chosen']|length > 0 -%}\n            {{- message['chosen'][0] -}}\n        {%- endif -%}\n        {%- if add_thoughts and 'thought' in message and message['thought'] is not none -%}\n            {{- '<thinking>' + message['thought'] + '</thinking>' -}}\n        {%- endif -%}\n        {%- if message['tool_calls'] is not none and message['tool_calls']|length > 0 -%}\n            {{- '\n<tool_calls>[\' -}}\n            {%- for tool_call in message["tool_calls"] -%}\n                {{- '{"name": "' + tool_call['function']['name'] + '", "arguments": ' + tool_call['function']['arguments']|string -}}\n                {%- if add_tool_id == true -%}\n                    {{- ', "id": "' + tool_call['id'] + '"' -}}\n                {%- endif -%}\n                {{- '}' -}}\n                {%- if not loop.last -%}{{- ', ' -}}{%- endif -%}\n            {%- endfor -%}\n            {{- ']</tool_calls>' -}}\n        {%- endif -%}\n        {{- '\n<|end|>\n' + eos_token -}}\n    {%- elif message['role'] == 'tool' -%}\n        {%- if message['content'] is string %}\n            {%- set tool_message = message['content'] %}\n        {%- else %}\n            {%- set tool_message = message['content'][0]['text'] %}\n        {%- endif -%}\n        {{- '<|tool_result|>\n' + tool_message|string + '\n<|end|>\n' -}}\n    {%- endif -%}\n    {%- if loop.last and add_generation_prompt and message['role'] != 'assistant' -%}\n        {{- '<|assistant|>\n' -}}\n    {%- endif -%}\n{%- endfor -%}\n"""

JAMBA_TEMPLATE = """
{%- set thinking_prefix = thinking_prefix or 'Begin by thinking about the reasoning process in the mind within <think> </think> tags and then proceed to give your response.\n'

-%}
{%- if bos_token is defined and bos_token is not none %}{{- bos_token -}}{%- endif %}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages|length > 0 and messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages|length > 0 and messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {%- set prefix = '' %}
        {%- if message.role == 'user' and '<think>' not in message.content %}
            {%- if loop.last %}
                {%- set prefix = thinking_prefix %}
            {%- endif %}
            {%- if not loop.last %}
                {%- if loop.nextitem.role == 'assistant' and loop.nextitem.content.startswith('<think>') or loop.nextitem.reasoning_content is defined and loop.nextitem.reasoning_content is not none %}
                    {%- set prefix = thinking_prefix %}
                {%- endif %}
            {%- endif %}
        {%- endif %}
        {%- if message.role == 'user' and not loop.last and loop.nextitem.role == 'assistant' and loop.nextitem.content.startswith('<think>') and '<think>' not in message.content %}
            {%- set prefix = thinking_prefix %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n' + prefix + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if reasoning_content %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{\"name\": "' }}
                {{- tool_call.name }}
                {{- '", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {{- '<think>\n' }}
{%- endif -%}"""

models = {}
model_states = {}
model_load_times = {}
models_lock = Lock()
current_backend = "llama.cpp" # or "ollama"

# llama.cpp server subprocess tracking
llamacpp_server_process = None
llamacpp_server_port = 8081
llamacpp_server_host = "http://127.0.0.1:8081"

def load_model(
    model_path,
    backend="llama-cpp-python", 
    context_tokens=32678, 
    gpu_layers=0, 
    temperature=0.7, 
    system_prompt="", 
    provider=None, 
    quantization='none', 
    kv_cache_quant='fp16',
    use_flash_attention=False,
    dtype='auto',
    thinking_mode=False,
    thinking_level='medium',
    device_map='auto',
    **kwargs # Catch any other unused params
):
    global models, model_states, current_backend
    with models_lock:
        try:
            current_backend = backend
            model_id = model_path.replace("\\", "/")

            if model_id in models:
                print(f"🟢 MODEL ALREADY LOADED: {model_id}")
                return models[model_id]

            # Unload existing models
            for mid in list(models.keys()):
                print(f"🟠 Unloading model: {mid}")
                del models[mid]
                model_states[mid] = 'unloaded'
            gc.collect()

            # --- Thinking Style Classification ---
            model_name_lower = model_id.lower()
            thinking_style = 'none' # Default
            
            simple_thinking_keywords = ["magistral", "deepseek-r1", "deepseek-v3.1"]
            advanced_thinking_keywords = ["gpt-oss", "qwen2.5-omni"] # As per user, qwen3-2507 is advanced

            if any(keyword in model_name_lower for keyword in advanced_thinking_keywords) or ('qwen' in model_name_lower and '2507' in model_name_lower):
                thinking_style = 'advanced'
            elif any(keyword in model_name_lower for keyword in simple_thinking_keywords) or ('qwen' in model_name_lower and 'thinking' in model_name_lower):
                thinking_style = 'simple'

            # Store system prompt and thinking style in model state
            model_states[model_id] = {
                'status': 'active',
                'system_prompt': system_prompt or "You are a helpful AI assistant.",
                'context_tokens': context_tokens,
                'thinking_style': thinking_style
            }

            print(f"🟢 LOADING MODEL: {model_id} with backend {backend} and provider {provider}")

            if backend == "llama-cpp-python":
                model_states[model_id]['uses_chat_handler'] = False
                model_basename = os.path.basename(model_id).lower()

                # Determine chat format from model name or metadata
                chat_format = None  # None = use GGUF embedded template
                
                # Only specify chat_format if we need to override the GGUF template
                # Most modern GGUF files have correct templates embedded
                if "qwen" in model_basename and "chatml" not in model_basename:
                    chat_format = "chatml"
                elif "gemma" in model_basename:
                    chat_format = "gemma"
                elif "llama-3" in model_basename or "llama3" in model_basename:
                    chat_format = "llama-3"
                # For DeepSeek and other models, let the GGUF template handle it (chat_format=None)
                
                if chat_format:
                    print(f"🗨️ Using specified chat format: {chat_format}")
                else:
                    print("🗨️ Using GGUF-embedded chat template (auto-detect)")

                # KV cache quantization
                type_k = GGML_TYPE_F16 if kv_cache_quant == 'fp16' else GGML_TYPE_Q4_0 if kv_cache_quant == 'int4' else GGML_TYPE_Q8_0
                type_v = GGML_TYPE_F16 if kv_cache_quant == 'fp16' else GGML_TYPE_Q4_0 if kv_cache_quant == 'int4' else GGML_TYPE_Q8_0

                # Load model WITHOUT custom chat_handler - let llama-cpp-python handle it
                models[model_id] = Llama(
                    model_path=model_id,
                    n_ctx=context_tokens,
                    n_gpu_layers=gpu_layers,
                    temperature=temperature,
                    flash_attn=True,
                    verbose=True,
                    type_k=type_k,
                    type_v=type_v,
                    chat_format=chat_format  # None means use GGUF embedded template
                )
            elif backend == "ollama":
                model_name = setup_ollama_kv_cache(model_id, kv_cache_quant)
                print(f"📦 Preparing to load Ollama model with tag: {model_name}")
                models[model_id] = model_name # Store model name
            
            elif backend == "api":
                if provider == "google":
                    api_key = os.getenv("GOOGLE_API_KEY")


                    if not api_key: raise ValueError("GOOGLE_API_KEY not set.")
                    genai.configure(api_key=api_key)
                    models[model_id] = genai.GenerativeModel(model_id, system_instruction=system_prompt)
                
                elif provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key: raise ValueError("OPENAI_API_KEY not set.")
                    openai.api_key = api_key
                    models[model_id] = model_id # Store model name
                
                # For other API providers, we often just need the model name for the API call
                # The actual client initialization might happen in the stream function
                else:
                    models[model_id] = model_id

            elif backend == "transformers":
                try:
                    from transformers import AutoTokenizer, TextStreamer, AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
                    from qwen_omni_utils import process_mm_info # Ensure this is available
                    import torch
                    
                    model_dir = model_path
                    is_omni_model = 'qwen2.5-omni' in model_dir.lower()
                    is_gpt_oss = 'gpt-oss' in model_dir.lower()

                    if is_omni_model:
                        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                        print("✨ Detected Qwen 2.5 Omni model. Using specialized loader.")
                        
                        processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
                        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                            model_dir,
                            dtype="auto",
                            device_map="auto",
                            load_in_4bit=True # Assuming 4-bit for Omni as per old config
                        )
                        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                        models[model_id] = (model, processor.tokenizer, streamer, processor)
                        
                    elif is_gpt_oss:
                        from transformers import kernels # Corrected typo from kernals
                        print("🔍 DETECTED GPT-OSS MODEL (MXFP4 QUANTIZED) - USING SPECIALIZED LOADER")
                        try:
                            # CRITICAL: NO BITSANDBYTES CONFIGURATION (MXFP4 is baked-in)
                            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                            
                            # Load WITHOUT quantization config (MXFP4 is already baked in)
                            model = AutoModelForCausalLM.from_pretrained(
                                model_dir,
                                dtype="auto",  # Handles MXFP4 automatically
                                device_map="auto",
                                trust_remote_code=True,
                                # NO quantization_config here - GPT-OSS uses MXFP4, not BitsAndBytes
                            )
                            
                            # GPT-OSS specific streamer settings (from examples)
                            streamer = TextIteratorStreamer(
                                tokenizer,
                                skip_prompt=True,
                                skip_special_tokens=True,
                                max_new_tokens=256  # Matches examples
                            )
                            
                            # Store with same structure as your Qwen2.5 Omni handler
                            models[model_id] = (model, tokenizer, streamer, processor)
                            print("✅ GPT-OSS model loaded successfully with MXFP4 compatibility")
                        except Exception as e:
                            print(f"🔴 GPT-OSS load error: {str(e)}")
                            raise e

                    else:
                        # --- Fallback for other SafeTensors models ---
                        config_path = os.path.join(BASE_DIR, "config.json")
                        with open(config_path, "r") as f:
                            all_configs = json.load(f)
                        model_config = all_configs.get(model_dir, {})
                        
                        quantization = model_config.get('quantization', 'none')
                        dtype_str = model_config.get('dtype', 'float16')
                        dtype = getattr(torch, dtype_str, torch.float16)
                        load_in_4bit = (quantization == '4bit')
                        load_in_8bit = (quantization == '8bit')
                        
                        quantization_config = None
                        if load_in_4bit:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                        elif load_in_8bit:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                        print(f"🔄 Loading generic SafeTensors model from {model_dir}")
                        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_dir,
                            dtype=dtype,
                            device_map="auto",
                            quantization_config=quantization_config,
                            trust_remote_code=True
                        )
                        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        models[model_id] = (model, tokenizer, streamer, processor)

                    print(f"✅ SafeTensors model loaded successfully!")
                    return models[model_id]
                except Exception as e:
                    print(f"🔴 SafeTensors load error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None

            elif backend == "vllm":
                if not VLLM_AVAILABLE:
                    error_msg = (
                        "vLLM backend is not available. vLLM requires CUDA extensions (_C module) which are missing. "
                        "This typically means vLLM needs to be built from source with proper CUDA support. "
                        "For now, please use the 'transformers' backend instead for HuggingFace models."
                    )
                    print(f"🔴 {error_msg}")
                    raise ValueError(error_msg)

                try:
                    print(f"🚀 Loading vLLM model from {model_path}")

                    # Get vLLM-specific configs
                    tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
                    gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
                    max_tokens = kwargs.get('max_tokens', 4096)

                    # Initialize vLLM engine
                    vllm_model = LLM(
                        model=model_path,
                        tensor_parallel_size=tensor_parallel_size,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=context_tokens,
                        trust_remote_code=True
                    )

                    # Store the model and tokenizer
                    models[model_id] = vllm_model

                    print(f"✅ vLLM model loaded successfully!")
                    return models[model_id]

                except Exception as e:
                    print(f"🔴 vLLM load error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None

            elif backend == "llama.cpp":
                global llamacpp_server_process, llamacpp_server_port, llamacpp_server_host

                try:
                    # Kill any existing server process
                    if llamacpp_server_process is not None:
                        print("🟡 Stopping existing llama.cpp server...")
                        try:
                            llamacpp_server_process.terminate()
                            llamacpp_server_process.wait(timeout=5)
                        except:
                            llamacpp_server_process.kill()
                        llamacpp_server_process = None

                    print(f"🚀 Starting llama.cpp server for {model_path}")

                    # Find llama-server executable
                    server_dir = os.path.join(BASE_DIR, "llama-cpp-server", "bin")
                    if os.name == 'nt':  # Windows
                        server_exe = os.path.join(server_dir, "llama-server.exe")
                    else:  # Unix-like
                        server_exe = os.path.join(server_dir, "llama-server")

                    if not os.path.exists(server_exe):
                        raise FileNotFoundError(
                            f"llama-server not found at {server_exe}\n"
                            f"Download from: https://github.com/ggerganov/llama.cpp/releases\n"
                            f"Place the binary in: {server_dir}"
                        )

                    # Build command
                    cmd = [
                        server_exe,
                        "--model", model_path,
                        "--port", str(llamacpp_server_port),
                        "--ctx-size", str(context_tokens),
                        "--n-gpu-layers", str(gpu_layers),
                        "--threads", str(kwargs.get('threads', 8)),
                    ]

                    # Add flash attention if requested
                    if kwargs.get('use_flash_attention', False):
                        cmd.append("--flash-attn")

                    print(f"🔧 Command: {' '.join(cmd)}")

                    # Start the server process
                    llamacpp_server_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )

                    # Wait for server to be ready (check health endpoint)
                    print("⏳ Waiting for llama.cpp server to start...")
                    import requests
                    max_retries = 30
                    for i in range(max_retries):
                        try:
                            response = requests.get(f"{llamacpp_server_host}/health", timeout=1)
                            if response.status_code == 200:
                                print(f"✅ llama.cpp server ready at {llamacpp_server_host}")
                                break
                        except:
                            pass
                        time.sleep(1)
                    else:
                        raise TimeoutError("llama.cpp server failed to start within 30 seconds")

                    # Store the server info
                    models[model_id] = {
                        "type": "llamacpp-server",
                        "host": llamacpp_server_host,
                        "process": llamacpp_server_process
                    }

                    print(f"✅ llama.cpp server loaded successfully!")
                    return models[model_id]

                except Exception as e:
                    print(f"🔴 llama.cpp server load error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Cleanup on failure
                    if llamacpp_server_process is not None:
                        try:
                            llamacpp_server_process.terminate()
                        except:
                            pass
                        llamacpp_server_process = None
                    return None

            return models.get(model_id)
        except Exception as e:
            print(f"🔴 Model load error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
def unload_model(model_obj, model_path, backend="llama.cpp"):
    global models, model_states
    model_id = model_path.replace("\\", "/")
    with models_lock:
        models.pop(model_id, None)
        model_states.pop(model_id, None)
        print(f"🟡 Unloading model: {model_id} from backend {backend}")

        # Safetensors backend
        if backend == "safetensors" and model_obj and isinstance(model_obj, tuple):
            model, tokenizer, streamer, processor = model_obj
            if torch.cuda.is_available():
                print("[DEBUG] VRAM summary before unload:")
                print(torch.cuda.memory_summary(device=None, abbreviated=True))
            model.to('cpu')
            del model
            del tokenizer
            del streamer
            del processor
            del model_obj

        # Ollama (just signal)
        elif backend == "ollama":
            try:
                import requests
                payload = {"model": model_id, "prompt": "Unloading model", "keep_alive": 0}
                requests.post('http://localhost:11434/api/generate', json=payload, timeout=0.5)
            except Exception as e:
                print("ℹ️ Ollama unload issue:", e)

        # vLLM backend
        elif backend == "vllm":
            if model_obj is not None:
                try:
                    # vLLM doesn't have a specific unload method, just delete the object
                    del model_obj
                    print("🟡 vLLM model unloaded")
                except Exception as e:
                    print(f"⚠️ vLLM unload issue: {e}")

        # llama.cpp server backend
        elif backend == "llama.cpp":
            global llamacpp_server_process
            try:
                if llamacpp_server_process is not None:
                    print("🟡 Stopping llama.cpp server...")
                    try:
                        # Try graceful shutdown first
                        llamacpp_server_process.terminate()
                        llamacpp_server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        print("⚠️ Forcefully killing llama.cpp server...")
                        llamacpp_server_process.kill()
                        llamacpp_server_process.wait()
                    llamacpp_server_process = None
                    print("✅ llama.cpp server stopped")
            except Exception as e:
                print(f"⚠️ llama.cpp server unload issue: {e}")

        else:
            # For llama.cpp or other models loaded in memory
            if model_obj is not None:
                del model_obj

        # Force cleanup
        print("Running final garbage collection...")
        gc.collect()

        if torch.cuda.is_available():
            # first, move any remaining tensors or buffers off GPU if possible
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print("⚠️ empty_cache error:", e)

            # reset memory tracking stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                # newer versions may not expose it
                pass

            print("[DEBUG] VRAM summary after unload attempt:")
            print(torch.cuda.memory_summary(device=None, abbreviated=True))

        print(f"✅ Model unload process attempted for: {model_id}")
        return True

def detect_model_family(model_instance, model_path: str) -> str:
    """Detect model family from GGUF metadata or fallback to filename."""
    try:
        metadata = model_instance.metadata
        if "chat_template" in metadata:
            return metadata["chat_template"]  # some GGUFs store exact format here
    except Exception:
        pass

    name = model_path.lower()
    if "qwen" in name:
        return "qwen"
    if "deepseek" in name:
        return "deepseek"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name:
        return "mistral"
    if "llama" in name or "vicuna" in name or "alpaca" in name:
        return "llama"
    return "generic"

def build_ollama_prompt_messages(
    user_input: str,
    conversation_history: list[dict],
    memory_context: str = "None",
    system_prompt: str = "You are a helpful AI companion.",
    image_data: str = None
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
        messages.append({"role": "system", "content": f"Relevant Memories:\n{memory_context}"})

    # The conversation_history already contains the latest user message
    for msg in conversation_history:
        new_msg = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("image"):
            # The image is already a raw base64 string
            new_msg["images"] = [msg["image"]]
        messages.append(new_msg)
    
    return messages

def build_gemini_prompt(
    user_input: str,
    conversation_history: list[dict],
    memory_context: str = "None",
    image_data: str = None,
    system_prompt: str = "You are a helpful AI assistant."
) -> list[dict]:
    from PIL import Image
    import io
    import base64

    # --- New Context Injection Logic ---
    system_context_parts = [system_prompt]
    if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
        system_context_parts.append(f"Relevant Memories:\n{memory_context}")

    # Check if the first message is a web search result from the orchestrator
    if conversation_history and conversation_history[0]['role'] == 'user' and "--- Web Search Results ---" in conversation_history[0]['content']:
        web_search_context = conversation_history.pop(0)['content']
        forceful_instruction = (
            "You have been provided with the following real-time web search results to answer the user's query. "
            "You MUST use this information to form your answer and ignore any conflicting internal knowledge."
        )
        system_context_parts.insert(0, f"{forceful_instruction}\n\n{web_search_context}")
    
    full_system_prompt = "\n\n".join(system_context_parts)
    # --- End New Logic ---

    # Process history to ensure alternating roles
    processed_history = []
    for msg in conversation_history:
        # Skip empty messages
        if not msg.get("content", "").strip() and not msg.get("image"):
            continue
            
        role = "model" if msg["role"] == "assistant" else "user"
        
        if processed_history and processed_history[-1]["role"] == role:
            # Merge with previous message of the same role
            processed_history[-1]["parts"][0]["text"] += "\n" + msg.get("content", "")
            if msg.get("image"):
                 try:
                    base64_data = msg["image"].split(',')[1]
                    image_bytes = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(image_bytes))
                    processed_history[-1]["parts"].append(img)
                 except Exception as e:
                    print(f"⚠️ Could not process image for Gemini history: {e}")
        else:
            parts = [{"text": msg.get("content", "")}]
            if msg.get("image"):
                try:
                    base64_data = msg["image"].split(',')[1]
                    image_bytes = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(image_bytes))
                    parts.append(img)
                except Exception as e:
                    print(f"⚠️ Could not process image for Gemini history: {e}")
            processed_history.append({"role": role, "parts": parts})

    # Prepend the full system context to the first user message in the processed history
    if processed_history and processed_history[0]["role"] == "user":
        processed_history[0]["parts"][0]["text"] = f"{full_system_prompt}\n\n--- User Request ---\n{processed_history[0]['parts'][0]['text']}"

    return processed_history

def build_gpt_oss_prompt(
    user_input: str,
    conversation_history: list[dict],
    memory_context: str = "None",
    system_prompt: str = "You are a helpful AI assistant.",
    reasoning_level: str = "medium"  # low, medium, high
) -> str:
    """Build prompt for GPT-OSS models using harmony format"""
    if not memory_context or memory_context.strip().lower() in ["none", "null", "[]"]:
        memory_context = "No relevant memories were found."
    
    # Harmony format for GPT-OSS
    messages = [
        {"role": "system", "content": f"{system_prompt}\nReasoning: {reasoning_level}"},
        {"role": "system", "content": f"Relevant Memories:\n{memory_context}"}
    ]
    
    for msg in conversation_history:
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_input})
    
    # Apply harmony chat template
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        # Fallback if tokenizer doesn't have apply_chat_template
        harmony_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            harmony_prompt += f"<|{role}|>\n{content}<|end|>\n"
        harmony_prompt += "<|assistant|>\n"
        return harmony_prompt

def build_safetensors_prompt(
    user_input: str,
    conversation_history: list[dict],
    memory_context: str = "None",
    system_prompt: str = "You are a helpful AI assistant.",
    model_family: str = "generic"
) -> str:
    """Universal prompt builder for ALL safetensors models"""
    
    if not memory_context or memory_context.strip().lower() in ["none", "null", "[]"]:
        memory_context = "No relevant memories were found."
    
    # Model-specific templates
    # The conversation_history already contains the latest user message
    templates = {
        "qwen": (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>system\n"
            f"Relevant Memories:\n"
            f"{memory_context}<|im_end|>\n"
            f"{ ''.join([f'<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n' for msg in conversation_history])}"
            f"<|im_start|>assistant\n"
        ),
        "llama": (
            f"<<SYS>>\n"
            f"{system_prompt}\n"
            f"Relevant Memories:\n"
            f"{memory_context}\n"
            f"<</SYS>>\n\n"
            f"{ ''.join([f'{msg['role'].capitalize()}: {msg['content']}\n' for msg in conversation_history])}"
            f"Assistant:"
        ),
        "gemma": (
            f"<start_of_turn>system\n"
            f"{system_prompt}<end_of_turn>\n"
            f"<start_of_turn>system\n"
            f"Relevant Memories:\n"
            f"{memory_context}<end_of_turn>\n"
            f"{ ''.join([f'<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n' for msg in conversation_history])}"
            f"<start_of_turn>model\n"
        ),
        "generic": (
            f"System: {system_prompt}\n"
            f"System: Relevant Memories:\n"
            f"{memory_context}\n\n"
            f"{ ''.join([f'{msg['role'].capitalize()}: {msg['content']}\n' for msg in conversation_history])}"
            f"Assistant:"
        )
    }
    
    return templates.get(model_family, templates["generic"])

def manage_context_window(conversation_history: list[dict], max_tokens: int = 8192, model_tokenizer=None) -> list[dict]:
    """
    Manages the conversation history to fit within the model's context window.
    It preserves the system prompt and the most recent messages.
    """
    if not conversation_history:
        return []

    # Work on a shallow copy to avoid mutating original
    history_copy = list(conversation_history)

    def _coerce_content_to_text(content):
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

    # Separate the system prompt from the rest of the conversation
    system_prompt = {}
    if history_copy and isinstance(history_copy[0], dict) and history_copy[0].get('role') == 'system':
        system_prompt = history_copy.pop(0)

    # Estimate token count for each message (a simple heuristic)
    def estimate_tokens(message_text):
        if not message_text:
            return 0
        return len(str(message_text).split())

    current_token_count = 0
    truncated_history = []

    # Always include the most recent messages first
    for message in reversed(history_copy):
        content_text = _coerce_content_to_text(message.get('content'))
        message_token_count = estimate_tokens(content_text)
        if current_token_count + message_token_count < max_tokens:
            truncated_history.insert(0, dict(message))
            current_token_count += message_token_count
        else:
            break  # Stop adding messages once the limit is reached

    # Re-add the system prompt at the beginning
    if system_prompt:
        truncated_history.insert(0, system_prompt)

    return truncated_history

def validate_gpt_oss_model(model_dir):
    """Validate GPT-OSS model installation and configuration"""
    print(f"🔍 Validating GPT-OSS model: {model_dir}")
    
    # Check for required files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            print(f"❌ Missing required file: {file}")
            return False
    
    # Check config for GPT-OSS specific settings
    try:
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        if config.get("model_type") != "gpt_oss":
            print("⚠️ Model type not explicitly gpt_oss, but continuing...")
        
        # Check for harmony format support
        if "chat_template" not in config and "harmony" not in str(config).lower():
            print("⚠️ No harmony format detected in config - may need manual template setup")
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False
    
    print("✅ GPT-OSS model validation passed")
    return True

def clean_and_normalize_history(conversation_history: list[dict], user_input: str = None) -> list[dict]:
    """
    Converts frontend message format to a backend-compatible format, merges
    consecutive messages from the same role, and appends the latest user input.
    """
    if not conversation_history:
        conversation_history = []

    # Create a mutable copy to avoid modifying the original list
    history_copy = [msg.copy() for msg in conversation_history]

    # Append the current user input if it exists
    if user_input:
        history_copy.append({'type': 'user', 'message': user_input})

    # First, normalize the roles and filter out irrelevant messages
    normalized = []
    for msg in history_copy:
        role = ""
        if msg.get('type') == 'user':
            role = 'user'
        elif msg.get('type') == 'ai':
            role = 'assistant'
        
        content = msg.get('message', '')
        if role and (content or msg.get('imageB64')):
            entry = {'role': role, 'content': content}
            if msg.get('imageB64'):
                entry['image'] = msg.get('imageB64')
            normalized.append(entry)

    if not normalized:
        return []

    # Now, merge consecutive messages
    merged = [normalized[0]]
    for i in range(1, len(normalized)):
        if normalized[i]['role'] == merged[-1]['role']:
            # Merge content
            merged[-1]['content'] += "\n" + normalized[i]['content']
            # Merge images (if any)
            if 'image' in normalized[i]:
                if 'image' in merged[-1]:
                    if isinstance(merged[-1]['image'], list):
                        merged[-1]['image'].append(normalized[i]['image'])
                    else:
                        merged[-1]['image'] = [merged[-1]['image'], normalized[i]['image']]
                else:
                    merged[-1]['image'] = normalized[i]['image']
        else:
            merged.append(normalized[i])
            
    return merged

# A global to hold the configurations loaded from config.json
model_configs = {}

def load_configs():
    """Loads the model configurations from config.json."""
    global model_configs
    try:
        with open('config.json', 'r') as f:
            model_configs = json.load(f)
    except FileNotFoundError:
        print("config.json not found. Using default settings.")
        model_configs = {}
    except json.JSONDecodeError:
        print("Error decoding config.json. Using default settings.")
        model_configs = {}

def get_system_prompt(model_path):
    """Helper to safely get the system prompt from the loaded configs."""
    if not model_configs:
        load_configs()
    return model_configs.get(model_path, {}).get('system_prompt', '')

def detect_safetensors_model_family(model_path):
    """Universal model family detection for ANY model"""
    try:
        # Check for config.json to determine model type
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            model_type = config.get("model_type", "").lower()
            architecture = config.get("architectures", [""])[0].lower() if config.get("architectures") else ""
            
            # Universal model family mapping
            if any(x in model_type for x in ["qwen", "chatml"]):
                return "qwen"
            elif any(x in model_type for x in ["llama", "llama3"]):
                return "llama"
            elif "gemma" in model_type:
                return "gemma"
            elif "mistral" in model_type:
                return "mistral"
            elif "phi" in model_type:
                return "phi"
            elif "deepseek" in model_type:
                return "deepseek"
            elif "yi" in model_type:
                return "yi"
            elif "internlm" in model_type:
                return "internlm"
        
        # Fallback: check directory name patterns
        model_name = os.path.basename(model_path).lower()
        model_families = {
            "qwen": "qwen",
            "llama": "llama", 
            "gemma": "gemma",
            "mistral": "mistral",
            "phi": "phi",
            "deepseek": "deepseek",
            "yi": "yi",
            "internlm": "internlm",
            "gpt": "openai",
            "gpt-oss": "openai"
        }
        
        for pattern, family in model_families.items():
            if pattern in model_name:
                return family
                
    except Exception as e:
        print(f"⚠️ Error detecting model family: {e}")
    
    return "generic"  # Fallback for unknown models

def _get_openai_models_from_api():
    """
    Try both new and legacy openai SDK styles for maximum compatibility:
    - New: from openai import OpenAI; client = OpenAI(); client.models.list()
    - Legacy: openai.api_key = ..., openai.Model.list()
    Returns a sorted list of model IDs or a fallback list on error.
    """
    try:
        # prefer new-style client if available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.models.list()
            models = [m.id for m in resp.data] if hasattr(resp, 'data') else []
        except Exception:
            # fallback to classic openai package interface
            import openai as _openai
            _openai.api_key = os.getenv("OPENAI_API_KEY")
            resp = _openai.Model.list()
            models = [m.id for m in resp['data']] if isinstance(resp, dict) and 'data' in resp else [m.id for m in resp.data]
        # filter and sort, return unique
        models = sorted(list(set(models)))
        return models
    except Exception as e:
        print(f"🔴 Could not fetch OpenAI models: {e}")
        # Fallback safe list
        return ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]

def _get_deepseek_models_from_api():
    """
    Fetch available DeepSeek models using the OpenAI-compatible API format.
    Returns a sorted list of model IDs, or a fallback list if the API fails.
    """
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("⚠️ DEEPSEEK_API_KEY not set.")
            return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            resp = client.models.list()
            models = [m.id for m in resp.data] if hasattr(resp, 'data') else []
        except Exception:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"}
            r = requests.get("https://api.deepseek.com/v1/models", headers=headers, timeout=10)
            if r.status_code == 200 and "data" in r.json():
                models = [m["id"] for m in r.json()["data"]]
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

        models = sorted(list(set(models)))
        return models or ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

    except Exception as e:
        print(f"🔴 Could not fetch DeepSeek models: {e}")
        # Fallback safe list (based on known public models)
        return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

def get_available_models(backend="llama-cpp-python", provider=None):
    print(f"🔍 Getting available models for backend: {backend}, provider: {provider}")

    if backend == "llama-cpp-python":
        model_dir = os.path.join(BASE_DIR, "models", "llama")
        models = []
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith(".gguf"):
                    models.append(os.path.join(model_dir, filename).replace("\\", "/"))
        print(f"📁 Found {len(models)} llama-cpp-python models")
        return models

    elif backend == "ollama":
        try:
            response = ollama._client.list()
            model_list = [m['model'] for m in response.get('models', []) if 'model' in m]
            print(f"📁 Found {len(model_list)} Ollama models")
            return model_list
        except Exception as e:
            print(f"🔴 Failed to fetch Ollama models: {e}")
            return []

    elif backend == "api":
        if provider == "google":
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key: raise Exception("API key not set")
                genai.configure(api_key=api_key)
                models = [m.name.replace('models/', '') for m in genai.list_models() 
                       if 'generateContent' in m.supported_generation_methods]
                print(f"📁 Found {len(models)} Google models")
                return models
            except Exception as e:
                print(f"🔴 Could not fetch Google models, returning defaults: {e}")
                return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        elif provider == "openai":
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key: raise Exception("API key not set")
                client = OpenAIClient(api_key=api_key)
                return sorted([m.id for m in client.models.list().data])
            except Exception as e:
                print(f"🔴 Could not fetch OpenAI models, returning defaults: {e}")
                return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

        elif provider == "deepseek":
            try:
                # Use the OpenAI-compatible DeepSeek models endpoint
                return _get_deepseek_models_from_api()
            except Exception:
                # Fallback to known model IDs
                return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
        
        elif provider == "anthropic":
            # Anthropic doesn't have a model list endpoint, so we always return known models
            return [
                # Claude 4 family (latest - November 2024)
                "claude-sonnet-4-5-20250929",      # Sonnet 4.5 - Most intelligent
                "claude-haiku-4-5-20251001",       # Haiku 4.5 - Fastest, cheapest
                "claude-opus-4-20250514",          # Opus 4.1 - Most powerful
                "claude-opus-4-20241229",          # Opus 4 - Powerful creative
                
                # Claude 3.5 family (still available)
                "claude-3-5-sonnet-20241022",      # Sonnet 3.5 v2 - Improved
                "claude-3-5-sonnet-20240620",      # Sonnet 3.5 v1 - Original
                "claude-3-5-haiku-20241022",       # Haiku 3.5 - Fast
                
                # Claude 3 family (legacy but available)
                "claude-3-opus-20240229",          # Opus 3 - Legacy
                "claude-3-sonnet-20240229",        # Sonnet 3 - Legacy
                "claude-3-haiku-20240307",         # Haiku 3 - Legacy
            ]

        elif provider == "meta":
            try:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key: raise Exception("API key not set")
                client = Groq(api_key=api_key)
                models = client.models.list()
                return sorted([model.id for model in models.data if 'llama' in model.id.lower()])
            except Exception as e:
                print(f"🔴 Could not fetch Meta/Groq models, returning defaults: {e}")
                return ["llama3-70b-8192", "llama3-8b-8192", "llama3.1-70b-versatile", "llama3.1-8b-instant"]

        # ... [Add similar try/except blocks with default fallbacks for other providers] ...
        
        elif provider == "xai":
            # xAI (Grok) - Use OpenAI-compatible endpoint
            try:
                api_key = os.getenv("XAI_API_KEY")
                if not api_key:
                    raise Exception("XAI_API_KEY not set")
                
                # xAI uses OpenAI-compatible API
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get("https://api.x.ai/v1/models", headers=headers, timeout=10)
                response.raise_for_status()
                
                models_data = response.json()
                if "error" in models_data:
                    raise Exception(models_data["error"].get("message", "Unknown xAI error"))
                model_ids = [m["id"] for m in models_data.get("data", [])]
                print(f"📁 Found {len(model_ids)} xAI models")
                return sorted(model_ids)
            except Exception as e:
                print(f"🔴 Could not fetch xAI models, returning defaults: {e}")
                return ["grok-beta", "grok-vision-beta"]

        elif provider == "qwen":
            # Qwen/Alibaba Cloud DashScope - International-Friendly Loading
            try:
                api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    raise Exception("QWEN_API_KEY or DASHSCOPE_API_KEY not set")
                
                # Try the OpenAI-compatible model list endpoint first
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(
                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
                    headers=headers,
                    timeout=10
                )
                
                # If it works, return the models from the response
                response.raise_for_status()
                models_data = response.json()
                
                # Handle both possible response formats
                if "data" in models_data and "models" in models_data["data"]:
                    # Format 1: {"data": {"models": [...]}}
                    model_ids = [m["id"] for m in models_data["data"]["models"]]
                elif "data" in models_data and isinstance(models_data["data"], list):
                    # Format 2: {"data": [...]}
                    model_ids = [m["id"] for m in models_data["data"]]
                elif "models" in models_data:
                    # Format 3: {"models": [...]}
                    model_ids = [m["id"] for m in models_data["models"]]
                else:
                    # If no models found in response, fall back to known models
                    print("📁 Qwen API returned unexpected format, using known models")
                    model_ids = [
                        "qwen-max",
                        "qwen-plus", 
                        "qwen-turbo",
                        "qwen-long",
                        "qwq-32b-preview"
                    ]
                
                print(f"📁 Found {len(model_ids)} Qwen models via API")
                return sorted(model_ids)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    print("🔴 Qwen API: 401 Unauthorized - International endpoint not enabled yet")
                    print("📁 Using known Qwen models as fallback")
                elif e.response.status_code == 404:
                    print("🔴 Qwen API: 404 - OpenAI-compatible endpoint not available")
                    print("📁 Using known Qwen models as fallback")
                else:
                    print(f"🔴 Qwen API HTTP Error: {e}")
                
                # Return known models with a note that API access is limited
                return [
                    "qwen-max",
                    "qwen-plus",
                    "qwen-turbo", 
                    "qwen-long",
                    "qwq-32b-preview"
                ]
                
            except requests.exceptions.RequestException as e:
                print(f"🔴 Qwen API Network Error (likely 401 Unauthorized): {e}")
                print("📁 Using known Qwen models as fallback - International access may be limited")
                return [
                    "qwen-max",
                    "qwen-plus",
                    "qwen-turbo",
                    "qwen-long", 
                    "qwq-32b-preview"
                ]
                
            except Exception as e:
                print(f"🔴 Unexpected error fetching Qwen models: {e}")
                print("📁 Using known Qwen models as fallback")
                return [
                    "qwen-max",
                    "qwen-plus", 
                    "qwen-turbo",
                    "qwen-long",
                    "qwq-32b-preview"
                ]

        elif provider == "perplexity":
            # Perplexity - Static list (they don't have a models endpoint)
            return [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-huge-128k-online",
                "llama-3.1-sonar-small-128k-chat",
                "llama-3.1-sonar-large-128k-chat",
                "llama-3.1-8b-instruct",
                "llama-3.1-70b-instruct"
            ]

        elif provider == "openrouter":
            # OpenRouter - They have a models endpoint
            try:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise Exception("OPENROUTER_API_KEY not set")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "SolaceHomeUI"
                }
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
                response.raise_for_status()
                
                models_data = response.json()
                model_ids = [m["id"] for m in models_data.get("data", [])]
                print(f"📁 Found {len(model_ids)} OpenRouter models")
                return sorted(model_ids)
            except Exception as e:
                print(f"🔴 Could not fetch OpenRouter models, returning defaults: {e}")
                return [
                    "anthropic/claude-3.5-sonnet",
                    "openai/gpt-4o",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.1-70b-instruct"
                ]
        
        else:
            return [] # No provider selected or unknown

    elif backend == "transformers":
        model_dir = os.path.join(BASE_DIR, "models", "safetensors")
        models = []
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # Check if this is a valid model directory
                    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(item_path))
                    has_model_files = any(f.endswith((".bin", ".pt", ".pth", ".index.json")) for f in os.listdir(item_path))
                    if has_safetensors or has_model_files:
                        models.append(item_path.replace("\\", "/"))
        print(f"📁 Found {len(models)} HuggingFace Transformers models")
        return models

    elif backend == "vllm":
        # vLLM uses the same model directory as safetensors
        # since both load HuggingFace format models
        model_dir = os.path.join(BASE_DIR, "models", "safetensors")
        models = []
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # Check if this is a valid model directory
                    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(item_path))
                    has_model_files = any(f.endswith((".bin", ".pt", ".pth", ".index.json")) for f in os.listdir(item_path))
                    has_config = os.path.exists(os.path.join(item_path, "config.json"))
                    if (has_safetensors or has_model_files) and has_config:
                        models.append(item_path.replace("\\", "/"))
        print(f"📁 Found {len(models)} vLLM models")
        return models

    elif backend == "llama.cpp":
        # llama.cpp server uses the same GGUF models as llama.cpp
        model_dir = os.path.join(BASE_DIR, "models", "llama")
        models = []
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith(".gguf"):
                    models.append(os.path.join(model_dir, filename).replace("\\", "/"))
        print(f"📁 Found {len(models)} llama.cpp server models (GGUF)")
        return models

    return []

def setup_ollama_kv_cache(model_name, kv_cache_quant="fp16"):
    """Configure KV cache quantization for Ollama"""
    # Ollama uses quantized models by default, but we can specify
    quant_suffix = ""
    if kv_cache_quant == "int4":
        quant_suffix = ":4bit"
    elif kv_cache_quant == "int8":
        quant_suffix = ":8bit"
    elif kv_cache_quant == "fp8":
        quant_suffix = ""  # FP8 not directly supported
    
    return f"{model_name}{quant_suffix}"

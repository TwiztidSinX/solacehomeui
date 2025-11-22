import json, os, time, gc, re
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

def load_model(
    model_path, 
    backend="llama.cpp", 
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

            if backend == "llama.cpp":
                model_states[model_id]['uses_chat_handler'] = True
                model_basename = os.path.basename(model_id).lower()
                chat_handler = None
                
                if "lfm2" in model_basename:
                    print("🗨️ Using custom chat format: LFM2")
                    chat_handler = Jinja2ChatFormatter(template=LFM2_TEMPLATE, bos_token="<|startoftext|>", eos_token="<|im_end|>")
                elif "apriel" in model_basename:
                    print("🗨️ Using custom chat format: Apriel")
                    chat_handler = Jinja2ChatFormatter(template=APRIEL_TEMPLATE, bos_token="<|im_start|>", eos_token="<|im_end|>")
                elif "jamba" in model_basename:
                    print("🗨️ Using custom chat format: Jamba")
                    chat_handler = Jinja2ChatFormatter(template=JAMBA_TEMPLATE, bos_token="<|startoftext|>", eos_token="<|endoftext|>")
                else:
                    chat_format_name = "llama-2"  # Default
                    if "qwen" in model_basename:
                        chat_format_name = "chatml"
                    elif "gemma" in model_basename:
                        chat_format_name = "gemma"
                    elif "llama-3" in model_basename or "llama3" in model_basename:
                        chat_format_name = "llama-3"
                    print(f"🗨️ Using built-in chat format: {chat_format_name}")
                    chat_handler = get_chat_completion_handler(chat_format_name)

                type_k = GGML_TYPE_F16 if kv_cache_quant == 'fp16' else GGML_TYPE_Q4_0 if kv_cache_quant == 'int4' else GGML_TYPE_Q8_0
                type_v = GGML_TYPE_F16 if kv_cache_quant == 'fp16' else GGML_TYPE_Q4_0 if kv_cache_quant == 'int4' else GGML_TYPE_Q8_0
                models[model_id] = Llama(
                    model_path=model_id,
                    n_ctx=context_tokens,
                    n_gpu_layers=gpu_layers,
                    temperature=temperature,
                    flash_attn=True,
                    verbose=True,
                    type_k=type_k,
                    type_v=type_v,
                    chat_handler=chat_handler
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

            elif backend == "safetensors":
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

    # Separate the system prompt from the rest of the conversation
    system_prompt = {}
    if conversation_history[0]['role'] == 'system':
        system_prompt = conversation_history.pop(0)

    # Estimate token count for each message (a simple heuristic)
    def estimate_tokens(message_text):
        return len(message_text.split())

    current_token_count = 0
    truncated_history = []

    # Always include the most recent messages first
    for message in reversed(conversation_history):
        message_token_count = estimate_tokens(message['content'])
        if current_token_count + message_token_count < max_tokens:
            truncated_history.insert(0, message)
            current_token_count += message_token_count
        else:
            break # Stop adding messages once the limit is reached

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

def stream_google(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    try:
        from datetime import datetime
        from PIL import Image
        import io
        import base64

        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."
        
        # Inject time message into the history
        conversation_history.insert(0, {"role": "system", "content": time_message})
        
        # Manage the context window
        conversation_history = manage_context_window(conversation_history, 32768 - 2048) # Gemini 1.5 Pro has 32k context

        # Add the current image to the last user message
        if image_data:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    if 'image' not in msg:
                         msg['image'] = []
                    msg['image'].append(image_data)
                    break

        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")
        messages = build_gemini_prompt(user_input, conversation_history, memory_context, image_data, system_prompt)

        stream = model_instance.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
            # The final chunk may not have any parts, causing chunk.text to fail.
            if chunk.parts and hasattr(chunk.parts[0], 'text'):
                yield {'type': 'reply', 'token': chunk.text}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- GOOGLE STREAM ERROR --- \n{error_details}\n--------------------------")
        yield {'type': 'error', 'token': f"[STREAM ERROR (Google): {repr(e)} - Check logs for details]"}

def stream_gpt(model, model_path, user_input, conversation_history, should_stop, backend, provider=None, image_data=None, timezone='UTC', tools=None, debug_mode=False, thinking_level='medium'):
    """
    Streams a response from a GPT-style model.
    Handles different backends and prompt formatting.
    """
    # --- Prompt Construction ---
    messages = []
    
    if debug_mode:
        print("\n--- RAW HISTORY FROM FRONTEND ---")
        print(json.dumps(conversation_history, indent=2))
        print("---------------------------------\n")

    system_prompt_text = get_system_prompt(model_path)
    
    # The frontend history now includes the latest user message, so we use it directly
    # We also need to normalize the roles from 'sender'/'type' to 'role'
    normalized_history = []
    for entry in conversation_history:
        role = "user" if entry.get("sender", "").lower() == "user" or entry.get("type", "").lower() == "user" else "assistant"
        content = entry.get("message", "")
        print("Point 1")
        if content:
            normalized_history.append({"role": role, "content": content})

    # --- Gemma System Prompt Fix ---
    is_gemma = 'gemma' in model_path.lower()
    if is_gemma and system_prompt_text:
        if normalized_history and normalized_history[0]['role'] == 'user':
            normalized_history[0]['content'] = f"{system_prompt_text}\n\n{normalized_history[0]['content']}"
        else:
            print("⚠️ Gemma model detected, but no initial user message to prepend system prompt to.")
    elif system_prompt_text:
        messages.insert(0, {"role": "system", "content": system_prompt_text})
    
    messages.extend(normalized_history)

    # CRITICAL FIX: Append the current user input to the messages list
    if user_input:
        messages.append({"role": "user", "content": user_input})
    print("Point 2")
    if debug_mode:
        print("\n--- FINAL PROMPT TO MODEL ---")
        print(json.dumps(messages, indent=2))
        print("-----------------------------\n")

    # --- Model Streaming ---
    try:
        if backend == "llama.cpp":
            # The chat_handler set during model load will correctly format the prompt.
            response_generator = model.create_chat_completion(
                messages=messages,
                temperature=0.7,
                stream=True,
                tools=tools,
                tool_choice="auto"
            )
            print("Point 3")
            # This loop now correctly handles both text and tool call responses from llama.cpp
            for chunk in response_generator:
                if should_stop(): break
                delta = chunk['choices'][0].get('delta', {})
                if not delta: continue

                if 'tool_calls' in delta and delta['tool_calls']:
                    # NOTE: This handles streaming tool calls. A more robust implementation
                    # might need to aggregate chunks if a single call is split.
                    full_tool_call = delta['tool_calls'][0]
                    yield {'type': 'tool_call', 'tool_call': full_tool_call}
                    continue

                token = delta.get('content')
                if token:
                    yield {'type': 'reply', 'token': token}
        
        elif backend == "ollama":
            # For Ollama, the 'model' is the model name string
            # Explicitly create a client with the correct host to override any faulty defaults
            client = ollama.Client(host='http://127.0.0.1:11434')

            # CRITICAL FIX: Ensure the system prompt is the first message for Ollama
            if not messages or messages[0].get('role') != 'system':
                messages.insert(0, {"role": "system", "content": system_prompt_text})

            stream = client.chat(
                model=model,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if should_stop(): break
                token = chunk['message']['content']
                if token:
                    yield {'type': 'reply', 'token': token}

        elif backend == "safetensors":
            # For safetensors, 'model' is a tuple (model, tokenizer, streamer, processor)
            _model, tokenizer, streamer, _ = model
            
            # Use the tokenizer's chat template for robust and accurate prompt formatting
            prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt_string, return_tensors="pt").to(_model.device)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=4096)
            
            thread = Thread(target=_model.generate, kwargs=generation_kwargs)
            thread.start()

            for token in streamer:
                if should_stop(): break
                if token:
                    yield {'type': 'reply', 'token': token}
            thread.join()
        
    except Exception as e:
        print(f"🔴 Streaming Error: {e}")
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"An error occurred: {e}"}

def stream_llamacpp(model_instance, model_id_str, user_input, conversation_history, should_stop,
                    image_data=None, timezone='UTC', tools=None, tool_outputs=None, debug_mode=False, thinking_level='medium'):
    try:
        import re
        import json
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        # --- Hybrid Thinking Logic ---
        model_state = model_states.get(model_id_str, {})
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
        # --- End Hybrid Logic ---

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
        
        if conversation_history and conversation_history[0].get("role") == "user" and "web_search" in conversation_history[0].get("content", ""):
            web_search_context = conversation_history.pop(0)["content"]
            full_system_prompt = (
                "You have been provided with the following real-time web search results..."
            )
        
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

        if image_data:
            for msg in reversed(messages):
                if msg['role'] == 'user' and isinstance(msg['content'], str):
                    msg['content'] = [{"type": "text", "text": msg['content']}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]
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
        # --- FINAL ROBUST GUARD FIX: Handle malformed response object ---
        # If 'stream' is not a generator, we process it manually.
        # Check if it lacks the generator's __iter__ method OR if it has the full object's 'choices' attribute.
        if not hasattr(stream, '__iter__') or hasattr(stream, 'choices'):
            
            full_response = stream
            
            # CRITICAL: We must guard the access to 'choices' because the object 
            # might be a malformed ChatFormatterResponse with no valid structure.
            try:
                # 1. Prepend <think> if required
                if force_think_prepend:
                    yield {'type': 'reply', 'token': '<think>'}

                # 2. Extract and yield content
                # This is the line that caused the crash, now safely inside try block.
                content = full_response.choices[0].message.get("content", "") 
                if content:
                    if "apriel" in model_id_str.lower():
                        content = re.sub(r'\s*\[/?(BEGIN|END) FINAL RESPONSE\]\s*', '', content, flags=re.IGNORECASE).strip()
                    yield {'type': 'reply', 'token': content}
                
                # 3. Extract and yield tool calls
                tool_calls = full_response.choices[0].message.get("tool_calls", [])
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
            
            except (AttributeError, IndexError) as e:
                # Catch the 'AttributeError: ... has no attribute choices' or IndexError
                # if choices is empty. This means the model template process failed.
                if debug_mode:
                    print(f"DEBUG: Caught expected non-streaming object error: {e}")
                # Yield a specific error token so the UI doesn't hang.
                yield {'type': 'error', 'token': "[Chat Format Error: Apriel template failed to generate a valid response structure.]"}

            # CRITICAL: Exit the function if it was not a generator!
            return 
        # --- END FINAL ROBUST GUARD FIX ---
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
                if "apriel" in model_id_str.lower():
                    token = re.sub(r'\[/?(BEGIN|END) FINAL RESPONSE\]', '', token, flags=re.IGNORECASE)
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

def stream_ollama(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium'):
    try:
        from datetime import datetime
        import json
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        # --- Hybrid Thinking Logic ---
        model_state = model_states.get(model_id_str, {})
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
        # Encourage consistent think-tag usage across providers
        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."
        # Per user request, we don't force-prepend <think> for Ollama.
        # --- End Hybrid Logic ---

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        messages = manage_context_window(messages, 8192 - 1024)

        # Force a think block for GPT-OSS models via Ollama so reasoning is visible
        force_think_prepend = ('gpt-oss' in str(model_id_str).lower()) or ('gpt_oss' in str(model_id_str).lower())

        if debug_mode:
            print("\n--- OLLAMA PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("---------------------------\n")

        stream = ollama._client.chat(
            model=model_instance,
            messages=messages,
            stream=True
        )

        opened_think = False
        if force_think_prepend:
            opened_think = True
            yield {'type': 'reply', 'token': '<think>'}

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            token = str(chunk['message']['content'])
            if token:
                yield {'type': 'reply', 'token': token}

        if opened_think:
            # Ensure the think block is closed if the model didn't
            yield {'type': 'reply', 'token': '</think>'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (Ollama): {str(e)}]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def stream_safetensors(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC', debug_mode=False, thinking_level='medium'):
    """Universal streaming for ALL safetensors models with hybrid thinking support."""
    try:
        from transformers import TextIteratorStreamer
        from threading import Thread
        from PIL import Image
        import io
        import base64
        from datetime import datetime
        from qwen_omni_utils import process_mm_info
        import json

        if len(model_instance) == 4:
            model, tokenizer, _, processor = model_instance  # streamer is recreated each time
        elif len(model_instance) == 3:
            model, tokenizer, _ = model_instance  # streamer is recreated each time
            processor = None
        else:
            raise ValueError("Invalid model_instance tuple size")

        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        # --- Hybrid Thinking Logic ---
        model_state = model_states.get(model_id_str, {})
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
        # Encourage consistent think-tag usage
        system_prompt += "\n\nWhen you need to reason, wrap your internal reasoning between <think> and </think> tags, then provide the final answer after the closing tag."
        # --- End Hybrid Logic ---

        now = datetime.now()
        time_message = f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: {timezone})"
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = clean_and_normalize_history(conversation_history, user_input)

        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            messages.insert(0, {"role": "system", "content": f"Relevant Memories:\n{memory_context}"})
        messages.insert(0, {"role": "system", "content": system_prompt_with_time})

        model_context_window = model_states.get(model_id_str, {}).get('context_tokens', 8192)
        messages = manage_context_window(messages, model_context_window - 1024)

        if debug_mode:
            print("\n--- SafeTensors PROMPT DEBUG ---")
            print(json.dumps(messages, indent=2))
            print("--------------------------------\n")

        if image_data and processor:
            try:
                for msg in reversed(messages):
                    if msg['role'] == 'user':
                        if isinstance(msg['content'], str):
                            msg['content'] = [{"type": "text", "text": msg['content']}]
                        image_bytes = base64.b64decode(image_data)
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

        # Create fresh streamer for this generation (cannot be reused)
        # Increased timeout to 60s for first token (4B models can be slow with long prompts)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

        print(f"🔧 DEBUG: Setting up SafeTensors generation for {model_id_str}")
        print(f"🔧 DEBUG: Input shape: {inputs['input_ids'].shape}")

        # Build generation kwargs properly without unpacking inputs directly
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

        # Remove None values
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        print(f"🔧 DEBUG: Generation kwargs keys: {list(generation_kwargs.keys())}")

        # Start generation in background thread
        print(f"🔧 DEBUG: Starting generation thread...")
        thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        print(f"🔧 DEBUG: Generation thread started, waiting for tokens...")

        if force_think_prepend:
            yield {'type': 'reply', 'token': '<think>'}

        token_count = 0
        try:
            # Non-blocking polling of streamer queue (safetensors-specific)
            from queue import Empty
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

            # Legacy blocking iteration (kept for reference)
            return
            # Stream tokens as they're generated
            for token in streamer:
                token_count += 1
                if token_count == 1:
                    print(f"🔧 DEBUG: First token received!")

                if should_stop():
                    print(f"🔧 DEBUG: Stop signal received after {token_count} tokens")
                    yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                    break

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

        # Wait for generation to complete (with timeout)
        thread.join(timeout=300)  # 5 minute max
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

def stream_google(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    try:
        from datetime import datetime
        from PIL import Image
        import io
        import base64

        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        
        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."
        
        # Inject time message into the history
        conversation_history.insert(0, {"role": "system", "content": time_message})
        
        # Manage the context window
        conversation_history = manage_context_window(conversation_history, 32768 - 2048) # Gemini 1.5 Pro has 32k context

        # Add the current image to the last user message
        if image_data:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    if 'image' not in msg:
                         msg['image'] = []
                    msg['image'].append(image_data)
                    break

        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")
        messages = build_gemini_prompt(user_input, conversation_history, memory_context, image_data, system_prompt)

        stream = model_instance.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
            # The final chunk may not have any parts, causing chunk.text to fail.
            if chunk.parts and hasattr(chunk.parts[0], 'text'):
                yield {'type': 'reply', 'token': str(chunk.parts[0].text)}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- GOOGLE STREAM ERROR --- \n{error_details}\n--------------------------")
        yield {'type': 'error', 'token': f"[STREAM ERROR (Google): {repr(e)} - Check logs for details]"}

def stream_openai(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    import openai
    import os
    try:
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        # Normalize history from frontend shape to OpenAI format
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

        # If there's image data, append to the last user message
        if image_data and normalized_history:
            for msg in reversed(normalized_history):
                if msg.get('role') == 'user':
                    if isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    break
        # Ensure the latest user input is present at the end
        if user_input:
            if not normalized_history or normalized_history[-1].get('content') != user_input or normalized_history[-1].get('role') != 'user':
                normalized_history.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": time_message},
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Relevant Memories:\n{memory_context}"},
            *manage_context_window(normalized_history, 128000 - 4096), # GPT-4o has 128k context
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

def stream_anthropic(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    try:
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        # Build messages in Anthropic format
        messages = []
        for msg in manage_context_window(conversation_history, 200000 - 4096): # Claude 3 has 200k context
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        with client.messages.stream(
            model=model_instance,
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

def stream_meta(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY", 
        image_data=image_data
    )

def stream_xai(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        image_data=image_data
    )

def stream_qwen(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="QWEN_API_KEY",
        image_data=image_data
    )

def stream_deepseek(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        image_data=image_data,
        timezone=timezone
    )

def stream_perplexity(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
        image_data=image_data
    )

def stream_openrouter(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    return stream_openai_compatible(
        model_instance, model_id_str, user_input, conversation_history, should_stop,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_headers={"HTTP-Referer": "NovaAI", "X-Title": "NovaAI"},
        image_data=image_data
    )

def stream_openai_compatible(model_instance, model_id_str, user_input, conversation_history, should_stop, 
                           base_url, api_key_env, extra_headers=None, image_data=None, timezone='UTC'):
    try:
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        # Normalize history from frontend shape (sender/message/type) to OpenAI format (role/content)
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

        # If there's an image attached to this turn, append it to the last user message
        if image_data and normalized_history:
            for msg in reversed(normalized_history):
                if msg.get('role') == 'user':
                    if isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    break

        # Ensure the latest user input is present at the end
        if user_input:
            if not normalized_history or normalized_history[-1].get('content') != user_input or normalized_history[-1].get('role') != 'user':
                normalized_history.append({"role": "user", "content": user_input})

        truncated_history = manage_context_window(normalized_history, 8192 - 2048)
        messages = [
            {"role": "system", "content": time_message},
            {"role": "system", "content": f"{system_prompt}\nRelevant Memories:\n{memory_context}"},
            *truncated_history,
        ]


        api_key = os.getenv(api_key_env)
        if not api_key:
            yield f"[ERROR: {api_key_env} not set]"
            return

        client = OpenAIClient(
            api_key=api_key, 
            base_url=base_url,
            default_headers=extra_headers if extra_headers else None
        )

        stream = client.chat.completions.create(
            model=model_instance,
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

def get_available_models(backend="llama.cpp", provider=None):
    print(f"🔍 Getting available models for backend: {backend}, provider: {provider}")
    
    if backend == "llama.cpp":
        model_dir = os.path.join(BASE_DIR, "models", "llama")
        models = []
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith(".gguf"):
                    models.append(os.path.join(model_dir, filename).replace("\\", "/"))
        print(f"📁 Found {len(models)} llama.cpp models")
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
            return ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", 
                   "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                   "claude-2.1", "claude-2.0", "claude-instant-1.2"]

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
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
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

    elif backend == "safetensors":
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
        print(f"📁 Found {len(models)} SafeTensors models")
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

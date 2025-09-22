import json, os, time, gc, re
from llama_cpp import Llama, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16
from llama_cpp.llama_chat_format import get_chat_completion_handler
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
    torch_dtype='auto',
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

            # Store system prompt in model state
            model_states[model_id] = {
                'status': 'active',
                'system_prompt': system_prompt or "You are a helpful AI assistant.",
                'context_tokens': context_tokens
            }

            print(f"🟢 LOADING MODEL: {model_id} with backend {backend} and provider {provider}")

            if backend == "llama.cpp":
                model_states[model_id]['uses_chat_handler'] = True
                model_basename = os.path.basename(model_id).lower()
                chat_format_name = "llama-2"  # Default
                if "qwen" in model_basename:
                    chat_format_name = "chatml"
                elif "gemma" in model_basename:
                    chat_format_name = "gemma"
                elif "llama-3" in model_basename or "llama3" in model_basename:
                    chat_format_name = "llama-3"

                print(f"🗨️ Using chat format: {chat_format_name}")
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
                    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
                    import torch
                    
                    model_dir = model_path
                    
                    # --- The Correct, Proven Configuration ---
                    # Derived from the working multimodal.py
                    
                    # 1. Define the compute dtype
                    compute_dtype = torch.bfloat16

                    # 2. Create the precise BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    print("✅ Applying 4-bit quantization with bfloat16 compute dtype.")

                    # 3. Load the model with the master dtype and the quantization config
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        device_map="auto",
                        torch_dtype=compute_dtype, # Master dtype switch
                        quantization_config=quantization_config, # The detailed instructions
                        trust_remote_code=True
                    )
                    
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    models[model_id] = (model, tokenizer, streamer, None) 

                    print(f"✅ SafeTensors model loaded successfully and correctly quantized!")
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
        
def unload_model(model_path, backend="llama.cpp"):
    global current_model, current_model_path_string
    model_id = model_path.replace("\\", "/")
    with models_lock:
        if model_id in models:
            print(f"🟡 Unloading model: {model_id} from backend {backend}")

            if backend == "ollama":
                try:
                    # To unload, we can send a request with keep_alive: 0
                    # The ollama library does not expose a direct unload.
                    # We will use a direct http request.
                    import requests
                    requests.post('http://localhost:11434/api/generate', json={"model": model_id, "keep_alive": 0})
                    print(f"✅ Sent unload request to Ollama for model: {model_id}")
                except Exception as e:
                    print(f"⚠️ Could not send unload request to Ollama: {e}")
            
                        # Special handling for SafeTensors models
            if backend == "safetensors" and isinstance(models[model_id], tuple):
                model_tuple = models[model_id]
                # Explicitly delete components to free memory
                for item in model_tuple:
                    del item
                del models[model_id]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                del models[model_id]

            model_states.pop(model_id, None)
            
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            current_model = None
            current_model_path_string = None
            print(f"✅ MODEL UNLOADED: {model_id}")
            return True
    return False

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

def clean_and_normalize_history(conversation_history: list[dict]) -> list[dict]:
    """
    Converts frontend message format to a backend-compatible format AND
    merges consecutive messages from the same role to ensure alternation.
    """
    if not conversation_history:
        return []
    
    # First, normalize the roles and filter out irrelevant messages
    normalized = []
    for msg in conversation_history:
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
                    # This case is ambiguous, but we'll just append for now
                    # A more robust solution might handle multiple images differently
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
        if content:
            normalized_history.append({"role": role, "content": content})

    # --- Add Thinking Instructions (Robust Method) ---
    if thinking_level and thinking_level != 'none':
        thinking_instructions = "Before you respond, you must think about the user's query and your response plan. Wrap all of your thoughts in <think>...</think> tags. The user will not see your thoughts."
        # Prepend to the first user message to ensure it's not ignored by any chat handlers
        if normalized_history and normalized_history[0]['role'] == 'user':
            normalized_history[0]['content'] = f"{thinking_instructions}\n\nUser Query: {normalized_history[0]['content']}"
        else:
             # Fallback for safety, though this case is unlikely
            system_prompt_text = f"{thinking_instructions}\n\n{system_prompt_text}"

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

        # ... (rest of the function for API backends)
        
    except Exception as e:
        print(f"🔴 Streaming Error: {e}")
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"An error occurred: {e}"}

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

def stream_llamacpp(model_instance, model_id_str, user_input, conversation_history, should_stop,
                    image_data=None, timezone='UTC', tools=None, tool_outputs=None):
    try:
        import re
        import json
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)

        # Load system prompt and config
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")
        config_path = os.path.join(BASE_DIR, "config.json")
        with open(config_path, "r", encoding="utf-8-sig") as f:
            all_configs = json.load(f)
        config = all_configs.get(model_id_str, {})

        now = datetime.now()
        time_message = f"Current Time: {now.strftime('%H:%M')} | Current Date: {now.strftime('%m/%d/%Y')} | User Timezone: {timezone}"

        full_system_prompt = f"{time_message}\n\n{system_prompt}"
        if memory_context and memory_context.strip().lower() not in ["none", "null", "[]"]:
            full_system_prompt += f"\n\nRelevant Memories:\n{memory_context}"

        # --- MODIFIED LOGIC V2 ---
        # If the history already contains messages (e.g., from the orchestrator),
        # prepend the system context to the first message to avoid role conflicts.
        if conversation_history:
            # Combine system prompt with the content of the first message
            conversation_history[0]['content'] = f"{full_system_prompt}\n\n--- User Request ---\n{conversation_history[0]['content']}"
            messages = conversation_history
        # If the history is empty, create a new system message. This is the standard
        # case for the first turn of a conversation.
        else:
            messages = [{"role": "system", "content": full_system_prompt}]
            # Since we are not extending with history, we need to add the user input here for non-history cases.
            # This part of the logic seems to be missing, let's assume the user_input is added later or should be here.
            # For now, let's stick to the prompt builder's responsibility. The calling function adds the user input.
        # --- END MODIFIED LOGIC V2 ---

        # The user_input is now expected to be the last item in conversation_history
        # Let's ensure the calling function `stream_gpt` handles this correctly.
        # The logic in `stream_gpt` normalizes history, and the user_input is passed separately.
        # The prompt builders should combine them. Let's adjust.

        # If history is not empty AND starts with a user message (from orchestrator), it contains web search results.
        if conversation_history and conversation_history[0]['role'] == 'user':
            # Extract the web search results from the first message.
            web_search_context = conversation_history.pop(0)['content']
            
            # Prepend a forceful instruction and the web context to the main system prompt.
            full_system_prompt = (
                "You have been provided with the following real-time web search results to answer the user's query. "
                "You MUST use this information to form your answer and ignore any conflicting internal knowledge.\n\n"
                f"--- Web Search Results ---\n{web_search_context}\n--- End Web Search Results ---\n\n"
                f"{full_system_prompt}"
            )
        
        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(conversation_history)

        if tool_outputs:
            for tool_output in tool_outputs:
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_output['output'])
                })

        # Stricter Context Window Management: Use only 40% of the configured context for history.
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
        if model_states.get(model_id_str, {}).get('uses_chat_handler'):
            for msg in messages:
                if isinstance(msg.get('content'), str):
                    msg['content'] = strip_outer_template_wrappers(msg['content'])
        active_tool_calls = {}
        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if not delta: continue

            if delta.get("content"):
                yield {'type': 'reply', 'token': delta["content"]}

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

def stream_ollama(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    try:
        from datetime import datetime
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")

        now = datetime.now()
        time_message = f"This is a system message. Do not refer to this message unless asked. This message is to give you a live update of the Current Date and Time. The User's Timezone is {timezone}. The Current time is {now.strftime('%H:%M')} and The Current Date is {now.strftime('%m/%d/%Y')}."

        # Prepend the time message to the system prompt for Ollama
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        messages = build_ollama_prompt_messages(user_input, conversation_history, memory_context, system_prompt_with_time, image_data)
        messages = manage_context_window(messages, 8192 - 1024)

        print("\n--- OLLAMA PROMPT DEBUG ---")
        import json
        print(json.dumps(messages, indent=2))
        print("---------------------------\n")

        stream = ollama._client.chat(
            model=model_instance,
            messages=messages,
            stream=True
        )

        # Simplified streaming logic
        for chunk in stream:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break

            token = chunk['message']['content']
            if token:
                # Yield each token immediately as a 'reply'
                yield {'type': 'reply', 'token': token}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (Ollama): {str(e)}]"}
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
                yield {'type': 'reply', 'token': chunk.text}

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

        # The conversation_history already includes the latest user message.
        # If there's image data, we need to find the last user message and append the image to it.
        if image_data:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    break
        
        messages = [
            {"role": "system", "content": time_message},
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Relevant Memories:\n{memory_context}"},
            *manage_context_window(conversation_history, 128000 - 4096), # GPT-4o has 128k context
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
                yield {'type': 'reply', 'token': delta}
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
                yield {'type': 'reply', 'token': text}

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

def stream_deepseek(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None):
    try:
        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")

        messages = [
            {"role": "system", "content": f"{system_prompt}\nRelevant Memories:\n{memory_context}"},
            *conversation_history,
        ]

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            yield f"[ERROR: DEEPSEEK_API_KEY not set]"
            return

        client = OpenAIClient(
            api_key=api_key, 
            base_url="https://api.deepseek.com/v1"
        )

        stream = client.chat.completions.create(
            model=model_instance,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if should_stop():
                yield "\n[Generation stopped by user]"
                break
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"[STREAM ERROR (DeepSeek): {str(e)} ]"

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

        # The conversation_history already includes the latest user message.
        # If there's image data, we need to find the last user message and append the image to it.
        if image_data:
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        msg['content'] = [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    break

        messages = [
            {"role": "system", "content": time_message},
            {"role": "system", "content": f"{system_prompt}\nRelevant Memories:\n{memory_context}"},
            *manage_context_window(conversation_history, 8192 - 2048), # Assume 8k context for generic compatible APIs
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
                yield {'type': 'reply', 'token': chunk.choices[0].delta.content}

    except Exception as e:
        yield {'type': 'error', 'token': f"[STREAM ERROR ({api_key_env}): {str(e)}]"}

def stream_safetensors(model_instance, model_id_str, user_input, conversation_history, should_stop, image_data=None, timezone='UTC'):
    """Universal streaming for ALL safetensors models, now with vision support."""
    try:
        from transformers import TextIteratorStreamer
        from threading import Thread
        from PIL import Image
        import io
        import base64
        from datetime import datetime
        from qwen_omni_utils import process_mm_info

        if len(model_instance) == 4:
            model, tokenizer, _, processor = model_instance
        elif len(model_instance) == 3:
            model, tokenizer, _ = model_instance
            processor = None
        else:
            raise ValueError("Invalid model_instance tuple size")

        memory_context = get_context_for_model(user_input, model_id=model_id_str)
        system_prompt = model_states.get(model_id_str, {}).get("system_prompt", "You are a helpful AI assistant.")
        now = datetime.now()
        time_message = f"System Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: {timezone})"
        system_prompt_with_time = f"{time_message}\n\n{system_prompt}"

        conversation = [{"role": "system", "content": [{"type": "text", "text": f"{system_prompt_with_time}\nRelevant Memories:\n{memory_context}"}]}]
        conversation.extend(conversation_history)
        # Get the correct context window size for this model
        model_context_window = model_states.get(model_id_str, {}).get('context_tokens', 8192)
        # Reserve 1k tokens for the response
        conversation = manage_context_window(conversation, model_context_window - 1024)

        if image_data and processor:
            try:
                current_user_message = conversation[-1]
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                current_user_message['content'] = [{"type": "text", "text": current_user_message['content']}, {"type": "image", "image": pil_image}]
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                _, images, _ = process_mm_info(conversation, use_audio_in_video=False)
                inputs = processor(text=text_prompt, images=images, return_tensors="pt").to(model.device)
            except Exception as e:
                yield {'type': 'error', 'token': f"[ERROR processing image: {e}]"}
                return
        else:
            model_family = detect_safetensors_model_family(model_id_str)
            prompt = build_safetensors_prompt(user_input, conversation_history, memory_context, system_prompt_with_time, model_family)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": 4096, "temperature": 0.7, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            if should_stop():
                yield {'type': 'reply', 'token': "\n[Generation stopped by user]"}
                break
            if token:
                yield {'type': 'reply', 'token': token}
        
        thread.join()

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'token': f"[STREAM ERROR (SafeTensors): {str(e)} ]"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
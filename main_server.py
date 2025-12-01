import asyncio
import base64
import json
import logging
import os
import re
import struct
import tempfile
import threading
import time
import webbrowser
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
import eventlet
import networkx as nx
import numpy as np
import pyaudio
import pvporcupine
import requests
import torch
import urllib.parse
import sounddevice as sd
from faster_whisper import WhisperModel
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from scipy.spatial import distance as ssd
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer
from llama_agent_integration import LlamaCppAgenticOrchestrator
from model_loader import (
    get_available_models, load_model, unload_model,
    stream_gpt, stream_llamacpp, stream_ollama, stream_safetensors,
    stream_google, stream_openai, stream_anthropic, stream_meta,
    stream_xai, stream_qwen, stream_deepseek, stream_perplexity, stream_openrouter
)
from upgraded_memory_manager import memory_manager as memory, beliefs_manager as beliefs, db
from token_utils import (
    TokenTracker,
    estimate_input_tokens,
    normalize_model_name,
    calculate_cost,
    MODEL_PRICING,
)
from orchestrator import load_orchestrator_model, get_summary_for_title, parse_command, get_tool_call, summarize_text, get_orchestrator_response, score_response_confidence, summarize_for_memory, should_perform_web_search_intelligent
from tools import dispatch_tool, TOOLS_SCHEMA
import orchestrator as orchestrator_module
from orchestrator import build_parliament_prompt  # helper we will add for merging
from parliament_voting import ParliamentVoter  # Parliament voting system
from agent_coding_socket import register_agent_coding_handlers

# Initialize Parliament voter
parliament_voter = ParliamentVoter()

is_speaking = False
# --- Database Setup for Chat History ---
try:
    chat_sessions_collection = db['chat_sessions']
    # Create indexes for faster queries
    chat_sessions_collection.create_index("user_id")
    chat_sessions_collection.create_index("timestamp")
    print("✅ Chat sessions collection configured.")
except Exception as e:
    print(f"🔴 FAILED to configure chat sessions collection: {e}")
    chat_sessions_collection = None
# --- End Database Setup ---

# --- Database Setup for Usage Events ---
try:
    usage_events_collection = db['llm_usage_events'] if db is not None else None
    if usage_events_collection is not None:
        usage_events_collection.create_index('timestamp')
        usage_events_collection.create_index([('model', 1), ('provider', 1)])
        usage_events_collection.create_index('sessionId')
        print("バ. Usage events collection configured.")
except Exception as e:
    print(f"dY\"' FAILED to configure usage events collection: {e}")
    usage_events_collection = None
# --- End Usage Events Setup ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = Path(BASE_DIR).resolve()
ORIGINAL_WORKSPACE_ROOT = WORKSPACE_ROOT

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(BASE_DIR, "nova.log"),
    format='[%(asctime)s] %(levelname)s %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

system_prompts = {
    "conversation": "You are Nova...",
    "coding": "You are NovaCoder...",
    "default": "You are Nova, a helpful AI assistant..."
}

listening_enabled = False
speak_enabled = False
def speak(text):
    global is_speaking
    is_speaking = True
    audio = kokoro_tts(text)
    sd.play(audio, samplerate=24000)
    sd.wait()  # wait for playback to finish
    time.sleep(0.2)
    with audio_q.mutex:
        audio_q.queue.clear()
    is_speaking = False

def listen():
    if is_speaking:
        return None  # <-- prevents self-interrupt
    audio = sd.rec(int(16000 * 2), samplerate=16000, channels=1)
    sd.wait()
    text = kyutai_stt(audio)
    return text
def get_voice_state():
    return listening_enabled, speak_enabled

def set_voice_state(listen, speak):
    global listening_enabled, speak_enabled
    listening_enabled = listen
    speak_enabled = speak
    print(f"🎚️ Voice State Updated — Listening: {listen} | Speaking: {speak}")


def extract_text_from_content(content):
    """
    Safely extract plain text from message content that might be a string or
    a multimodal list of parts (e.g., [{'type': 'text', ...}, {'type': 'image_url', ...}]).
    """
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(p for p in parts if p).strip()
    if isinstance(content, str):
        return content
    return ""


def merge_text_into_content(content, extra_text):
    """
    Inject extra_text into an existing content structure without losing multimodal parts.
    For list content, prepend the text block; for string content, prepend as plain text.
    """
    if not extra_text:
        return content

    if isinstance(content, list):
        updated = []
        text_merged = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and not text_merged:
                merged = dict(part)
                merged["text"] = f"{extra_text}\n\n{part.get('text', '')}".strip()
                updated.append(merged)
                text_merged = True
            else:
                updated.append(part)
        if not text_merged:
            updated.insert(0, {"type": "text", "text": extra_text})
        return updated

    base_text = extract_text_from_content(content)
    return f"{extra_text}\n\n{base_text}".strip()


def is_question_like(text: str) -> bool:
    """
    Rough heuristic to decide if a user input is an info-seeking query.
    Helps prevent unnecessary auto-search on casual statements.
    """
    if not text:
        return False
    lowered = text.lower().strip()
    if '?' in lowered:
        return True
    starters = ("who ", "what ", "where ", "when ", "why ", "how ", "which ", "can you", "could you", "would you", "should you", "tell me", "explain", "give me")
    return lowered.startswith(starters)


def build_search_query(user_text: str, max_words: int = 32) -> str:
    """
    Reduce verbose user inputs to a compact search query.
    - Prefer the first sentence; otherwise the first N words.
    """
    if not user_text:
        return ""
    first_sentence_end = user_text.find(".")
    if first_sentence_end != -1 and first_sentence_end < 240:
        candidate = user_text[: first_sentence_end + 1]
    else:
        words = user_text.split()
        candidate = " ".join(words[:max_words])
    return candidate.strip()


def extract_uncertain_span(response_text: str, max_words: int = 24) -> str:
    """
    Try to pull the most uncertain sentence/phrase from the model response
    to use as a targeted search query.
    """
    if not response_text:
        return ""
    markers = [
        "not sure", "don't know", "uncertain", "unclear",
        "might be", "could be", "probably", "maybe",
        "as far as i know", "my knowledge", "not certain"
    ]
    # Naive sentence split
    import re
    sentences = re.split(r"[\\.\\?!]", response_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    for sent in sentences:
        lower = sent.lower()
        if any(m in lower for m in markers):
            words = sent.split()
            return " ".join(words[:max_words]).strip()
    # fallback: first sentence snippet
    if sentences:
        words = sentences[0].split()
        return " ".join(words[:max_words]).strip()
    return ""

app = Flask(__name__, static_folder='static/react', static_url_path='')
# Add cache-busting configuration for development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   ping_timeout=60000, # 60 seconds
                   ping_interval=25000, # 25 seconds
                   max_http_buffer_size=100 * 1024 * 1024) # 100MB limit

# Resolve the model to use for agentic coding (Nova-style coding agent).
def get_agent_coding_model_client():
    """
    Return a model client suitable for agentic coding tool calls.
    Prefer the lightweight orchestrator model when available, otherwise
    fall back to the currently loaded main model.
    """
    try:
        if orchestrator_module.orchestrator_model is not None:
            return orchestrator_module.orchestrator_model
        return current_model
    except Exception as e:
        print(f"[AgentCoding] Failed to resolve model client: {e}")
        return None

# Register agentic coding socket handlers (Nova-style coding agent)
agent_coding_handler = register_agent_coding_handlers(
    socketio,
    model_client_factory=get_agent_coding_model_client
)

memory_graph = nx.DiGraph()
_model = None
_model_lock = threading.Lock()

# Agentic orchestrator (ReAct-style multi-step reasoning)
agentic_orchestrator = None
auto_agent_enabled = True

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    return _model

def build_graph(threshold=0.75):
    memory_graph.clear()
    all_memories = list(memory.collection.find())
    for mem in all_memories:
        memory_graph.add_node(str(mem["_id"]), **{
            "content": mem["content"],
            "timestamp": mem.get("created_at"),
            "source": mem.get("source", "unknown"),
            "emotion": mem.get("metadata", {}).get("emotion", {}).get("vader_mood", "neutral"),
            "score": mem.get("score", 0),
        })
    for i, mem1 in enumerate(all_memories):
        for j, mem2 in enumerate(all_memories):
            if i >= j: continue
            sim = 1 - ssd.cosine(mem1.get("embedding"), mem2.get("embedding")) if mem1.get("embedding") and mem2.get("embedding") else 0
            if sim >= threshold:
                memory_graph.add_edge((str(mem1["_id"])), str(mem2["_id"]), weight=sim, type="semantic_link")
    return memory_graph

def export_graph_json():
    valid_nodes = []
    node_ids = set()
    for n in memory_graph.nodes:
        try:
            node_id = str(n)
            content = memory_graph.nodes[n].get("content", "")
            emotion = memory_graph.nodes[n].get("emotion", "neutral")
            if not content: continue
            node = {
                "id": node_id,
                "label": content[:40] + "...",
                "title": content,
                "group": emotion,
            }
            valid_nodes.append(node)
            node_ids.add(node_id)
        except Exception as e:
            print(f"⚠️ Node export error: {e}")
    valid_edges = []
    for u, v in memory_graph.edges:
        try:
            from_id = str(u)
            to_id = str(v)
            if from_id in node_ids and to_id in node_ids:
                valid_edges.append({
                    "from": from_id,
                    "to": to_id,
                    "label": memory_graph[u][v].get("type", "link"),
                    "value": round(memory_graph[u][v].get("weight", 1.0), 2)
                })
        except Exception as e:
            print(f"⚠️ Edge export error: {e}")
    return {"nodes": valid_nodes, "edges": valid_edges}

def text_to_speech_stream(text: str):
    """
    Acts as a client to the dedicated voice server.
    Forwards the TTS request and streams back the audio response.
    """
    try:
        # Load the voice settings to determine which model to request
        with open('voice_settings.json', 'r') as f:
            settings = json.load(f).get('tts', {})
        
        # The 'model' field in settings should correspond to the model folder name
        model_name = settings.get('model', 'Kyutai-TTS-0.75B') 
        speaker = settings.get('voice') # Pass speaker if specified

        voice_server_url = 'http://localhost:8880/tts'
        payload = {
            "text": text,
            "model_name": model_name,
            "speaker": speaker
        }
        
        response = requests.post(voice_server_url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=4096):
            yield chunk

    except requests.exceptions.RequestException as e:
        print(f"❌ Voice server TTS error: {e}")
        yield b'' # Return empty bytes on error
    except Exception as e:
        print(f"❌ An unexpected error occurred in TTS streaming: {e}")
        yield b''

def listen_for_hotword():
    keyword_path = os.path.join(BASE_DIR, "porcupine_models", "hey_nova.ppn")
    try:
        porcupine = pvporcupine.create(access_key=os.getenv("PICOVOICE_API_KEY"), keyword_paths=[keyword_path])
    except Exception as e:
        print(f"❌ Failed to create Porcupine instance: {e}")
        raise
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
    print("Waiting for wake word 'Hey Nova'...")
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        result = porcupine.process(pcm)
        if result >= 0:
            print("Hotword detected!")
            return

def set_auto_agent_enabled(enabled: bool):
    """Enable or disable automatic agentic orchestration for chat messages."""
    global auto_agent_enabled
    auto_agent_enabled = bool(enabled)
    status = "enabled" if auto_agent_enabled else "disabled"
    print(f"[Agentic] Automatic agentic orchestration {status}.")

def init_agentic_orchestrator_if_needed():
    """
    Lazily initialize the agentic orchestrator using the existing
    small orchestrator model (Qwen) if available.
    """
    global agentic_orchestrator

    if agentic_orchestrator is not None:
        return

    try:
        if orchestrator_module.orchestrator_model is None:
            # Orchestrator model not loaded yet; skip initialization for now
            print("[Agentic] Orchestrator model not loaded; cannot initialize agentic orchestrator yet.")
            return

        agentic_orchestrator = LlamaCppAgenticOrchestrator(
            model=orchestrator_module.orchestrator_model,
            temperature=0.1,
            max_tokens=2048,
        )
        print("[Agentic] Agentic orchestrator initialized and ready.")
    except Exception as e:
        print(f"[Agentic] Failed to initialize agentic orchestrator: {e}")
        agentic_orchestrator = None

def detect_needs_agentic_approach(user_input: str) -> bool:
    """
    Decides when to use the agentic orchestrator.
    VERY conservative - requires explicit multi-step research indicators.
    """
    user_lower = user_input.lower()
    
    # === EXCLUSIONS (Never trigger agentic) ===
    
    # 1. Conversational/casual messages
    conversational_phrases = [
        "how are you", "how's it going", "what's up", "hey", "hi ", "hello",
        "thank you", "thanks for", "my day", "i'm ", "i am ",
        "just thought", "by the way", "btw", "fyi", "you there",
        "weird", "hmm", "ugh", "wow", "cool", "nice"
    ]
    if any(phrase in user_lower for phrase in conversational_phrases):
        return False
    
    # 2. Meta-questions ABOUT the system
    meta_phrases = [
        "test the", "can you test", "let's test", "to test",
        "give me a prompt", "prompt idea", "example of",
        "how do i", "how to", "what should i"
    ]
    if any(phrase in user_lower for phrase in meta_phrases):
        return False
    
    # 3. Short direct questions
    if ("are you" in user_lower or "can you" in user_lower or "do you" in user_lower):
        if len(user_input.split()) < 20:
            return False
    
    # 4. User TELLING information (not ASKING)
    telling_indicators = [
        r'\b(built|created|made|developed|wrote|coded|fixed|updated|added)\b',
        r'\b(was|were|has been|have been)\b',
        r'^(here is|here are|this is|these are|let me tell)',
        r'\d+%',  # Percentages
    ]
    import re
    is_telling = any(re.search(pattern, user_lower) for pattern in telling_indicators)
    if is_telling and '?' not in user_input:
        return False
    
    # === TRIGGERS (Must meet specific criteria) ===
    
    # Only trigger on EXPLICIT multi-step research phrases
    explicit_research_triggers = [
        "comprehensive research on",
        "in-depth research into",
        "thoroughly research",
        "research and compare",
        "deep dive into",
        "investigate thoroughly",
        "comprehensive analysis of",
        "detailed comparison of",
        "research multiple sources",
    ]
    
    if any(trigger in user_lower for trigger in explicit_research_triggers):
        print(f"🔍 Agentic trigger: Explicit research phrase detected")
        return True
    
    # Multi-part complex queries (very long with multiple questions)
    if len(user_input.split()) > 80 and user_input.count("?") > 3:
        print(f"🔍 Agentic trigger: Complex multi-part query")
        return True
    
    # Comparison + temporal + long form
    has_comparison = any(word in user_lower for word in ["compare", "versus", " vs ", "difference between"])
    has_temporal = any(word in user_lower for word in ["latest", "recent", "current", "new"])
    if has_comparison and has_temporal and len(user_input.split()) > 40:
        print(f"🔍 Agentic trigger: Comparison of recent info (complex)")
        return True
    
    return False
def run_agentic_chat_task(data, sid, forced: bool = False):
    """
    Background task that runs the agentic orchestrator and streams
    reasoning + final answer over the existing Socket.IO stream channel.
    """
    global agentic_orchestrator

    session_id = data.get("session_id")
    user_input = extract_text_from_content(data.get("text", ""))
    current_sender = data.get("aiName", "Nova")

    full_thought = ""
    final_response = ""

    try:
        # Ensure orchestrator is initialized before first use
        init_agentic_orchestrator_if_needed()

        if agentic_orchestrator is None:
            # Fallback: use normal streaming pipeline if agentic is unavailable
            stream_response(data, sid)
            return

        emit_payload = {"sender": current_sender}
        socketio.emit("stream_start", emit_payload, room=sid)
        socketio.sleep(0)

        socketio.emit('stream', {'text': '<think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
        socketio.sleep(0)

        result = agentic_orchestrator.run_agent_loop(user_query=user_input)

        reasoning_trace = result.get("reasoning_trace") or []
        for step in reasoning_trace:
            step_text = str(step)
            full_thought += step_text + "\n"
            socketio.emit("stream", step_text + "\n", room=sid)
            socketio.sleep(0)

        socketio.emit('stream', {'text': '</think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
        socketio.sleep(0)

        final_response = result.get("answer") or ""
        if final_response:
            socketio.emit("stream", final_response, room=sid)
            socketio.sleep(0)

    except Exception as e:
        error_message = f"Agentic orchestrator error: {e}"
        print(error_message)
        socketio.emit('stream', {'text': '</think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
        socketio.emit("stream", error_message, room=sid)
        final_response = error_message
    finally:
        socketio.emit("stream_end", {}, room=sid)

        if session_id and final_response:
            ai_message = {
                "sender": current_sender,
                "message": final_response,
                "type": "ai",
                "thought": full_thought,
            }
            _save_message_to_db(session_id, ai_message)

current_model = None
current_model_path_string = None
current_backend = "llama.cpp"
model_lock = threading.Lock()
model_configs = {}
stop_streaming = False
stop_lock = threading.Lock()

@socketio.on('connect')
def handle_connect():
    global current_backend
    with model_lock:
        backend = model_configs.get("backend", "llama.cpp")
        provider = model_configs.get("api_provider") if backend == "api" else None
        models = get_available_models(backend, provider=provider)
        emit('models', {'backend': backend, 'models': models})
        emit('configs', model_configs)
        emit('backend_set', {'backend': backend})
        emit('nova_settings_loaded', nova_settings) # Emit loaded settings
        if current_model_path_string:
            emit('model_loaded', {'model': current_model_path_string})

def fetch_models_task(backend, provider, sid):
    """Background task to fetch models without blocking."""
    print(f"📦 Fetching models for backend: {backend}, provider: {provider}")
    try:
        models = get_available_models(backend, provider=provider)
        print(f"✅ Found {len(models)} models: {models}")
        socketio.emit('models', {'backend': backend, 'models': models}, room=sid)
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        socketio.emit('error', {'message': f"Failed to fetch models for {backend}: {e}"}, room=sid)

@socketio.on('set_backend')
def handle_set_backend(data):
    global current_backend, model_configs
    sid = request.sid
    with model_lock:
        backend = data.get('backend')
        provider = data.get('provider')
        
        print(f"🎯 Received set_backend: backend={backend}, provider={provider}")
        
        if backend not in ["llama.cpp", "ollama", "api", "safetensors"]:
            emit('error', {'message': 'Invalid backend selected'})
            return

        current_backend = backend
        model_configs["backend"] = backend

        if backend == "api":
            if provider:
                model_configs["api_provider"] = provider
            if 'api_key' in data:
                api_key = data['api_key']
                key_name = f"{provider}_api_key"
                if api_key:
                    os.environ[f"{provider.upper()}_API_KEY"] = api_key
                    model_configs[key_name] = api_key
                    print(f"🔑 Set API key for {provider}")
                else:
                    model_configs.pop(key_name, None)
                    os.environ.pop(f"{provider.upper()}_API_KEY", None)
                    print(f"🔑 Cleared API key for {provider}")

        with open('config.json', 'w') as f:
            json.dump(model_configs, f, indent=4)

        emit('backend_set', {'backend': current_backend})
        
        provider_for_models = model_configs.get("api_provider") if current_backend == "api" else None
        socketio.start_background_task(fetch_models_task, current_backend, provider_for_models, sid)

def load_model_task(data, sid):
    """Background task to load a model without blocking."""
    global current_model, current_model_path_string, current_backend
    with model_lock:
        try:
            # If a model is already loaded, unload it using the proper task.
            if current_model:
                print("Unloading existing model before loading new one...")
                # This is a blocking call within the locked task, which is what we want.
                unload_model_task(sid)
            
            print("Loading new model...")
            current_model = load_model(**data)
            current_model_path_string = data['model_path']
            current_backend = data.get('backend', current_backend) # Ensure backend is updated
            
            socketio.emit('model_loaded', {'model': data['model_path']}, room=sid)
            print(f"✅ New model loaded: {data['model_path']}")
        except Exception as e:
            socketio.emit('error', {'message': f"Load failed: {str(e)}"}, room=sid)
            import traceback
            traceback.print_exc()

@socketio.on('load_model')
def handle_load_model(data):
    sid = request.sid
    # The backend is now sent with the request, ensuring the correct one is used.
    backend = data.get('backend', current_backend)
    
    # --- Hotfix for frontend parameter name mismatch ---
    if 'gpuLayers' in data:
        data['gpu_layers'] = data.pop('gpuLayers')
    # --- End Hotfix ---

    print(f"LOAD MODEL REQUEST: Using backend '{backend}' with data: {data}")
    socketio.start_background_task(load_model_task, data, sid)

# --- New Chat History Socket.IO Handlers ---
@socketio.on('summarize_and_rename')
def handle_summarize_and_rename(data):
    """Summarizes the first message to create a chat title."""
    session_id = data.get('session_id')
    text = data.get('text')
    user_id = data.get('user_id', 'default_user')
    sid = request.sid
    if not all([session_id, text]): return

    try:
        summary_title = get_summary_for_title(text)
        if summary_title:
            result = chat_sessions_collection.update_one(
                {'_id': ObjectId(session_id), 'user_id': user_id},
                {'$set': {'name': summary_title}}
            )
            if result.modified_count > 0:
                # Emit to all clients so the sidebar updates everywhere
                socketio.emit('session_renamed', {'session_id': session_id, 'new_name': summary_title})
    except Exception as e:
        # Don't bother the user with an error if this fails, just log it
        print(f"⚠️ Failed to summarize and rename session: {e}")

@socketio.on('get_sessions')
def handle_get_sessions(data):
    """Fetches all chat sessions for a user from MongoDB."""
    user_id = data.get('user_id', 'default_user') # Assume a default user for now
    sid = request.sid
    if chat_sessions_collection is None:
        emit('error', {'message': 'Chat history database not configured.'}, room=sid)
        return

    try:
        sessions_cursor = chat_sessions_collection.find(
            {'user_id': user_id},
            {'_id': 1, 'name': 1, 'timestamp': 1} # Projection: only get needed fields
        ).sort('timestamp', -1) # Sort by most recent
        
        sessions = [
            {**s, '_id': str(s['_id']), 'timestamp': s['timestamp'].isoformat()}
            for s in sessions_cursor
        ]
        emit('sessions_loaded', {'sessions': sessions}, room=sid)
    except PyMongoError as e:
        emit('error', {'message': f'Failed to load chat sessions: {e}'}, room=sid)

@socketio.on('get_session_messages')
def handle_get_session_messages(data):
    """Fetches all messages for a specific chat session."""
    session_id = data.get('session_id')
    user_id = data.get('user_id', 'default_user')
    sid = request.sid

    if not session_id:
        emit('error', {'message': 'No session_id provided.'}, room=sid)
        return
    if chat_sessions_collection is None:
        emit('error', {'message': 'Chat history database not configured.'}, room=sid)
        return

    try:
        session = chat_sessions_collection.find_one(
            {'_id': ObjectId(session_id), 'user_id': user_id}
        )
        if session:
            # Ensure messages have a consistent format if needed (e.g., converting timestamps)
            emit('session_messages_loaded', {'session_id': session_id, 'messages': session.get('messages', [])}, room=sid)
        else:
            emit('error', {'message': 'Session not found or access denied.'}, room=sid)
    except PyMongoError as e:
        emit('error', {'message': f'Failed to load messages: {e}'}, room=sid)
    except Exception as e: # Catch potential ObjectId errors
        emit('error', {'message': f'Invalid session_id format: {e}'}, room=sid)

@socketio.on('create_session')
def handle_create_session(data):
    """Creates a new chat session in MongoDB."""
    user_id = data.get('user_id', 'default_user')
    name = data.get('name', 'New Chat')
    sid = request.sid
    if chat_sessions_collection is None:
        emit('error', {'message': 'Chat history database not configured.'}, room=sid)
        return

    try:
        new_session = {
            "user_id": user_id,
            "name": name,
            "messages": [],
            "timestamp": datetime.now()
        }
        result = chat_sessions_collection.insert_one(new_session)
        new_session['_id'] = str(result.inserted_id)
        new_session['timestamp'] = new_session['timestamp'].isoformat() # Convert datetime to string
        emit('session_created', {'session': new_session}, room=sid)
    except PyMongoError as e:
        emit('error', {'message': f'Failed to create new session: {e}'}, room=sid)

@socketio.on('delete_session')
def handle_delete_session(data):
    """Deletes a chat session from MongoDB."""
    session_id = data.get('session_id')
    user_id = data.get('user_id', 'default_user')
    sid = request.sid
    if not session_id: return

    try:
        result = chat_sessions_collection.delete_one(
            {'_id': ObjectId(session_id), 'user_id': user_id}
        )
        if result.deleted_count > 0:
            emit('session_deleted', {'session_id': session_id}, room=sid)
        else:
            emit('error', {'message': 'Session not found or you do not have permission to delete it.'}, room=sid)
    except Exception as e:
        emit('error', {'message': f'Failed to delete session: {e}'}, room=sid)

@socketio.on('rename_session')
def handle_rename_session(data):
    """Renames a chat session in MongoDB."""
    session_id = data.get('session_id')
    new_name = data.get('new_name')
    user_id = data.get('user_id', 'default_user')
    sid = request.sid
    if not all([session_id, new_name]): return

    try:
        result = chat_sessions_collection.update_one(
            {'_id': ObjectId(session_id), 'user_id': user_id},
            {'$set': {'name': new_name, 'timestamp': datetime.now()}}
        )
        if result.modified_count > 0:
            emit('session_renamed', {'session_id': session_id, 'new_name': new_name}, room=sid)
        else:
            emit('error', {'message': 'Session not found or you do not have permission to rename it.'}, room=sid)
    except Exception as e:
        emit('error', {'message': f'Failed to rename session: {e}'}, room=sid)

def _save_message_to_db(session_id, message, user_id='default_user'):
    """Internal helper to save a message to the database."""
    if not all([session_id, message]): return False
    try:
        result = chat_sessions_collection.update_one(
            {'_id': ObjectId(session_id), 'user_id': user_id},
            {
                '$push': {'messages': message},
                '$set': {'timestamp': datetime.now()}
            }
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"🔴 Failed to save message internally: {e}")
        return False

@socketio.on('save_message')
def handle_save_message(data):
    """Saves a message to a chat session in MongoDB."""
    sid = request.sid
    if not _save_message_to_db(data.get('session_id'), data.get('message'), data.get('user_id', 'default_user')):
        emit('error', {'message': 'Could not save message: Session not found.'}, room=sid)

@socketio.on('agent_command')
def handle_agent_command(data):
    """Handles a direct command to the agent/orchestrator."""
    user_input = data.get('text', '')
    session_id = data.get('session_id')
    sid = request.sid
    
    if not user_input:
        return

    # --- Save User's Agent Command to DB ---
    if session_id:
        user_message = {
            "sender": data.get('userName', 'User'),
            "message": f"[Agent Command] {user_input}",
            "type": "user"
        }
        _save_message_to_db(session_id, user_message)

    # Prefer the new agentic orchestrator when auto-agent is enabled
    if auto_agent_enabled:
        socketio.start_background_task(run_agentic_chat_task, data, sid, True)
        return

    def run_agent_task():
        full_thought = ""
        final_response = ""
        orchestrator_name = "Nova"  # Orchestrator always uses "Nova"
        current_sender = data.get('aiName', 'Nova')  # Get the actual AI name
        try:
            socketio.emit('stream_start', {}, room=sid)
            socketio.sleep(0)

            # 1. Get the list of tool calls from the orchestrator
            tool_calls = get_tool_call(user_input)

            socketio.emit('stream', {'text': '<think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
            socketio.sleep(0)

            if not tool_calls:
                final_response = "I wasn't able to determine a tool to use for that command."
                full_thought = "Orchestrator did not select a tool."
                socketio.emit('stream', full_thought, room=sid)
                socketio.sleep(0)
            else:
                # 2. Iterate through the sequence of tool calls
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments")
                    
                    thought_for_dispatch = f"Step {i+1}: Executing tool `{tool_name}` with arguments: `{tool_args}`...\n"
                    full_thought += thought_for_dispatch
                    socketio.emit('stream', thought_for_dispatch, room=sid)
                    socketio.sleep(0)

                    # 3. Dispatch the tool and stream the result
                    tool_result = dispatch_tool(tool_name, tool_args)
                    result_text = f"Step {i+1} Result: {tool_result}\n"
                    full_thought += result_text
                    final_response += result_text
                    socketio.emit('stream', result_text, room=sid)
                    socketio.sleep(0)

            # 4. Stream the final aggregated response
            socketio.emit('stream', '</think>', room=sid)
            socketio.sleep(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"An error occurred while running the agent command: {e}"
            socketio.emit('stream', '</think>', room=sid) 
            socketio.emit('stream', error_message, room=sid)
            final_response = error_message
        finally:
            socketio.emit('stream_end', {}, room=sid)
            # --- Save Agent's Final Response to DB ---
            if session_id and final_response:
                ai_message = {
                    "sender": orchestrator_name,  # Always use "Nova" for orchestrator
                    "message": final_response,
                    "type": "ai",
                    "thought": full_thought
                }
                _save_message_to_db(session_id, ai_message)

    socketio.start_background_task(run_agent_task)

@socketio.on('chat')
def handle_chat(data):
    backend = data.get('backend', current_backend)
    provider = data.get('provider')
    timezone = data.get('timezone', 'UTC')
    raw_user_input = data.get('text', '')
    user_input_text = extract_text_from_content(raw_user_input)
    attachments = data.get('attachments') or []
    attachment_context = ""
    if attachments:
        blocks = []
        for attachment in attachments:
            name = attachment.get('name', 'attachment')
            content = attachment.get('content', '')
            blocks.append(f"### {name} ###\n{content}")
        attachment_context = "\n\n--- ATTACHED FILES ---\n" + "\n\n".join(blocks)
        context_block = attachment_context.strip()
        user_input_text = f"{attachment_context}\n\n{user_input_text}".strip()
        raw_user_input = merge_text_into_content(raw_user_input, context_block)
        # Keep the augmented input in the payload for downstream steps (preserve multimodal parts)
        data['text'] = raw_user_input
    else:
        data['text'] = raw_user_input
    
    # --- Save User Message to DB ---
    session_id = data.get('session_id')
    if session_id:
        user_message = {
            "sender": data.get('userName', 'User'),
            "message": user_input_text,
            "type": "user",
            "imageB64": data.get('image_base_64'),
            "attachments": attachments
        }
        _save_message_to_db(session_id, user_message)

    # --- Hybrid Orchestration Logic ---
    # 0. Allow forcing agentic mode with /agent prefix
    if auto_agent_enabled:
        stripped_input = user_input_text.strip()
        if stripped_input.startswith('/agent '):
            forced_query = stripped_input[7:].strip()
            if forced_query:
                data_for_agent = dict(data)
                data_for_agent['text'] = forced_query
                sid = request.sid
                socketio.start_background_task(run_agentic_chat_task, data_for_agent, sid, True)
                return

    # 1. Always check for slash commands first (instant)
    command = parse_command(user_input_text)
    if command:
        query = command.get('query')
        command_name = command.get('command')

        # List of commands handled by the special iframe/frontend logic
        frontend_commands = ['browser']

        if command_name in frontend_commands:
            response_payload = {'type': 'error', 'message': f"Unknown command: {command_name}", 'sender': 'Nova'}
            try:
                if command_name == 'browser':
                    # Treat the query as a URL (or domain/keyword) and open via the proxy-enabled browser panel.
                    target = (query or "").strip()
                    if target and not target.startswith(("http://", "https://")):
                        target = f"https://{target}"
                    response_payload = {
                        'type': 'iframe',
                        'url': target,
                        'message': f"Opening browser for: `{query}`",
                        'sender': 'Nova'
                    }

                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return # Stop further processing

            except Exception as e:
                response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return # Stop further processing

        elif command_name == 'search':
            # NEW: Enhanced multi-panel intelligent search
            try:
                encoded_query = urllib.parse.quote(query)
                aggregated_context = f"User searched for: {query}\n\n"

                # 1. YouTube Search first (so embed shows quickly even with single SearXNG instance)
                try:
                    youtube_url = f"http://localhost:8088/search?q={encoded_query}&engines=youtube&format=json"
                    youtube_response = requests.get(youtube_url, timeout=10).json()
                    first_video = next((r for r in youtube_response.get('results', []) if 'youtube.com/watch' in r.get('url', '') or 'youtu.be/' in r.get('url', '')), None)

                    if first_video:
                        url = first_video.get('url', '')
                        vid_match = re.search(r'(?:v=|youtu\.be/)([^&\\s]{6,})', url)
                        video_id = vid_match.group(1) if vid_match else None
                        if video_id:
                            emit('command_response', {
                                'type': 'youtube_embed',
                                'video_id': video_id,
                                'message': f"Related video for '{query}':",
                                'sender': 'Nova'
                            })
                            aggregated_context += "YouTube Result:\n"
                            aggregated_context += f"Video: {first_video.get('title', 'No title')}\n\n"
                    else:
                        print("No YouTube result found via SearXNG")
                except Exception as e:
                    print(f"YouTube search failed: {e}")

                # Slight delay to avoid hammering single SearXNG instance
                time.sleep(1.5)

                # 2. SearXNG Web Search
                try:
                    search_url = f"http://localhost:8088/search?q={encoded_query}&format=json"
                    search_response = requests.get(search_url, timeout=10).json()
                    results = search_response.get('results', [])[:5]  # Top 5 results

                    # Emit to open search panel
                    emit('search_results', {
                        'query': query,
                        'results': [{'title': r.get('title', ''), 'url': r.get('url', ''), 'snippet': r.get('content', '')} for r in results]
                    })

                    # Add to context
                    if results:
                        aggregated_context += "Web Search Results:\n"
                        for i, result in enumerate(results, 1):
                            aggregated_context += f"{i}. {result.get('title', 'No title')}\n"
                            aggregated_context += f"   {result.get('content', 'No description')[:200]}...\n\n"
                except Exception as e:
                    print(f"Web search failed: {e}")

                # 3. Send aggregated context to Solace for synthesis
                # Continue to normal chat flow with enhanced context
                data['text'] = f"{aggregated_context}\n\nBased on the search results above, please provide a helpful summary about: {query}"

                # Don't return - let it continue to normal chat processing

            except Exception as e:
                response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return # Stop further processing

        elif command_name == 'youtube':
            try:
                encoded_query = urllib.parse.quote(query)
                search_url = f"http://localhost:8088/search?q={encoded_query}&engines=youtube&format=json"
                response = requests.get(search_url, timeout=10).json()
                first_video = next((r for r in response.get('results', []) if 'youtube.com/watch' in r.get('url', '')), None)
                if first_video:
                    video_id = first_video['url'].split('v=')[1].split('&')[0]
                    response_payload = {'type': 'youtube_embed', 'video_id': video_id, 'message': f"Here is the top YouTube result for `{query}`:", 'sender': 'Nova'}
                else:
                    response_payload = {'type': 'error', 'message': f"No YouTube results found for '{query}'.", 'sender': 'Nova'}
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
            emit('command_response', response_payload)
            if session_id:
                _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
            return
        elif command_name == 'image':
            try:
                response_payload = {
                    'type': 'image_generation', 
                    'message': f"Opening image generator for: '{query}'",
                    'prompt': query,
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'media':
            try:
                response_payload = {
                    'type': 'media_browser', 
                    'message': f"Opening media browser for: '{query}'",
                    'query': query,
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'maps':
            try:
                encoded_location = urllib.parse.quote(query)
                maps_url = f"https://www.google.com/maps/embed/v1/place?key=AIzaSyD&q={encoded_location}"
                # Note: Replace 'AIzaSyD' with actual Google Maps API key or use search mode
                maps_url = f"https://www.google.com/maps?q={encoded_location}"  # Fallback to direct link
                response_payload = {
                    'type': 'iframe',
                    'url': maps_url,
                    'message': f"📍 Map for: {query}",
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Maps error: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'weather':
            try:
                # Use wttr.in for simple weather data (no API key needed)
                encoded_location = urllib.parse.quote(query if query else 'auto')  # auto = IP-based location
                weather_url = f"https://wttr.in/{encoded_location}?format=j1"
                weather_response = requests.get(weather_url, timeout=10).json()

                current = weather_response['current_condition'][0]
                temp_c = current['temp_C']
                temp_f = current['temp_F']
                feels_like_c = current['FeelsLikeC']
                feels_like_f = current['FeelsLikeF']
                weather_desc = current['weatherDesc'][0]['value']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']

                location = weather_response['nearest_area'][0]['areaName'][0]['value']
                country = weather_response['nearest_area'][0]['country'][0]['value']

                weather_message = f"""🌤️ **Weather for {location}, {country}**

**Current Conditions:** {weather_desc}
**Temperature:** {temp_c}°C ({temp_f}°F)
**Feels Like:** {feels_like_c}°C ({feels_like_f}°F)
**Humidity:** {humidity}%
**Wind Speed:** {wind_speed} km/h"""

                response_payload = {
                    'type': 'text',
                    'message': weather_message,
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Weather lookup failed: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'wiki':
            try:
                encoded_topic = urllib.parse.quote(query)
                wiki_url = f"https://en.wikipedia.org/wiki/{encoded_topic}"
                response_payload = {
                    'type': 'iframe',
                    'url': wiki_url,
                    'message': f"📚 Wikipedia: {query}",
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Wikipedia error: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'wiki-summarize':
            try:
                # Fetch Wikipedia content via API
                wiki_api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(query)}&format=json"
                wiki_response = requests.get(wiki_api_url, timeout=10).json()

                pages = wiki_response.get('query', {}).get('pages', {})
                page = next(iter(pages.values()))

                if 'extract' in page:
                    wiki_text = page['extract']
                    # Limit to first 2000 chars for summarization
                    wiki_text = wiki_text[:2000] if len(wiki_text) > 2000 else wiki_text

                    # Use Nova to summarize
                    summary = summarize_text(wiki_text)

                    response_payload = {
                        'type': 'text',
                        'message': f"📚 **Wikipedia Summary: {query}**\n\n{summary}",
                        'sender': 'Nova'
                    }
                else:
                    response_payload = {
                        'type': 'error',
                        'message': f"Could not find Wikipedia article for '{query}'",
                        'sender': 'Nova'
                    }

                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Wikipedia summarization failed: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'calc':
            try:
                # Safe calculator using ast.literal_eval
                import ast
                import operator as op

                # Supported operators
                operators = {
                    ast.Add: op.add,
                    ast.Sub: op.sub,
                    ast.Mult: op.mul,
                    ast.Div: op.truediv,
                    ast.Pow: op.pow,
                    ast.USub: op.neg
                }

                def safe_eval(expr):
                    """Safely evaluate a mathematical expression"""
                    def eval_(node):
                        if isinstance(node, ast.Num):  # number
                            return node.n
                        elif isinstance(node, ast.BinOp):  # binary operation
                            return operators[type(node.op)](eval_(node.left), eval_(node.right))
                        elif isinstance(node, ast.UnaryOp):  # unary operation
                            return operators[type(node.op)](eval_(node.operand))
                        else:
                            raise TypeError(node)

                    return eval_(ast.parse(expr, mode='eval').body)

                result = safe_eval(query)
                response_payload = {
                    'type': 'text',
                    'message': f"🧮 **Calculator**\n\n`{query}` = **{result}**",
                    'sender': 'Nova'
                }
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Calculation error: Invalid expression", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

        elif command_name == 'play':
            try:
                # Spotify integration placeholder
                # This requires Spotify Web API credentials
                response_payload = {
                    'type': 'error',
                    'message': f"🎵 Spotify integration not yet configured. Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your environment.",
                    'sender': 'Nova'
                }
                # TODO: Implement Spotify search and playback
                # 1. Search Spotify for the query
                # 2. Return embedded player or link
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return
            except Exception as e:
                response_payload = {'type': 'error', 'message': f"Spotify error: {e}", 'sender': 'Nova'}
                emit('command_response', response_payload)
                if session_id:
                    _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
                return

    # 1b. If no handled slash command, decide whether to use agentic orchestrator
    use_agentic = auto_agent_enabled
    if use_agentic and detect_needs_agentic_approach(user_input_text):
        sid = request.sid
        socketio.start_background_task(run_agentic_chat_task, data, sid, False)
        return

    # 2. If no slash command, intelligently detect if web search is needed
    # Use intelligent detection (heuristics + Nova) as primary method
    search_decision = should_perform_web_search_intelligent(user_input_text)
    should_orchestrate = search_decision['should_search']

    # Fallback: Also check hardcoded keywords for robustness
    if not should_orchestrate:
        ORCHESTRATOR_KEYWORDS = [
            "latest news", "breaking news", "current events", "today's news",
            "weather", "forecast", "temperature", "air quality",
            "price of", "stock price", "share price", "exchange rate", "crypto price", "market cap",
            "sports score", "game results", "match score", "league standings",
            "flight status", "train schedule", "bus times",
            "traffic", "road conditions"
        ]
        should_orchestrate = any(keyword in user_input_text.lower() for keyword in ORCHESTRATOR_KEYWORDS)
        if should_orchestrate:
            print(f"🔍 Web search triggered by keyword fallback")
    else:
        print(f"🧠 Web search triggered intelligently: {search_decision['reasoning']} (confidence: {search_decision['confidence']:.0%})")

    if should_orchestrate:
        print("Orchestration keywords detected. Checking for tool call...")
        tool_calls = get_tool_call(user_input_text)
        if tool_calls:
            tool_call = tool_calls[0] # Assuming one tool call for now
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments")
            
            # ONLY DO THIS FOR TOOL CALLS - NOT FOR THE FIRST MESSAGE
            if tool_name == "direct_chat":
                print("Orchestrator decided on direct chat, bypassing tool execution and sending to main model.")
                # Fall through to the main model logic below
                pass
            else:
                # Execute the tool
                tool_result = dispatch_tool(tool_name, tool_args)
                # Prepend the tool result with a forceful instruction for the main model.
                forceful_instruction = (
                    "You have been provided with the following real-time information to answer the user's query. "
                    "You MUST use this information to form your answer and ignore any conflicting internal knowledge."
                )
                user_input_text = f"{forceful_instruction}\n\n--- Information ---\n{tool_result}\n--- End Information ---\n\nUser Query: {user_input_text}"
                # IMPORTANT: Update the 'text' in the data payload that gets passed to the streaming task.
                data['text'] = user_input_text
                # Fall through to the main model logic below

    # 3. If neither, bypass orchestrator and go directly to the main model
    if not current_model:
        emit('error', {'message': 'No model loaded.'})
        return
    
    # Get the config for the current model to pass thinking level
    if current_model_path_string in model_configs:
        data['thinking_level'] = model_configs[current_model_path_string].get('thinking_level', 'medium')

    # Ensure backend and provider are correctly set for streaming
    data['backend'] = model_configs.get("backend", "llama.cpp")
    if data['backend'] == "api":
        data['provider'] = model_configs.get("api_provider")
    else:
        data['provider'] = None # Clear provider if not API backend

    # Note: Memory learning now happens AFTER response via Nova summarization
    # This extracts key facts from the full exchange instead of storing raw text

    sid = request.sid
    socketio.start_background_task(stream_response, data, sid)


def _detect_provider_from_model(model_name, provider_hint=None, backend_hint=None):
    """
    Resolve provider name using explicit hint, backend, or model pattern.
    """
    if provider_hint:
        return provider_hint
    name = (model_name or "").lower()
    if "deepseek" in name:
        return "deepseek"
    if "qwen" in name:
        return "qwen"
    if "gpt" in name or "openai" in name or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    if "claude" in name or "anthropic" in name:
        return "anthropic"
    if "gemini" in name:
        return "google"
    if "grok" in name or "xai" in name:
        return "xai"
    if "llama" in name or "meta" in name:
        return "meta"
    if "mistral" in name:
        return "mistral"
    if "sonar" in name or "perplexity" in name:
        return "perplexity"
    if backend_hint:
        return backend_hint
    return "unknown"


def stream_response(data, sid):
    global stop_streaming
    full_response = ""
    full_thought = ""
    session_id = data.get('session_id')
    in_thought_block = False
    current_sender = data.get('aiName', 'Nova')  # Get the actual AI name

    # Initialize token tracking
    model_name = data.get('model_name', current_model_path_string)
    token_tracker = TokenTracker(model_name=model_name)
    raw_user_input = data.get('text', '')
    user_input_text = extract_text_from_content(raw_user_input)

    try:
        # Start timing for TTFT calculation
        token_tracker.start()

        # Estimate input tokens from conversation history
        conversation_history = data.get('history', [])
        input_token_count = estimate_input_tokens(conversation_history, raw_user_input, model_name)
        token_tracker.add_input_tokens(input_token_count)

        socketio.emit('stream_start', {'sender': current_sender}, room=sid)
        socketio.sleep(0)

        backend = data.get('backend', current_backend)
        provider = data.get('provider')
        
        # --- Stream Router ---
        streamer_map = {
            "llama.cpp": stream_llamacpp,
            "ollama": stream_ollama,
            "safetensors": stream_safetensors,
            "api": {
                "google": stream_google,
                "openai": stream_openai,
                "anthropic": stream_anthropic,
                "meta": stream_meta,
                "xai": stream_xai,
                "qwen": stream_qwen,
                "deepseek": stream_deepseek,
                "perplexity": stream_perplexity,
                "openrouter": stream_openrouter
            }
        }

        streamer = None
        if backend == 'api':
            streamer = streamer_map.get('api', {}).get(provider)
        else:
            streamer = streamer_map.get(backend)

        if not streamer:
            raise ValueError(f"No valid streamer found for backend '{backend}' and provider '{provider}'")

        # --- Prepare Arguments for All Streamers ---
        image_payload = None
        if data.get('image_base_64'):
            image_payload = {
                "data": data.get('image_base_64'),
                "media_type": data.get('image_mime_type') or "image/png"
            }

        args = {
            "model_instance": current_model if backend != 'ollama' else current_model_path_string,
            "model_id_str": current_model_path_string,
            "user_input": data['text'],
            "conversation_history": data.get('history', []),
            "should_stop": lambda: stop_streaming,
            "image_data": image_payload,
            "timezone": data.get('timezone', 'UTC'),
            "debug_mode": data.get('debug_mode', False),
            "thinking_level": data.get('thinking_level', 'medium') # Pass thinking level
        }
        
        # Add backend-specific arguments
        if backend == 'llama.cpp':
            args["tools"] = TOOLS_SCHEMA
        
        # Filter args to only those accepted by the target streamer to avoid unexpected kwargs
        try:
            import inspect
            sig = inspect.signature(streamer)
            accepted = set(sig.parameters.keys())
            filtered_args = {k: v for k, v in args.items() if k in accepted}
        except Exception:
            # Fallback to original args if inspection fails
            filtered_args = args

        model_response_generator = streamer(**filtered_args)

        # --- Universal Response Handling with Consistent Thinking Blocks ---
        for chunk in model_response_generator:
            if stop_streaming:
                break

            # Robustly support different streamer outputs:
            # - dict chunks with keys {'type', 'token'} (preferred)
            # - plain strings (some providers)
            # - dicts missing 'type' treated as reply
            if isinstance(chunk, dict):
                chunk_type = chunk.get('type', 'reply')
                token = chunk.get('token', '')
            else:
                chunk_type = 'reply'
                token = str(chunk)

            if chunk_type == 'tool_call':
                # Handle tool calls if needed
                pass
            elif chunk_type == 'error':
                socketio.emit('error', {'message': token}, room=sid)
            else:
                # Normalize think handling: respect explicit 'thought' type
                # and also detect inline <think>...</think> tags in token streams.
                def emit_thought_text(text):
                    nonlocal full_thought
                    if text:
                        full_thought += text
                        socketio.emit('stream', {'text': text, 'aiName': current_sender, 'type': 'thought'}, room=sid)

                def emit_reply_text(text):
                    nonlocal full_response
                    if text:
                        full_response += text
                        # Track output tokens (approximate 1 token per chunk on average)
                        token_tracker.add_output_token(text)
                        socketio.emit('stream', {'text': text, 'aiName': current_sender, 'type': 'reply'}, room=sid)

                if chunk_type == 'thought':
                    if not in_thought_block:
                        socketio.emit('stream', {'text': '<think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
                        in_thought_block = True
                    emit_thought_text(token)
                else:
                    # chunk_type is reply or unknown — parse for inline think tags
                    remaining = token or ''
                    while remaining:
                        if in_thought_block:
                            close_idx = remaining.find('</think>')
                            if close_idx == -1:
                                emit_thought_text(remaining)
                                remaining = ''
                            else:
                                emit_thought_text(remaining[:close_idx])
                                socketio.emit('stream', '</think>', room=sid)
                                in_thought_block = False
                                remaining = remaining[close_idx + len('</think>'):]
                        else:
                            open_idx = remaining.find('<think>')
                            if open_idx == -1:
                                emit_reply_text(remaining)
                                remaining = ''
                            else:
                                # Emit any leading reply text before think
                                emit_reply_text(remaining[:open_idx])
                                socketio.emit('stream', {'text': '<think>', 'aiName': current_sender, 'type': 'control'}, room=sid)
                                in_thought_block = True
                                remaining = remaining[open_idx + len('<think>'):]

            socketio.sleep(0)

        # Ensure thinking block is closed at the end
        if in_thought_block:
            socketio.emit('stream', '</think>', room=sid)

    except Exception as e:
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f"Stream error: {str(e)}"}, room=sid)
    finally:
        with stop_lock:
            stop_streaming = False

        # Get token metrics
        token_metrics = token_tracker.get_metrics()

        # Persist usage metrics for analytics (one event per streamed response)
        if usage_events_collection is not None:
            try:
                normalized_model = normalize_model_name(model_name)
                provider_value = _detect_provider_from_model(
                    model_name, data.get('provider'), data.get('backend', current_backend)
                )
                input_tokens = token_metrics.get('inputTokens', 0)
                output_tokens = token_metrics.get('outputTokens', 0)
                total_tokens = input_tokens + output_tokens
                estimated_cost = calculate_cost(input_tokens, output_tokens, normalized_model)
                usage_events_collection.insert_one({
                    "timestamp": int(time.time() * 1000),
                    "sessionId": session_id,
                    "model": normalized_model,
                    "provider": provider_value,
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": total_tokens,
                    "timeToFirstTokenMs": token_metrics.get('timeToFirstToken'),
                    "tokensPerSecond": token_metrics.get('tokensPerSecond'),
                    "estimatedCost": estimated_cost,
                    "meta": {
                        "aiName": current_sender,
                        "backend": data.get('backend', current_backend),
                        "timezone": data.get('timezone'),
                        "rawModelName": model_name,
                    }
                })
            except Exception as e:
                logger.error(f"Failed to persist usage event: {e}")

        socketio.emit('stream_end', {
            'aiName': current_sender,
            'tokenMetrics': token_metrics
        }, room=sid)
        
        # Fix: Use the actual sender name instead of hardcoding "Nova"
        if full_response and session_id:
            ai_message = {
                "sender": current_sender,  # Use the dynamic AI name
                "message": full_response,
                "type": "ai",
                "thought": full_thought,
                "tokenMetrics": token_metrics
            }
            _save_message_to_db(session_id, ai_message)

        # MEMORY SUMMARIZATION: Extract key facts via Nova instead of storing raw text
        user_id = data.get('user_id', 'default_user')

        if full_response and user_input_text:
            # Let Nova extract concise facts from the conversation exchange
            facts = summarize_for_memory(user_input_text, full_response)

            if facts:
                print(f" Processing {len(facts)} facts extracted by Nova...")
                # Store the summarized facts (much cleaner than raw text)
                stored_count = 0
                for fact in facts:
                    result = memory.learn_from_text(
                        fact,
                        source="conversation_summary",
                        model_id=current_model_path_string,
                        user_id=user_id
                    )
                    if result:
                        stored_count += 1

                if stored_count > 0:
                    print(f"✅ Successfully stored {stored_count}/{len(facts)} facts to memory")
                else:
                    print(f"⚠️ No facts stored ({len(facts)} extracted, all below threshold or duplicates)")
            else:
                # Fallback: If summarization fails, store raw (old behavior)
                print("⚠️ Summarization returned no facts, using fallback storage")
                memory.learn_from_text(
                    full_response,
                    source=current_sender.lower(),
                    model_id=current_model_path_string,
                    user_id=user_id
                )

        # CONFIDENCE SCORING: Score the main model's response
        # Skip scoring for: Nova's own responses, empty responses, command responses, or already-enhanced retries
        is_nova_response = current_sender.lower() == 'nova'
        is_command = user_input_text.strip().startswith('/')
        is_enhanced = data.get('confidence_enhanced', False)

        if full_response and not is_nova_response and not is_command and not is_enhanced and len(full_response.strip()) > 10:
            try:
                confidence_result = score_response_confidence(full_response, user_input_text)
                confidence_score = confidence_result.get('score', 100)
                reasoning = confidence_result.get('reasoning', 'n/a')
                should_search = confidence_result.get('should_search', False)

                # Log the confidence score
                if confidence_score < 50:
                    print(f"🔴 LOW CONFIDENCE ({confidence_score:.1f}%): {reasoning}")
                elif confidence_score < 80:
                    print(f"🟡 MODERATE CONFIDENCE ({confidence_score:.1f}%): {reasoning}")
                else:
                    print(f"🟢 HIGH CONFIDENCE ({confidence_score:.1f}%): {reasoning}")

                # Only act on low confidence + search recommendation + question-like input
                # (threshold relaxed to 60 to allow legit lookups without being over-aggressive)
                if confidence_score < 60 and should_search and is_question_like(user_input_text):
                    print(f"⚠️ Low confidence ({confidence_score:.1f}%) detected for query: {user_input_text[:100]}")
                    try:
                        # Auto-trigger a single web search and enhanced response (no UI spam, no loops)
                        search_query = extract_uncertain_span(full_response) or build_search_query(user_input_text)
                        search_results = dispatch_tool("search_web", {"query": search_query})
                        if search_results and not str(search_results).lower().startswith("error"):
                            enhanced_prompt = (
                                "The previous response had low confidence. Here is additional context from web search:\n\n"
                                f"{search_results}\n\n"
                                f"Original query: {user_input_text}\n\n"
                                "Please provide a more informed answer using the search results above."
                            )
                            enhanced_data = dict(data)
                            enhanced_data['text'] = enhanced_prompt
                            enhanced_data['confidence_enhanced'] = True
                            print("✅ Triggering enhanced response with web search context")
                            socketio.start_background_task(stream_response, enhanced_data, sid)
                        else:
                            print("⚠️ Web search returned no results; skipping enhancement")
                    except Exception as enhance_err:
                        print(f"❌ Auto-search/reenhance failed: {enhance_err}")

            except Exception as e:
                print(f"⚠️ Confidence scoring error: {e}")

@socketio.on('stop')
def handle_stop():
    global stop_streaming
    with stop_lock:
        stop_streaming = True
    print("🛑 Stop request received.")

def unload_model_task(sid):
    """Background task to unload a model without blocking."""
    global current_model, current_model_path_string, current_backend
    with model_lock:
        # Allow unload as long as we have a tracked model path, even if the in-memory object was GC'ed
        if current_model_path_string:
            model_path_to_unload = current_model_path_string
            backend_to_unload = current_backend
            model_to_unload = current_model

            print(f"[DEBUG] Unloading task: current_model object id is {id(model_to_unload)}")

            # Set globals to None *before* cleanup to break the main reference
            current_model = None
            current_model_path_string = None
            
            if unload_model(model_to_unload, model_path_to_unload, backend=backend_to_unload):
                socketio.emit('model_unloaded', {'model': model_path_to_unload}, room=sid)
                print(f"✅ Globals cleared and unload process finished for: {model_path_to_unload}")
            else:
                # This block might not be reachable if unload_model always returns True
                socketio.emit('error', {'message': f'Failed to unload model: {model_path_to_unload}'}, room=sid)
        else:
            socketio.emit('error', {'message': 'No model is currently loaded.'}, room=sid)

@socketio.on('unload_model')
def handle_unload_model():
    sid = request.sid
    socketio.start_background_task(unload_model_task, sid)

@socketio.on('save_config')
def handle_save_config(data):
    global model_configs
    model_path = data.get('model_path')
    if model_path:
        # Ensure the entry for the model exists
        if model_path not in model_configs:
            model_configs[model_path] = {}
        
        # Update the existing config with new values from the frontend
        for key, value in data.items():
            if key != 'model_path': # Don't save the model_path key inside its own entry
                model_configs[model_path][key] = value
        
        with open('config.json', 'w') as f:
            json.dump(model_configs, f, indent=4)
        emit('config_saved', {'message': f'Configuration for {model_path} saved.'})
        # Emit the updated configs to all clients to keep them in sync
        emit('configs', model_configs)

@socketio.on('toggle_auto_agent')
def handle_toggle_auto_agent(data):
    enabled = data.get('enabled', False)
    set_auto_agent_enabled(enabled)
    emit('auto_agent_status', {'enabled': enabled})

@socketio.on('get_graph_data')
def handle_graph_data():
    try:
        build_graph()
        graph_data = export_graph_json()
        emit('graph_data', graph_data)
    except Exception as e:
        emit('error', {'message': f"Graph error: {str(e)}"})

@socketio.on('tts')
def handle_tts(data):
    text = data.get('text')
    if text:
        for chunk in text_to_speech_stream(text):
            emit('voice_stream', {'audio': chunk})
        emit('voice_stream_end')

@socketio.on('transcribe')
def handle_transcribe(data):
    audio_data = data.get('audio')
    if not audio_data:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
        fp.write(audio_data)
        audio_path = fp.name
    
    try:
        transcription = transcribe_audio(audio_path)
        # After transcription, populate the user's input field
        # For this, we might need a new event or just send it back as a result
        emit('transcription_result', {'text': transcription})
    except Exception as e:
        print(f"Transcription failed: {e}")
        emit('error', {'message': f"Transcription failed: {e}"})
    finally:
        os.remove(audio_path)

def transcribe_audio(audio_path):
    with open('voice_settings.json', 'r') as f:
        settings = json.load(f).get('stt', {})
    
    if settings.get('type') == 'local':
        model_size = settings.get('model', 'tiny.en')
        model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
        return "".join(segment.text for segment in segments)
    elif settings.get('type') == 'cloud':
        provider = settings.get('provider', 'openai') # Default to openai
        if provider == 'google':
            from google.cloud import speech
            client = speech.SpeechClient()
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            response = client.recognize(config=config, audio=audio)
            return "".join(result.alternatives[0].transcript for result in response.results)
        else: # Placeholder for other cloud providers like OpenAI Whisper API
            return "Cloud STT for this provider is not yet implemented."

@socketio.on('save_tool_settings')
def handle_save_tool_settings(data):
    with open('tool_settings.json', 'w') as f:
        json.dump(data, f, indent=4)
    emit('tool_settings_saved', {'message': 'Tool settings saved.'})

@socketio.on('save_voice_settings')
def handle_save_voice_settings(data):
    with open('voice_settings.json', 'w') as f:
        json.dump(data, f, indent=4)
    emit('voice_settings_saved', {'message': 'Voice settings saved.'})
        
@socketio.on('save_nova_settings')
def handle_save_nova_settings(data):
    """Saves the Nova customization settings."""
    try:
        with open('nova_settings.json', 'w') as f:
            json.dump(data, f, indent=4)
        emit('nova_settings_saved', {'message': 'Nova settings saved successfully.'})
    except Exception as e:
        emit('error', {'message': f'Failed to save Nova settings: {e}'})

@socketio.on('reload_orchestrator')
def handle_reload_orchestrator(data):
    """Reload the orchestrator model with new settings."""
    try:
        print("\n" + "="*60)
        print("🔄 ORCHESTRATOR RELOAD REQUEST RECEIVED")
        print("="*60)
        
        # Import here to avoid circular imports
        from orchestrator import reload_orchestrator_model, ORCHESTRATOR_CONFIG
        
        print("🔄 Reloading orchestrator model...")
        reload_orchestrator_model()
        
        print(f"✅ Orchestrator reloaded successfully!")
        print(f"   Type: {ORCHESTRATOR_CONFIG['type']}")
        if ORCHESTRATOR_CONFIG['type'] == 'api':
            print(f"   Provider: {ORCHESTRATOR_CONFIG['api_provider']}")
            print(f"   Model: {ORCHESTRATOR_CONFIG['api_model']}")
        print("="*60 + "\n")
        
        emit('orchestrator_reloaded', {
            'success': True,
            'config': ORCHESTRATOR_CONFIG,
            'message': f"Orchestrator reloaded: {ORCHESTRATOR_CONFIG['type']} mode"
        })
    except Exception as e:
        print(f"❌ Failed to reload orchestrator: {e}")
        import traceback
        traceback.print_exc()
        
        emit('orchestrator_reloaded', {
            'success': False,
            'error': str(e)
        })

@socketio.on('manage_ollama')
def handle_manage_ollama(data):
    import subprocess
    action = data.get('action')
    env_vars = data.get('env', {})
    
    if action == 'stop':
        print("🔌 Stopping Ollama server...")
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, text=True)
        emit('ollama_status', {'status': 'stopped'})
        print("✅ Ollama server stopped.")
    
    elif action == 'restart':
        print("🔌 Restarting Ollama server...")
        # Stop the server first
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, text=True)
        time.sleep(5) # Give it a moment to die
        
        # Set environment variables using the batch script for persistence
        kv_cache_type = env_vars.get('OLLAMA_KV_CACHE_TYPE', 'f16')
        print(f"   - Setting OLLAMA_KV_CACHE_TYPE={kv_cache_type}")
        
        # Use a batch script to set env var and start ollama
        # This is more reliable on Windows for ensuring the new process gets the var
        script_path = os.path.join(BASE_DIR, "run_ollama.bat")
        subprocess.Popen([script_path, kv_cache_type], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        # Give Ollama a generous amount of time to restart
        time.sleep(15)
        
        emit('ollama_status', {'status': 'running', 'env': env_vars})
        print("✅ Ollama server is restarting in the background.")
print(f"📂 Static folder being used: {app.static_folder}")
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    print(f"📦 Request for path: '{path}'")
    
    static_file_path = os.path.join(app.static_folder, path)
    print(f"📁 Looking for file: {static_file_path}")
    print(f"📁 File exists: {os.path.exists(static_file_path)}")
    
    if path != "" and os.path.exists(static_file_path) and os.path.isfile(static_file_path):
        print(f"✅ Serving static file: {path}")
        return send_from_directory(app.static_folder, path)
    
    print(f"🎯 Serving index.html for React routing (path: {path})")
    return send_from_directory(app.static_folder, 'index.html')

import websockets
import uuid

# ... (rest of the imports)

async def get_image(prompt, image_gen_url):
    server_address = image_gen_url.replace('http://', '').replace('https://', '').split('/')[0]
    client_id = str(uuid.uuid4())

    ws_url = f"ws://{server_address}/ws?clientId={client_id}"
    async with websockets.connect(ws_url) as websocket:
        response = requests.post(f"http://{server_address}/prompt", json=prompt)
        prompt_id = response.json()['prompt_id']

        while True:
            out = await websocket.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
        
        history_response = requests.get(f"http://{server_address}/history/{prompt_id}")
        history = history_response.json()
        
        for o in history[prompt_id]['outputs']:
            for node_id in history[prompt_id]['outputs']:
                node_output = history[prompt_id]['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        image_url = f"http://{server_address}/view?filename={image['filename']}&subfolder={image['subfolder']}&type={image['type']}"
                        return image_url
    return None

@socketio.on('generate_image')
def handle_generate_image(data):
    """Handles image generation requests by sending a prompt to a ComfyUI server."""
    def task():
        try:
            prompt = data.get('prompt')
            settings = data.get('settings', {})
            session_id = data.get('session_id')

            # Load image generation URL from settings
            with open('nova_settings.json', 'r') as f:
                nova_settings = json.load(f)
            image_gen_url = nova_settings.get('imageGenUrl')

            if not image_gen_url:
                emit('error', {'message': 'Image generation URL not configured in nova_settings.json'})
                return

            # Qwen-Image ComfyUI API prompt workflow
            comfy_prompt = {
                "39": {
                    "inputs": {
                        "vae_name": "qwen_image_vae.safetensors"
                    },
                    "class_type": "VAELoader"
                },
                "38": {
                    "inputs": {
                        "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                        "type": "qwen_image",
                        "device": "default"
                    },
                    "class_type": "CLIPLoader"
                },
                "58": {
                    "inputs": {
                        "width": settings.get('width', 512),
                        "height": settings.get('height', 512),
                        "batch_size": 1
                    },
                    "class_type": "EmptySD3LatentImage"
                },
                "6": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["38", 0]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "7": {
                    "inputs": {
                        "text": "",
                        "clip": ["38", 0]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "60": {
                    "inputs": {
                        "filename_prefix": "SolaceHomeUI",
                        "images": ["8", 0]
                    },
                    "class_type": "SaveImage"
                },
                "66": {
                    "inputs": {
                        "model": ["73", 0],
                        "shift": 3.1
                    },
                    "class_type": "ModelSamplingAuraFlow"
                },
                "73": {
                    "inputs": {
                        "model": ["37", 0],
                        "lora_name": "Qwen-Image-Lightning-8steps-V1.0.safetensors",
                        "strength_model": 1.0
                    },
                    "class_type": "LoraLoaderModelOnly"
                },
                "8": {
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["39", 0]
                    },
                    "class_type": "VAEDecode"
                },
                "3": {
                    "inputs": {
                        "model": ["66", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["58", 0],
                        "seed": random.randint(0, 999999999999999),
                        "steps": settings.get('steps', 8),
                        "cfg": settings.get('cfgScale', 2.5),
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1.0
                    },
                    "class_type": "KSampler"
                },
                "37": {
                    "inputs": {
                        "unet_name": "qwen_image_fp8_e4m3fn.safetensors",
                        "weight_dtype": "default"
                    },
                    "class_type": "UNETLoader"
                }
            }

            image_url = asyncio.run(get_image({'prompt': comfy_prompt}, image_gen_url))

            if image_url:
                result = {
                    'type': 'image_generated',
                    'message': f'Generated image for: "{prompt}"',
                    'image_url': image_url,
                    'prompt': prompt
                }
            else:
                result = {
                    'type': 'error',
                    'message': 'Failed to generate image.',
                }
            
            emit('command_response', result)
            if session_id:
                _save_message_to_db(session_id, {"sender": "Nova", **result, "type": "ai"})

        except Exception as e:
            emit('error', {'message': f'Image generation failed: {str(e)}'})

    socketio.start_background_task(target=task)

@socketio.on('play_media')
def handle_play_media(data):
    """Handles media playback requests by returning an embeddable URL."""
    try:
        media_id = data.get('mediaId')
        session_id = data.get('session_id')

        # Load media server settings
        with open('nova_settings.json', 'r') as f:
            nova_settings = json.load(f)
        media_server_url = nova_settings.get('mediaServerUrl')
        media_server_api_key = nova_settings.get('mediaServerApiKey')

        if not media_server_url or not media_server_api_key:
            emit('error', {'message': 'Media server URL or API key not configured in nova_settings.json'})
            return

        # Construct the Jellyfin direct stream URL
        embed_url = f"{media_server_url.rstrip('/')}/Videos/{media_id}/stream?api_key={media_server_api_key}"

        result = {
            'type': 'media_embed',
            'message': f'Playing media item {media_id}.',
            'embed_url': embed_url
        }
        
        emit('command_response', result)
        if session_id:
            _save_message_to_db(session_id, {"sender": "Nova", **result, "type": "ai"})

    except Exception as e:
        emit('error', {'message': f'Media playback failed: {str(e)}'})

# ---- Code panel file operations ----
def _is_safe_path(target: Path) -> bool:
    try:
        target = target.resolve()
        return str(target).startswith(str(WORKSPACE_ROOT))
    except Exception:
        return False

def _build_file_tree(path: Path, depth: int = 10, max_entries: int = 9999):
    """
    Build a file tree up to a given depth and entry count.
    """
    root = path.resolve()
    entries_left = max_entries
    
    SKIP_DIRS = {
        '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
        'env', 'dist', 'build', '.next', '.cache',
        'llama-gpu', '.pytest_cache', '.mypy_cache', '.tox',
        'htmlcov', '.coverage', 'site-packages', 'Lib', 'searxng-docker'
    }

    def walk(node: Path, current_depth: int):
        nonlocal entries_left
        if entries_left <= 0:
            return None
        try:
            is_dir = node.is_dir()
        except Exception as e:
            return None
            
        if is_dir:
            is_root_node = (node == root)
            should_skip = (not is_root_node) and (node.name in SKIP_DIRS or node.name.startswith('.'))
            
            if should_skip:
                return None
                
            children = []
            if is_root_node or current_depth < depth:
                try:
                    all_items = list(node.iterdir())
                    
                    # IMPORTANT: Process FILES first, then DIRECTORIES
                    # This ensures root-level files are added before we recurse deep into folders
                    files = [item for item in all_items if item.is_file() and not item.name.startswith('.')]
                    dirs = [item for item in all_items if item.is_dir() and not item.name.startswith('.') and item.name not in SKIP_DIRS]
                    
                    # Sort files alphabetically
                    files.sort(key=lambda p: p.name.lower())
                    # Sort dirs alphabetically  
                    dirs.sort(key=lambda p: p.name.lower())
                    
                    # Process files FIRST
                    for child in files:
                        if entries_left <= 0:
                            break
                        child_info = walk(child, current_depth + 1)
                        if child_info:
                            children.append(child_info)
                    
                    # Then process directories
                    for child in dirs:
                        if entries_left <= 0:
                            break
                        child_info = walk(child, current_depth + 1)
                        if child_info:
                            children.append(child_info)
                            
                except PermissionError:
                    pass
                    
            entries_left -= 1
            return {
                "name": node.name,
                "path": str(node.relative_to(WORKSPACE_ROOT)).replace('\\', '/'),
                "type": "dir",
                "children": children,
            }
        else:
            entries_left -= 1
            return {
                "name": node.name,
                "path": str(node.relative_to(WORKSPACE_ROOT)).replace('\\', '/'),
                "type": "file",
            }

    return walk(root, 0)
    print(f"✅ File tree built. Entries remaining: {entries_left}/{max_entries}")
    return result

@socketio.on('set_workspace_root')
def set_workspace_root(data):
    """
    Update the workspace root to a user-selected directory.
    """
    global WORKSPACE_ROOT
    new_root = data.get('path')
    try:
        if not new_root:
            emit('file_error', {'message': 'Path is required'})
            return
        new_path = Path(new_root).resolve()
        if not new_path.exists() or not new_path.is_dir():
            emit('file_error', {'message': 'Directory not found'})
            return
        WORKSPACE_ROOT = new_path
        emit('workspace_root_changed', {'root': str(WORKSPACE_ROOT)})
    except Exception as e:
        emit('file_error', {'message': f'Failed to set workspace: {e}'})

@socketio.on('list_files')
def handle_list_files(data):
    path = data.get('path', '.')
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    if not target.exists() or not target.is_dir():
        emit('file_error', {'message': 'Directory not found'})
        return
    tree = _build_file_tree(target, depth=8, max_entries=2000)
    emit('file_tree', {'tree': [tree] if tree else [], 'workspaceRoot': str(WORKSPACE_ROOT)})

@socketio.on('read_file')
def handle_read_file(data):
    path = data.get('path')
    if not path:
        emit('file_error', {'message': 'Path is required'})
        return
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    if not target.exists() or not target.is_file():
        emit('file_error', {'message': 'File not found'})
        return
    
    # Check if file is a binary type that Monaco can't edit
    binary_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', 
                        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.pdf', '.zip', 
                        '.tar', '.gz', '.exe', '.dll', '.so', '.dylib', '.bin'}
    file_ext = target.suffix.lower()
    
    if file_ext in binary_extensions:
        emit('file_error', {'message': f'Cannot open binary file type: {file_ext}. Monaco editor only supports text files.'})
        return
        
    try:
        content = target.read_text(encoding='utf-8')
        emit('file_content', {'path': str(path), 'content': content})
    except UnicodeDecodeError:
        emit('file_error', {'message': f'File is not UTF-8 text. This file may be binary or use a different encoding.'})
    except Exception as e:
        emit('file_error', {'message': f'Error reading file: {e}'})

@socketio.on('save_file')
def handle_save_file(data):
    path = data.get('path')
    content = data.get('content', '')
    if not path:
        emit('file_error', {'message': 'Path is required'})
        return
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        emit('file_saved', {'path': str(path)})
    except Exception as e:
        emit('file_error', {'message': f'Error saving file: {e}'})

@socketio.on('create_file')
def handle_create_file(data):
    path = data.get('path')
    if not path:
        emit('file_error', {'message': 'Path is required'})
        return
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('', encoding='utf-8')
        emit('file_saved', {'path': str(path)})
        handle_list_files({'path': '.'})
    except Exception as e:
        emit('file_error', {'message': f'Error creating file: {e}'})

@socketio.on('create_folder')
def handle_create_folder(data):
    path = data.get('path')
    if not path:
        emit('file_error', {'message': 'Path is required'})
        return
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        emit('file_saved', {'path': str(path)})
        handle_list_files({'path': '.'})
    except Exception as e:
        emit('file_error', {'message': f'Error creating folder: {e}'})

@socketio.on('rename_path')
def handle_rename_path(data):
    old_path = data.get('old_path')
    new_path = data.get('new_path')
    if not old_path or not new_path:
        emit('file_error', {'message': 'Old and new paths are required'})
        return
    old_target = (WORKSPACE_ROOT / old_path).resolve()
    new_target = (WORKSPACE_ROOT / new_path).resolve()
    if not _is_safe_path(old_target) or not _is_safe_path(new_target):
        emit('file_error', {'message': 'Access denied'})
        return
    try:
        old_target.rename(new_target)
        emit('file_saved', {'path': str(new_path)})
        handle_list_files({'path': '.'})
    except Exception as e:
        emit('file_error', {'message': f'Error renaming: {e}'})

@socketio.on('delete_path')
def handle_delete_path(data):
    path = data.get('path')
    if not path:
        emit('file_error', {'message': 'Path is required'})
        return
    target = (WORKSPACE_ROOT / path).resolve()
    if not _is_safe_path(target):
        emit('file_error', {'message': 'Access denied'})
        return
    try:
        if target.is_dir():
          for child in target.rglob('*'):
            if child.is_file():
              child.unlink()
          target.rmdir()
        else:
          target.unlink()
        emit('file_saved', {'path': str(path)})
        handle_list_files({'path': '.'})
    except Exception as e:
        emit('file_error', {'message': f'Error deleting: {e}'})

# ---- Parliament orchestration ----
@socketio.on('parliament_request')
def handle_parliament_request(data):
    """
    Fan out the user's message to enabled parliament roles and return merged context.
    Emits:
      - parliament_update { key, status, payload? }
      - parliament_summary { roles: [{key, name, model, provider, prompt, response}], merged_prompt }
    """
    print("\n" + "="*80)
    print("🏛️ PARLIAMENT REQUEST RECEIVED")
    print("="*80)

    try:
        message = data.get('message', '')
        roles = data.get('roles', [])
        session_id = data.get('session_id')

        print(f"📝 Message: {message}")
        print(f"👥 Roles: {len(roles)} roles configured")
        for role in roles:
            print(f"   - {role.get('name')} ({role.get('provider')}/{role.get('model')})")

        if not message or not roles:
            print("⚠️ Missing message or roles, aborting")
            emit('parliament_summary', {'roles': [], 'merged_prompt': ''})
            return

        results = []
        results_lock = threading.Lock()  # Thread safety for results list

        # Stream router map (same as regular chat)
        streamer_map = {
            "llama.cpp": stream_llamacpp,
            "ollama": stream_ollama,
            "safetensors": stream_safetensors,
            "api": {
                "google": stream_google,
                "openai": stream_openai,
                "anthropic": stream_anthropic,
                "meta": stream_meta,
                "xai": stream_xai,
                "qwen": stream_qwen,
                "deepseek": stream_deepseek,
                "perplexity": stream_perplexity,
                "openrouter": stream_openrouter
            }
        }

        # Define function to process each role in parallel
        def process_role(role):
            key = role.get('key')
            prompt = role.get('prompt', '')
            model = role.get('model', 'default')
            provider = role.get('provider', 'cloud')
            backend = role.get('backend', 'api')
            name = role.get('name', key)

            try:
                print(f"🔄 Parliament: Processing {name} ({provider}/{model})...")

                # Build the full prompt for this role - ONLY role prompt + user question, NO HISTORY
                role_prompt = f"{prompt}\n\nUser request:\n{message}"

                # Get the appropriate streamer function
                streamer = None
                if backend == 'api':
                    streamer = streamer_map.get('api', {}).get(provider)
                else:
                    streamer = streamer_map.get(backend)

                if not streamer:
                    raise ValueError(f"No valid streamer found for backend '{backend}' and provider '{provider}'")

                # Collect the full response (non-streaming for Parliament)
                # Some providers don't accept timezone parameter
                providers_with_timezone = ['google', 'openai', 'anthropic', 'deepseek']

                # CRITICAL: Pass None for model_id to skip memory manager initialization
                # Parliament doesn't need chat history/memories, and parallel loads cause PyTorch errors
                streamer_args = {
                    'model_instance': model,
                    'model_id_str': None,  # None = skip memory context loading
                    'user_input': role_prompt,
                    'conversation_history': [],  # Empty - no history for Parliament
                    'should_stop': lambda: False,
                    'image_data': None
                }

                if provider in providers_with_timezone:
                    streamer_args['timezone'] = 'UTC'

                response_chunks = []
                for chunk in streamer(**streamer_args):
                    # Extract text from chunk
                    if isinstance(chunk, dict):
                        token = chunk.get('token', '')
                    else:
                        token = str(chunk)
                    response_chunks.append(token)

                response_text = ''.join(response_chunks)

                print(f"✅ Parliament: {name} responded ({len(response_text)} chars)")

                # Assign confidence score based on provider
                # API models typically have higher confidence than local models
                confidence = 0.90 if provider.lower() in ['cloud', 'api', 'openai', 'anthropic', 'google', 'xai', 'qwen', 'deepseek', 'perplexity', 'openrouter', 'meta'] else 0.85

                result = {
                    'key': key,
                    'name': name,
                    'model': model,
                    'provider': provider,
                    'prompt': prompt,
                    'response': response_text,
                    'confidence': confidence,
                    'status': 'done'
                }

                return result

            except Exception as e:
                # Errors get lower confidence
                error_msg = str(e)
                print(f"❌ Parliament error for {name} ({provider}/{model}): {error_msg}")

                error_result = {
                    'key': key,
                    'name': name,
                    'model': model,
                    'provider': provider,
                    'prompt': prompt,
                    'response': f"Error: {error_msg}",
                    'confidence': 0.1,
                    'status': 'error'
                }

                return error_result

        # Send "working" status for all roles
        for role in roles:
            emit('parliament_update', {'key': role.get('key'), 'status': 'working'})

        # Execute all roles in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

        print(f"\n🏛️ Parliament: Starting parallel execution for {len(roles)} roles...")

        with ThreadPoolExecutor(max_workers=len(roles)) as executor:
            # Submit all tasks
            future_to_role = {executor.submit(process_role, role): role for role in roles}

            # Process results as they complete and emit updates
            # Timeout after 60 seconds per role to prevent hanging
            for future in as_completed(future_to_role, timeout=60):
                try:
                    result = future.result()

                    # Add to results list (thread-safe)
                    with results_lock:
                        results.append(result)

                    # Emit update from main thread (works properly here)
                    emit('parliament_update', {
                        'key': result['key'],
                        'status': result['status'],
                        'payload': result['response']
                    })

                except Exception as e:
                    role = future_to_role[future]
                    print(f"❌ Thread execution error for {role.get('name')}: {e}")

                    # Create error result
                    error_result = {
                        'key': role.get('key'),
                        'name': role.get('name'),
                        'model': role.get('model'),
                        'provider': role.get('provider'),
                        'prompt': role.get('prompt'),
                        'response': f"Thread error: {e}",
                        'confidence': 0.1,
                        'status': 'error'
                    }

                    with results_lock:
                        results.append(error_result)

                    emit('parliament_update', {
                        'key': error_result['key'],
                        'status': 'error',
                        'payload': error_result['response']
                    })

        print(f"✅ Parliament: All {len(results)} responses collected")

        # Perform voting on responses to find consensus
        vote_result = None
        if len(results) > 0:
            try:
                vote_result = parliament_voter.vote(results)
                print(f"Parliament voting complete: {vote_result['votes']} votes, {vote_result['total_clusters']} clusters")
            except Exception as e:
                print(f"Parliament voting failed: {e}")
                vote_result = {
                    'winning_answer': '',
                    'winning_model': '',
                    'confidence': 0.0,
                    'votes': 0,
                    'total_clusters': 0,
                    'cluster_details': []
                }

        merged_prompt = build_parliament_prompt(message, results)
        emit('parliament_summary', {
            'roles': results,
            'merged_prompt': merged_prompt,
            'vote_result': vote_result
        })

        if session_id:
            _save_message_to_db(session_id, {"sender": "Parliament", "message": merged_prompt, "type": "info"})

        # Now send the winning answer to the main model as context for final response
        if vote_result and vote_result.get('winning_answer'):
            print(f"\n🏛️ Building consensus context for main model...")
            print(f"   Winning model: {vote_result['winning_model']}")
            print(f"   Votes: {vote_result['votes']}/{len(results)}")
            print(f"   Confidence: {vote_result['confidence']:.2f}")

            consensus_context = f"""Parliament Consensus ({vote_result['votes']}/{len(results)} models agreed):
{vote_result['winning_answer']}

Original Question: {message}

Please provide your own response to the user's question, informed by the Parliament consensus above."""

            print(f"📤 Emitting parliament_complete event...")

            # Trigger regular chat with the consensus context
            # This will be handled by emitting a chat event back to trigger normal chat flow
            emit('parliament_complete', {
                'consensus_context': consensus_context,
                'original_message': message,
                'vote_result': vote_result
            })

            print(f"✅ Parliament complete event sent!")

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("\n" + "="*80)
        print("❌ PARLIAMENT ERROR")
        print("="*80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        print(error_trace)
        print("="*80)
        emit('parliament_update', {'status': 'error', 'message': str(e)})


def _default_usage_window(days: int):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    return {
        "from": start.isoformat() + "Z",
        "to": now.isoformat() + "Z",
        "totalCost": 0.0,
        "totalTokens": 0,
        "byDay": [],
        "byModel": [],
    }


@app.route('/api/usage/summary')
def get_usage_summary():
    """Return aggregated usage metrics for the given window."""
    try:
        days = int(request.args.get('days', 7))
    except Exception:
        days = 7

    if usage_events_collection is None:
        return jsonify(_default_usage_window(days))

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)

    events = list(usage_events_collection.find({"timestamp": {"$gte": start_ms}}))

    total_cost = 0.0
    total_tokens = 0
    by_day_map = {}
    by_model_map = {}

    for evt in events:
        tokens = int(evt.get("totalTokens", evt.get("inputTokens", 0) + evt.get("outputTokens", 0)))
        cost = float(evt.get("estimatedCost", 0) or 0)
        total_cost += cost
        total_tokens += tokens

        ts = evt.get("timestamp")
        if ts:
            day_key = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            if day_key not in by_day_map:
                by_day_map[day_key] = {"date": day_key, "totalTokens": 0, "totalCost": 0.0}
            by_day_map[day_key]["totalTokens"] += tokens
            by_day_map[day_key]["totalCost"] += cost

        model_key = evt.get("model", "unknown")
        provider = evt.get("provider", "unknown")
        key = f"{provider}:{model_key}"
        if key not in by_model_map:
            by_model_map[key] = {
                "model": model_key,
                "provider": provider,
                "totalTokens": 0,
                "totalCost": 0.0,
                "messages": 0,
            }
        by_model_map[key]["totalTokens"] += tokens
        by_model_map[key]["totalCost"] += cost
        by_model_map[key]["messages"] += 1

    return jsonify({
        "from": start.isoformat() + "Z",
        "to": now.isoformat() + "Z",
        "totalCost": round(total_cost, 6),
        "totalTokens": total_tokens,
        "byDay": sorted(by_day_map.values(), key=lambda x: x["date"]),
        "byModel": sorted(by_model_map.values(), key=lambda x: x["totalTokens"], reverse=True),
    })


@app.route('/api/usage/model/<path:model_name>')
def get_usage_for_model(model_name):
    """Return usage metrics for a specific model."""
    try:
        days = int(request.args.get('days', 7))
    except Exception:
        days = 7

    normalized_model = normalize_model_name(model_name)

    if usage_events_collection is None:
        return jsonify({
            "model": normalized_model,
            "provider": _detect_provider_from_model(model_name),
            "days": days,
            "totalTokens": 0,
            "totalCost": 0.0,
            "messages": 0,
            "avgTokensPerMessage": 0,
            "avgCostPerMessage": 0.0,
            "byDay": [],
        })

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)

    events = list(usage_events_collection.find({
        "timestamp": {"$gte": start_ms},
        "model": normalized_model
    }))

    total_tokens = 0
    total_cost = 0.0
    by_day_map = {}

    for evt in events:
        tokens = int(evt.get("totalTokens", evt.get("inputTokens", 0) + evt.get("outputTokens", 0)))
        cost = float(evt.get("estimatedCost", 0) or 0)
        total_tokens += tokens
        total_cost += cost

        ts = evt.get("timestamp")
        if ts:
            day_key = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            if day_key not in by_day_map:
                by_day_map[day_key] = {"date": day_key, "totalTokens": 0, "totalCost": 0.0}
            by_day_map[day_key]["totalTokens"] += tokens
            by_day_map[day_key]["totalCost"] += cost

    messages = len(events)
    provider = events[0].get("provider") if events else _detect_provider_from_model(normalized_model)
    avg_tokens = int(total_tokens / messages) if messages else 0
    avg_cost = round(total_cost / messages, 6) if messages else 0.0

    return jsonify({
        "model": normalized_model,
        "provider": provider,
        "days": days,
        "totalTokens": total_tokens,
        "totalCost": round(total_cost, 6),
        "messages": messages,
        "avgTokensPerMessage": avg_tokens,
        "avgCostPerMessage": avg_cost,
        "byDay": sorted(by_day_map.values(), key=lambda x: x["date"]),
    })


@app.route('/api/models/pricing/<path:model_name>')
def get_model_pricing(model_name):
    """Return pricing info for a given model using backend pricing table."""
    normalized_model = normalize_model_name(model_name)
    pricing = MODEL_PRICING.get(normalized_model, MODEL_PRICING.get('default', {}))
    return jsonify({
        "model": normalized_model,
        "pricing": pricing,
    })

def background_loop():
    logger.info("🔄 Maintenance thread started")
    while True:
        try:
            now = datetime.now()
            if now.hour == 3 and now.minute < 5:  # 3:00-3:05 AM window
                logger.info("🌙 Beginning nightly belief maintenance")
                
                # Get a stable copy of beliefs to iterate over
                current_beliefs = list(beliefs.get_all_beliefs())
                adjusted = 0
                
                for belief in current_beliefs:
                    if random.random() < 0.2:
                        # This is a placeholder for the actual update logic
                        # beliefs.collection.update_one({'_id': belief['_id']}, {'$mul': {'confidence': 0.8}})
                        logger.debug(f"Belief adjusted: {belief['content'][:50]}...")
                        adjusted += 1
                
                logger.info(f"🌙 Nightly maintenance complete. Adjusted {adjusted}/{len(current_beliefs)} beliefs")
            
            time.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.critical(f"Maintenance loop crashed: {str(e)}", exc_info=True)
            time.sleep(60)  # Wait before retry)

def migrate_and_load_config():
    """Loads config, migrates old keys, and sets environment variables."""
    global model_configs
    if not os.path.exists('config.json'):
        return

    with open('config.json', 'r') as f:
        model_configs = json.load(f)

    migrated = False
    # Migrate old gemini_api_key to new google_api_key
    if 'gemini_api_key' in model_configs:
        print("Migrating gemini_api_key to google_api_key...")
        model_configs['google_api_key'] = model_configs.pop('gemini_api_key')
        migrated = True

    # Set environment variables for all provider keys found
    for key, value in model_configs.items():
        if key.endswith("_api_key") and value:
            provider = key.replace("_api_key", "").upper()
            os.environ[f"{provider}_API_KEY"] = value
            print(f"Set env var for {provider}")

    if migrated:
        with open('config.json', 'w') as f:
            json.dump(model_configs, f, indent=4)
        print("Migration complete, config saved.")

def load_nova_settings():
    """Loads Nova customization settings from JSON file."""
    try:
        if os.path.exists('nova_settings.json'):
            with open('nova_settings.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load nova_settings.json: {e}")
    return {} # Return empty dict if file doesn't exist or fails to load

if __name__ == "__main__":
    migrate_and_load_config()
    nova_settings = load_nova_settings()
    load_orchestrator_model()

    # Start maintenance thread
    threading.Thread(target=background_loop, daemon=True, name="NovaMaintenance").start()

    # Debug: confirm static folder
    print(f"📂 Static folder being used: {app.static_folder}")

    # Run directly instead of in a thread
    print("🚀 Starting WebUI on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)

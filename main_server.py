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
from datetime import datetime
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
from flask import Flask, render_template, request, send_from_directory
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
from orchestrator import load_orchestrator_model, get_summary_for_title, parse_command, get_tool_call, summarize_text, get_orchestrator_response, score_response_confidence, summarize_for_memory, should_perform_web_search_intelligent
from tools import dispatch_tool, TOOLS_SCHEMA
import orchestrator as orchestrator_module
from orchestrator import build_parliament_prompt  # helper we will add for merging
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
    user_input = data.get("text", "")
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
    user_input = data.get('text', '')
    
    # --- Save User Message to DB ---
    session_id = data.get('session_id')
    if session_id:
        user_message = {
            "sender": data.get('userName', 'User'),
            "message": user_input,
            "type": "user",
            "imageB64": data.get('image_base_64')
        }
        _save_message_to_db(session_id, user_message)

    # --- Hybrid Orchestration Logic ---
    # 0. Allow forcing agentic mode with /agent prefix
    if auto_agent_enabled:
        stripped_input = user_input.strip()
        if stripped_input.startswith('/agent '):
            forced_query = stripped_input[7:].strip()
            if forced_query:
                data_for_agent = dict(data)
                data_for_agent['text'] = forced_query
                sid = request.sid
                socketio.start_background_task(run_agentic_chat_task, data_for_agent, sid, True)
                return

    # 1. Always check for slash commands first (instant)
    command = parse_command(user_input)
    if command:
        query = command.get('query')
        command_name = command.get('command')

        # List of commands handled by the special iframe/frontend logic
        frontend_commands = ['search']

        if command_name in frontend_commands:
            response_payload = {'type': 'error', 'message': f"Unknown command: {command_name}", 'sender': 'Nova'}
            try:
                if command_name == 'search':
                    encoded_query = urllib.parse.quote(query)
                    search_url = f"http://localhost:8088/?q={encoded_query}"
                    response_payload = {'type': 'iframe', 'url': search_url, 'message': f"Searching for: `{query}`", 'sender': 'Nova'}
                
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
    if use_agentic and detect_needs_agentic_approach(user_input):
        sid = request.sid
        socketio.start_background_task(run_agentic_chat_task, data, sid, False)
        return

    # 2. If no slash command, intelligently detect if web search is needed
    # Use intelligent detection (heuristics + Nova) as primary method
    search_decision = should_perform_web_search_intelligent(user_input)
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
        should_orchestrate = any(keyword in user_input.lower() for keyword in ORCHESTRATOR_KEYWORDS)
        if should_orchestrate:
            print(f"🔍 Web search triggered by keyword fallback")
    else:
        print(f"🧠 Web search triggered intelligently: {search_decision['reasoning']} (confidence: {search_decision['confidence']:.0%})")

    if should_orchestrate:
        print("Orchestration keywords detected. Checking for tool call...")
        tool_calls = get_tool_call(user_input)
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
                user_input = f"{forceful_instruction}\n\n--- Information ---\n{tool_result}\n--- End Information ---\n\nUser Query: {user_input}"
                # IMPORTANT: Update the 'text' in the data payload that gets passed to the streaming task.
                data['text'] = user_input
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

def stream_response(data, sid):
    global stop_streaming
    full_response = ""
    full_thought = ""
    session_id = data.get('session_id')
    in_thought_block = False
    current_sender = data.get('aiName', 'Nova')  # Get the actual AI name

    try:
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
        args = {
            "model_instance": current_model if backend != 'ollama' else current_model_path_string,
            "model_id_str": current_model_path_string,
            "user_input": data['text'],
            "conversation_history": data.get('history', []),
            "should_stop": lambda: stop_streaming,
            "image_data": data.get('image_base_64'),
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
        
        socketio.emit('stream_end', {'aiName': current_sender}, room=sid)
        
        # Fix: Use the actual sender name instead of hardcoding "Nova"
        if full_response and session_id:
            ai_message = {
                "sender": current_sender,  # Use the dynamic AI name
                "message": full_response,
                "type": "ai",
                "thought": full_thought
            }
            _save_message_to_db(session_id, ai_message)

        # MEMORY SUMMARIZATION: Extract key facts via Nova instead of storing raw text
        user_id = data.get('user_id', 'default_user')
        user_input = data.get('text', '')

        if full_response and user_input:
            # Let Nova extract concise facts from the conversation exchange
            facts = summarize_for_memory(user_input, full_response)

            if facts:
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

                print(f"💾 Stored {stored_count} summarized facts to memory")
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
        # Skip scoring for: Nova's own responses, empty responses, or command responses
        # (user_input already extracted above at line 866)
        is_nova_response = current_sender.lower() == 'nova'
        is_command = user_input.strip().startswith('/')

        if full_response and not is_nova_response and not is_command and len(full_response.strip()) > 10:
            try:
                confidence_result = score_response_confidence(full_response, user_input)
                score = confidence_result['score']
                reasoning = confidence_result['reasoning']
                should_search = confidence_result['should_search']

                # Log the confidence score
                if score < 50:
                    print(f"🔴 LOW CONFIDENCE ({score:.1f}%): {reasoning}")
                elif score < 80:
                    print(f"🟡 MODERATE CONFIDENCE ({score:.1f}%): {reasoning}")
                else:
                    print(f"🟢 HIGH CONFIDENCE ({score:.1f}%): {reasoning}")

                # If confidence is low, log recommendation for web search
                if should_search:
                    print(f"💡 RECOMMENDATION: Consider web search for query: '{user_input[:100]}'")
                    # TODO: In future, auto-trigger reactive web search here
                    # For now, just log it so we don't break anything

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

def _build_file_tree(path: Path, depth: int = 6, max_entries: int = 800):
    """
    Build a file tree up to a given depth and entry count.
    Simpler traversal to ensure files are not skipped.
    """
    root = path.resolve()
    entries_left = max_entries

    def walk(node: Path, current_depth: int):
        nonlocal entries_left
        if entries_left <= 0:
            return None
        try:
          is_dir = node.is_dir()
        except Exception:
          return None
        if is_dir:
            children = []
            if current_depth < depth:
                for child in sorted(node.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                    if entries_left <= 0:
                        break
                    if child.name.startswith('.'):
                        continue  # hide dotfiles
                    child_info = walk(child, current_depth + 1)
                    if child_info:
                        children.append(child_info)
            entries_left -= 1
            return {
                "name": node.name,
                "path": str(node.relative_to(WORKSPACE_ROOT)),
                "type": "dir",
                "children": children,
            }
        else:
            entries_left -= 1
            return {
                "name": node.name,
                "path": str(node.relative_to(WORKSPACE_ROOT)),
                "type": "file",
            }

    return walk(root, 0)

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
    tree = _build_file_tree(target, depth=4, max_entries=400)
    emit('file_tree', {'tree': [tree] if tree else []})

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
    try:
        content = target.read_text(encoding='utf-8')
        emit('file_content', {'path': str(path), 'content': content})
    except UnicodeDecodeError:
        emit('file_error', {'message': 'File is not UTF-8 text'})
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
    try:
        message = data.get('message', '')
        roles = data.get('roles', [])
        session_id = data.get('session_id')
        if not message or not roles:
            emit('parliament_summary', {'roles': [], 'merged_prompt': ''})
            return

        results = []

        for role in roles:
          key = role.get('key')
          prompt = role.get('prompt', '')
          model = role.get('model', 'default')
          provider = role.get('provider', 'cloud')
          name = role.get('name', key)
          emit('parliament_update', {'key': key, 'status': 'working'})
          try:
              # Use orchestrator to fetch a completion for this role
              role_prompt = f"{prompt}\n\nUser request:\n{message}\n\nReturn JSON only."
              response_text = orchestrator_module.get_orchestrator_response(
                  role_prompt,
                  model_override=model,
                  provider_override=provider,
                  stream=False
              )
              results.append({
                  'key': key,
                  'name': name,
                  'model': model,
                  'provider': provider,
                  'prompt': prompt,
                  'response': response_text
              })
              emit('parliament_update', {'key': key, 'status': 'done', 'payload': response_text})
          except Exception as e:
              results.append({
                  'key': key,
                  'name': name,
                  'model': model,
                  'provider': provider,
                  'prompt': prompt,
                  'response': f"Error: {e}"
              })
              emit('parliament_update', {'key': key, 'status': 'done', 'payload': f"Error: {e}"})

        merged_prompt = build_parliament_prompt(message, results)
        emit('parliament_summary', {'roles': results, 'merged_prompt': merged_prompt})

        if session_id:
            _save_message_to_db(session_id, {"sender": "Parliament", "message": merged_prompt, "type": "info"})

    except Exception as e:
        emit('parliament_update', {'status': 'error', 'message': str(e)})

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

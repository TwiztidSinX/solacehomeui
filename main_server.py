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
from datetime import datetime
import eventlet
import networkx as nx
import numpy as np
import pyaudio
import pvporcupine
import requests
import torch
import urllib.parse # <-- Add this import
from faster_whisper import WhisperModel
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from scipy.spatial import distance as ssd
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer
from model_loader import get_available_models, load_model, unload_model, stream_gpt
from upgraded_memory_manager import memory_manager as memory, beliefs_manager as beliefs, db
from orchestrator import load_orchestrator_model, get_summary_for_title, parse_command, get_tool_call, summarize_text
from tools import dispatch_tool, TOOLS_SCHEMA

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

def get_voice_state():
    return listening_enabled, speak_enabled

def set_voice_state(listen, speak):
    global listening_enabled, speak_enabled
    listening_enabled = listen
    speak_enabled = speak
    print(f"🎚️ Voice State Updated — Listening: {listen} | Speaking: {speak}")

app = Flask(__name__, static_folder='static/react', static_url_path='/')
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
    with open('voice_settings.json', 'r') as f:
        settings = json.load(f).get('tts', {})

    if settings.get('type') == 'local':
        url = settings.get('url', 'http://localhost:8880/v1/audio/speech')
        payload = {"text": text, "language": "en"}
        try:
            response = requests.post(url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4096):
                yield chunk
        except requests.exceptions.RequestException as e:
            print(f"❌ Local TTS error: {e}")
            yield b''
    elif settings.get('type') == 'cloud':
        provider = settings.get('provider', 'openai') # Default to openai if not specified
        if provider == 'openai':
            api_key = settings.get('apiKey')
            model = settings.get('model', 'tts-1')
            voice = settings.get('voice', 'alloy')
            url = 'https://api.openai.com/v1/audio/speech'
            headers = {'Authorization': f'Bearer {api_key}'}
            payload = {'model': model, 'input': text, 'voice': voice, 'response_format': 'mp3'}
            try:
                response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=4096):
                    yield chunk
            except requests.exceptions.RequestException as e:
                print(f"❌ OpenAI TTS error: {e}")
                yield b''
        elif provider == 'google':
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            yield response.audio_content
    else:
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
            if current_model:
                unload_model(current_model_path_string, backend=current_backend)
            
            # The 'data' dictionary from the frontend now contains all necessary arguments.
            # We can pass it directly using the **kwargs syntax.
            current_model = load_model(**data)
            
            current_model_path_string = data['model_path']
            
            socketio.emit('model_loaded', {'model': data['model_path']}, room=sid)
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

    def run_agent_task():
        full_thought = ""
        final_response = ""
        try:
            socketio.emit('stream_start', {}, room=sid)
            socketio.sleep(0)

            # 1. Get the list of tool calls from the orchestrator
            tool_calls = get_tool_call(user_input)

            socketio.emit('stream', '<think>', room=sid)
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
                    final_response += result_text # Aggregate results
                    socketio.emit('stream', result_text, room=sid)
                    socketio.sleep(0)

            # 4. Stream the final aggregated response
            socketio.emit('stream', '</think>', room=sid) # Close thinking block
            # We don't need to stream the final response again as it was streamed step-by-step
            socketio.sleep(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"An error occurred while running the agent command: {e}"
            # Ensure think tag is closed on error, if it was ever opened
            socketio.emit('stream', '</think>', room=sid) 
            socketio.emit('stream', error_message, room=sid)
            final_response = error_message
        finally:
            socketio.emit('stream_end', {}, room=sid)
            # --- Save Agent's Final Response to DB ---
            if session_id and final_response:
                ai_message = {
                    "sender": "Nova", # Or a dedicated agent name
                    "message": final_response,
                    "type": "ai",
                    "thought": full_thought
                }
                _save_message_to_db(session_id, ai_message)

    # Run the agent task in the background
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
    # 1. Always check for slash commands first (instant)
    command = parse_command(user_input)
    if command:
        query = command['query']
        response_payload = {'type': 'error', 'message': f"Unknown command: {command['command']}", 'sender': 'Nova'}
        try:
            # ... (existing command handling logic) ...
            if command['command'] == 'search':
                encoded_query = urllib.parse.quote(query)
                search_url = f"http://localhost:8088/?q={encoded_query}"
                response_payload = {'type': 'iframe', 'url': search_url, 'message': f"Searching for: `{query}`", 'sender': 'Nova'}
            
            elif command['command'] == 'youtube':
                encoded_query = urllib.parse.quote(f"!yt {query}")
                search_url = f"http://localhost:8088/search?q={encoded_query}&format=json"
                response = requests.get(search_url, timeout=10).json()
                first_video = next((r for r in response.get('results', []) if 'youtube.com/watch' in r.get('url', '')), None)
                if first_video:
                    video_id = first_video['url'].split('v=')[1].split('&')[0]
                    response_payload = {'type': 'youtube_embed', 'video_id': video_id, 'message': f"Here is the top YouTube result for `{query}`:", 'sender': 'Nova'}
                else:
                    response_payload = {'type': 'error', 'message': f"No YouTube results found for '{query}'.", 'sender': 'Nova'}
            
            elif command['command'] == 'read':
                response = requests.get(query, timeout=15).text
                clean_text = re.sub(r'<style.*?>.*?</style>|<script.*?>.*?</script>|<!--.*?-->', '', response, flags=re.DOTALL)
                clean_text = re.sub(r'<.*?>', ' ', clean_text)
                clean_text = ' '.join(clean_text.split())
                summary = summarize_text(clean_text[:8000])
                response_payload = {'type': 'info', 'message': f"**Summary of {query}:**\n\n{summary}", 'sender': 'Nova'}

            elif command['command'] == 'calc':
                answer = summarize_text(f"Calculate the following expression and provide only the numerical answer: {query}")
                response_payload = {'type': 'info', 'message': f"`{query}` = **{answer}**", 'sender': 'Nova'}

        except requests.exceptions.RequestException as e:
            response_payload = {'type': 'error', 'message': f"Command failed due to a network error: {e}", 'sender': 'Nova'}
        except Exception as e:
            response_payload = {'type': 'error', 'message': f"An unexpected error occurred: {e}", 'sender': 'Nova'}
        
        emit('command_response', response_payload)
        if session_id:
            _save_message_to_db(session_id, {"sender": "Nova", **response_payload, "type": "ai"})
        return # Stop further processing

    # 2. If no slash command, check for keywords (fast)
    ORCHESTRATOR_KEYWORDS = [
        "search", "what is", "who is", "how to", "current", "latest", "news", 
        "weather", "price of", "stock", "define", "wiki", "map of", 
        "look up", "find me", "scrape", "read the article"
    ]
    
    # More robust check: see if any keyword is a substring of the user input
    should_orchestrate = any(keyword in user_input.lower() for keyword in ORCHESTRATOR_KEYWORDS)

    if should_orchestrate:
        print("Orchestration keywords detected. Checking for tool call...")
        tool_call = get_tool_call(user_input)
        if tool_call:
            # A tool was selected by the orchestrator
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments")
            
            # Execute the tool
            tool_result = dispatch_tool(tool_name, tool_args)
            
            # Prepend the result to the user's input for the main model
            user_input = f"--- Tool Result for '{tool_name}' ---\n{tool_result}\n--- End Result ---\n\nPlease use this information to answer my original query: {user_input}"
            # Update the 'text' in the data payload for the streamer
            data['text'] = user_input

    # 3. If neither, bypass orchestrator and go directly to the main model
    if not current_model:
        emit('error', {'message': 'No model loaded.'})
        return
    
    # Get the config for the current model to pass thinking level
    if current_model_path_string in model_configs:
        data['thinking_level'] = model_configs[current_model_path_string].get('thinking_level', 'medium')

    sid = request.sid
    socketio.start_background_task(stream_response, data, sid)

def stream_response(data, sid):
    global stop_streaming
    full_response = ""
    full_thought = ""
    local_backends = ['llama.cpp', 'ollama', 'safetensors']
    backend = data.get('backend', current_backend)
    provider = data.get('provider')
    timezone = data.get('timezone', 'UTC')
    session_id = data.get('session_id')
    in_thought_block = False

    # --- Debug Mode ---
    if data.get('debug_mode'):
        print("\n--- DEBUG MODE: INCOMING PAYLOAD ---")
        # Avoid printing massive base64 image strings
        debug_data = {k: v for k, v in data.items() if k not in ['image_base_64']}
        if data.get('image_base_64'):
            debug_data['image_base_64'] = "Image data present (omitted for brevity)"
        print(json.dumps(debug_data, indent=2))
        print("-------------------------------------\n")
    # --- End Debug Mode ---

    try:
        socketio.emit('stream_start', {}, room=sid)
        socketio.sleep(0)
        
        model_instance = current_model
        if backend == "ollama":
            model_instance = current_model_path_string

        if backend in local_backends:
            model_response_generator = stream_gpt(
                model_instance, current_model_path_string, data['text'], 
                conversation_history=data.get('history', []),
                should_stop=lambda: stop_streaming,
                backend=backend, provider=provider, image_data=data.get('image_base_64'),
                timezone=timezone, tools=TOOLS_SCHEMA,
                debug_mode=data.get('debug_mode', False),
                thinking_level=data.get('thinking_level', 'medium') # Pass thinking level
            )

            tool_calls = []
            for chunk in model_response_generator:
                chunk_type = chunk.get('type')
                token = chunk.get('token', '')

                if chunk_type == 'tool_call':
                    tool_calls.append(chunk.get('tool_call'))
                elif chunk_type == 'thought':
                    if not in_thought_block:
                        socketio.emit('stream', '<think>', room=sid)
                        in_thought_block = True
                    full_thought += token
                    socketio.emit('stream', token, room=sid)
                elif chunk_type == 'reply':
                    if in_thought_block:
                        socketio.emit('stream', '</think>', room=sid)
                        in_thought_block = False
                    full_response += token
                    socketio.emit('stream', token, room=sid)
                
                socketio.sleep(0)
                if stop_streaming: break
            
            if in_thought_block: # Close any open thought tag
                socketio.emit('stream', '</think>', room=sid)
                in_thought_block = False
            
            if tool_calls:
                # Handle tool calls...
                pass
        else: # API Backends
            streamer = stream_gpt(
                model_instance, current_model_path_string, data['text'], 
                conversation_history=data.get('history', []),
                should_stop=lambda: stop_streaming,
                backend=backend, provider=provider, image_data=data.get('image_base_64'),
                timezone=timezone,
                debug_mode=data.get('debug_mode', False)
            )
            for chunk in streamer:
                chunk_type = chunk.get('type')
                token = chunk.get('token', '')

                if chunk_type == 'thought':
                    if not in_thought_block:
                        socketio.emit('stream', '<think>', room=sid)
                        in_thought_block = True
                    full_thought += token
                    socketio.emit('stream', token, room=sid)
                elif chunk_type == 'reply':
                    if in_thought_block:
                        socketio.emit('stream', '</think>', room=sid)
                        in_thought_block = False
                    full_response += token
                    socketio.emit('stream', token, room=sid)

                socketio.sleep(0)
                if stop_streaming: break
            
            if in_thought_block: # Close any open thought tag
                socketio.emit('stream', '</think>', room=sid)
                in_thought_block = False

        socketio.emit('stream_end', {}, room=sid)
    except Exception as e:
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f"Stream error: {str(e)}"}, room=sid)
    finally:
        with stop_lock:
            stop_streaming = False
        
        if full_response and session_id:
            ai_message = {
                "sender": data.get('aiName', 'Solace'),
                "message": full_response,
                "type": "ai",
                "thought": full_thought
            }
            _save_message_to_db(session_id, ai_message)

        if data.get('text'):
            memory.learn_from_text(data['text'], source="user", model_id=current_model_path_string)
        if full_response:
            memory.learn_from_text(full_response, source="nova", model_id=current_model_path_string)

@socketio.on('stop')
def handle_stop():
    global stop_streaming
    with stop_lock:
        stop_streaming = True
    print("🛑 Stop request received.")

def unload_model_task(sid):
    """Background task to unload a model without blocking."""
    global current_model, current_model_path_string
    with model_lock:
        if current_model_path_string:
            model_path_to_unload = current_model_path_string
            backend_to_unload = current_backend
            if unload_model(model_path_to_unload, backend=backend_to_unload):
                current_model = None
                current_model_path_string = None
                socketio.emit('model_unloaded', {'model': model_path_to_unload}, room=sid)
                print(f"✅ Model unloaded and globals cleared: {model_path_to_unload}")
            else:
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
        fp.write(data['audio'])
        audio_path = fp.name
    
    transcription = transcribe_audio(audio_path)
    os.remove(audio_path)
    emit('transcription_result', {'text': transcription})

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

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

    threading.Thread(target=background_loop, daemon=True, name="NovaMaintenance").start()

    web_ui_thread = threading.Thread(target=lambda: socketio.run(app, host='0.0.0.0', port=5000), daemon=True)
    web_ui_thread.start()
    print("WebUI server started in background thread.")

    webbrowser.open("http://localhost:5000")

    while True:
        time.sleep(1)
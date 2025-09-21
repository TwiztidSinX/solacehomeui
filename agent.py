import json
import os
import random
import re
import time
from typing import List
from datetime import datetime
from threading import Thread

import pyautogui
import pytesseract
from PIL import Image

from upgraded_memory_manager import analyze_emotion

from abilities import run_tool
from abilities import run_tool
from model_loader import stream_gpt, models

IDLE_BANTER_FILE = "idle_banter.json"

def get_random_line_from(category: str):
    """
    Retrieves a random line from a specified category in the idle_banter.json file.
    """
    if os.path.exists(IDLE_BANTER_FILE):
        with open(IDLE_BANTER_FILE, "r") as f:
            data = json.load(f)
            lines = data.get(category, [])
            if lines:
                return random.choice(lines)
    return "..."

_global_tom = None

def init_cognitive_loop(user_id: str):
    """Initialize the cognitive loop with a user profile"""
    global _global_tom
    from upgraded_memory_manager import ToMProfile, memory_manager
    _global_tom = ToMProfile(memory_manager.db, user_id)

def process_interaction(user_input: str, nova_response: str):
    """
    Processes an interaction between user and Nova, updating:
    - Theory of Mind profile
    - Emotional analysis
    - Belief systems
    - Pattern recognition
    """
    if not _global_tom:
        raise RuntimeError("Cognitive loop not initialized. Call init_cognitive_loop() first.")
    
    try:
        # Unified interaction processing
        interaction_summary = f"User: {user_input[:50]}... Nova: {nova_response[:50]}..."
        _global_tom.update_from_interaction(interaction_summary)
        
        # Emotional analysis
        user_emotion = analyze_emotion(user_input)
        
        # Pattern detection
        if user_emotion["vader_mood"] in ["angry", "frustrated"]:
            _global_tom.observe_pattern(
                "user_frustration",
                f"User frustrated when Nova responds: {nova_response[:30]}..."
            )
        
        return True
        
    except Exception as e:
        print(f"COG LOOP FAILURE: {str(e)}")
        return False

TASK_FILE = "tasks.json"

def get_task_context():
    """
    Retrieves the current tasks from the tasks.json file.
    """
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            return json.load(f)
    return []

INTENT_PATTERNS = [
    (r"\bscroll (down|more|further)\b", "scroll_down", None),
    (r"\bscroll up\b", "scroll_up", None),
    (r"\bchatgpt\b.*(focus|open|switch)", "focus_window", "ChatGPT"),
    (r"\bwhat windows.*open\b", "list_windows", None),
    (r"\bwindow.*info\b", "get_active_window_info", None),
    (r"\b(type|write).*(screen|chat)\b\s*(.*)", "type_screen_text", 3),
    (r"\b(read|extract).*screen", "read_screen_text", None),
    (r"\b(search|look up|find)\b\s*(for\s*)?(.*)", "web_search", 3),
    (r"\bhow (hot|cold|windy).*outside\b", "web_search", 0), # The whole match is the query
    (r"\bwho (sings|wrote|is the artist of)\b\s*(.*)", "web_search", 2)
]

def extract_intent_and_execute(text: str):
    """
    Looks for patterns in the text and executes the corresponding tool.
    """
    text_lower = text.strip().lower()
    for pattern, tool_name, arg_group in INTENT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            print(f"🧠 Intent matched: '{tool_name}' from pattern '{pattern}'")
            
            argument = ""
            if arg_group is not None:
                if arg_group == 0:
                    argument = match.group(0) # Full match
                else:
                    argument = match.group(arg_group).strip()
            
            # If the argument is still empty for a search, use the whole text
            if tool_name == "web_search" and not argument:
                argument = text

            try:
                return run_tool(tool_name, argument)
            except Exception as e:
                return f"❌ Tool '{tool_name}' failed: {str(e)}"
    return None

AUTO_AGENT_ENABLED = False  # Disabled by default
_auto_thread = None

def start_auto_loop_once():
    """
    Starts the agent's autonomous loop in a separate thread if it's not already running.
    """
    global _auto_thread
    if not (_auto_thread and _auto_thread.is_alive()):
        _auto_thread = Thread(target=agent_loop, daemon=True)
        _auto_thread.start()
        print("🧠 Auto agent loop started.")

def agent_loop():
    """
    The main autonomous loop for the agent.
    """
    print("✅ Nova Auto-Agent initialized.")
    while True:
        if not AUTO_AGENT_ENABLED:
            time.sleep(10)
            continue

        # 1. Recall recent memories
        recent_memories = memory.recall("recent events", limit=10)
        
        # 2. Get current tasks
        tasks = get_task_context()
        
        # 3. Formulate a thought
        prompt = f"Recent memories: {recent_memories}\nCurrent tasks: {tasks}\nWhat should I do next?"
        
        # 4. Get a response from the model
        if models:
            model_instance = next(iter(models.values()))
            model_id_str = next(iter(models.keys()))
            response_generator = stream_gpt(model_instance, model_id_str, prompt)
            response = "".join(response_generator)
            
            # 5. Parse the response for actions
            # This is a simplified example. A more robust implementation would use a tool-calling model.
            if "search_web" in response:
                query = response.split("search_web(")[1].split(")")[0]
                run_tool("search_web", query)
            elif "open_url" in response:
                url = response.split("open_url(")[1].split(")")[0]
                run_tool("open_url", url)

        time.sleep(30) # Wait before the next loop

def set_auto_agent_enabled(enabled: bool):
    """
    Enables or disables the auto agent loop.
    """
    global AUTO_AGENT_ENABLED
    AUTO_AGENT_ENABLED = enabled
    print(f"🤖 Auto agent loop {'enabled' if enabled else 'disabled'}.")

def simulate_responses(prompt: str, context: List[dict], num_alternatives: int = 3) -> List[str]:
    """
    Simulate alternative responses Nova could say.
    Placeholder logic 
    can later use real LLM scoring.
    """
    options = [
        f"One way to look at it is: {context[0]['content'] if context else '...'}",
        f"If I were to approach this differently, I'd say: {prompt[::-1][:60]}...",
        f"Another angle: {context[1]['content'] if len(context) > 1 else '...'}"
    ]
    return options[:num_alternatives]



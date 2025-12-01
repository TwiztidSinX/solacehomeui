import json
import os
import urllib.parse
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import time
import pygetwindow as gw
import pyautogui
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from gnews import GNews
from bson import ObjectId
import platform
from typing import Optional, Dict, Any
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
try:
    import psutil
except ImportError:
    psutil = None

# Import tool state tracking and introspection tools
from tool_state import record_tool_execution, log_missing_tool_request
from tools_introspection import INTROSPECTION_TOOLS_SCHEMA, INTROSPECTION_TOOL_REGISTRY
# Memory manager
try:
    from upgraded_memory_manager import memory_manager as memory
    import upgraded_memory_manager as upgraded_memory_module
except Exception:
    memory = None
    upgraded_memory_module = None

# Orchestrator summarization helper (used for research summaries)
try:
    from orchestrator import summarize_text
except Exception:
    summarize_text = None

from agentic_coding import CODING_TOOLS_SCHEMA
from agent_coding_socket import get_coding_tool_registry

# Path for dynamically registered tools
DYNAMIC_TOOLS_PATH = Path(__file__).resolve().parent / 'dynamic_tools.json'
# Cache of dynamic callables so we can rebuild the registry without leaking old defs
_DYNAMIC_TOOL_FUNCS = {}

# --- Tool Definitions (for the model) ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "direct_chat",
            "description": "ONLY use this when: (1) No main model is currently loaded, OR (2) The user explicitly asks to speak with Nova/the orchestrator. This bypasses the main model and uses Nova (the orchestrator) to respond directly. For all normal chat, questions, and conversations, let the main model handle it by NOT calling any tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The user's message to respond to."
                    }
                },
                "required": ["message"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Performs a web search using DuckDuckGo for general queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Searches for recent news articles when the user asks about current events, news, or headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for in the news.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrapes the text content of a given URL. Useful for summarizing articles or getting information from a specific webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_open_windows",
            "description": "Lists the titles of all open windows on the desktop.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_window",
            "description": "Brings a window to the foreground based on a substring of its title. Use 'list_open_windows' first to see available window titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_substring": {
                        "type": "string",
                        "description": "A substring of the title of the window to focus (e.g., 'notepad', 'chrome').",
                    }
                },
                "required": ["title_substring"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_down",
            "description": "Scrolls the active window down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "integer",
                        "description": "The number of pixels to scroll down. Defaults to 500.",
                        "default": 500
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_up",
            "description": "Scrolls the active window up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "integer",
                        "description": "The number of pixels to scroll up. Defaults to 500.",
                        "default": 500
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_window_info",
            "description": "Gets information about the currently active window, such as its title, size, and position.",
            "parameters": { "type": "object", "properties": {}, "required": [] },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_screen_text",
            "description": "Reads text from the entire screen using OCR. Useful for understanding the content of the current view.",
            "parameters": { "type": "object", "properties": {}, "required": [] },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_screen_text",
            "description": "Types the given text into the currently active window, character by character.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates an image using AI image generation models like Stable Diffusion via ComfyUI or Automatic1111.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt describing the image to generate.",
                    },
                    "model": {
                        "type": "string",
                        "description": "The specific model to use for generation.",
                        "default": "default"
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Store a text snippet into long-term memory with optional metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "source": {"type": "string", "default": "tool"},
                    "model_id": {"type": "string"},
                    "metadata": {"type": "object"}
                },
                "required": ["content"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search long-term memory for relevant entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a specific memory by id or fetch the most recent memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 3}
                }
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "MongoDB _id of the memory to delete."}
                },
                "required": ["memory_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": "Lists running processes (name and pid).",
            "parameters": { "type": "object", "properties": {} }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kill_process",
            "description": "Kills a process by pid or name substring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": { "type": "integer" },
                    "name": { "type": "string" }
                }
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_subsystem",
            "description": "Restarts a known subsystem (ollama, searxng, tts, llama.cpp) by stopping related processes. Start commands are stubbed and should be wired per environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Subsystem name (ollama|searxng|tts|llama.cpp)" }
                },
                "required": ["name"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "environment_info",
            "description": "Returns OS, python version, and basic hardware info.",
            "parameters": { "type": "object", "properties": {} }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists files in a directory (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current working directory.",
                        "default": "."
                    }
                }
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads a text file and returns its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read."
                    }
                },
                "required": ["path"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes text to a file (overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write."
                    }
                },
                "required": ["path", "content"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Appends text to a file (creates if missing).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append."
                    }
                },
                "required": ["path", "content"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Deletes a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete."
                    }
                },
                "required": ["path"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Searches for a string in files under a directory (recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search. Defaults to current working directory.",
                        "default": "."
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for."
                    }
                },
                "required": ["query"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python snippet in a subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "Python code to run." },
                    "timeout": { "type": "number", "default": 30 }
                },
                "required": ["code"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetches the content of a URL (HTML) for downstream summarization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                },
                "required": ["url"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_summary",
            "description": "Summarizes the content of a fetched URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                },
                "required": ["url"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Performs multi-step research: web search + scrape + summarize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "depth": { "type": "integer", "default": 3 },
                    "time_limit": { "type": "integer", "default": 60 }
                },
                "required": ["query"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_javascript",
            "description": "Executes JavaScript using node (if available) and returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": { "type": "string" },
                    "timeout": { "type": "number", "default": 20 }
                },
                "required": ["code"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_audio",
            "description": "Calls the voice server TTS endpoint to synthesize speech.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "voice": { "type": "string", "description": "Optional voice/speaker id" }
                },
                "required": ["text"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Stub for image analysis (returns a placeholder unless wired to a vision API).",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": { "type": "string" },
                    "prompt": { "type": "string", "description": "What to analyze" }
                },
                "required": ["image_url"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stream_audio_chunk",
            "description": "Placeholder: would stream audio chunks to frontend; currently returns a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_b64": { "type": "string", "description": "Base64-encoded audio chunk" }
                },
                "required": ["chunk_b64"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_environment_audio",
            "description": "Placeholder: would start recording mic; currently returns a stub message.",
            "parameters": { "type": "object", "properties": {} }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribes an audio file using local Whisper configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to audio file to transcribe" }
                },
                "required": ["path"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_goal",
            "description": "Sets a high-level goal for the current session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": { "type": "string" },
                    "session_id": { "type": "string" }
                },
                "required": ["goal", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Updates the current multi-step plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": { "type": "array", "items": { "type": "string" } },
                    "session_id": { "type": "string" }
                },
                "required": ["plan", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reflect",
            "description": "Stores a reflection or note about performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": { "type": "string" },
                    "session_id": { "type": "string" }
                },
                "required": ["note", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_answer",
            "description": "Revises a prior answer with new guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": { "type": "string" },
                    "session_id": { "type": "string" }
                },
                "required": ["answer", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Finalizes the current response/plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": { "type": "string" },
                    "session_id": { "type": "string" }
                },
                "required": ["summary", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_state",
            "description": "Returns current system state (backend, model, settings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" }
                }
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_system_state",
            "description": "Sets simple system flags (currently stub).",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": { "type": "object" },
                    "session_id": { "type": "string" }
                },
                "required": ["state", "session_id"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_settings",
            "description": "Returns nova_settings.json content.",
            "parameters": { "type": "object", "properties": {} }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_settings",
            "description": "Saves nova_settings.json content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "settings": { "type": "object" }
                },
                "required": ["settings"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jellyfin_control",
            "description": "Controls Jellyfin playback (play/pause) via embed URL generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "description": "play|pause|info" },
                    "media_id": { "type": "string" }
                },
                "required": ["action"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comfyui_generate",
            "description": "Triggers a ComfyUI generation via existing workflow stub (uses imageGenUrl).",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string" },
                    "width": { "type": "integer", "default": 512 },
                    "height": { "type": "integer", "default": 512 }
                },
                "required": ["prompt"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "searx_query",
            "description": "Alias to search_web for agents that expect searx_query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_iframe",
            "description": "Returns an embeddable iframe URL for the frontend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                },
                "required": ["url"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Runs a shell command and returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run."
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in seconds.",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abort",
            "description": "Immediately aborts the current agentic flow.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_tool_params",
            "description": "Validates parameters for a proposed tool call and echoes them back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": { "type": "string" },
                    "params": { "type": "object" }
                },
                "required": ["tool_name", "params"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_media",
            "description": "Browses and plays media content from Jellyfin/Plex/Emby media servers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The media category to browse (movies, tvshows, music, etc).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific title or search query for media.",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Clicks an element on the current browser page. Use CSS selectors to target elements (e.g., 'button.submit', '#login-btn', 'a[href=\"/tickets\"]'). The browser must already be open via /browser command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the element to click (e.g., 'button.buy-now', '#submit', 'a.movie-title')",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill",
            "description": "Fills a form field on the current browser page. Works with input, textarea, and select elements. Triggers input and change events for compatibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the form field (e.g., 'input[name=\"email\"]', '#password', 'select[name=\"quantity\"]')",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to enter into the field",
                    }
                },
                "required": ["selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_content",
            "description": "Gets the current page's content including title, URL, visible text, and HTML. Use this to understand what's on the page before taking actions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_links",
            "description": "Gets all clickable links on the current page with their text and URLs. Useful for navigation and finding specific links.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_form_fields",
            "description": "Gets all form fields (inputs, selects, textareas) on the current page with their names, types, values, IDs, and placeholders. Use this to understand what information a form needs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll_to",
            "description": "Scrolls to a specific element on the page, bringing it into view. Useful for accessing elements that are off-screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the element to scroll to",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_exec",
            "description": "Executes custom JavaScript code in the browser page context. Advanced tool for complex interactions. Returns the result of the code execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "JavaScript code to execute (e.g., 'document.querySelectorAll(\".item\").length')",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigates the browser to a new URL. Opens the browser panel if not already open.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (e.g., 'https://example.com', 'google.com')",
                    }
                },
                "required": ["url"],
            },
        },
    }
] + INTROSPECTION_TOOLS_SCHEMA  # Add introspection tools dynamically
# Agentic coding tool schema (Nova-style coding agent)
TOOLS_SCHEMA.extend(CODING_TOOLS_SCHEMA)

# --- Dynamic tools storage ---

def _load_dynamic_tools_from_disk() -> list:
    """
    Load dynamic tool definitions from disk.
    """
    if not DYNAMIC_TOOLS_PATH.exists():
        return []
    try:
        return json.loads(DYNAMIC_TOOLS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load dynamic tools: {e}")
        return []

def _persist_dynamic_tools(definitions: list):
    """
    Persist dynamic tool definitions to disk.
    """
    try:
        DYNAMIC_TOOLS_PATH.write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Failed to persist dynamic tools: {e}")

def _compile_dynamic_callable(name: str, code: str, handler_name: Optional[str] = None):
    """
    Compile a dynamic tool from source code into a callable.
    Expects a function named `handler` or the provided handler_name/name.
    """
    namespace: dict = {}
    exec(code, namespace)
    func = None
    if handler_name and callable(namespace.get(handler_name)):
        func = namespace[handler_name]
    elif callable(namespace.get("handler")):
        func = namespace["handler"]
    elif callable(namespace.get(name)):
        func = namespace[name]
    if not callable(func):
        raise ValueError(f"No callable found for dynamic tool '{name}'. Define a function named 'handler' or '{name}'.")
    return func

def _register_dynamic_callable(name: str, func):
    """
    Attach a callable to the registry and cache.
    """
    _DYNAMIC_TOOL_FUNCS[name] = func
    TOOL_REGISTRY[name] = func

def register_dynamic_tool(name: str, description: str, parameters: dict, code: str, handler_name: Optional[str] = None):
    """
    Register a new tool at runtime, persist it, and expose it to the registry/schema.
    """
    # Compile first to ensure the code is valid before persisting
    func = _compile_dynamic_callable(name, code, handler_name)

    # Persist definition
    current_defs = _load_dynamic_tools_from_disk()
    # Replace existing entry if same name
    filtered = [d for d in current_defs if d.get("function", {}).get("name") != name]
    new_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
        "code": code,
        "handler_name": handler_name,
    }
    filtered.append(new_def)
    _persist_dynamic_tools(filtered)

    # Update in-memory schema and registry
    # Drop any prior schema entries with same name
    global TOOLS_SCHEMA
    TOOLS_SCHEMA = [t for t in TOOLS_SCHEMA if t.get("function", {}).get("name") != name]
    TOOLS_SCHEMA.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
    _register_dynamic_callable(name, func)
    return {"status": "ok", "name": name}

def load_dynamic_tools():
    """
    Load and register all dynamic tools from disk.
    """
    dynamic_defs = _load_dynamic_tools_from_disk()
    for entry in dynamic_defs:
        fn = entry.get("function", {})
        code = entry.get("code", "")
        handler_name = entry.get("handler_name")
        name = fn.get("name")
        description = fn.get("description", "")
        parameters = fn.get("parameters", {"type": "object", "properties": {}, "required": []})
        if not name:
            continue
        try:
            func = _compile_dynamic_callable(name, code, handler_name)
            _register_dynamic_callable(name, func)
            # Ensure schema contains the dynamic entry (avoid duplicates)
            if not any(t.get("function", {}).get("name") == name for t in TOOLS_SCHEMA):
                TOOLS_SCHEMA.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    }
                })
        except Exception as e:
            print(f"Failed to register dynamic tool '{name}': {e}")


# --- Tool Implementations (for the backend) ---

def direct_chat(message: str):
    """
    A placeholder for direct chat. The orchestrator will handle this specially.
    """
    # The orchestrator should see this and just return the response to the user.
    # This function's return value might not even be used.
    return message

def search_web(query: str):
    """
    Performs a web search using the user's configured SearXNG instance.
    """
    print(f"Executing web search for: {query}")
    try:
        # Load SearXNG URL from settings
        settings_path = os.path.join(os.path.dirname(__file__), 'nova_settings.json')
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        searxng_url = settings.get('searxngUrl')

        if not searxng_url:
            return "Error: SearXNG URL is not configured in settings."

        # Construct the full API URL
        api_url = f"{searxng_url.rstrip('/')}/search?q={urllib.parse.quote(query)}&format=json"
        
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        results = response.json()

        # Always return snippets for now
        snippets = [r.get('content', '') for r in results.get('results', [])[:5]]
        return "\n".join(snippets) if snippets else "No results found."

    except Exception as e:
        print(f"Web search tool failed: {e}")
        return f"Error searching web: {e}"

def search_news(query: str):
    """
    Searches for news articles using GNews.
    """
    print(f"Executing news search for: {query}")
    try:
        gnews = GNews()
        results = gnews.get_news(query)
        # Format the results into a readable string for the model
        formatted_results = [f"Title: {r['title']}\nSource: {r['publisher']['title']}\nURL: {r['url']}" for r in results[:3]] # Get top 3
        return "\n\n".join(formatted_results) if formatted_results else "No news found on that topic."
    except Exception as e:
        print(f"News search tool failed: {e}")
        return f"Error searching news: {e}"

def scrape_website(url: str):
    """
    Scrapes the text content of a website.
    """
    print(f"Executing scrape for URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Get text and remove excessive newlines
        text = ' '.join(soup.get_text().split())
        return text[:8000] # Return the first 8000 characters to avoid context overload
    except Exception as e:
        print(f"Scrape website tool failed: {e}")
        return f"Error scraping website: {e}"

def list_open_windows():
    """
    Lists the titles of all open (non-empty) windows.
    """
    print("Executing list_open_windows")
    try:
        windows = gw.getAllTitles()
        # Filter out empty strings and join into a single newline-separated string
        return "\n".join([w for w in windows if w])
    except Exception as e:
        print(f"List open windows tool failed: {e}")
        return f"Error listing windows: {e}"

def focus_window(title_substring: str):
    """
    Brings a window to the foreground.
    """
    print(f"Executing focus_window with substring: {title_substring}")
    try:
        windows = gw.getWindowsWithTitle(title_substring)
        if not windows:
            return f"Error: No window found with title containing '{title_substring}'."
        # Get the first matching window and activate it
        win = windows[0]
        # A more reliable way to bring a window to the front on modern Windows
        win.minimize()
        win.restore()
        return f"Successfully focused window: {win.title}"
    except Exception as e:
        print(f"Focus window tool failed: {e}")
        return f"Error focusing window: {e}"

def scroll_down(pixels: int = 500):
    """
    Scrolls the active window down.
    """
    print(f"Executing scroll_down by {pixels} pixels")
    try:
        pyautogui.scroll(-pixels)
        return "Scrolled down."
    except Exception as e:
        print(f"Scroll down tool failed: {e}")
        return f"Error scrolling down: {e}"

def scroll_up(pixels: int = 500):
    """
    Scrolls the active window up.
    """
    print(f"Executing scroll_up by {pixels} pixels")
    try:
        pyautogui.scroll(pixels)
        return "Scrolled up."
    except Exception as e:
        print(f"Scroll up tool failed: {e}")
        return f"Error scrolling up: {e}"

def get_active_window_info():
    """
    Gets information about the currently active window.
    """
    print("Executing get_active_window_info")
    try:
        active_window = gw.getActiveWindow()
        if not active_window:
            return "No active window found."
        return json.dumps({
            "title": active_window.title,
            "size": (active_window.width, active_window.height),
            "position": (active_window.left, active_window.top)
        })
    except Exception as e:
        print(f"Get active window info tool failed: {e}")
        return f"Error getting active window info: {e}"

def read_screen_text():
    """
    Performs OCR on the entire screen and returns the extracted text.
    """
    print("Executing read_screen_text (OCR)")
    try:
        screenshot = pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)
        return text.strip() if text else "No text found on screen."
    except Exception as e:
        print(f"Read screen text tool failed: {e}")
        # Provide a more helpful error if Tesseract is not installed
        if "tesseract is not installed" in str(e).lower():
            return "Error: Tesseract OCR is not installed or not in your system's PATH. This tool cannot function."
        return f"Error reading screen text: {e}"

def type_screen_text(text: str):
    """
    Types the given text into the active window.
    """
    print(f"Executing type_screen_text with text: {text[:50]}...")
    try:
        pyautogui.write(text, interval=0.05)
        return "Successfully typed text."
    except Exception as e:
        print(f"Type screen text tool failed: {e}")
        return f"Error typing text: {e}"

# Add the new tool implementations:
def generate_image(prompt: str, model: str = "default"):
    """
    Generates an image using ComfyUI/A1111 and returns the image data.
    """
    print(f"Executing generate_image with prompt: {prompt}, model: {model}")
    try:
        # Load media server settings
        settings_path = os.path.join(os.path.dirname(__file__), 'nova_settings.json')
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        image_gen_url = settings.get('imageGenUrl')
        if not image_gen_url:
            return "Error: Image generation URL is not configured in settings."

        # For now, return a command response that will trigger the frontend UI
        # The actual image generation will be handled by the frontend
        return {
            "type": "image_generation",
            "prompt": prompt,
            "model": model,
            "message": f"Ready to generate image with prompt: '{prompt}'. Opening image generation panel..."
        }

    except Exception as e:
        print(f"Image generation tool failed: {e}")
        return f"Error generating image: {e}"

def browse_media(category: str, query: str = ""):
    """
    Browses media content from Jellyfin/Plex/Emby.
    """
    print(f"Executing browse_media with category: {category}, query: {query}")
    try:
        # Load media server settings
        settings_path = os.path.join(os.path.dirname(__file__), 'nova_settings.json')
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        media_server_url = settings.get('mediaServerUrl')
        if not media_server_url:
            return "Error: Media server URL is not configured in settings."

        # Return a command response that will trigger the frontend media browser
        return {
            "type": "media_browser",
            "category": category,
            "query": query,
            "message": f"Opening media browser for {category}..."
        }

    except Exception as e:
        print(f"Media browsing tool failed: {e}")
        return f"Error browsing media: {e}"

# --- Knowledge / research helpers ---

def browse_url(url: str):
    """
    Fetches raw HTML content from a URL.
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text[:20000]  # cap to avoid huge payloads
    except Exception as e:
        return f"Error fetching URL: {e}"

def get_page_summary(url: str):
    """
    Fetches and summarizes a page (simple heuristic summary).
    """
    html = browse_url(url)
    if isinstance(html, str) and html.startswith("Error"):
        return html
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(soup.get_text().split())
        return text[:2000] if text else "No text content found."
    except Exception as e:
        return f"Error summarizing page: {e}"

def deep_research(query: str, depth: int = 3, time_limit: int = 60):
    """
    Multi-round research: search -> scrape top results -> summarize.
    """
    start_time = time.time()
    rounds = []
    sources_seen = set()

    def _search_with_urls(q, limit):
        try:
            settings_path = os.path.join(os.path.dirname(__file__), 'nova_settings.json')
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            searxng_url = settings.get('searxngUrl')
            if not searxng_url:
                return []
            api_url = f"{searxng_url.rstrip('/')}/search?q={urllib.parse.quote(q)}&format=json&language=en"
            resp = requests.get(api_url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get('results', [])[:limit]
        except Exception as e:
            print(f"deep_research search failed: {e}")
            return []

    def _summarize(text):
        if summarize_text:
            return summarize_text(text)
        # fallback: truncate
        return text[:800]

    try:
        max_rounds = max(1, min(5, depth))
        per_round = max(1, min(5, depth))
        for round_idx in range(max_rounds):
            if time.time() - start_time > time_limit:
                break
            results = _search_with_urls(query, per_round)
            if not results:
                break
            round_summaries = []
            for r in results:
                url = r.get('url')
                if not url or url in sources_seen:
                    continue
                sources_seen.add(url)
                page_summary = get_page_summary(url)
                round_summaries.append({
                    "url": url,
                    "title": r.get('title'),
                    "snippet": r.get('content'),
                    "summary": page_summary,
                })
            rounds.append({
                "round": round_idx + 1,
                "query": query,
                "findings": round_summaries,
            })
            # simple stop if we gathered enough summaries
            if len(sources_seen) >= depth:
                break

        flattened = []
        for r in rounds:
            for f in r["findings"]:
                flattened.append(f"Source: {f.get('title','')} {f.get('url','')}\nSummary: {f.get('summary','')}")
        final_summary = _summarize("\n\n".join(flattened)) if flattened else "No findings."
        # Persist a snapshot to memory if available
        if memory:
            try:
                memory.learn_from_text(final_summary, source="deep_research", force=True, user_id="default_user")
            except Exception:
                pass
        return {
            "rounds": rounds,
            "sources": list(sources_seen),
            "final_summary": final_summary
        }
    except Exception as e:
        return f"Error in deep research: {e}"

def run_javascript(code: str, timeout: float = 20):
    """
    Runs JavaScript using node (if installed).
    """
    try:
        completed = subprocess.run(
            ["node", "-e", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = completed.stdout.strip()
        err = completed.stderr.strip()
        result = f"exit_code: {completed.returncode}\n"
        if out:
            result += f"stdout:\n{out}\n"
        if err:
            result += f"stderr:\n{err}"
        return result.strip()
    except subprocess.TimeoutExpired:
        return f"JavaScript snippet timed out after {timeout} seconds."
    except FileNotFoundError:
        return "Node.js not found. Please install Node to use run_javascript."
    except Exception as e:
        return f"Error running JavaScript: {e}"

def generate_audio(text: str, voice: str = None):
    """
    Calls the local voice server TTS endpoint.
    """
    try:
        with open('voice_settings.json', 'r') as f:
            settings = json.load(f).get('tts', {})
        model_name = settings.get('model', 'Kyutai-TTS-0.75B')
        speaker = voice or settings.get('voice')
        voice_server_url = 'http://localhost:8880/tts'
        payload = {"text": text, "model_name": model_name, "speaker": speaker}
        resp = requests.post(voice_server_url, json=payload, timeout=30)
        resp.raise_for_status()
        return "TTS request sent."
    except Exception as e:
        return f"Error generating audio: {e}"

def analyze_image(image_url: str, prompt: str = None):
    """
    Stub for image analysis (no vision backend wired); returns a placeholder.
    """
    return f"Image analysis not configured. Received url={image_url}, prompt={prompt or ''}"

def stream_audio_chunk(chunk_b64: str):
    """
    Placeholder for streaming audio; returns an acknowledgement.
    """
    return "Audio chunk received (streaming not implemented in backend)."

def record_environment_audio():
    """
    Placeholder for recording environment audio.
    """
    return "Recording not implemented in backend."

def transcribe_audio(path: str):
    """
    Transcribes an audio file using local Whisper if configured.
    """
    try:
        from main_server import transcribe_audio as server_transcribe
        return server_transcribe(path)
    except Exception as e:
        return f"Error transcribing audio: {e}"

# --- Memory helpers (existing) ---
def _ensure_memory():
    if memory is None:
        return "Memory manager is unavailable."
    return None

def memory_write(content: str, source: str = "tool", model_id: str = None, metadata: dict = None):
    """
    Store a text snippet in long-term memory.
    """
    mem_err = _ensure_memory()
    if mem_err:
        return mem_err
    try:
        # learn_from_text also updates ToM profiles when source == user
        memory.learn_from_text(content, source=source, model_id=model_id, force=True, user_id="default_user")
        # Also store as raw memory to ensure persistence even if fact extraction skips
        memory.collection.insert_one({
            "content": content,
            "source": source,
            "created_at": datetime.now(),
            "metadata": {"model_id": model_id, **(metadata or {})}
        })
        return "Memory stored."
    except Exception as e:
        return f"Error storing memory: {e}"

def memory_search(query: str, limit: int = 5):
    """
    Recall relevant memories using embedding search.
    """
    mem_err = _ensure_memory()
    if mem_err:
        return mem_err
    try:
        results = memory.recall(query, limit=limit)
        if not results:
            return "No memories found."
        lines = []
        for mem in results:
            lines.append(f"- {mem.get('content','')[:400]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching memories: {e}"

def memory_read(memory_id: str = None, limit: int = 3):
    """
    Read a specific memory by id or fetch the most recent ones.
    """
    mem_err = _ensure_memory()
    if mem_err:
        return mem_err
    try:
        if memory_id:
            doc = memory.collection.find_one({"_id": ObjectId(memory_id)})
            if not doc:
                return f"Memory {memory_id} not found."
            return doc.get("content", "")
        # recent memories
        docs = list(memory.collection.find().sort("created_at", -1).limit(limit))
        if not docs:
            return "No memories available."
        return "\n".join([f"- {d.get('content','')[:400]}" for d in docs])
    except Exception as e:
        return f"Error reading memory: {e}"

def memory_delete(memory_id: str):
    """
    Delete a memory by id.
    """
    mem_err = _ensure_memory()
    if mem_err:
        return mem_err
    try:
        res = memory.collection.delete_one({"_id": ObjectId(memory_id)})
        if res.deleted_count:
            return f"Deleted memory {memory_id}"
        return f"Memory {memory_id} not found."
    except Exception as e:
        return f"Error deleting memory: {e}"

def list_processes():
    """
    Lists running processes.
    """
    try:
        procs = []
        if psutil:
            for p in psutil.process_iter(attrs=["pid", "name"]):
                procs.append(f"{p.info.get('name','')} (pid={p.info.get('pid')})")
        else:
            completed = subprocess.run("tasklist", shell=True, capture_output=True, text=True)
            procs = completed.stdout.splitlines()
        procs = procs[:100]  # cap output
        return "\n".join(procs) if procs else "No processes found."
    except Exception as e:
        return f"Error listing processes: {e}"

def kill_process(pid: int = None, name: str = None):
    """
    Kills a process by pid or name substring.
    """
    if not pid and not name:
        return "Provide pid or name."
    try:
        if psutil:
            if pid:
                p = psutil.Process(pid)
                p.kill()
                return f"Killed pid {pid}"
            else:
                killed = []
                for p in psutil.process_iter(attrs=["pid", "name"]):
                    if name.lower() in (p.info.get("name") or "").lower():
                        p.kill()
                        killed.append(p.info.get("pid"))
                if killed:
                    return f"Killed processes: {killed}"
                return f"No process matching '{name}'"
        else:
            cmd = None
            if pid:
                cmd = f"taskkill /PID {pid} /F"
            else:
                cmd = f'taskkill /F /IM "*{name}*"'
            completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if completed.returncode == 0:
                return completed.stdout.strip() or "Process killed."
            return completed.stderr.strip() or "Kill command failed."
    except Exception as e:
        return f"Error killing process: {e}"

def restart_subsystem(name: str):
    """
    Attempts to restart a subsystem by killing known processes.
    """
    target = name.lower()
    killed = []
    if target in ["ollama"]:
        killed.append(kill_process(name="ollama"))
    elif target in ["searxng", "searx"]:
        killed.append(kill_process(name="searxng"))
    elif target in ["tts", "voice"]:
        killed.append(kill_process(name="kokoro"))
        killed.append(kill_process(name="voice-server"))
    elif target in ["llama", "llama.cpp"]:
        killed.append(kill_process(name="llama"))
        killed.append(kill_process(name="main"))
    else:
        return f"Unknown subsystem: {name}"
    return " | ".join(killed)

def environment_info():
    """
    Returns OS and basic hardware info.
    """
    info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version()
    }
    if psutil:
        try:
            info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass
    return json.dumps(info)

# Agentic meta/state tools (persisted per session)

def _agent_collection():
    try:
        if memory:
            return memory.collection.database["agent_state"]
        if upgraded_memory_module and hasattr(upgraded_memory_module, "db") and upgraded_memory_module.db:
            return upgraded_memory_module.db["agent_state"]
    except Exception:
        return None
    return None

def _get_agent_state(session_id: str):
    coll = _agent_collection()
    if not coll:
        return None
    return coll.find_one({"session_id": session_id}) or {"session_id": session_id}

def set_goal(goal: str, session_id: str):
    coll = _agent_collection()
    if not coll:
        return "Agent state store unavailable."
    coll.update_one(
        {"session_id": session_id},
        {"$set": {"goal": goal, "updated_at": datetime.now()}, "$setOnInsert": {"created_at": datetime.now()}},
        upsert=True,
    )
    return f"Goal set for session {session_id}"

def update_plan(plan: list, session_id: str):
    coll = _agent_collection()
    if not coll:
        return "Agent state store unavailable."
    coll.update_one(
        {"session_id": session_id},
        {"$set": {"plan": plan, "updated_at": datetime.now()}, "$setOnInsert": {"created_at": datetime.now()}},
        upsert=True,
    )
    return {"plan": plan}

def reflect(note: str, session_id: str):
    coll = _agent_collection()
    if not coll:
        return "Agent state store unavailable."
    coll.update_one(
        {"session_id": session_id},
        {"$push": {"reflections": {"note": note, "timestamp": datetime.now()}}, "$setOnInsert": {"created_at": datetime.now()}},
        upsert=True,
    )
    return "Reflection stored."

def revise_answer(answer: str, session_id: str):
    coll = _agent_collection()
    if coll:
        coll.update_one(
            {"session_id": session_id},
            {"$set": {"last_revision": {"answer": answer, "timestamp": datetime.now()}}, "$setOnInsert": {"created_at": datetime.now()}},
            upsert=True,
        )
    return f"Revised answer: {answer}"

def finalize(summary: str, session_id: str):
    coll = _agent_collection()
    if coll:
        coll.update_one(
            {"session_id": session_id},
            {"$set": {"final_summary": summary, "finalized_at": datetime.now()}, "$setOnInsert": {"created_at": datetime.now()}},
            upsert=True,
        )
    return f"Finalized: {summary}"

def get_system_state(session_id: str = None):
    agent_state = {}
    if session_id:
        agent_state = _get_agent_state(session_id) or {}
    try:
        from main_server import current_backend, current_model_path_string
        return {"backend": current_backend, "model": current_model_path_string, "agent_state": agent_state}
    except Exception:
        return {"backend": None, "model": None, "agent_state": agent_state}

def set_system_state(state: dict, session_id: str):
    coll = _agent_collection()
    if coll:
        coll.update_one(
            {"session_id": session_id},
            {"$set": {"system_state": state, "updated_at": datetime.now()}, "$setOnInsert": {"created_at": datetime.now()}},
            upsert=True,
        )
    return {"system_state": state}

def get_settings():
    try:
        with open('nova_settings.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return f"Error reading settings: {e}"

def save_settings(settings: dict):
    try:
        with open('nova_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        return "Settings saved."
    except Exception as e:
        return f"Error saving settings: {e}"

def jellyfin_control(action: str, media_id: str = None):
    """
    Simplified Jellyfin control using embed URL pattern.
    """
    try:
        with open('nova_settings.json', 'r') as f:
            nova_settings = json.load(f)
        media_server_url = nova_settings.get('mediaServerUrl')
        media_server_api_key = nova_settings.get('mediaServerApiKey')
        if not media_server_url or not media_server_api_key:
            return "Media server URL or API key not configured."
        if action == "play" and media_id:
            embed_url = f"{media_server_url.rstrip('/')}/Videos/{media_id}/stream?api_key={media_server_api_key}"
            return {"type": "media_embed", "embed_url": embed_url, "message": f"Playing media {media_id}"}
        elif action == "pause":
            return "Pause not implemented; use frontend player controls."
        return "Unsupported action."
    except Exception as e:
        return f"Error controlling Jellyfin: {e}"

def comfyui_generate(prompt: str, width: int = 512, height: int = 512):
    """
    Triggers ComfyUI generation via existing imageGenUrl settings; returns a command response.
    """
    try:
        with open('nova_settings.json', 'r') as f:
            nova_settings = json.load(f)
        image_gen_url = nova_settings.get('imageGenUrl')
        if not image_gen_url:
            return "Image generation URL not configured."
        return {
            "type": "image_generation",
            "prompt": prompt,
            "message": f"Opening image generator for '{prompt}' ({width}x{height})"
        }
    except Exception as e:
        return f"Error triggering ComfyUI: {e}"

def searx_query(query: str):
    """
    Alias to search_web.
    """
    return search_web(query)

def browser_iframe(url: str):
    """
    Returns an iframe payload for the frontend.
    """
    return {"type": "iframe", "url": url, "message": f"Opening {url}"}

def list_files(path: str = "."):
    """
    Lists files in a directory (non-recursive).
    """
    try:
        entries = [p.name for p in Path(path).iterdir()]
        return "\n".join(entries) if entries else "No entries found."
    except Exception as e:
        return f"Error listing files: {e}"

def read_file(path: str):
    """
    Reads a UTF-8 text file.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str):
    """
    Writes content to a file (overwrites).
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def append_file(path: str, content: str):
    """
    Appends content to a file (creates if missing).
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} characters to {path}"
    except Exception as e:
        return f"Error appending file: {e}"

def delete_file(path: str):
    """
    Deletes a file.
    """
    try:
        Path(path).unlink()
        return f"Deleted {path}"
    except Exception as e:
        return f"Error deleting file: {e}"

def search_in_files(query: str, path: str = "."):
    """
    Recursively searches for a string in text files.
    """
    matches = []
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception:
                    continue  # Skip binary or unreadable files
                if query in content:
                    matches.append(str(p))
        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"Error searching files: {e}"

def run_shell(command: str, timeout: float = 30):
    """
    Runs a shell command and returns stdout/stderr.
    """
    try:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        output = completed.stdout.strip()
        err = completed.stderr.strip()
        status = completed.returncode
        result = f"exit_code: {status}\n"
        if output:
            result += f"stdout:\n{output}\n"
        if err:
            result += f"stderr:\n{err}"
        return result.strip()
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error running command: {e}"

def abort():
    """
    Signals an immediate abort of the current flow.
    """
    return "Abort requested. Halting current operation."

def run_python(code: str, timeout: float = 30):
    """
    Runs a Python snippet in a subprocess to avoid state leakage.
    """
    try:
        completed = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = completed.stdout.strip()
        err = completed.stderr.strip()
        result = f"exit_code: {completed.returncode}\n"
        if out:
            result += f"stdout:\n{out}\n"
        if err:
            result += f"stderr:\n{err}"
        return result.strip()
    except subprocess.TimeoutExpired:
        return f"Python snippet timed out after {timeout} seconds."
    except Exception as e:
        return f"Error running python: {e}"

def validate_tool_params(tool_name: str, params: dict):
    """
    Echoes tool parameters back for validation.
    """
    return {
        "tool": tool_name,
        "validated_params": params
    }

# --- Browser Control Tools ---

def browser_click(selector: str):
    """
    Clicks an element on the current browser page using the injected control bridge.
    Returns a command payload for the frontend to execute.
    """
    print(f"Executing browser_click with selector: {selector}")
    return {
        "type": "browser_control",
        "action": "click",
        "selector": selector,
        "message": f"Clicking element: {selector}"
    }

def browser_fill(selector: str, value: str):
    """
    Fills a form field on the current browser page.
    Returns a command payload for the frontend to execute.
    """
    print(f"Executing browser_fill with selector: {selector}, value: {value[:50]}...")
    return {
        "type": "browser_control",
        "action": "fill",
        "selector": selector,
        "value": value,
        "message": f"Filling {selector} with value"
    }

def browser_get_content():
    """
    Gets the current page's content (title, URL, text, HTML).
    Returns a command payload for the frontend to execute.
    """
    print("Executing browser_get_content")
    return {
        "type": "browser_control",
        "action": "getContent",
        "message": "Getting page content"
    }

def browser_get_links():
    """
    Gets all links on the current page.
    Returns a command payload for the frontend to execute.
    """
    print("Executing browser_get_links")
    return {
        "type": "browser_control",
        "action": "getLinks",
        "message": "Getting all links on page"
    }

def browser_get_form_fields():
    """
    Gets all form fields on the current page.
    Returns a command payload for the frontend to execute.
    """
    print("Executing browser_get_form_fields")
    return {
        "type": "browser_control",
        "action": "getFormFields",
        "message": "Getting all form fields on page"
    }

def browser_scroll_to(selector: str):
    """
    Scrolls to a specific element on the page.
    Returns a command payload for the frontend to execute.
    """
    print(f"Executing browser_scroll_to with selector: {selector}")
    return {
        "type": "browser_control",
        "action": "scrollTo",
        "selector": selector,
        "message": f"Scrolling to element: {selector}"
    }

def browser_exec(code: str):
    """
    Executes custom JavaScript code in the browser context.
    Returns a command payload for the frontend to execute.
    """
    print(f"Executing browser_exec with code: {code[:100]}...")
    return {
        "type": "browser_control",
        "action": "exec",
        "code": code,
        "message": "Executing custom JavaScript"
    }

def browser_navigate(url: str):
    """
    Navigates the browser to a new URL.
    Returns a command payload for the frontend to execute.
    """
    print(f"Executing browser_navigate to: {url}")
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    return {
        "type": "browser_navigate",
        "url": url,
        "message": f"Navigating to {url}"
    }

# --- Tool Dispatcher ---

TOOL_REGISTRY = {
    "direct_chat": direct_chat,
    "search_web": search_web,
    "search_news": search_news,
    "scrape_website": scrape_website,
    "list_open_windows": list_open_windows,
    "focus_window": focus_window,
    "scroll_up": scroll_up,
    "scroll_down": scroll_down,
    "get_active_window_info": get_active_window_info,
    "read_screen_text": read_screen_text,
    "type_screen_text": type_screen_text,
    "generate_image": generate_image,
    "browse_media": browse_media,
    "memory_write": memory_write,
    "memory_search": memory_search,
    "memory_read": memory_read,
    "memory_delete": memory_delete,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "delete_file": delete_file,
    "search_in_files": search_in_files,
    "run_shell": run_shell,
    "run_python": run_python,
    "abort": abort,
    "validate_tool_params": validate_tool_params,
    "list_processes": list_processes,
    "kill_process": kill_process,
    "restart_subsystem": restart_subsystem,
    "environment_info": environment_info,
    "browse_url": browse_url,
    "get_page_summary": get_page_summary,
    "deep_research": deep_research,
    "run_javascript": run_javascript,
    "generate_audio": generate_audio,
    "analyze_image": analyze_image,
    "stream_audio_chunk": stream_audio_chunk,
    "record_environment_audio": record_environment_audio,
    "transcribe_audio": transcribe_audio,
    "set_goal": set_goal,
    "update_plan": update_plan,
    "reflect": reflect,
    "revise_answer": revise_answer,
    "finalize": finalize,
    "get_system_state": get_system_state,
    "set_system_state": set_system_state,
    "get_settings": get_settings,
    "save_settings": save_settings,
    "jellyfin_control": jellyfin_control,
    "comfyui_generate": comfyui_generate,
    "searx_query": searx_query,
    "browser_iframe": browser_iframe,
    "browser_click": browser_click,
    "browser_fill": browser_fill,
    "browser_get_content": browser_get_content,
    "browser_get_links": browser_get_links,
    "browser_get_form_fields": browser_get_form_fields,
    "browser_scroll_to": browser_scroll_to,
    "browser_exec": browser_exec,
    "browser_navigate": browser_navigate,
    **INTROSPECTION_TOOL_REGISTRY  # Add introspection tools dynamically
}

# Load and register dynamic tools after base registry is defined
load_dynamic_tools()

# Register agentic coding tools (filesystem-aware coding operations)
try:
    TOOL_REGISTRY.update(get_coding_tool_registry(workspace_root="."))
except Exception as e:
    print(f"Failed to register agentic coding tools: {e}")

def dispatch_tool(tool_name: str, arguments: dict):
    """
    Calls the appropriate Python function based on the tool name provided by the model.
    Tracks execution history for introspection and emergence learning.
    """
    try:
        if tool_name in TOOL_REGISTRY:
            result = TOOL_REGISTRY[tool_name](**arguments)
            # Record successful execution
            record_tool_execution(tool_name, arguments, result, success=True)
            return result
        else:
            # Log missing tool request for emergence learning
            log_missing_tool_request(tool_name, arguments)
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(list(TOOL_REGISTRY.keys())[:5])}..."
            record_tool_execution(tool_name, arguments, error_msg, success=False)
            return error_msg
    except Exception as e:
        # Log failed execution
        error_msg = f"Error executing {tool_name}: {str(e)}"
        record_tool_execution(tool_name, arguments, error_msg, success=False)
        print(f"❌ Tool execution error: {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

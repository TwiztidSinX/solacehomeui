import json
import os
import urllib.parse
import requests
import pygetwindow as gw
import pyautogui
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from gnews import GNews

# --- Tool Definitions (for the model) ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "direct_chat",
            "description": "Engages in a direct conversation with the user. Use this when the user wants to chat, ask a question, or give a command that doesn't match any other tool.",
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
    }
]

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
}

def dispatch_tool(tool_name: str, arguments: dict):
    """
    Calls the appropriate Python function based on the tool name provided by the model.
    """
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name](**arguments)
    else:
        return f"Error: Tool '{tool_name}' not found."

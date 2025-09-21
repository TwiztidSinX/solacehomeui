import json
import os
import webbrowser
import pyautogui
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from gnews import GNews
from googletrans import Translator
import pygetwindow as gw
import pytesseract
from PIL import Image

class Action: # Base class for all actions
    def __init__(self, name, description, usage):
        self.name = name
        self.description = description
        self.usage = usage

    def execute(self, **kwargs):
        raise NotImplementedError

class OpenUrlAction(Action):
    def __init__(self):
        super().__init__(
            name="open_url",
            description="Opens a URL in the default web browser.",
            usage="open_url(url='https://example.com')"
        )

    def execute(self, url):
        try:
            webbrowser.open(url)
            return f"Successfully opened {url}"
        except Exception as e:
            return f"Error opening URL: {e}"

def search_web(query):
    """Searches the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return json.dumps(results)
    except Exception as e:
        return f"Error searching web: {e}"

def search_news(query):
    """Searches for news articles using GNews."""
    try:
        gnews = GNews()
        results = gnews.get_news(query)
        return json.dumps(results)
    except Exception as e:
        return f"Error searching news: {e}"

def scrape_website(url):
    """Scrapes the text content of a website."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error scraping website: {e}"

def translate_text(text, dest_lang):
    """Translates text to a specified language."""
    try:
        translator = Translator()
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        return f"Error translating text: {e}"

def capture_screen(region=None):
    """Captures the screen or a region of the screen."""
    try:
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        # The screenshot object needs to be handled (e.g., saved or processed)
        # For now, let's just return a success message.
        return "Screen captured successfully."
    except Exception as e:
        return f"Error capturing screen: {e}"

def list_open_windows():
    windows = gw.getAllTitles()
    return [w for w in windows if w.strip()]

def focus_window(title_substring):
    windows = gw.getWindowsWithTitle(title_substring)
    if not windows:
        return f"❌ No window found with title containing: {title_substring}"
    win = windows[0]
    win.activate()
    return f"✅ Focused window: {win.title}"

def scroll_down(pixels=500):
    pyautogui.scroll(-pixels)
    return "⬇️ Scrolled down."

def scroll_up(pixels=500):
    pyautogui.scroll(pixels)
    return "⬆️ Scrolled up."

def get_active_window_info():
    try:
        active = gw.getActiveWindow()
        if not active:
            return "⚠️ No active window."
        return {
            "title": active.title,
            "size": (active.width, active.height),
            "position": (active.left, active.top)
        }
    except Exception as e:
        return f"Error: {str(e)}"

def read_screen_text(region=None):
    """Extract text from the screen (optionally cropped region)"""
    image = pyautogui.screenshot(region=region)
    gray = image.convert("L")  # Convert to grayscale
    text = pytesseract.image_to_string(gray)
    return text.strip()

def type_screen_text(text):
    """Type message into current input field and hit enter"""
    pyautogui.write(text, interval=0.05)
    pyautogui.press("enter")
    print(f"[GUI] Sent message: {text}")

def generate_tool_prompt(tools):
    """Generates a prompt for the AI to use tools."""
    prompt = "You have access to the following tools:\n"
    for tool in tools:
        prompt += f"- {tool.name}: {tool.description} Usage: {tool.usage}\n"
    prompt += "\nPlease select a tool to use based on the user's request."
    return prompt

TOOL_REGISTRY = {
    "open_url": OpenUrlAction(),
    "search_web": search_web,
    "search_news": search_news,
    "scrape_website": scrape_website,
    "translate_text": translate_text,
    "capture_screen": capture_screen,
    "list_open_windows": list_open_windows,
    "focus_window": focus_window,
    "scroll_down": scroll_down,
    "scroll_up": scroll_up,
    "get_active_window_info": get_active_window_info,
    "read_screen_text": read_screen_text,
    "type_screen_text": type_screen_text,
}

def get_tool(name):
    return TOOL_REGISTRY.get(name)

def get_all_tools():
    return list(TOOL_REGISTRY.values())

def run_tool(tool_name, argument):
    tool = get_tool(tool_name)
    if tool:
        return tool.execute(argument) if isinstance(tool, Action) else tool(argument)
    return f"Tool '{tool_name}' not found."

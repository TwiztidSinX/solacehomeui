import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from gnews import GNews

# --- Tool Definitions (for the model) ---

TOOLS_SCHEMA = [
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
    }
]

# --- Tool Implementations (for the backend) ---

def search_web(query: str):
    """
    The actual Python function that gets called when the model decides to use the 'search_web' tool.
    """
    print(f"Executing web search for: {query}")
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=5)]
            return "\n".join(results) if results else "No results found."
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


# --- Tool Dispatcher ---

TOOL_REGISTRY = {
    "search_web": search_web,
    "search_news": search_news,
    "scrape_website": scrape_website,
}

def dispatch_tool(tool_name: str, arguments: dict):
    """
    Calls the appropriate Python function based on the tool name provided by the model.
    """
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name](**arguments)
    else:
        return f"Error: Tool '{tool_name}' not found."

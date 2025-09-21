import json
from duckduckgo_search import DDGS

# --- Tool Definitions (for the model) ---

# This is the JSON schema that the model will see. It's how the model knows what functions are available.
# We will expand this list as we migrate more abilities.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Performs a web search using DuckDuckGo when the user asks a question that requires up-to-date information, specific facts, or knowledge beyond your internal cutoff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"],
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
            # We'll return a simplified list of snippets for the model to read.
            results = [r['body'] for r in ddgs.text(query, max_results=5)]
            # Join the snippets into a single string for the model to easily parse.
            return "\n".join(results) if results else "No results found."
    except Exception as e:
        print(f"Web search tool failed: {e}")
        return f"Error searching web: {e}"


# --- Tool Dispatcher ---

# This is a simple router that maps the tool name from the model to the actual Python function.
def dispatch_tool(tool_name: str, arguments: dict):
    """
    Calls the appropriate Python function based on the tool name provided by the model.
    """
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name](**arguments)
    else:
        return f"Error: Tool '{tool_name}' not found."

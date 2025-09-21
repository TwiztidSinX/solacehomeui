from llama_cpp import Llama
import os
import json
import re

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORCHESTRATOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "llama", "Qwen3-1.7B-Q8_0.gguf").replace("\\", "/")

orchestrator_model = None

def load_orchestrator_model():
    """Loads the small Qwen model into RAM for orchestration tasks."""
    global orchestrator_model
    if orchestrator_model is None:
        print("Loading Orchestrator Model...")
        try:
            orchestrator_model = Llama(
                model_path=ORCHESTRATOR_MODEL_PATH,
                n_ctx=2048,
                n_gpu_layers=0,  # Force to RAM
                verbose=False
            )
            print("Orchestrator Model Loaded.")
        except Exception as e:
            print(f"FAILED to load Orchestrator Model: {e}")
            orchestrator_model = None

from llama_cpp import Llama
import os
import json
import re

# Import the new tool schema
from tools import TOOLS_SCHEMA

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORCHESTRATOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "llama", "Qwen3-1.7B-Q8_0.gguf").replace("\\", "/")

orchestrator_model = None

def load_orchestrator_model():
    """Loads the small Qwen model into RAM for orchestration tasks."""
    global orchestrator_model
    if orchestrator_model is None:
        print("Loading Orchestrator Model...")
        try:
            orchestrator_model = Llama(
                model_path=ORCHESTRATOR_MODEL_PATH,
                n_ctx=2048,
                n_gpu_layers=0,  # Force to RAM
                verbose=False
            )
            print("Orchestrator Model Loaded.")
        except Exception as e:
            print(f"FAILED to load Orchestrator Model: {e}")
            orchestrator_model = None

def get_tool_call(user_input: str):
    """
    Uses the orchestrator model to decide if a tool call is necessary.
    Returns the tool call details if needed, otherwise None.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping tool call check.")
        return None

    # Create the prompt for the orchestrator
    messages = [
        {"role": "system", "content": "You are a helpful assistant that decides if a function call is needed to answer the user's question. Use the provided tools to answer questions that require up-to-date information or external capabilities. Do not answer the question yourself, only call a tool if necessary."},
        {"role": "user", "content": user_input}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0
        )

        tool_calls = response.get('choices', [{}])[0].get('message', {}).get('tool_calls')
        if tool_calls:
            # For now, we only handle the first tool call
            tool_call = tool_calls[0]
            tool_name = tool_call.get('function', {}).get('name')
            arguments = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
            
            print(f"Orchestrator decided to call tool: {tool_name} with args: {arguments}")
            return {"name": tool_name, "arguments": arguments}

    except Exception as e:
        print(f"Orchestrator tool call check failed: {e}")

    return None

def get_summary_for_title(text: str):
    """
    Uses the orchestrator model with a few-shot prompt to generate a short,
    concise, and correctly formatted title for a chat session.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping title summary.")
        return "New Chat"

    # A more forceful prompt with a very specific instruction in the system message.
    prompt = f"""---
Text: "Hey, can you help me figure out why my Python script for web scraping isn't working? I'm getting a 403 error."
New Chat Title: "Python Web Scraping 403 Error"
---
Text: "what's the weather like in new york city right now?"
New Chat Title: "New York City Weather Inquiry"
---
Text: "Hello again, how are you today?"
New Chat Title: "Friendly Greeting"
---
Text: "{text}"
New Chat Title:"""

    messages = [
        {"role": "system", "content": "You are a machine that only generates chat titles. You will be given examples and a final text. Your entire output will be only the final line 'New Chat Title: \"...\"'. You will not think, explain, or say anything else."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=75,
            stop=["---"] # Stop the model from hallucinating more examples
        )
        raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        # A more specific regex that looks for the exact pattern we want.
        match = re.search(r'New Chat Title:\s*"(.*?)"', raw_content)
        if match:
            title = match.group(1).strip()
            print(f"Orchestrator generated title via few-shot: '{title}' from raw: '{raw_content}'")
            return title if title else "Chat Summary"
        else:
            # Fallback if the specific regex fails, which might happen if the model omits the label.
            # We take the last line and strip quotes.
            last_line = raw_content.splitlines()[-1]
            title = last_line.replace("New Chat Title:", "").strip().strip('"')
            print(f"Orchestrator generated title (fallback parsing): '{title}' from raw: '{raw_content}'")
            return title if title else "Chat Summary"

    except Exception as e:
        print(f"Orchestrator title summary failed: {e}")
        return "Chat Summary"

def sanitize_chat_title(title: str):
    """
    Checks if a title contains non-ASCII characters and asks the orchestrator to
    rewrite it in English if necessary.
    """
    # Check if any character in the title is outside the standard ASCII range
    if any(ord(char) > 127 for char in title):
        print(f"Non-ASCII title detected: '{title}'. Requesting English translation.")
        
        messages = [
            {"role": "system", "content": "You are an expert title translator. The user will provide a title, potentially in another language. Your task is to provide a concise, natural-sounding English equivalent. Respond with ONLY the translated English title and nothing else."},
            {"role": "user", "content": f"Translate this title to English: \"{title}\""}
        ]

        try:
            response = orchestrator_model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=25
            )
            new_title = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip().strip('\'"')
            
            print(f"Translated title: '{new_title}'")
            return new_title if new_title else title # Fallback to original title on failure
        except Exception as e:
            print(f"Title translation failed: {e}")
            return title # Fallback to original title on failure
            
    # If the title is already ASCII, return it as is
    return title

def summarize_text(text: str):
    """Uses the orchestrator model to generate a summary of a given text."""
    if orchestrator_model is None:
        return "Orchestrator model not loaded. Cannot summarize."

    messages = [
        {"role": "system", "content": "You are an expert at summarizing web content. Provide a concise summary of the following text."},
        {"role": "user", "content": text}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512 
        )
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        return summary
    except Exception as e:
        print(f"Orchestrator summary failed: {e}")
        return "Failed to generate summary."

def parse_command(user_input: str):
    """
    Parses user input for slash commands.
    Returns a dictionary with the command and query, or None if no command is found.
    """
    match = re.match(r"^\s*/(\w+)\s*(.*)", user_input)
    if match:
        command = match.group(1).lower()
        query = match.group(2).strip()
        
        # Define all valid commands
        valid_commands = [
            "search", "youtube", "go", "read", "image", 
            "map", "wiki", "weather", "calc", "define", "play"
        ]
        
        if command in valid_commands:
            print(f"Orchestrator parsed command: /{command} with query: '{query}'")
            return {"command": command, "query": query}
    return None


if __name__ == '__main__':
    # For testing purposes
    load_orchestrator_model()
    if orchestrator_model:
        test_queries = [
            "What's the weather in New York today?",
            "Who is the current president of France?",
            "What is the capital of California?",
            "Summarize the plot of Hamlet.",
            "Search for the latest news on AI.",
            "look up the price of bitcoin"
        ]
        for query in test_queries:
            print(f"\n--- Testing Query: '{query}' ---")
            search_query = should_perform_web_search(query)
            if search_query:
                print(f"Result: Web search needed for '{search_query}'")
            else:
                print("Result: No web search needed.")
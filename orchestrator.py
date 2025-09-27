from llama_cpp import Llama
import os
import json
import re

# Import the new tool schema
from tools import TOOLS_SCHEMA, search_web
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
                n_ctx=8192,
                n_gpu_layers=0,  # Force to RAM
                verbose=False
            )
            print("Orchestrator Model Loaded.")
        except Exception as e:
            print(f"FAILED to load Orchestrator Model: {e}")
            orchestrator_model = None

def get_tool_call(user_input: str):
    """
    Uses the orchestrator model to decide if a tool call is necessary by treating
    the task as a text-to-JSON translation.
    Returns the tool call details if needed, otherwise None.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping tool call check.")
        return None

    # A more direct, machine-like prompt that frames the task as a translation.
    system_prompt = """Your task is to translate user requests into function calls based on the provided tool descriptions. If a suitable function exists, output the JSON for the function call. If no function is suitable, output the word 'None'.

### Tools Available
{tools_json}

### Examples
User: What windows are open on the computer right now?
Assistant: {{"name": "list_open_windows", "arguments": {{}}}}

User: can you look up the latest news about nvidia?
Assistant: {{"name": "search_news", "arguments": {{"query": "latest news about nvidia"}}}}

User: bring the discord window to the front
Assistant: {{"name": "focus_window", "arguments": {{"title_substring": "discord"}}}}

User: Hey how are you doing?
Assistant: {{"name": "direct_chat", "arguments": {{}}}}"""

    # Inject the tool schema directly into the prompt
    prompt_with_tools = system_prompt.format(tools_json=json.dumps(TOOLS_SCHEMA, indent=2))

    messages = [
        {"role": "system", "content": prompt_with_tools},
        {"role": "user", "content": user_input}
    ]

    try:
        # Simplified call: ask for plain text generation, not a tool call.
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=512 # Increased token limit for safety
        )

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        print(f"Orchestrator raw output: {content}")

        if content.lower() == 'none':
            print("Orchestrator decided no tool call is needed.")
            return None
        
        # The model might output multiple JSON objects for sequential tool calls.
        # We need to find all of them.
        json_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        tool_calls = []
        
        for match in json_matches:
            json_string = match.group(0)
            try:
                tool_call_data = json.loads(json_string)
                # Basic validation of the parsed structure
                if "name" in tool_call_data and "arguments" in tool_call_data:
                    print(f"Orchestrator parsed tool call: {tool_call_data}")
                    tool_calls.append(tool_call_data)
                else:
                    print(f"Orchestrator generated JSON with missing keys: {json_string}")
            except json.JSONDecodeError:
                print(f"Orchestrator generated invalid JSON snippet: {json_string}")
                continue # Try the next match

        if not tool_calls:
            print("Orchestrator did not generate any valid tool calls.")
            return []

        # Filter out consecutive duplicate tool calls
        filtered_tool_calls = []
        for i, current_call in enumerate(tool_calls):
            if i == 0 or current_call != tool_calls[i-1]:
                filtered_tool_calls.append(current_call)
        tool_calls = filtered_tool_calls

        return tool_calls

    except Exception as e:
        print(f"Orchestrator tool call check failed: {e}")

    return []

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

def get_orchestrator_response(user_input: str, history: list = []):
    """
    Generates a direct response from the orchestrator model.
    """
    if orchestrator_model is None:
        return "Orchestrator model not loaded."

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep your responses brief and to the point."},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        return content
    except Exception as e:
        print(f"Orchestrator response generation failed: {e}")
        return "I'm sorry, I had a problem generating a response."

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
            "map", "wiki", "weather", "calc", "define", "play",
            "media"
        ]
        
        if command in valid_commands:
            print(f"Orchestrator parsed command: /{command} with query: '{query}'")
            if command == 'youtube':
                return {"command": "youtube", "query": query}
            elif command == 'image':
                return {"command": "image", "query": query}
            elif command == 'media':
                return {"command": "media", "query": query}
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
            search_query = search_web(query)
            if search_query:
                print(f"Result: Web search needed for '{search_query}'")
            else:
                print("Result: No web search needed.")
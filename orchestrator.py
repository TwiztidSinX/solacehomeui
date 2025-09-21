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

def should_perform_web_search(user_input: str):
    """
    Uses the orchestrator model to decide if a web search is necessary.
    Returns the search query if needed, otherwise None.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping web search check.")
        return None

    # Define the tool for the model to call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Performs a web search when the user asks a question that requires up-to-date information or knowledge beyond your internal cutoff.",
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
        }
    ]

    # Create the prompt for the orchestrator
    messages = [
        {"role": "system", "content": "Your sole purpose is to determine if a user's question requires a real-time web search. You must call the `web_search` tool if the user asks about current events, weather, or any topic that requires up-to-the-minute information. Do not answer the question yourself. Only call the tool or do nothing."},
        {"role": "user", "content": user_input}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )

        # --- DEBUGGING: Print the full response from the orchestrator ---
        # print("--- Orchestrator Full Response ---")
        # print(json.dumps(response, indent=2))
        # print("---------------------------------")

        # First, check the standard 'tool_calls' field
        tool_calls = response.get('choices', [{}])[0].get('message', {}).get('tool_calls')
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get('function', {}).get('name') == 'web_search':
                    arguments = json.loads(tool_call['function']['arguments'])
                    query = arguments.get('query')
                    print(f"Orchestrator decided to search for: {query} (via tool_calls)")
                    return query

        # Fallback: Check if the tool call is embedded in the 'content' field
        message_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        if '<tool_call>' in message_content:
            try:
                # Extract the JSON part from the content string
                tool_call_str = message_content.split('<tool_call>')[1].split('</tool_call>')[0].strip()
                tool_call_json = json.loads(tool_call_str)
                if tool_call_json.get('name') == 'web_search':
                    query = tool_call_json.get('arguments', {}).get('query')
                    if query:
                        print(f"Orchestrator decided to search for: {query} (via content fallback)")
                        return query
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Failed to parse tool call from content: {e}")

    except Exception as e:
        print(f"Orchestrator check failed: {e}")

    return None

def get_summary_for_title(text: str):
    """
    Uses the orchestrator model to generate a short, concise title for a chat session.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping title summary.")
        return "New Chat"

    messages = [
        {"role": "system", "content": "You are an expert at summarizing text into a short, concise title. The title must be no more than 5 words. Do not use punctuation or quotes. Your final line of output should be ONLY the title."},
        {"role": "user", "content": f"Summarize this text for a chat title: {text}"}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=25
        )
        raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        # --- New, More Robust Title Extraction ---
        # Take the last non-empty line from the output.
        lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
        summary = lines[-1] if lines else ""
        
        # Clean up any potential model artifacts
        summary = summary.replace('"', '').replace("'", '').strip()
        
        print(f"Orchestrator generated title: '{summary}' from raw: '{raw_content}'")
        return summary if summary else "Chat Summary"
    except Exception as e:
        print(f"Orchestrator title summary failed: {e}")
        return "Chat Summary"

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
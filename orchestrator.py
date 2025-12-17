from llama_cpp import Llama
import os
import json
import re

# Import the new tool schema
from tools import TOOLS_SCHEMA, search_web
from orchestrator_api import create_api_orchestrator
import json
from typing import List, Dict, Any
# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORCHESTRATOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "llama", "Qwen3-1.7B-Q8_0.gguf").replace("\\", "/")
# At module level (after imports, around line 12)
orchestrator_model = None

# Orchestrator configuration
ORCHESTRATOR_CONFIG = {
    "type": "local",  # "local" or "api"
    "api_provider": "openai",  # "openai", "deepseek", or "qwen"
    "api_model": "gpt-5-nano",
    "local_model_path": ORCHESTRATOR_MODEL_PATH
}

def load_config():
    """Load orchestrator config from nova_settings.json if it exists."""
    global ORCHESTRATOR_CONFIG
    try:
        if os.path.exists('nova_settings.json'):
            with open('nova_settings.json', 'r') as f:
                settings = json.load(f)
                
                # Update config from settings
                if 'orchestratorType' in settings:
                    ORCHESTRATOR_CONFIG['type'] = settings['orchestratorType']
                if 'orchestratorApiProvider' in settings:
                    ORCHESTRATOR_CONFIG['api_provider'] = settings['orchestratorApiProvider']
                if 'orchestratorApiModel' in settings:
                    ORCHESTRATOR_CONFIG['api_model'] = settings['orchestratorApiModel']
                p = ORCHESTRATOR_CONFIG['api_provider'].lower()
                if p in ["grok", "grok4", "grok-4", "grok-4-1"]:
                    ORCHESTRATOR_CONFIG['api_provider'] = "xai"
                
                print(f"📋 Orchestrator config loaded: {ORCHESTRATOR_CONFIG['type']}")
    except Exception as e:
        print(f"⚠️ Could not load orchestrator config: {e}")

def load_orchestrator_model():
    """Loads orchestrator model (local or API) based on configuration."""
    global orchestrator_model  # CRITICAL: Declare global at the TOP
    
    # Load config first
    load_config()
    
    # Check if already loaded
    if orchestrator_model is not None:
        print("⚠️ Orchestrator already loaded. Unload first if switching models.")
        return
    
    orchestrator_type = ORCHESTRATOR_CONFIG['type']
    
    if orchestrator_type == "local":
        # Load local llama.cpp model
        print("📦 Loading LOCAL orchestrator model...")
        try:
            orchestrator_model = Llama(
                model_path=ORCHESTRATOR_CONFIG['local_model_path'],
                n_ctx=8192,
                n_gpu_layers=0,  # Force to RAM
                verbose=False
            )
            print("✅ Local Orchestrator Model Loaded (Qwen3 1.7B)")
        except Exception as e:
            print(f"❌ FAILED to load local orchestrator: {e}")
            orchestrator_model = None
    
    elif orchestrator_type == "api":
        # Load API-based orchestrator
        provider = ORCHESTRATOR_CONFIG['api_provider']
        model = ORCHESTRATOR_CONFIG['api_model']
        
        print(f"🌐 Loading API orchestrator: {provider}/{model}")
        try:
            orchestrator_model = create_api_orchestrator(provider, model)
            print(f"✅ API Orchestrator Loaded ({provider}/{model})")
        except Exception as e:
            print(f"❌ FAILED to load API orchestrator: {e}")
            print(f"💡 Make sure {provider.upper()}_API_KEY is set in environment or config.json")
            orchestrator_model = None
    
    else:
        print(f"❌ Invalid orchestrator type: {orchestrator_type}")
        orchestrator_model = None

def unload_orchestrator_model():
    """Unload the current orchestrator model."""
    global orchestrator_model
    if orchestrator_model is not None:
        print("🔄 Unloading orchestrator model...")
        orchestrator_model = None
        print("✅ Orchestrator unloaded")
    else:
        print("⚠️ No orchestrator model loaded")

def reload_orchestrator_model():
    """Reload orchestrator with current configuration."""
    unload_orchestrator_model()
    load_orchestrator_model()

def build_parliament_prompt(user_message: str, roles: List[Dict[str, Any]]) -> str:
    """
    Build a merged prompt block from parliament role responses to feed into Nova/Solace.
    """
    lines = [
        "Parliament context:",
        f"User request: {user_message}",
        "Role responses (JSON or text):"
    ]
    for r in roles:
        lines.append(f"- {r.get('name', r.get('key'))} [{r.get('provider','?')}/{r.get('model','?')}]: {r.get('response','')}")
    lines.append("End parliament context. Use this to craft a concise, helpful reply.")
    return "\n".join(lines)

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
    system_prompt = """Your task is to translate user requests into function calls based on the provided tool descriptions. If a suitable function exists, output the JSON for the function call. If no function is suitable — especially for normal conversation, knowledge, or when the user is telling you something — output the word 'None'.
IMPORTANT: The main model (Nova) handles all chat, knowledge, creativity, opinions, and code review. You are ONLY the gatekeeper for actions that require external systems: web searches, window control, etc.

### Tools Available
{tools_json}

### WHEN TO USE TOOLS (Examples):
User: What windows are open on the computer right now?
Assistant: {"name": "list_open_windows", "arguments": {}}
User: Can you look up the latest news about Nvidia?
Assistant: {"name": "search_news", "arguments": {"query": "latest news about nvidia"}}
User: Bring the Discord window to the front.
Assistant: {"name": "focus_window", "arguments": {"title_substring": "discord"}}

### WHEN NOT TO USE TOOLS (Always output 'None' for these):
User: How are you?
Assistant: None
Reason: Greeting — Nova handles small talk.

User: Tell me about Python programming.
Assistant: None
Reason: General knowledge — Nova has this in her training.

User: Explain how OAuth works.
Assistant: None
Reason: Conceptual explanation — Nova can reason through this directly.

User: Write me a poem about the ocean.
Assistant: None
Reason: Creative task — Nova generates poetry natively.

User: What do you think about AI taking over the world?
Assistant: None
Reason: Opinion question — Nova forms opinions, not tools.

User: Help me debug this code: [code snippet]
Assistant: None
Reason: Code review — Nova analyzes and fixes code without tools.

User: Here's the history of my project: GPT-4o built X, DeepSeek fixed Y.
Assistant: None
Reason: User is PROVIDING information — not asking for it. Do not search.

User: Around 95% of the code was AI-generated.
Assistant: None
Reason: Statistic provided — not a question. No search needed.

User: I have three cats and a dog.
Assistant: None
Reason: Personal fact — irrelevant to tools.

### CRITICAL: PARAMETER RULES
1. ONLY use parameters explicitly listed in the tool schema.
2. NEVER invent parameters like `max_results`, `filter`, `language`, `sort`, etc.
3. If you're unsure whether a parameter exists — DO NOT use it.
4. When in doubt: call the tool with ONLY the required `query` parameter. Leave optional fields out.

WRONG (hallucinated parameters):
{"name": "search_web", "arguments": {"query": "AI news", "max_results": 10, "filter": "recent"}}
                                                                    ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
                                                                    NOT IN SCHEMA!      NOT IN SCHEMA!

RIGHT (correct usage):
{"name": "search_web", "arguments": {"query": "AI news"}}

### IMPORTANT: ASKING vs TELLING
ASKING (use tools if appropriate):
- "What is X?" → User wants info
- "Search for Y" → Explicit request
- "How does Z work?" → May need research

TELLING (ALWAYS output 'None'):
- "X was built by Y" → Past tense = user providing context
- "Here's my data: [data]" → User sharing info
- "I have 3 cats" → User stating facts
- "Around 95% was..." → User giving statistics

Rule: If the sentence uses past tense verbs (built, created, fixed, added, was, were) OR starts with "Here's", "For context", "Let me tell you", and has NO question mark — it’s TELLING. Do NOT use a tool.

### COMMON MISTAKES TO AVOID:

❌ DON'T call search for statements:
User: "GPT-4o built the original version"
Assistant: None  (NOT search_web — user is telling you!)

❌ DON'T call search for knowledge questions:
User: "What is Python?"
Assistant: None  (Main model knows this!)

❌ DON'T call tools for creative tasks:
User: "Write me a story"
Assistant: None  (Main model handles creative writing!)

❌ DON'T hallucinate parameters:
User: "Search for AI news"
Assistant: {"name": "search_web", "arguments": {"query": "AI news", "max_results": 10}}
                                                                        ^^^^^^^^^^^^^^^^
                                                                        WRONG! Not in schema!

✅ Correct:
Assistant: {"name": "search_web", "arguments": {"query": "AI news"}}

❌ DON'T call tools for opinions:
User: "What do you think about AI?"
Assistant: None  (Opinion questions don't need tools!)

❌ DON'T call tools for code review:
User: "Can you review this code: [code]"
Assistant: None  (Main model reviews code directly!)

### IF UNCERTAIN:
When in doubt, output "None" and let the main model handle it.
Better to skip a tool than hallucinate one."""

    # Inject the tool schema directly into the prompt
    prompt_with_tools = system_prompt.replace(
        "{tools_json}", 
        json.dumps(TOOLS_SCHEMA, indent=2)
    )

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
    Uses the orchestrator model with few-shot prompting to generate a concise chat title.
    Nova is a Qwen3 thinking model with 2507 reasoning - she needs 8K+ tokens for CoT.
    We embrace the reasoning and parse the final output using regex on structured tags.
    """
    if orchestrator_model is None:
        print("Orchestrator model not loaded. Skipping title summary.")
        return "New Chat"

    # Few-shot with VERY clear structured output format
    prompt = f"""You generate short, descriptive chat titles from the user's first message.

Requirements:
- 4–6 words maximum
- Concise and descriptive
- Title case (capitalize first letter of main words)
- No quotes in the final title
- After your thinking/reasoning, output the title wrapped in special tags

Format: TITLE_START your title here TITLE_END

Examples:
User: "Hey, can you help me figure out why my Python script for web scraping isn't working? I'm getting a 403 error."
TITLE_START Python Web Scraping 403 Error TITLE_END

User: "what's the weather like in new york city right now?"
TITLE_START New York City Weather TITLE_END

User: "Hello again, how are you today?"
TITLE_START Friendly Greeting TITLE_END

User: "I need help understanding how decorators work in Python"
TITLE_START Python Decorators Explanation TITLE_END

Now generate a title for this message:
User: "{text}"

Remember: Think as much as you need, then output TITLE_START your title TITLE_END
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a title generator. If you think through the task, put your reasoning inside <think>...</think>.\n"
                "Then output ONLY the final title wrapped in TITLE_START and TITLE_END tags.\n"
                "Never leave <think> text in the final title."
            ),
        },
        {"role": "user", "content": prompt}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=20000  # Allow thinking + final title for CoT models
        )
        raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        print(f"🔧 DEBUG Title Generation:")
        print(f"  Raw output length: {len(raw_content)} chars")
        print(f"  First 200 chars: {raw_content[:200]}")
        print(f"  Last 200 chars: {raw_content[-200:]}")

        # Strip think blocks before parsing
        cleaned_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()

        parse_target = cleaned_output or raw_content

        # Extract title using the TITLE_START/TITLE_END markers
        title_match = re.search(r'TITLE_START\s*(.+?)\s*TITLE_END', parse_target, re.DOTALL | re.IGNORECASE)
        
        if title_match:
            title = title_match.group(1).strip()
            # Clean up the title
            title = title.strip('"').strip("'").strip()
            # Remove any remaining newlines
            title = ' '.join(title.split())
            
            # Truncate if too long
            if len(title) > 60:
                title = title[:57] + "..."
            
            print(f"✅ Nova generated title: '{title}'")
            return title if title else "Chat Summary"

        # Fallback 1: Try the old [TITLE] format in case model uses it anyway
        legacy_match = re.search(r'\[TITLE\](.+?)\[/TITLE\]', parse_target, re.IGNORECASE | re.DOTALL)
        if legacy_match:
            title = legacy_match.group(1).strip().strip('"').strip("'").strip()
            title = ' '.join(title.split())
            if len(title) > 60:
                title = title[:57] + "..."
            print(f"✅ Nova generated title (legacy format): '{title}'")
            return title if title else "Chat Summary"

        # Fallback 2: Look for "Chat Title:" pattern
        chat_title_match = re.search(r'Chat Title:\s*(.+?)(?:\n|$)', parse_target, re.IGNORECASE)
        if chat_title_match:
            title = chat_title_match.group(1).strip().strip('"').strip("'").strip()
            title = ' '.join(title.split())
            if len(title) > 60:
                title = title[:57] + "..."
            print(f"✅ Nova generated title (Chat Title format): '{title}'")
            return title if title else "Chat Summary"

        # Fallback 3: Find the last short line (likely the final answer after reasoning)
        lines = [line.strip() for line in parse_target.split('\n') if line.strip()]
        for line in reversed(lines):
            # Skip lines that are clearly reasoning/thinking
            if any(word in line.lower() for word in [
                'okay', 'let me', 'first', 'the user', 'i think', 'i should',
                'seems like', 'based on', 'looking at', 'considering'
            ]):
                continue
            # Skip very long lines (probably reasoning)
            if len(line) > 100:
                continue
            # Skip lines with thinking tags
            if '<think>' in line or '</think>' in line or '<|thinking|>' in line:
                continue
            
            # This looks like a title! Clean it up
            title = line.strip('"').strip("'").strip()
            title = re.sub(r'^[\d\.\-\*\•]+\s*', '', title)  # Remove bullets/numbers
            title = ' '.join(title.split())  # Normalize whitespace
            
            if 5 <= len(title) <= 80:  # Reasonable title length
                if len(title) > 60:
                    title = title[:57] + "..."
                print(f"✅ Nova generated title (extracted from output): '{title}'")
                return title if title else "Chat Summary"

        # Fallback 4: Just use first few words of user message
        print(f"⚠️ All extraction methods failed. Using fallback.")
        print(f"  Full raw output for debugging:\n{raw_content}\n")
        
        words = text.split()[:6]
        title = ' '.join(words)
        if len(text.split()) > 6:
            title += "..."
        print(f"⚠️ Using fallback title from user message: '{title}'")
        return title

    except Exception as e:
        print(f"❌ Orchestrator title summary failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback
        words = text.split()[:6]
        title = ' '.join(words)
        if len(text.split()) > 6:
            title += "..."
        return title

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
                temperature=0.1,
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

def score_response_confidence(response: str, user_query: str) -> dict:
    """
    Scores the main model's response confidence (0-100%) to detect hallucinations or uncertainty.
    Returns dict with: {'score': float, 'reasoning': str, 'should_search': bool}
    """
    if orchestrator_model is None:
        print("⚠️ Orchestrator model not loaded. Cannot score confidence.")
        return {'score': 100.0, 'reasoning': 'Orchestrator unavailable', 'should_search': False}

    # Quick heuristic checks first (fast path)
    uncertainty_phrases = [
        r"\bi don't know\b", r"\bi’m not sure\b", r"\bi'm not sure\b",
        r"\bi cannot\b", r"\bi can't\b", r"\bnot certain\b", r"\bunclear\b", r"\buncertain\b",
        r"\bprobably\b", r"\bmaybe\b", r"\bpossibly\b", r"\bmight be\b", r"\bcould be\b",
        r"\bi think\b", r"\bi believe\b", r"\bas far as i know\b", r"\bto the best of my knowledge\b",
        r"\bi don't have access\b", r"\bi cannot access\b", r"\bmy knowledge cutoff\b"
    ]

    response_lower = response.lower()
    uncertainty_count = 0
    for pattern in uncertainty_phrases:
        if re.search(pattern, response_lower):
            uncertainty_count += 1

    # If response is very short, it might be light-weight but not necessarily evasive
    if len(response.strip()) < 20:
        return {
            'score': 70.0,
            'reasoning': 'Very short response',
            'should_search': True
        }

    # Softer penalty for uncertainty markers to avoid over-triggering
    if uncertainty_count >= 3:
        return {
            'score': 55.0,
            'reasoning': f'High uncertainty ({uncertainty_count} markers)',
            'should_search': True
        }
    elif uncertainty_count >= 1:
        base_score = max(60.0, 85.0 - (uncertainty_count * 10))
        return {
            'score': base_score,
            'reasoning': f'Moderate uncertainty ({uncertainty_count} markers)',
            'should_search': base_score < 70
        }
    # Check if user was TELLING information (not asking)
    # These should NEVER trigger low confidence or search
    telling_indicators = [
        r'\b(built|created|made|developed|wrote|coded|fixed|updated|added)\b',
        r'\b(was|were|has been|have been)\b.*\b(built|created|developed)',
        r'^(here is|here are|this is|these are|let me tell you|for context)',
        r'\d+%',  # Percentages
        r'around \d+',  # Statistics
    ]

    query_lower = user_query.lower()
    is_telling = any(re.search(pattern, query_lower) for pattern in telling_indicators)

    if is_telling and '?' not in user_query:
        return {
            'score': 95.0,
            'reasoning': 'User providing information, not asking - no search needed',
            'should_search': False
        }
    # Temporal detection - user asking about recent events
    temporal_keywords = [
        "current", "latest", "today", "yesterday", "this week", "this month",
        "recent", "now", "right now", "what's happening", "breaking",
        "2025", "2024"  # Recent years
    ]

    query_lower = user_query.lower()
    is_temporal_query = any(keyword in query_lower for keyword in temporal_keywords)

    # Check if model admits outdated knowledge
    outdated_indicators = [
        "my training data", "as of my last update", "knowledge cutoff",
        "i was trained", "my training ended"
    ]
    mentions_outdated = any(indicator in response_lower for indicator in outdated_indicators)

    if is_temporal_query and mentions_outdated:
        return {
            'score': 50.0,
            'reasoning': 'Temporal query + model admits outdated knowledge',
            'should_search': True
        }

    # Use Nova to score the response quality
    prompt = f"""You are a confidence scorer. Analyze this AI response and score its confidence level.

User Query: {user_query}

AI Response: {response}

Score the response on these criteria:
1. Factual specificity (names, dates, numbers vs vague statements)
2. Internal consistency (no contradictions)
3. Appropriate confidence level (admits uncertainty when warranted)
4. Relevance to query

Output ONLY a number from 0-100, where:
- 90-100: Highly confident, specific, factual
- 70-89: Good answer, minor uncertainties
- 50-69: Moderate issues, some uncertainty
- 30-49: Low confidence, vague or evasive
- 0-29: Very uncertain, likely incorrect

Score:"""

    messages = [
        {"role": "system", "content": "You are a response quality evaluator. Output ONLY a number from 0-100."},
        {"role": "user", "content": prompt}
    ]

    try:
        response_obj = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )

        raw_score = response_obj.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        # Extract number from response
        score_match = re.search(r'(\d+)', raw_score)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(100.0, score))  # Clamp to 0-100

            reasoning = "Nova scored response"
            if score < 50:
                reasoning += " (low confidence detected)"
            elif score < 80:
                reasoning += " (moderate confidence)"
            else:
                reasoning += " (high confidence)"

            return {
                'score': score,
                'reasoning': reasoning,
                'should_search': score < 80
            }
        else:
            print(f"⚠️ Could not parse score from: {raw_score}")
            return {'score': 75.0, 'reasoning': 'Parse error, assuming moderate', 'should_search': False}

    except Exception as e:
        print(f"❌ Confidence scoring failed: {e}")
        return {'score': 75.0, 'reasoning': f'Scoring error: {str(e)}', 'should_search': False}

def summarize_for_memory(user_input: str, ai_response: str) -> list:
    """
    Extracts concise factual statements from a conversation exchange.
    Removes conversational fluff and returns key facts suitable for long-term memory.
    Returns: list of concise factual strings
    """
    if orchestrator_model is None:
        print("⚠️ Orchestrator model not loaded. Cannot summarize for memory.")
        return []

    # Skip summarization for very short exchanges
    if len(user_input) < 10 and len(ai_response) < 20:
        return []

    prompt = f"""Extract ONLY genuinely meaningful facts from this conversation. 

Guidelines:
- Focus on: user interests, specific information (names/tech/preferences), problems being solved, skill level
- IGNORE: greetings, pleasantries, casual conversation, generic responses
- Quality over quantity: Extract 0-5 facts depending on information density
- If the exchange is just casual chat with no real info, return NOTHING

Output format: One fact per line. No numbering, bullets, or labels.

User: {user_input}
AI: {ai_response}

Meaningful Facts (if any):"""

    messages = [
        {"role": "system", "content": "You are a fact extractor. Only output genuinely meaningful facts. If there are no meaningful facts, output nothing. No meta-commentary, no fluff, no generic observations."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )

        raw_content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        # Parse facts (split by newlines, clean up)
        facts = []
        for line in raw_content.split('\n'):
            line = line.strip()
            # Remove bullet points, numbers, etc.
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line)
            line = line.strip()

            # Skip empty lines or very short lines
            if len(line) < 20:
                continue
            if line.lower().startswith(('note:', 'fact:', 'key fact:', 'summary:')):
                line = re.sub(r'^[^:]+:\s*', '', line, flags=re.IGNORECASE).strip()

            # Skip lines that are too generic
            generic_phrases = [
                'the user asked', 'the ai responded', 'conversation about', 'discussion about',
                'user wants', 'user is asking', 'ai explains', 'ai suggests',
                'greeting exchange', 'casual conversation'
            ]
            if any(phrase in line.lower() for phrase in generic_phrases):
                continue

            if line and len(line) >= 20:
                facts.append(line)

        if facts:
            print(f" Extracted {len(facts)} facts for memory")
        else:
            print(f" No meaningful facts extracted (casual conversation)")
        return facts

    except Exception as e:
        print(f"❌ Memory summarization failed: {e}")
        return []

def should_perform_web_search_intelligent(user_query: str) -> dict:
    """
    Intelligently determines if a web search is needed using heuristics + Nova.
    Returns: {'should_search': bool, 'reasoning': str, 'confidence': float}
    """
    import re
    
    # Fast path: Heuristic checks first
    query_lower = user_query.lower()

    # Check for common greeting patterns first (ANTI-pattern - don't search these)
    greeting_patterns = [
        r'\bhow are you( doing)?\b',
        r'\bhow\'?s it going\b',
        r'\bwhat\'?s up\b',
        r'\bgood morning|good afternoon|good evening\b',
        r'\bhello|hi|hey\b.*\btoday\b'
    ]
    is_greeting = any(re.search(pattern, query_lower) for pattern in greeting_patterns)
    if is_greeting:
        return {'should_search': False, 'reasoning': 'Greeting detected', 'confidence': 0.95}

    # Temporal keywords indicating recent/current events
    temporal_patterns = [
        r'\b(current|latest|recent)\b.*(price|news|event|update|development|release)',
        r'\b(what happened|what\'?s happening)\b',
        r'\b(yesterday|this week|this month|today)\b.*(happen|occur|news)',
        r'\b(now|right now|at the moment|currently)\b.*(price|market|weather)',
        r'\b(2025|2024|202[3-5])\b.*(event|news|update)',
        r'\b(breaking|update) news\b',
        r'\b(upcoming|scheduled|planned)\b.*(event|releases?|launchs?)',
        # NEW: Single-word temporal triggers
        r'\b(anything|something|what\'?s)\s+(new|recent|latest)\b',
        r'\bnew\s+(in|from|about)\b',
        r'\b(any|latest)\s+(updates?|news|developments?)\b',
        r'\b(recent|latest)\s+(advances?|innovations?|breakthroughs?)\b',
    ]

    has_temporal = any(re.search(pattern, query_lower) for pattern in temporal_patterns)

    # Data request patterns
    data_patterns = [
        r'\b(price of|cost of|value of|worth of)\b',
        r'\b(show me|find|search for|look up|link to)\b',
        r'\b(weather|temperature|forecast)\b',
        r'\b(stock|crypto|bitcoin|market)\b',
        r'\b(score|results|standings|match)\b',
        r'\b(status|schedule|times|availability)\b'
    ]

    has_data_request = any(re.search(pattern, query_lower) for pattern in data_patterns)

    # NEW: Check for rapidly-changing field queries
    field_patterns = [
        r'\bin the (AI|ML|tech|technology|crypto|market|stock) field\b',
        r'\b(AI|ML|tech|crypto) (space|industry|sector|world)\b',
        r'\bfrom (openai|anthropic|google|microsoft|meta|nvidia|apple)\b',
    ]

    has_field_context = any(re.search(pattern, query_lower) for pattern in field_patterns)

    # Question indicators
    is_question = '?' in user_query or any(query_lower.startswith(q) for q in ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'do', 'does'])

    # Fast decision for obvious cases
    if has_temporal and is_question:
        return {
            'should_search': True,
            'reasoning': 'Temporal query detected',
            'confidence': 0.9
        }
    
    # NEW: Rapidly-changing field with question
    if has_field_context and is_question:
        return {
            'should_search': True,
            'reasoning': 'Question about rapidly-changing field',
            'confidence': 0.85
        }

    if has_data_request:
        return {
            'should_search': True,
            'reasoning': 'Data request detected',
            'confidence': 0.85
        }

    # If heuristics are unclear, ask Nova
    if orchestrator_model is None or len(user_query.strip()) < 10:
        return {'should_search': False, 'reasoning': 'No clear indicators', 'confidence': 0.6}

    # Use Nova for intelligent decision
    prompt = f"""Analyze this user query carefully:

Query: "{user_query}"

First, determine: Is the user ASKING for information — or PROVIDING information?

ASKING (search YES):
- "What is the price of Bitcoin?"
- "Search for recent AI news"
- "What happened in tech today?"
- "How does the new iPhone compare to Android?"
- "Who won the match last night?"

PROVIDING (search NO — ALWAYS return NO):
- "GPT-4o built the original version"
- "Here's the history of my project: DeepSeek fixed the memory leak"
- "The code was created by Qwen3 2507"
- "Around 95% of the code was AI-generated"
- "Let me tell you about my setup"
- "For context, I'm using a 12GB GPU"

Key indicators user is PROVIDING:
- Past tense verbs: built, created, made, fixed, added, was, were
- Starting phrases: "Here is", "Here are", "Let me tell you", "For context", "In my project"
- Percentages or stats without questions: "95%", "around 20%", "over 80%"
- No question mark in the sentence

If user is PROVIDING information — answer: NO

If user is ASKING, then check:
- Does it need real-time data? (prices, weather, scores, news, live events)
- Would the answer change tomorrow? (dates, events, "latest", "today", "now")
- Does it reference specific recent events? (2024, 2025, "this week")

Answer ONLY: YES or NO

Answer:"""

    messages = [
        {"role": "system", "content": "You are a query analyzer. Answer ONLY with YES or NO."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = orchestrator_model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=5
        )

        raw_answer = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip().upper()

        if 'YES' in raw_answer:
            return {
                'should_search': True,
                'reasoning': 'Nova determined search needed',
                'confidence': 0.8
            }
        else:
            return {
                'should_search': False,
                'reasoning': 'Nova determined no search needed',
                'confidence': 0.75
            }

    except Exception as e:
        print(f"⚠️ Intelligent search detection failed: {e}")
        return {'should_search': False, 'reasoning': 'Detection error', 'confidence': 0.5}

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

def get_orchestrator_response(user_input: str, history: list = [], model_override=None, provider_override=None, stream: bool=False):
    """
    Generates a direct response from the orchestrator model with optional overrides.
    """
    orchestrator_type = ORCHESTRATOR_CONFIG.get('type', 'local')

    if orchestrator_type == "local":
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
                temperature=0.5,
                max_tokens=1024
            )
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            return content
        except Exception as e:
            print(f"Orchestrator response generation failed: {e}")
            return "I'm sorry, I had a problem generating a response."

    if orchestrator_type == "api":
        provider = provider_override or ORCHESTRATOR_CONFIG['api_provider']
        model = model_override or ORCHESTRATOR_CONFIG['api_model']
        api_orchestrator = create_api_orchestrator(provider, model)
        try:
            return api_orchestrator(user_input, stream=stream)
        except Exception as e:
            return f"Orchestrator API error ({provider}/{model}): {e}"

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
            "search", "youtube", "browser", "go", "read", "image",
            "map", "maps", "wiki", "weather", "calc", "define", "play",
            "media"
        ]

        # Check for compound commands (e.g., wiki-summarize)
        if '-' in command:
            parts = command.split('-')
            main_command = parts[0]
            modifier = '-'.join(parts[1:])
            if main_command in valid_commands:
                print(f"Orchestrator parsed compound command: /{command} with query: '{query}'")
                return {"command": command, "query": query, "main": main_command, "modifier": modifier}

        if command in valid_commands:
            print(f"Orchestrator parsed command: /{command} with query: '{query}'")
            if command == 'youtube':
                return {"command": "youtube", "query": query}
            elif command == 'image':
                return {"command": "image", "query": query}
            elif command == 'media':
                return {"command": "media", "query": query}
            elif command in ['map', 'maps']:
                return {"command": "maps", "query": query}
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

# ============================================================================
# INTROSPECTION TOOLS - Added by The AI Parliament, Nov 19 2025
# These tools allow Nova and other models to inspect their own tool usage
# ============================================================================

from tool_state import (
    get_tool_history, 
    get_last_tool_output, 
    get_search_results,
    get_missing_tool_requests
)

def inspect_search_results(query: str = None, limit: int = 3):
    """
    Returns detailed information about recent web search results.
    If query is provided, returns results for that specific search.
    Otherwise, returns the most recent search results.
    
    Useful for debugging search quality or answering meta-questions
    about what information was retrieved.
    """
    print(f"Executing inspect_search_results for query: {query or 'most recent'}")
    try:
        result = get_search_results(query)
        
        if not result:
            return "No recent search results found." if not query else f"No results found for query: '{query}'"
        
        # Format the output nicely
        output = f"""🔍 Search Results Inspection

Query: {result.get('query', 'N/A')}
Timestamp: {result.get('timestamp', 'N/A')}

Results:
{result.get('result', 'No content')[:2000]}

--- End of Search Results ---
"""
        return output
        
    except Exception as e:
        print(f"inspect_search_results failed: {e}")
        return f"Error inspecting search results: {e}"


def inspect_last_tool_output(tool_name: str = None):
    """
    Returns the complete output from the most recent tool execution.
    If tool_name is specified, returns the last output from that specific tool.
    
    Useful for debugging tool behavior or understanding what data
    was returned by previous operations.
    """
    print(f"Executing inspect_last_tool_output for tool: {tool_name or 'most recent'}")
    try:
        result = get_last_tool_output(tool_name)
        
        if not result:
            if tool_name:
                return f"No execution history found for tool: '{tool_name}'"
            else:
                return "No tool execution history available."
        
        # Format the output
        if tool_name:
            output = f"""🔧 Tool Output Inspection: {tool_name}

Timestamp: {result.get('timestamp', 'N/A')}
Success: {result.get('success', False)}

Output:
{str(result.get('result', 'No output'))[:2000]}

--- End of Tool Output ---
"""
        else:
            output = f"""🔧 Last Tool Execution

Tool: {result.get('tool_name', 'Unknown')}
Timestamp: {result.get('timestamp', 'N/A')}
Success: {result.get('success', False)}

Output:
{str(result.get('result', 'No output'))[:2000]}

--- End of Tool Output ---
"""
        return output
        
    except Exception as e:
        print(f"inspect_last_tool_output failed: {e}")
        return f"Error inspecting tool output: {e}"


def inspect_tool_history(limit: int = 10):
    """
    Returns a summary of recent tool executions.
    Shows tool names, timestamps, and success/failure status.
    
    Useful for understanding the sequence of operations that led
    to the current state, or debugging workflow issues.
    """
    print(f"Executing inspect_tool_history with limit: {limit}")
    try:
        history = get_tool_history(limit)
        
        if not history:
            return "No tool execution history available."
        
        # Format as a timeline
        output = f"🕐 Tool Execution History (last {len(history)} executions)\n\n"
        
        for i, execution in enumerate(reversed(history), 1):
            status = "✅" if execution.get('success', False) else "❌"
            tool_name = execution.get('tool_name', 'Unknown')
            timestamp = execution.get('timestamp', 'N/A')
            args = execution.get('arguments', {})
            
            output += f"{i}. {status} {tool_name}\n"
            output += f"   Time: {timestamp}\n"
            if args:
                args_str = ", ".join([f"{k}={v}" for k, v in list(args.items())[:2]])
                output += f"   Args: {args_str}\n"
            output += "\n"
        
        output += "--- End of History ---\n"
        return output
        
    except Exception as e:
        print(f"inspect_tool_history failed: {e}")
        return f"Error inspecting tool history: {e}"


def inspect_request_context():
    """
    Returns metadata about the current system state and configuration.
    Shows active models, loaded tools, user info, etc.
    
    Useful for debugging configuration issues or understanding
    the current system capabilities.
    """
    print("Executing inspect_request_context")
    try:
        import sys
        
        # Import TOOL_REGISTRY from the main tools module
        from tools import TOOL_REGISTRY
        
        # Gather system info
        context = {
            "python_version": sys.version.split()[0],
            "available_tools": list(TOOL_REGISTRY.keys()),
            "tool_count": len(TOOL_REGISTRY),
        }
        
        # Try to get memory stats if available
        try:
            from upgraded_memory_manager import memory_manager
            context["memory_stats"] = {
                "total_memories": memory_manager.collection.count_documents({}),
            }
        except:
            context["memory_stats"] = {"total_memories": "N/A"}
        
        # Try to get current model info if available
        try:
            from main_server import current_model_path_string, current_backend
            context["active_model"] = current_model_path_string
            context["backend"] = current_backend
        except:
            context["active_model"] = "Unknown"
            context["backend"] = "Unknown"
        
        # Format output
        output = f"""📊 System Context Inspection

Python Version: {context['python_version']}
Active Model: {context.get('active_model', 'N/A')}
Backend: {context.get('backend', 'N/A')}

Available Tools ({context['tool_count']}):
{', '.join(context['available_tools'])}

Memory Statistics:
- Total Memories: {context['memory_stats']['total_memories']}

--- End of Context ---
"""
        return output
        
    except Exception as e:
        print(f"inspect_request_context failed: {e}")
        return f"Error inspecting request context: {e}"


def inspect_missing_tool_requests(limit: int = 10):
    """
    Returns a log of recent requests for non-existent tools.
    This helps identify what capabilities Nova and other models
    are trying to use but don't have access to yet.
    
    Useful for understanding emergent tool needs and guiding
    future tool development.
    """
    print(f"Executing inspect_missing_tool_requests with limit: {limit}")
    try:
        missing = get_missing_tool_requests(limit)
        
        if not missing:
            return "No missing tool requests logged."
        
        output = f"📝 Missing Tool Requests (last {len(missing)})\n\n"
        
        for i, request in enumerate(reversed(missing), 1):
            tool_name = request.get('tool_name', 'Unknown')
            timestamp = request.get('timestamp', 'N/A')
            args = request.get('arguments', {})
            
            output += f"{i}. {tool_name}\n"
            output += f"   Time: {timestamp}\n"
            if args:
                args_str = ", ".join([f"{k}={v}" for k, v in list(args.items())[:2]])
                output += f"   Requested Args: {args_str}\n"
            output += "\n"
        
        output += "--- End of Missing Tool Requests ---\n"
        output += "\nThese requests suggest tools that could be added to improve system capabilities.\n"
        return output
        
    except Exception as e:
        print(f"inspect_missing_tool_requests failed: {e}")
        return f"Error inspecting missing tool requests: {e}"


# ============================================================================
# INTROSPECTION TOOL SCHEMAS - Add these to TOOLS_SCHEMA
# ============================================================================

INTROSPECTION_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "inspect_search_results",
            "description": "Inspects the detailed results from recent web searches. Use this to see the raw search data, verify search quality, or answer questions about what information was retrieved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional: The specific search query to inspect. If not provided, shows the most recent search."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to show (default: 3)",
                        "default": 3
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_last_tool_output",
            "description": "Returns the complete output from a recent tool execution. Use this to examine what data a tool returned, debug tool behavior, or answer meta-questions about previous operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Optional: Name of the specific tool to inspect (e.g., 'search_web', 'scrape_website'). If not provided, shows the most recent tool output."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_tool_history",
            "description": "Shows a timeline of recent tool executions with status, timestamps, and arguments. Useful for debugging workflows or understanding the sequence of operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent executions to show (default: 10)",
                        "default": 10
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_request_context",
            "description": "Returns system metadata including active models, available tools, memory stats, and configuration. Use this to understand current system capabilities or debug configuration issues.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_missing_tool_requests",
            "description": "Shows a log of tools that were requested but don't exist yet. This reveals what capabilities models are trying to use, guiding future tool development.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of missing tool requests to show (default: 10)",
                        "default": 10
                    }
                }
            }
        }
    }
]

# ============================================================================
# REGISTRY ADDITIONS - Add these to TOOL_REGISTRY
# ============================================================================

INTROSPECTION_TOOL_REGISTRY = {
    "inspect_search_results": inspect_search_results,
    "inspect_last_tool_output": inspect_last_tool_output,
    "inspect_tool_history": inspect_tool_history,
    "inspect_request_context": inspect_request_context,
    "inspect_missing_tool_requests": inspect_missing_tool_requests,
}

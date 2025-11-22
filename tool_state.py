"""
Global state tracker for tool executions.
Allows introspection tools to access recent tool history.
"""

from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
import threading

# Thread-safe storage for tool execution history
_lock = threading.Lock()
_tool_history = deque(maxlen=50)  # Keep last 50 tool executions
_search_results_cache = {}  # Cache search results by query
_last_tool_output = {}  # Store last output per tool type

def record_tool_execution(tool_name: str, arguments: Dict[str, Any], result: Any, success: bool = True):
    """Record a tool execution for later introspection."""
    with _lock:
        execution = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        _tool_history.append(execution)
        
        # Update last output cache
        _last_tool_output[tool_name] = {
            "result": result,
            "timestamp": execution["timestamp"],
            "success": success
        }
        
        # Cache search results
        if tool_name == "search_web" and success:
            query = arguments.get("query", "")
            _search_results_cache[query] = {
                "result": result,
                "timestamp": execution["timestamp"]
            }

def get_tool_history(limit: int = 10) -> list:
    """Get recent tool execution history."""
    with _lock:
        return list(_tool_history)[-limit:]

def get_last_tool_output(tool_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the last output from a specific tool, or the most recent tool overall."""
    with _lock:
        if tool_name:
            return _last_tool_output.get(tool_name)
        else:
            # Return most recent from history
            if _tool_history:
                last = _tool_history[-1]
                return {
                    "tool_name": last["tool_name"],
                    "result": last["result"],
                    "timestamp": last["timestamp"],
                    "success": last["success"]
                }
            return None

def get_search_results(query: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get cached search results for a query, or the most recent search."""
    with _lock:
        if query:
            return _search_results_cache.get(query)
        else:
            # Return most recent search
            for execution in reversed(_tool_history):
                if execution["tool_name"] == "search_web" and execution["success"]:
                    return {
                        "query": execution["arguments"].get("query"),
                        "result": execution["result"],
                        "timestamp": execution["timestamp"]
                    }
            return None

def clear_history():
    """Clear all cached tool history (for testing/debugging)."""
    with _lock:
        _tool_history.clear()
        _search_results_cache.clear()
        _last_tool_output.clear()

# Track missing tool requests (for emergence learning)
_missing_tools = deque(maxlen=100)

def log_missing_tool_request(tool_name: str, arguments: Dict[str, Any]):
    """Log when Nova requests a tool that doesn't exist."""
    with _lock:
        _missing_tools.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        })
    print(f"📝 Missing tool request logged: {tool_name}")

def get_missing_tool_requests(limit: int = 20) -> list:
    """Get recent requests for non-existent tools."""
    with _lock:
        return list(_missing_tools)[-limit:]

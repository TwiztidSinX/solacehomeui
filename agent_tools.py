"""
Dedicated Tool System for SolaceOS Agentic Coder
=================================================

This module provides a dedicated, consistent, and robust set of tools for the agentic coder.
It is designed to be used by the `AgenticCodingOrchestrator` and is completely separate
from the main `tools.py` file used by the SolaceOS orchestrator.

Key Features:
- A dedicated `CODING_TOOLS_SCHEMA` for the agent.
- A `Workspace` class to manage the agent's file operations.
- A `ToolGuardian` that validates tool calls.
- The `agent_` prefix for all coding-related tools.
- Clear documentation for each tool.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import fnmatch
import re

# =========================================================================
# Workspace Management
# =========================================================================

class Workspace:
    """
    Manages the agent's workspace, ensuring all file operations are confined
    to the designated directory.
    """
    def __init__(self, workspace_root: str):
        self.root = Path(workspace_root).resolve()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        """
        Resolves a path relative to the workspace root and ensures it is
        within the workspace.
        """
        # Normalize slashes and remove any weirdness
        normalized_path_part = os.path.normpath(path)
        
        # Prevent path traversal attacks
        if '..' in normalized_path_part.split(os.sep):
            raise PermissionError("Path traversal ('..') is not allowed.")

        # Join the root with the normalized path part
        full_path = self.root.joinpath(normalized_path_part).resolve()


        # Final check to ensure the path is within the workspace
        if self.root not in full_path.parents and full_path != self.root:
            raise PermissionError(f"Attempted to access a path outside the workspace: {full_path}")
            
        return full_path

    def is_safe_path(self, path: str) -> bool:
        """
        Checks if a path is safe and within the workspace.
        """
        try:
            self.resolve_path(path)
            return True
        except PermissionError:
            return False

# =========================================================================
# Tool Guardian
# =========================================================================

class ToolGuardian:
    """
    Validates tool calls before execution to ensure they are safe and correct.
    """
    def __init__(self, workspace: Workspace, tool_schema: List[Dict[str, Any]]):
        self.workspace = workspace
        self.tool_schema = {item['function']['name']: item['function'] for item in tool_schema}

    def validate(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Validates a tool call. Returns an error message if invalid, otherwise None.
        """
        if tool_name not in self.tool_schema:
            return f"Unknown tool: '{tool_name}'"

        schema_params = self.tool_schema[tool_name].get('parameters', {}).get('properties', {})
        required_params = self.tool_schema[tool_name].get('parameters', {}).get('required', [])

        # Check for missing required parameters
        for param in required_params:
            if param not in params:
                return f"Missing required parameter '{param}' for tool '{tool_name}'"

        # Check for unknown parameters
        for param in params:
            if param not in schema_params:
                return f"Unknown parameter '{param}' for tool '{tool_name}'"

        # Check path parameters for safety
        for param_name, param_schema in schema_params.items():
            if "path" in param_name.lower() and param_name in params:
                if not self.workspace.is_safe_path(params[param_name]):
                    return f"Invalid path: '{params[param_name]}'. Path is outside the workspace."

        return None

# =========================================================================
# Tool Schema
# =========================================================================

CODING_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "agent_list_directory",
            "description": "List files and directories in a path. Shows a tree-like structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list relative to the workspace root.", "default": "."},
                    "recursive": {"type": "boolean", "description": "Whether to list recursively.", "default": False},
                    "max_depth": {"type": "integer", "description": "Maximum recursion depth.", "default": 2}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_read_file",
            "description": "Read the contents of a file. Use this to examine code before making changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file relative to the workspace root."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_write_file",
            "description": "Write content to a file. This will create the file if it does not exist, or overwrite it if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file relative to the workspace root."},
                    "content": {"type": "string", "description": "The complete content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_append_file",
            "description": "Append content to a file. Creates the file if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file relative to the workspace root."},
                    "content": {"type": "string", "description": "The content to append."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_delete_file",
            "description": "Delete a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to delete relative to the workspace root."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_search_in_files",
            "description": "Search for text content within files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex to search for."},
                    "directory": {"type": "string", "description": "Directory to search in, relative to the workspace root.", "default": "."},
                    "file_pattern": {"type": "string", "description": "Only search in files matching this pattern (e.g., '*.py').", "default": "*"},
                    "is_regex": {"type": "boolean", "description": "Whether the query is a regex.", "default": False}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_run_shell_command",
            "description": "Run a shell command within the workspace. Use sparingly and only for necessary operations like installing dependencies or running tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_task_complete",
            "description": "Signal that the coding task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "A brief summary of the changes made."}
                },
                "required": ["summary"]
            }
        }
    }
]

# =========================================================================
# Tool Implementations
# =========================================================================

def agent_list_directory(workspace: Workspace, path: str = ".", recursive: bool = False, max_depth: int = 2) -> Dict[str, Any]:
    """Lists directory contents."""
    try:
        full_path = workspace.resolve_path(path)
        if not full_path.is_dir():
            return {'success': False, 'error': f"Directory not found: {path}"}

        entries = []
        
        if recursive:
            for root, dirs, files in os.walk(full_path):
                current_depth = len(Path(root).relative_to(full_path).parts)
                if current_depth >= max_depth:
                    dirs.clear() # Prune recursion
                    continue

                # Skip common ignored directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'dist', 'build']]
                
                rel_root = Path(root).relative_to(full_path)
                
                for d in dirs:
                    entries.append(str(rel_root / d) + '/')
                for f in files:
                    entries.append(str(rel_root / f))
        else:
            for item in full_path.iterdir():
                if item.is_dir():
                    entries.append(item.name + '/')
                else:
                    entries.append(item.name)
        
        return {
            'success': True,
            'entries': entries,
            'count': len(entries)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_read_file(workspace: Workspace, path: str) -> Dict[str, Any]:
    """Reads a file's contents."""
    try:
        full_path = workspace.resolve_path(path)
        if not full_path.is_file():
            return {'success': False, 'error': f"File not found: {path}"}
        
        content = full_path.read_text(encoding='utf-8')
        return {
            'success': True,
            'content': content,
            'lines': len(content.splitlines())
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_write_file(workspace: Workspace, path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file."""
    try:
        full_path = workspace.resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
        return {
            'success': True,
            'bytesWritten': len(content.encode('utf-8'))
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_append_file(workspace: Workspace, path: str, content: str) -> Dict[str, Any]:
    """Appends content to a file."""
    try:
        full_path = workspace.resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return {
            'success': True,
            'bytesAppended': len(content.encode('utf-8'))
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_delete_file(workspace: Workspace, path: str) -> Dict[str, Any]:
    """Deletes a file."""
    try:
        full_path = workspace.resolve_path(path)
        if not full_path.is_file():
            return {'success': False, 'error': f"File not found: {path}"}
        full_path.unlink()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_search_in_files(workspace: Workspace, query: str, directory: str = ".", file_pattern: str = "*", is_regex: bool = False) -> Dict[str, Any]:
    """Searches for text content within files."""
    try:
        full_path = workspace.resolve_path(directory)
        matches = []

        if is_regex:
            pattern = re.compile(query)
        
        for p in full_path.rglob(file_pattern):
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8")
                    if is_regex:
                        if pattern.search(content):
                            matches.append(str(p.relative_to(workspace.root)))
                    else:
                        if query in content:
                            matches.append(str(p.relative_to(workspace.root)))
                except Exception:
                    continue # Skip binary or unreadable files
        
        return {
            'success': True,
            'matches': matches
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_run_shell_command(workspace: Workspace, command: str, timeout: int = 60) -> Dict[str, Any]:
    """Runs a shell command in the workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': f'Command timed out after {timeout}s'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def agent_task_complete(workspace: Workspace, summary: str) -> Dict[str, Any]:
    """Signals that the task is complete."""
    return {
        'success': True,
        'summary': summary
    }

# =========================================================================
# Tool Dispatcher
# =========================================================================

TOOL_REGISTRY = {
    "agent_list_directory": agent_list_directory,
    "agent_read_file": agent_read_file,
    "agent_write_file": agent_write_file,
    "agent_append_file": agent_append_file,
    "agent_delete_file": agent_delete_file,
    "agent_search_in_files": agent_search_in_files,
    "agent_run_shell_command": agent_run_shell_command,
    "agent_task_complete": agent_task_complete,
}

def execute_tool(tool_name: str, params: Dict[str, Any], workspace: Workspace) -> Dict[str, Any]:
    """
    Executes a tool after validation.
    """
    guardian = ToolGuardian(workspace, CODING_TOOLS_SCHEMA)
    error = guardian.validate(tool_name, params)
    if error:
        return {'success': False, 'error': error}

    if tool_name not in TOOL_REGISTRY:
        return {'success': False, 'error': f"Tool '{tool_name}' is not implemented."}

    handler = TOOL_REGISTRY[tool_name]
    
    # Pass the workspace object to the tool
    return handler(workspace, **params)

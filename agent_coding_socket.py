"""
Socket.IO handlers for the Agentic Coding System

This module adds socket event handlers to integrate the agentic coding
orchestrator with the main SolaceOS backend.
"""

import os
import uuid
from typing import Any, Dict, Optional
from flask_socketio import emit
from flask import request as flask_request
from agentic_coding import AgenticCodingOrchestrator, CODING_TOOLS_SCHEMA


class AgentCodingSocketHandler:
    """
    Socket.IO handler for agentic coding operations.
    
    Integrates with the main Flask-SocketIO app to provide:
    - Session management
    - Real-time streaming of agent actions
    - Approval workflows
    """
    
    def __init__(self, socketio, model_client_factory=None):
        """
        Initialize the socket handler.
        
        Args:
            socketio: Flask-SocketIO instance
            model_client_factory: Callable that returns a model client
        """
        self.socketio = socketio
        self.model_client_factory = model_client_factory
        self.orchestrators: Dict[str, AgenticCodingOrchestrator] = {}
        
        # Register socket handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all socket event handlers."""
        
        @self.socketio.on('agent_coding_start')
        def handle_start(data):
            """Start a new agent coding session."""
            request = data.get('request', '')
            workspace_root = data.get('workspaceRoot', '.')
            sid = flask_request.sid
            
            if not request:
                emit('agent_coding_message', {
                    'type': 'session_error',
                    'sessionId': None,
                    'timestamp': 0,
                    'data': {'error': 'No request provided'}
                })
                return
            
            # Create orchestrator with message callback that streams in real-time
            def on_message(msg):
                # Use the SocketIO instance directly since this callback runs
                # outside of a request context.
                self.socketio.emit('agent_coding_message', msg, room=sid)
                # Yield to allow socket to flush the message immediately
                try:
                    import eventlet
                    eventlet.sleep(0)
                except ImportError:
                    try:
                        import gevent
                        gevent.sleep(0)
                    except ImportError:
                        pass  # No async runtime, continue anyway
            
            # Get model client if factory provided
            model_client = None
            if self.model_client_factory:
                try:
                    model_client = self.model_client_factory()
                except Exception as e:
                    print(f"Failed to create model client: {e}")
            
            orchestrator = AgenticCodingOrchestrator(
                model_client=model_client,
                workspace_root=workspace_root,
                on_message=on_message,
                max_iterations=20
            )
            
            # Generate a temporary ID for tracking the orchestrator
            # The real session ID will be created by run()
            temp_id = str(uuid.uuid4())[:8]
            self.orchestrators[temp_id] = orchestrator
            
            # Run in background thread - pass the request directly to run()
            # This avoids creating duplicate sessions
            self.socketio.start_background_task(
                self._run_session_direct,
                orchestrator,
                temp_id,
                request,
                workspace_root
            )
        
        @self.socketio.on('agent_coding_stop')
        def handle_stop(data):
            """Stop an active agent coding session."""
            session_id = data.get('sessionId')
            sid = flask_request.sid
            
            # Find orchestrator by session ID
            orchestrator = None
            for oid, orch in self.orchestrators.items():
                if orch.current_session and orch.current_session.id == session_id:
                    orchestrator = orch
                    break
            
            if orchestrator and orchestrator.current_session:
                orchestrator.current_session.error = "Stopped by user"
                from agentic_coding import AgentPhase
                orchestrator.current_session.phase = AgentPhase.ERROR
                
                self.socketio.emit('agent_coding_message', {
                    'type': 'session_error',
                    'sessionId': session_id,
                    'timestamp': 0,
                    'data': {'error': 'Session stopped by user'}
                }, room=sid)
        
        @self.socketio.on('agent_coding_approve')
        def handle_approve(data):
            """Approve pending changes."""
            session_id = data.get('sessionId')
            change_ids = data.get('changeIds')
            approve_all = data.get('approveAll', False)
            sid = flask_request.sid
            
            # Find orchestrator by session ID
            for oid, orchestrator in self.orchestrators.items():
                if orchestrator.current_session and orchestrator.current_session.id == session_id:
                    result = orchestrator.approve_changes(
                        session_id,
                        change_ids if not approve_all else None,
                        approve_all
                    )
                    
                    self.socketio.emit('agent_coding_message', {
                        'type': 'approval_received',
                        'sessionId': session_id,
                        'timestamp': 0,
                        'data': result
                    }, room=sid)
                    return
        
        @self.socketio.on('agent_coding_reject')
        def handle_reject(data):
            """Reject pending changes."""
            session_id = data.get('sessionId')
            reason = data.get('reason', '')
            sid = flask_request.sid
            
            # Find orchestrator by session ID
            for oid, orchestrator in self.orchestrators.items():
                if orchestrator.current_session and orchestrator.current_session.id == session_id:
                    result = orchestrator.reject_changes(
                        session_id,
                        reason
                    )
                    
                    self.socketio.emit('agent_coding_message', {
                        'type': 'session_error',
                        'sessionId': session_id,
                        'timestamp': 0,
                        'data': {'error': f'Changes rejected: {reason}'}
                    }, room=sid)
                    return
        
        @self.socketio.on('agent_coding_get_tools')
        def handle_get_tools(data):
            """Get the available coding tools schema."""
            sid = flask_request.sid
            self.socketio.emit('agent_coding_tools', {
                'tools': CODING_TOOLS_SCHEMA
            }, room=sid)
    
    def _run_session_direct(self, orchestrator: AgenticCodingOrchestrator, temp_id: str, 
                            request: str, workspace_root: str):
        """Run a coding session in a background thread."""
        real_session_id = None
        try:
            # run() will create the session and use the on_message callback
            result = orchestrator.run(request, workspace_root)
            real_session_id = result.get('id')
            
            # Update orchestrator tracking with real session ID
            if real_session_id and real_session_id != temp_id:
                self.orchestrators[real_session_id] = orchestrator
                
        except Exception as e:
            print(f"Agent coding session error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to get session ID for error message
            session_id = real_session_id or temp_id
            if orchestrator.current_session:
                session_id = orchestrator.current_session.id
                
            self.socketio.emit('agent_coding_message', {
                'type': 'session_error',
                'sessionId': session_id,
                'timestamp': 0,
                'data': {'error': str(e)}
            })
        finally:
            # Cleanup temp_id entry
            if temp_id in self.orchestrators:
                del self.orchestrators[temp_id]


def register_agent_coding_handlers(socketio, model_client_factory=None):
    """
    Register agent coding handlers with the SocketIO instance.
    
    Call this from your main server setup:
    
        from agent_coding_socket import register_agent_coding_handlers
        register_agent_coding_handlers(socketio, lambda: get_model_client())
    
    Args:
        socketio: Flask-SocketIO instance
        model_client_factory: Callable that returns a model client for LLM calls
    
    Returns:
        AgentCodingSocketHandler instance
    """
    return AgentCodingSocketHandler(socketio, model_client_factory)


# =============================================================================
# STANDALONE TOOL EXECUTION (for use without full orchestrator)
# =============================================================================

class AgentCodingTools:
    """
    Standalone tool execution for agent coding operations.
    
    Use this when you want Nova or another orchestrator to call
    individual tools without the full ReAct loop.
    """
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self._orchestrator = AgenticCodingOrchestrator(
            workspace_root=workspace_root,
            on_message=lambda x: None  # Silent
        )
        
        # Create a dummy session for tool execution
        self._session = self._orchestrator.create_session(
            "standalone_tools",
            workspace_root
        )
    
    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> Dict[str, Any]:
        """Read a file."""
        return self._orchestrator._action_read_file(self._session, {
            'path': path,
            'startLine': start_line,
            'endLine': end_line
        })
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write a file."""
        return self._orchestrator._action_write_file(self._session, {
            'path': path,
            'content': content
        })
    
    def edit_file(self, path: str, edits: list) -> Dict[str, Any]:
        """Apply surgical edits to a file."""
        return self._orchestrator._action_edit_file(self._session, {
            'path': path,
            'edits': edits
        })
    
    def create_file(self, path: str, content: str) -> Dict[str, Any]:
        """Create a new file."""
        return self._orchestrator._action_create_file(self._session, {
            'path': path,
            'content': content
        })
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file."""
        return self._orchestrator._action_delete_file(self._session, {
            'path': path
        })
    
    def rename_file(self, old_path: str, new_path: str) -> Dict[str, Any]:
        """Rename/move a file."""
        return self._orchestrator._action_rename_file(self._session, {
            'oldPath': old_path,
            'newPath': new_path
        })
    
    def list_directory(self, path: str = ".", recursive: bool = False, max_depth: int = 2) -> Dict[str, Any]:
        """List directory contents."""
        return self._orchestrator._action_list_directory(self._session, {
            'path': path,
            'recursive': recursive,
            'maxDepth': max_depth
        })
    
    def search_files(self, pattern: str, directory: str = ".", exclude_patterns: list = None) -> Dict[str, Any]:
        """Search for files by name."""
        return self._orchestrator._action_search_files(self._session, {
            'pattern': pattern,
            'directory': directory,
            'excludePatterns': exclude_patterns or []
        })
    
    def search_in_files(self, query: str, directory: str = ".", file_pattern: str = None, is_regex: bool = False) -> Dict[str, Any]:
        """Search for text in files."""
        return self._orchestrator._action_search_in_files(self._session, {
            'query': query,
            'directory': directory,
            'filePattern': file_pattern,
            'isRegex': is_regex
        })
    
    def run_command(self, command: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Run a shell command."""
        return self._orchestrator._action_run_command(self._session, {
            'command': command,
            'cwd': cwd,
            'timeout': timeout
        })


# Tool registry for Nova integration
def get_coding_tool_registry(workspace_root: str = ".") -> Dict[str, callable]:
    """
    Get a tool registry compatible with Nova's tool system.
    
    Returns a dict mapping tool names to callables.
    """
    tools = AgentCodingTools(workspace_root)
    
    return {
        'agent_read_file': lambda **kwargs: tools.read_file(**kwargs),
        'agent_write_file': lambda **kwargs: tools.write_file(**kwargs),
        'agent_edit_file': lambda **kwargs: tools.edit_file(**kwargs),
        'agent_create_file': lambda **kwargs: tools.create_file(**kwargs),
        'agent_delete_file': lambda **kwargs: tools.delete_file(**kwargs),
        'agent_rename_file': lambda old_path, new_path: tools.rename_file(old_path, new_path),
        'agent_list_directory': lambda **kwargs: tools.list_directory(**kwargs),
        'agent_search_files': lambda **kwargs: tools.search_files(**kwargs),
        'agent_search_in_files': lambda **kwargs: tools.search_in_files(**kwargs),
        'agent_run_command': lambda **kwargs: tools.run_command(**kwargs),
    }


if __name__ == "__main__":
    # Example usage
    tools = AgentCodingTools(".")
    
    # List current directory
    result = tools.list_directory()
    print("Directory listing:", result)
    
    # Search for Python files
    result = tools.search_files("*.py")
    print("Python files:", result)

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
            
            # PHASE 1: Use the already-loaded models
            # The model_client_factory returns the user's selected model
            # which is already loaded and working in the main chat
            
            coding_model = None
            orchestrator_model = None
            
            if self.model_client_factory:
                try:
                    print("🔧 Getting model for agentic coding...")
                    # Just get the model - it's already configured and wrapped
                    coding_model = self.model_client_factory()
                    
                    # Use same model for orchestrator for now
                    # (both will be the user's selected model - deepseek-chat in your case)
                    orchestrator_model = coding_model
                    
                    print(f"✅ Using model: {getattr(coding_model, 'model', 'unknown')}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to get model: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create orchestrator
            if not coding_model or not orchestrator_model:
                emit('agent_coding_message', {
                    'type': 'session_error',
                    'sessionId': None,
                    'timestamp': 0,
                    'data': {'error': 'Failed to load models. Check console for details.'}
                })
                return
            
            orchestrator = AgenticCodingOrchestrator(
                coding_model=coding_model,
                orchestrator_model=orchestrator_model,
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
        
        @self.socketio.on('agent_coding_answer')
        def handle_answer(data):
            """
            PHASE 3: User provides answer to agent's question.
            """
            session_id = data.get('sessionId')
            question_id = data.get('questionId')
            answer = data.get('answer', '')
            sid = flask_request.sid
            
            if not session_id or not question_id:
                self.socketio.emit('agent_coding_message', {
                    'type': 'error',
                    'data': {'error': 'Missing sessionId or questionId'}
                }, room=sid)
                return
            
            # Find orchestrator by session ID
            for oid, orchestrator in self.orchestrators.items():
                if orchestrator.current_session and orchestrator.current_session.id == session_id:
                    result = orchestrator.answer_question(session_id, question_id, answer)
                    
                    if result.get('success'):
                        # Answer received, continue execution
                        # Re-run the session to continue from where it left off
                        self.socketio.start_background_task(
                            self._continue_session,
                            orchestrator,
                            session_id
                        )
                    else:
                        self.socketio.emit('agent_coding_message', {
                            'type': 'error',
                            'data': result
                        }, room=sid)
                    return
            
            # Session not found
            self.socketio.emit('agent_coding_message', {
                'type': 'error',
                'data': {'error': f'Session {session_id} not found'}
            }, room=sid)
        
        @self.socketio.on('agent_coding_pause')
        def handle_pause(data):
            """
            PHASE 3: User requests to pause execution.
            """
            session_id = data.get('sessionId')
            reason = data.get('reason')
            sid = flask_request.sid
            
            # Find orchestrator by session ID
            for oid, orchestrator in self.orchestrators.items():
                if orchestrator.current_session and orchestrator.current_session.id == session_id:
                    result = orchestrator.pause_session(session_id, reason)
                    
                    self.socketio.emit('agent_coding_message', {
                        'type': 'session_paused' if result.get('success') else 'error',
                        'sessionId': session_id,
                        'data': result
                    }, room=sid)
                    return
        
        @self.socketio.on('agent_coding_resume')
        def handle_resume(data):
            """
            PHASE 3: User requests to resume paused execution.
            """
            session_id = data.get('sessionId')
            sid = flask_request.sid
            
            # Find orchestrator by session ID
            for oid, orchestrator in self.orchestrators.items():
                if orchestrator.current_session and orchestrator.current_session.id == session_id:
                    result = orchestrator.resume_session(session_id)
                    
                    if result.get('success'):
                        # Session resumed, continue execution
                        self.socketio.start_background_task(
                            self._continue_session,
                            orchestrator,
                            session_id
                        )
                    else:
                        self.socketio.emit('agent_coding_message', {
                            'type': 'error',
                            'data': result
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
    
    def _continue_session(self, orchestrator: AgenticCodingOrchestrator, session_id: str):
        """
        PHASE 3: Continue a session after user answered a question or resumed from pause.
        
        This picks up where the session left off and continues the execution loop.
        """
        try:
            session = orchestrator.current_session
            if not session or session.id != session_id:
                print(f"[AgentCodingSocket] Cannot continue session {session_id} - not found")
                return
            
            print(f"[AgentCodingSocket] Continuing session {session_id} from phase {session.phase.value}")
            
            # The session is already set up, just need to re-run the execution loop
            # This will continue from where it left off based on the phase
            result = orchestrator.run(session.user_request, session.workspace_root)
            
        except Exception as e:
            print(f"[AgentCodingSocket] Error continuing session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            
            self.socketio.emit('agent_coding_message', {
                'type': 'session_error',
                'sessionId': session_id,
                'timestamp': 0,
                'data': {'error': f'Failed to continue: {str(e)}'}
            })


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
    
    This does NOT require models - it just provides direct file system operations.
    """
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        
        # PHASE 1 FIX: Don't create an orchestrator for standalone tools
        # Just store workspace root and use direct operations
        # The orchestrator is only needed for full ReAct loops
        
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to workspace root."""
        if not path:
            return self.workspace_root
        
        # Normalize path
        path = path.replace('\\\\', '\\')
        path = os.path.normpath(path)
        
        # Make absolute
        if os.path.isabs(path):
            return path
        
        return os.path.normpath(os.path.join(self.workspace_root, path))
    
    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> Dict[str, Any]:
        """Read a file."""
        try:
            full_path = self._resolve_path(path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            if start_line is not None:
                start_idx = start_line - 1
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]
                content = '\n'.join(lines)
            
            return {'success': True, 'content': content, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write a file."""
        try:
            full_path = self._resolve_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {'success': True, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def edit_file(self, path: str, edits: list) -> Dict[str, Any]:
        """Apply surgical edits to a file."""
        try:
            full_path = self._resolve_path(path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply edits (simple search/replace)
            for edit in edits:
                old_text = edit.get('oldText', '')
                new_text = edit.get('newText', '')
                content = content.replace(old_text, new_text, 1)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {'success': True, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_file(self, path: str, content: str) -> Dict[str, Any]:
        """Create a new file."""
        try:
            full_path = self._resolve_path(path)
            if os.path.exists(full_path):
                return {'success': False, 'error': f'File already exists: {path}'}
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {'success': True, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file."""
        try:
            full_path = self._resolve_path(path)
            os.remove(full_path)
            return {'success': True, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def rename_file(self, old_path: str, new_path: str) -> Dict[str, Any]:
        """Rename/move a file."""
        try:
            old_full = self._resolve_path(old_path)
            new_full = self._resolve_path(new_path)
            os.rename(old_full, new_full)
            return {'success': True, 'oldPath': old_path, 'newPath': new_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_directory(self, path: str = ".", recursive: bool = False, max_depth: int = 2) -> Dict[str, Any]:
        """List directory contents."""
        try:
            full_path = self._resolve_path(path)
            entries = []
            
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    entries.append({'name': item, 'type': 'directory'})
                else:
                    entries.append({'name': item, 'type': 'file', 'size': os.path.getsize(item_path)})
            
            return {'success': True, 'entries': entries, 'path': path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_files(self, pattern: str, directory: str = ".", exclude_patterns: list = None) -> Dict[str, Any]:
        """Search for files by name."""
        try:
            import fnmatch
            full_dir = self._resolve_path(directory)
            matches = []
            
            for root, dirs, files in os.walk(full_dir):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        rel_path = os.path.relpath(os.path.join(root, filename), self.workspace_root)
                        matches.append(rel_path)
            
            return {'success': True, 'matches': matches}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_in_files(self, query: str, directory: str = ".", file_pattern: str = None, is_regex: bool = False) -> Dict[str, Any]:
        """Search for text in files."""
        try:
            import fnmatch
            import re
            full_dir = self._resolve_path(directory)
            results = []
            
            pattern = re.compile(query) if is_regex else None
            
            for root, dirs, files in os.walk(full_dir):
                for filename in files:
                    if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if is_regex:
                                    if pattern.search(line):
                                        results.append({'file': filepath, 'line': line_num, 'content': line.strip()})
                                else:
                                    if query in line:
                                        results.append({'file': filepath, 'line': line_num, 'content': line.strip()})
                    except:
                        pass
            
            return {'success': True, 'results': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_command(self, command: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Run a shell command."""
    def run_command(self, command: str, cwd: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Run a shell command."""
        try:
            import subprocess
            work_dir = self._resolve_path(cwd) if cwd else self.workspace_root
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f'Command timed out after {timeout}s'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


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
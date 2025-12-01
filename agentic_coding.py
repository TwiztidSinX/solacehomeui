"""
Agentic Coding System for SolaceOS
===================================

This module implements a ReAct-style coding agent that can:
- Read, write, and edit files
- Apply surgical diffs
- Search codebase
- Run commands
- Plan multi-step operations

The agent runs through a loop of:
1. Observation (read files, understand context)
2. Reasoning (think about what to do)
3. Action (execute a tool)
4. Result (process the outcome)
"""

import os
import json
import uuid
import time
import difflib
import subprocess
import fnmatch
import re
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class AgentPhase(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class FileEdit:
    """A single edit operation within a file."""
    old_text: str
    new_text: str
    description: str = ""
    
    def to_dict(self):
        return {
            "oldText": self.old_text,
            "newText": self.new_text,
            "description": self.description
        }


@dataclass
class DiffHunk:
    """A hunk in a unified diff."""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UnifiedDiff:
    """Represents a unified diff between two versions of a file."""
    old_path: str
    new_path: str
    hunks: List[DiffHunk] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "oldPath": self.old_path,
            "newPath": self.new_path,
            "hunks": [
                {
                    "oldStart": h.old_start,
                    "oldLines": h.old_lines,
                    "newStart": h.new_start,
                    "newLines": h.new_lines,
                    "lines": h.lines
                }
                for h in self.hunks
            ]
        }


@dataclass
class PendingChange:
    """A change awaiting user approval."""
    id: str
    change_type: str  # create, modify, delete, rename
    path: str
    new_path: Optional[str] = None
    diff: Optional[UnifiedDiff] = None
    new_content: Optional[str] = None
    approved: bool = False
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.change_type,
            "path": self.path,
            "newPath": self.new_path,
            "diff": self.diff.to_dict() if self.diff else None,
            "newContent": self.new_content,
            "approved": self.approved
        }


@dataclass
class ReasoningStep:
    """A step in the agent's reasoning process."""
    timestamp: float
    phase: str
    thought: str
    observation: Optional[str] = None
    conclusion: Optional[str] = None
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "thought": self.thought,
            "observation": self.observation,
            "conclusion": self.conclusion
        }


@dataclass
class AgentAction:
    """Base class for agent actions."""
    id: str
    action_type: str
    reasoning: str
    timestamp: float
    status: str = "pending"  # pending, executing, completed, failed, awaiting_approval
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.action_type,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "status": self.status,
            "params": self.params,
            "result": self.result
        }


@dataclass
class PlanStep:
    """A step in the agent's plan."""
    index: int
    description: str
    status: str = "pending"  # pending, active, completed, skipped
    actions: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status,
            "actions": self.actions
        }


@dataclass
class CodingSession:
    """Represents a coding session with the agent."""
    id: str
    user_request: str
    workspace_root: str
    phase: AgentPhase = AgentPhase.IDLE
    plan: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    actions: List[AgentAction] = field(default_factory=list)
    pending_changes: List[PendingChange] = field(default_factory=list)
    reasoning: List[ReasoningStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None
    files_modified: List[str] = field(default_factory=list)
    scratchpad: Dict[str, Any] = field(default_factory=dict)  # Track state between calls
    files_read: List[str] = field(default_factory=list)  # Track files already read
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Full LLM history
    
    def to_dict(self):
        return {
            "id": self.id,
            "userRequest": self.user_request,
            "workspaceRoot": self.workspace_root,
            "phase": self.phase.value,
            "plan": [s.to_dict() for s in self.plan],
            "currentStepIndex": self.current_step_index,
            "actions": [a.to_dict() for a in self.actions],
            "pendingChanges": [c.to_dict() for c in self.pending_changes],
            "reasoning": [r.to_dict() for r in self.reasoning],
            "startTime": self.start_time,
            "endTime": self.end_time,
            "error": self.error,
            "filesModified": self.files_modified,
            "filesRead": self.files_read,
            "scratchpad": self.scratchpad
        }


class AgenticCodingOrchestrator:
    """
    Main orchestrator for agentic coding operations.
    
    This class manages coding sessions, executes actions, and
    coordinates with the LLM to perform multi-step coding tasks.
    """
    
    def __init__(
        self,
        model_client: Any = None,
        workspace_root: str = ".",
        on_message: Optional[Callable] = None,
        max_iterations: int = 20,
        require_approval_for: Optional[List[str]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model_client: LLM client for generating actions
            workspace_root: Root directory for file operations
            on_message: Callback for streaming messages to UI
            max_iterations: Maximum number of action iterations
            require_approval_for: Action types that require user approval
        """
        self.model_client = model_client
        self.workspace_root = os.path.abspath(workspace_root)
        self.on_message = on_message or (lambda x: None)
        self.max_iterations = max_iterations
        self.require_approval_for = require_approval_for or ['delete_file', 'run_command']
        
        self.current_session: Optional[CodingSession] = None
        self.sessions: Dict[str, CodingSession] = {}
        
        # File content cache for generating diffs
        self._file_cache: Dict[str, str] = {}
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def create_session(self, user_request: str, workspace_root: Optional[str] = None) -> CodingSession:
        """Create a new coding session."""
        session_id = str(uuid.uuid4())[:8]
        root = workspace_root or self.workspace_root
        
        session = CodingSession(
            id=session_id,
            user_request=user_request,
            workspace_root=os.path.abspath(root)
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        self._emit_message('session_start', {
            'userRequest': user_request,
            'workspaceRoot': session.workspace_root
        })
        
        return session
    
    def get_session(self, session_id: str) -> Optional[CodingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    # =========================================================================
    # CORE AGENT LOOP
    # =========================================================================
    
    def run(self, user_request: str, workspace_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point - run a complete coding session.
        
        This implements the ReAct loop:
        1. Planning - Create a plan for the task
        2. Execution - Execute actions step by step
        3. Reflection - Reflect on results and adjust
        """
        session = self.create_session(user_request, workspace_root)
        
        try:
            # Phase 1: Planning
            self._planning_phase(session)
            
            # Phase 2: Execution Loop
            iteration = 0
            while session.phase in [AgentPhase.EXECUTING, AgentPhase.AWAITING_APPROVAL] and iteration < self.max_iterations:
                iteration += 1
                
                # If waiting for approval, don't get new action - just wait
                if session.phase == AgentPhase.AWAITING_APPROVAL:
                    # Exit loop - UI will call approve_changes() which resumes
                    # This returns control to the UI to handle approval
                    break
                
                # Get next action from LLM
                action = self._get_next_action(session)
                
                if action is None:
                    break
                
                # Check if approval needed
                if action.action_type in self.require_approval_for:
                    session.phase = AgentPhase.AWAITING_APPROVAL
                    self._request_approval(session, action)
                    # Store the action so it can be executed after approval
                    session.scratchpad['pending_action'] = action
                    # Exit the loop - approval handler will resume
                    break
                
                # Execute the action (only if not waiting for approval)
                result = self._execute_action(session, action)
                
                # Check for completion
                if action.action_type == 'complete':
                    session.phase = AgentPhase.COMPLETE
                    session.end_time = time.time()
                    break
                
                # Check for abort
                if action.action_type == 'abort':
                    session.phase = AgentPhase.ERROR
                    session.error = action.params.get('reason', 'Aborted by agent')
                    session.end_time = time.time()
                    break
                
                # Reflect on the result
                self._reflect_on_result(session, action, result)
            
            if iteration >= self.max_iterations:
                session.error = f"Max iterations ({self.max_iterations}) reached"
                session.phase = AgentPhase.ERROR
            
            # Only finalize if not waiting for approval
            if session.phase != AgentPhase.AWAITING_APPROVAL:
                # Finalize
                self._emit_message('session_complete', {
                    'summary': self._generate_summary(session),
                    'filesModified': session.files_modified,
                    'duration': (session.end_time or time.time()) - session.start_time
                })
            
        except Exception as e:
            session.phase = AgentPhase.ERROR
            session.error = str(e)
            session.end_time = time.time()
            self._emit_message('session_error', {'error': str(e)})
            raise
        
        return session.to_dict()
    
    def _planning_phase(self, session: CodingSession):
        """Create a plan for the coding task."""
        session.phase = AgentPhase.PLANNING
        self._emit_message('planning_start', {})
        
        # Build context for planning
        context = self._build_planning_context(session)
        
        # Get plan from LLM
        plan_prompt = self._create_planning_prompt(session, context)
        plan_response = self._call_llm(plan_prompt, expect_json=True)
        
        # Parse plan
        if isinstance(plan_response, dict) and 'plan' in plan_response:
            for i, step_desc in enumerate(plan_response['plan']):
                session.plan.append(PlanStep(
                    index=i,
                    description=step_desc
                ))
        
        # Add reasoning
        session.reasoning.append(ReasoningStep(
            timestamp=time.time(),
            phase=AgentPhase.PLANNING.value,
            thought=plan_response.get('reasoning', 'Created execution plan'),
            conclusion=f"Plan has {len(session.plan)} steps"
        ))
        
        self._emit_message('plan_created', {
            'plan': [s.to_dict() for s in session.plan],
            'reasoning': plan_response.get('reasoning', '')
        })
        
        session.phase = AgentPhase.EXECUTING
    
    def _get_next_action(self, session: CodingSession) -> Optional[AgentAction]:
        """Get the next action from the LLM."""
        # Build context with current state
        context = self._build_execution_context(session)
        
        # Get action decision from LLM
        action_prompt = self._create_action_prompt(session, context)
        action_response = self._call_llm(action_prompt, expect_json=True)
        
        if not action_response or not isinstance(action_response, dict):
            return None
        
        action_type = action_response.get('action')
        if not action_type:
            return None
        
        action = AgentAction(
            id=str(uuid.uuid4())[:8],
            action_type=action_type,
            reasoning=action_response.get('reasoning', ''),
            timestamp=time.time(),
            params=action_response.get('params', {})
        )
        
        session.actions.append(action)
        
        self._emit_message('action_start', {
            'action': action.to_dict(),
            'stepIndex': session.current_step_index
        })
        
        return action
    
    def _execute_action(self, session: CodingSession, action: AgentAction) -> Dict[str, Any]:
        """Execute an action and return the result."""
        action.status = 'executing'
        
        try:
            result = self._dispatch_action(session, action)
            action.status = 'completed'
            action.result = result
        except Exception as e:
            action.status = 'failed'
            action.result = {'success': False, 'error': str(e)}
            result = action.result
        
        self._emit_message('action_complete', {
            'actionId': action.id,
            'result': result
        })
        
        return result
    
    def _reflect_on_result(self, session: CodingSession, action: AgentAction, result: Dict[str, Any]):
        """Reflect on an action result and update reasoning."""
        session.phase = AgentPhase.REFLECTING
        
        observation = self._summarize_result(action, result)
        
        reasoning = ReasoningStep(
            timestamp=time.time(),
            phase=AgentPhase.REFLECTING.value,
            thought=f"Executed {action.action_type}",
            observation=observation,
            conclusion="Proceeding to next action" if result.get('success') else "Need to handle error"
        )
        
        session.reasoning.append(reasoning)
        
        self._emit_message('reasoning', reasoning.to_dict())
        
        session.phase = AgentPhase.EXECUTING
    
    def _request_approval(self, session: CodingSession, action: AgentAction):
        """
        Request user approval for a destructive action.
        
        This emits an approval_required message to the UI and updates
        the action status to awaiting_approval.
        """
        action.status = 'awaiting_approval'
        
        # Create a pending change for this action
        change_type = 'modify'
        if action.action_type == 'delete_file':
            change_type = 'delete'
        elif action.action_type == 'create_file':
            change_type = 'create'
        elif action.action_type == 'rename_file':
            change_type = 'rename'
        elif action.action_type == 'run_command':
            change_type = 'command'
        
        pending = PendingChange(
            id=str(uuid.uuid4())[:8],
            change_type=change_type,
            path=action.params.get('path', action.params.get('command', 'unknown')),
            new_path=action.params.get('newPath'),
            approved=False
        )
        
        session.pending_changes.append(pending)
        
        # Build descriptive message
        if action.action_type == 'delete_file':
            message = f"The agent wants to delete: {action.params.get('path')}"
        elif action.action_type == 'run_command':
            message = f"The agent wants to run command: {action.params.get('command')}"
        elif action.action_type == 'rename_file':
            message = f"The agent wants to rename {action.params.get('oldPath')} to {action.params.get('newPath')}"
        else:
            message = f"The agent wants to perform: {action.action_type}"
        
        self._emit_message('approval_required', {
            'changes': [pending.to_dict()],
            'message': message,
            'action': action.to_dict()
        })
        
        # Add reasoning about waiting for approval
        session.reasoning.append(ReasoningStep(
            timestamp=time.time(),
            phase=AgentPhase.AWAITING_APPROVAL.value,
            thought=f"Requesting approval for {action.action_type}",
            observation=message,
            conclusion="Waiting for user approval"
        ))
    
    # =========================================================================
    # ACTION DISPATCH
    # =========================================================================
    
    def _dispatch_action(self, session: CodingSession, action: AgentAction) -> Dict[str, Any]:
        """Dispatch an action to the appropriate handler."""
        handlers = {
            'read_file': self._action_read_file,
            'write_file': self._action_write_file,
            'edit_file': self._action_edit_file,
            'create_file': self._action_create_file,
            'delete_file': self._action_delete_file,
            'rename_file': self._action_rename_file,
            'list_directory': self._action_list_directory,
            'search_files': self._action_search_files,
            'search_in_files': self._action_search_in_files,
            'run_command': self._action_run_command,
            'complete': self._action_complete,
            'request_approval': self._action_request_approval,
            'abort': self._action_abort,
        }
        
        handler = handlers.get(action.action_type)
        if not handler:
            return {'success': False, 'error': f"Unknown action type: {action.action_type}"}
        
        return handler(session, action.params)
    
    # =========================================================================
    # ACTION HANDLERS
    # =========================================================================
    
    def _action_read_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file's contents."""
        # Handle both 'path' and 'file_path' (LLM sometimes hallucinates parameter names)
        path = params.get('path') or params.get('file_path', '')
        
        if not path:
            return {'success': False, 'error': 'No file path provided. Use "path" parameter.'}
        
        full_path = self._resolve_path(session, path)
        
        if not os.path.exists(full_path):
            return {'success': False, 'error': f"File not found: {path} (resolved to: {full_path})"}
        
        # Check if already read this file
        if path in session.files_read:
            session.scratchpad[f"duplicate_read_{path}"] = session.scratchpad.get(f"duplicate_read_{path}", 0) + 1
            # Still allow the read, but note it in scratchpad
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache for diff generation
            self._file_cache[full_path] = content
            
            # Track that we read this file
            if path not in session.files_read:
                session.files_read.append(path)
            
            # Update scratchpad
            session.scratchpad[f"file_content_{path}"] = content[:500]  # Store preview
            
            lines = content.split('\n')
            start_line = params.get('startLine', 1) - 1
            end_line = params.get('endLine', len(lines))
            
            selected_lines = lines[start_line:end_line]
            
            return {
                'success': True,
                'content': '\n'.join(selected_lines),
                'totalLines': len(lines),
                'linesRead': len(selected_lines),
                'note': f"Previously read this file {session.scratchpad.get(f'duplicate_read_{path}', 0)} times" if path in session.files_read else None
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_write_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file (full replacement)."""
        path = params.get('path', '')
        content = params.get('content', '')
        full_path = self._resolve_path(session, path)
        
        try:
            # Create parent directories if needed
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Generate diff if file exists
            old_content = self._file_cache.get(full_path, '')
            if os.path.exists(full_path) and not old_content:
                with open(full_path, 'r', encoding='utf-8') as f:
                    old_content = f.read()
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update cache
            self._file_cache[full_path] = content
            
            # Track modification
            if path not in session.files_modified:
                session.files_modified.append(path)
            
            # Generate and emit diff
            diff = self._generate_diff(path, old_content, content)
            self._emit_message('diff_preview', {
                'path': path,
                'diff': diff.to_dict(),
                'originalContent': old_content,
                'newContent': content
            })
            
            return {
                'success': True,
                'bytesWritten': len(content.encode('utf-8')),
                'diff': diff.to_dict()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_edit_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply surgical edits to a file."""
        path = params.get('path', '')
        edits = params.get('edits', [])
        full_path = self._resolve_path(session, path)
        
        if not os.path.exists(full_path):
            return {'success': False, 'error': f"File not found: {path}"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            edits_applied = 0
            errors = []
            
            for edit in edits:
                old_text = edit.get('oldText', '')
                new_text = edit.get('newText', '')
                
                if old_text not in content:
                    errors.append(f"Could not find text: {old_text[:50]}...")
                    continue
                
                # Check for uniqueness
                if content.count(old_text) > 1:
                    errors.append(f"Text is not unique (found {content.count(old_text)} occurrences): {old_text[:50]}...")
                    continue
                
                content = content.replace(old_text, new_text, 1)
                edits_applied += 1
            
            if edits_applied > 0:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self._file_cache[full_path] = content
                
                if path not in session.files_modified:
                    session.files_modified.append(path)
                
                # Generate and emit diff
                diff = self._generate_diff(path, original_content, content)
                self._emit_message('diff_preview', {
                    'path': path,
                    'diff': diff.to_dict(),
                    'originalContent': original_content,
                    'newContent': content
                })
            
            return {
                'success': edits_applied > 0,
                'editsApplied': edits_applied,
                'editsRequested': len(edits),
                'errors': errors if errors else None
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_create_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new file."""
        path = params.get('path', '')
        content = params.get('content', '')
        full_path = self._resolve_path(session, path)
        
        if os.path.exists(full_path):
            return {'success': False, 'error': f"File already exists: {path}"}
        
        try:
            # Create parent directories if needed
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self._file_cache[full_path] = content
            session.files_modified.append(path)
            
            # Emit preview
            self._emit_message('file_preview', {
                'path': path,
                'content': content,
                'isNew': True
            })
            
            return {
                'success': True,
                'bytesWritten': len(content.encode('utf-8'))
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_delete_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file."""
        path = params.get('path', '')
        full_path = self._resolve_path(session, path)
        
        if not os.path.exists(full_path):
            return {'success': False, 'error': f"File not found: {path}"}
        
        try:
            os.remove(full_path)
            
            if full_path in self._file_cache:
                del self._file_cache[full_path]
            
            session.files_modified.append(path)
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_rename_file(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rename/move a file."""
        old_path = params.get('oldPath', '')
        new_path = params.get('newPath', '')
        
        old_full = self._resolve_path(session, old_path)
        new_full = self._resolve_path(session, new_path)
        
        if not os.path.exists(old_full):
            return {'success': False, 'error': f"File not found: {old_path}"}
        
        if os.path.exists(new_full):
            return {'success': False, 'error': f"Destination already exists: {new_path}"}
        
        try:
            os.makedirs(os.path.dirname(new_full), exist_ok=True)
            os.rename(old_full, new_full)
            
            # Update cache
            if old_full in self._file_cache:
                self._file_cache[new_full] = self._file_cache[old_full]
                del self._file_cache[old_full]
            
            session.files_modified.append(old_path)
            session.files_modified.append(new_path)
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_list_directory(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents."""
        path = params.get('path', '.')
        recursive = params.get('recursive', False)
        max_depth = params.get('maxDepth', 2)
        
        full_path = self._resolve_path(session, path)
        
        if not os.path.exists(full_path):
            return {'success': False, 'error': f"Directory not found: {path}"}
        
        try:
            entries = []
            
            if recursive:
                for root, dirs, files in os.walk(full_path):
                    depth = root.replace(full_path, '').count(os.sep)
                    if depth >= max_depth:
                        dirs.clear()
                        continue
                    
                    # Skip common ignored directories
                    dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'dist', 'build']]
                    
                    rel_root = os.path.relpath(root, full_path)
                    
                    for d in dirs:
                        entries.append({
                            'name': d,
                            'path': os.path.join(rel_root, d) if rel_root != '.' else d,
                            'type': 'directory'
                        })
                    
                    for f in files:
                        file_path = os.path.join(root, f)
                        entries.append({
                            'name': f,
                            'path': os.path.join(rel_root, f) if rel_root != '.' else f,
                            'type': 'file',
                            'size': os.path.getsize(file_path)
                        })
            else:
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    entry = {
                        'name': item,
                        'path': item,
                        'type': 'directory' if os.path.isdir(item_path) else 'file'
                    }
                    if entry['type'] == 'file':
                        entry['size'] = os.path.getsize(item_path)
                    entries.append(entry)
            
            return {
                'success': True,
                'entries': entries,
                'count': len(entries)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_search_files(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for files by name pattern."""
        pattern = params.get('pattern', '*')
        directory = params.get('directory', '.')
        exclude_patterns = params.get('excludePatterns', ['node_modules', '.git', '__pycache__'])
        
        full_path = self._resolve_path(session, directory)
        
        try:
            matches = []
            
            for root, dirs, files in os.walk(full_path):
                # Apply exclusions
                dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, ex) for ex in exclude_patterns)]
                
                rel_root = os.path.relpath(root, full_path)
                
                for f in files:
                    if fnmatch.fnmatch(f, pattern) or pattern.lower() in f.lower():
                        matches.append({
                            'path': os.path.join(rel_root, f) if rel_root != '.' else f,
                            'name': f
                        })
            
            return {
                'success': True,
                'matches': matches[:100],  # Limit results
                'totalMatches': len(matches)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_search_in_files(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for text content within files."""
        query = params.get('query', '')
        directory = params.get('directory', '.')
        file_pattern = params.get('filePattern', '*')
        is_regex = params.get('isRegex', False)
        
        full_path = self._resolve_path(session, directory)
        
        try:
            matches = []
            
            if is_regex:
                pattern = re.compile(query)
            
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__']]
                
                rel_root = os.path.relpath(root, full_path)
                
                for f in files:
                    if not fnmatch.fnmatch(f, file_pattern):
                        continue
                    
                    file_path = os.path.join(root, f)
                    rel_path = os.path.join(rel_root, f) if rel_root != '.' else f
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                            for line_num, line in enumerate(fp, 1):
                                if is_regex:
                                    match = pattern.search(line)
                                    if match:
                                        matches.append({
                                            'path': rel_path,
                                            'line': line_num,
                                            'column': match.start() + 1,
                                            'preview': line.strip()[:100],
                                            'matchedText': match.group()
                                        })
                                else:
                                    if query in line:
                                        col = line.index(query) + 1
                                        matches.append({
                                            'path': rel_path,
                                            'line': line_num,
                                            'column': col,
                                            'preview': line.strip()[:100],
                                            'matchedText': query
                                        })
                    except:
                        continue
            
            return {
                'success': True,
                'matches': matches[:100],
                'totalMatches': len(matches)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_run_command(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a shell command."""
        command = params.get('command', '')
        cwd = params.get('cwd', session.workspace_root)
        timeout = params.get('timeout', 30)
        
        if not command:
            return {'success': False, 'error': 'No command provided'}
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': result.returncode == 0,
                'exitCode': result.returncode,
                'stdout': result.stdout[:5000] if result.stdout else '',
                'stderr': result.stderr[:5000] if result.stderr else ''
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f'Command timed out after {timeout}s'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _action_complete(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mark session as complete."""
        return {
            'success': True,
            'summary': params.get('summary', ''),
            'filesModified': params.get('filesModified', session.files_modified)
        }
    
    def _action_request_approval(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Request user approval."""
        message = params.get('message', '')
        changes = params.get('changes', [])
        
        for change in changes:
            pending = PendingChange(
                id=str(uuid.uuid4())[:8],
                change_type=change.get('type', 'modify'),
                path=change.get('path', ''),
                new_path=change.get('newPath'),
                approved=False
            )
            session.pending_changes.append(pending)
        
        self._emit_message('approval_required', {
            'changes': [c.to_dict() for c in session.pending_changes],
            'message': message
        })
        
        return {
            'success': True,
            'message': 'Approval requested',
            'pendingCount': len(session.pending_changes)
        }
    
    def _action_abort(self, session: CodingSession, params: Dict[str, Any]) -> Dict[str, Any]:
        """Abort the session."""
        return {
            'success': True,
            'reason': params.get('reason', 'Aborted by agent')
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _resolve_path(self, session: CodingSession, path: str) -> str:
        """
        Resolve a path relative to workspace root.
        
        Handles:
        - Absolute paths (returned as-is)
        - Relative paths (joined with workspace root)
        - Windows and Unix path separators
        - Prevents double-escaping issues
        """
        if not path:
            return session.workspace_root
        
        # CRITICAL FIX: Remove any double backslashes FIRST
        # This prevents F:\\\\solace-home-ui-v3 issues
        path = path.replace('\\\\', '\\')
        
        # Normalize the path - convert forward slashes to OS separator
        path = path.replace('/', os.sep)
        
        # Now normalize to clean up any remaining issues
        path = os.path.normpath(path)
        
        # Check if it's an absolute path
        if os.path.isabs(path):
            return path
        
        # Join with workspace root and normalize
        full_path = os.path.join(session.workspace_root, path)
        return os.path.normpath(full_path)
    
    def _generate_diff(self, path: str, old_content: str, new_content: str) -> UnifiedDiff:
        """Generate a unified diff between two versions of content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = UnifiedDiff(old_path=path, new_path=path)
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        hunks = []
        
        for group in matcher.get_grouped_opcodes(3):
            hunk_lines = []
            first_old = group[0][1]
            last_old = group[-1][2]
            first_new = group[0][3]
            last_new = group[-1][4]
            
            for tag, i1, i2, j1, j2 in group:
                if tag == 'equal':
                    for i, line in enumerate(old_lines[i1:i2]):
                        hunk_lines.append({
                            'type': 'context',
                            'content': line.rstrip('\n'),
                            'oldLineNumber': i1 + i + 1,
                            'newLineNumber': j1 + i + 1
                        })
                elif tag == 'replace':
                    for i, line in enumerate(old_lines[i1:i2]):
                        hunk_lines.append({
                            'type': 'remove',
                            'content': line.rstrip('\n'),
                            'oldLineNumber': i1 + i + 1
                        })
                    for i, line in enumerate(new_lines[j1:j2]):
                        hunk_lines.append({
                            'type': 'add',
                            'content': line.rstrip('\n'),
                            'newLineNumber': j1 + i + 1
                        })
                elif tag == 'delete':
                    for i, line in enumerate(old_lines[i1:i2]):
                        hunk_lines.append({
                            'type': 'remove',
                            'content': line.rstrip('\n'),
                            'oldLineNumber': i1 + i + 1
                        })
                elif tag == 'insert':
                    for i, line in enumerate(new_lines[j1:j2]):
                        hunk_lines.append({
                            'type': 'add',
                            'content': line.rstrip('\n'),
                            'newLineNumber': j1 + i + 1
                        })
            
            if hunk_lines:
                hunk = DiffHunk(
                    old_start=first_old + 1,
                    old_lines=last_old - first_old,
                    new_start=first_new + 1,
                    new_lines=last_new - first_new,
                    lines=hunk_lines
                )
                diff.hunks.append(hunk)
        
        return diff
    
    def _emit_message(self, msg_type: str, data: Dict[str, Any]):
        """Emit a message to the UI."""
        message = {
            'type': msg_type,
            'sessionId': self.current_session.id if self.current_session else None,
            'timestamp': time.time(),
            'data': data
        }
        self.on_message(message)
    
    def _summarize_result(self, action: AgentAction, result: Dict[str, Any]) -> str:
        """Create a human-readable summary of an action result."""
        if result.get('success'):
            if action.action_type == 'read_file':
                return f"Read {result.get('linesRead', 0)} lines from file"
            elif action.action_type == 'write_file':
                return f"Wrote {result.get('bytesWritten', 0)} bytes"
            elif action.action_type == 'edit_file':
                return f"Applied {result.get('editsApplied', 0)} edits"
            elif action.action_type == 'list_directory':
                return f"Found {result.get('count', 0)} entries"
            elif action.action_type == 'search_files':
                return f"Found {result.get('totalMatches', 0)} matching files"
            elif action.action_type == 'search_in_files':
                return f"Found {result.get('totalMatches', 0)} text matches"
            elif action.action_type == 'run_command':
                return f"Command exited with code {result.get('exitCode', -1)}"
            else:
                return f"Action {action.action_type} completed successfully"
        else:
            return f"Action failed: {result.get('error', 'Unknown error')}"
    
    def _build_scratchpad_summary(self) -> str:
        """Build a summary of the scratchpad for LLM context."""
        if not self.current_session:
            return ""
        
        summary_parts = ["=== SCRATCHPAD (Your Memory) ==="]
        
        # Files already read
        if self.current_session.files_read:
            summary_parts.append(f"\nFILES YOU'VE ALREADY READ:")
            for f in self.current_session.files_read:
                duplicate_count = self.current_session.scratchpad.get(f"duplicate_read_{f}", 0)
                if duplicate_count > 0:
                    summary_parts.append(f"  - {f} (read {duplicate_count + 1} times - STOP REREADING THIS)")
                else:
                    summary_parts.append(f"  - {f}")
        
        # Files modified
        if self.current_session.files_modified:
            summary_parts.append(f"\nFILES YOU'VE MODIFIED:")
            for f in self.current_session.files_modified:
                summary_parts.append(f"  - {f}")
        
        # Recent actions summary
        if self.current_session.actions:
            summary_parts.append(f"\nLAST 5 ACTIONS:")
            for action in self.current_session.actions[-5:]:
                status_icon = "✓" if action.status == "completed" else "✗"
                summary_parts.append(f"  {status_icon} {action.action_type}")
        
        # Current task state
        if self.current_session.scratchpad:
            summary_parts.append(f"\nTASK STATE:")
            for key, value in list(self.current_session.scratchpad.items())[:10]:
                if not key.startswith("file_content_") and not key.startswith("duplicate_read_"):
                    summary_parts.append(f"  - {key}: {str(value)[:100]}")
        
        summary_parts.append("=== END SCRATCHPAD ===\n")
        return "\n".join(summary_parts)
    
    def _generate_summary(self, session: CodingSession) -> str:
        """Generate a summary of the session."""
        if session.error:
            return f"Session ended with error: {session.error}"
        
        completed_actions = [a for a in session.actions if a.status == 'completed']
        return f"Completed {len(completed_actions)} actions, modified {len(session.files_modified)} files"
    
    # =========================================================================
    # LLM INTEGRATION
    # =========================================================================
    
    def _call_llm(self, prompt: str, expect_json: bool = False) -> Any:
        """Call the LLM with a prompt using the configured model client."""
        if self.model_client is None:
            raise ValueError("No model client configured for agentic coding")

        # Add scratchpad context if session exists
        scratchpad_context = ""
        if self.current_session:
            scratchpad_context = self._build_scratchpad_summary()
        
        # Build messages with history
        messages = [
            {
                "role": "system",
                "content": "You are a precise coding agent. Return structured JSON with reasoning and the next action to take."
            }
        ]
        
        # Add conversation history if available
        if self.current_session and self.current_session.conversation_history:
            messages.extend(self.current_session.conversation_history[-10:])  # Last 10 messages
        
        # Add scratchpad context to current prompt
        full_prompt = prompt
        if scratchpad_context:
            full_prompt = f"{scratchpad_context}\n\n{prompt}"
        
        messages.append({"role": "user", "content": full_prompt})

        try:
            response = None

            def _supports_arg(fn, name: str) -> bool:
                try:
                    return name in inspect.signature(fn).parameters
                except Exception:
                    return False

            # Prefer OpenAI-style chat interface if available
            if hasattr(self.model_client, "chat"):
                fn = self.model_client.chat
                kwargs = {"messages": messages}
                if not expect_json and _supports_arg(fn, "tools"):
                    kwargs["tools"] = CODING_TOOLS_SCHEMA
                    if _supports_arg(fn, "tool_choice"):
                        kwargs["tool_choice"] = "auto"
                response = fn(**kwargs)
            # Fallback to llama.cpp/OpenAI compatible create_chat_completion
            elif hasattr(self.model_client, "create_chat_completion"):
                fn = self.model_client.create_chat_completion
                kwargs = {
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 800
                }
                if not expect_json and _supports_arg(fn, "tools"):
                    kwargs["tools"] = CODING_TOOLS_SCHEMA
                response = fn(**kwargs)
            else:
                raise ValueError("Model client does not support chat completions")

            content = ""
            if isinstance(response, dict):
                # OpenAI / llama.cpp style response
                message = response.get('choices', [{}])[0].get('message', {})
                content = message.get('content') or response.get('content') or ""
            else:
                content = str(response)

            # Store in conversation history
            if self.current_session:
                self.current_session.conversation_history.append({"role": "user", "content": full_prompt})
                self.current_session.conversation_history.append({"role": "assistant", "content": content})

            if expect_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Attempt to strip markdown fences before parsing
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        cleaned = "\n".join(
                            line for line in cleaned.splitlines()[1:]
                            if line.strip() != "```"
                        )
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return {"error": "Failed to parse JSON response", "raw": content}

            return content

        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            print(f"[AgenticCoding] {error_msg}")
            if expect_json:
                return {"error": error_msg}
            return error_msg
    
    def _build_planning_context(self, session: CodingSession) -> str:
        """Build context for the planning phase."""
        # List workspace structure
        entries = []
        try:
            for item in os.listdir(session.workspace_root):
                item_path = os.path.join(session.workspace_root, item)
                if os.path.isdir(item_path) and item not in ['node_modules', '.git', '__pycache__']:
                    entries.append(f"[DIR] {item}/")
                elif os.path.isfile(item_path):
                    entries.append(f"[FILE] {item}")
        except:
            pass
        
        return f"""
WORKSPACE ROOT: {session.workspace_root}

WORKSPACE CONTENTS:
{chr(10).join(entries[:50]) if entries else '(empty or inaccessible)'}
"""
    
    def _build_execution_context(self, session: CodingSession) -> str:
        """Build context for action execution."""
        recent_actions = session.actions[-5:] if session.actions else []
        action_history = ""
        for a in recent_actions:
            action_history += f"\n- {a.action_type}: {a.status}"
            if a.result:
                action_history += f" -> {self._summarize_result(a, a.result)}"
        
        return f"""
CURRENT PLAN STEP: {session.current_step_index + 1} of {len(session.plan)}
STEP DESCRIPTION: {session.plan[session.current_step_index].description if session.plan else 'No plan'}

RECENT ACTIONS:{action_history if action_history else ' None'}

FILES MODIFIED SO FAR: {', '.join(session.files_modified) if session.files_modified else 'None'}
"""
    
    def _create_planning_prompt(self, session: CodingSession, context: str) -> str:
        """Create the prompt for the planning phase."""
        return f"""You are an AI coding assistant. Create a step-by-step plan for the following task.

USER REQUEST:
{session.user_request}

{context}

Create a plan as a JSON object with:
{{
  "reasoning": "Your thought process",
  "plan": ["Step 1 description", "Step 2 description", ...]
}}

Keep the plan focused and actionable. Each step should be a concrete action.
"""
    
    def _create_action_prompt(self, session: CodingSession, context: str) -> str:
        """Create the prompt for action decision."""
        # Build list of files already read
        files_read_list = ""
        if session.files_read:
            files_read_list = "\n".join([f"  - {f}" for f in session.files_read])
        
        return f"""You are executing a coding task. Decide your next action.

ORIGINAL REQUEST:
{session.user_request}

{context}

FILES YOU'VE ALREADY READ (DO NOT READ AGAIN):
{files_read_list if files_read_list else "  (none yet)"}

⚠️ CRITICAL: If you've already read a file, you have its contents. ANALYZE it instead of reading it again!

Available actions (USE EXACT PARAMETER NAMES):
- read_file: Read file contents (ONLY if not already read!)
  Parameters: {{"path": "filename.py"}}
  
- write_file: Write/replace entire file
  Parameters: {{"path": "filename.py", "content": "..."}}
  
- edit_file: Apply surgical edits (find/replace)
  Parameters: {{"path": "filename.py", "edits": [{{"oldText": "...", "newText": "..."}}]}}
  
- create_file: Create new file
  Parameters: {{"path": "filename.py", "content": "..."}}
  
- delete_file: Delete file
  Parameters: {{"path": "filename.py"}}
  
- list_directory: List directory contents
  Parameters: {{"path": "."}}
  
- search_files: Search for files by name
  Parameters: {{"pattern": "*.py"}}
  
- search_in_files: Search for text in files
  Parameters: {{"query": "search term"}}
  
- run_command: Run shell command
  Parameters: {{"command": "ls -la"}}
  
- complete: Mark task as complete (use this when the bug is ACTUALLY FIXED)
  Parameters: {{"summary": "Fixed X by doing Y", "filesModified": ["file1.py"]}}
  
- abort: Abort with reason
  Parameters: {{"reason": "Cannot proceed because..."}}

⚠️ CRITICAL PARAMETER RULES:
- Use "path" NOT "file_path" or "filepath"
- Use "content" NOT "text" or "file_content"
- Use "command" NOT "cmd" or "shell_command"
- ONLY use parameters listed above - do not invent new ones!

Respond with JSON:
{{
  "reasoning": "Your thought process - explain what you learned and what you'll do next",
  "action": "action_name",
  "params": {{...EXACT parameters from above...}}
}}

REMEMBER: 
- If you've read a file, ANALYZE it and MAKE EDITS
- Don't get stuck in read loops
- Actually FIX bugs, don't just observe them
- Use EXACT parameter names from the list above
"""
    
    # =========================================================================
    # APPROVAL HANDLING
    # =========================================================================
    
    def approve_changes(self, session_id: str, change_ids: Optional[List[str]] = None, approve_all: bool = False) -> Dict[str, Any]:
        """Approve pending changes and resume execution."""
        session = self.sessions.get(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}
        
        approved_count = 0
        for change in session.pending_changes:
            if approve_all or (change_ids and change.id in change_ids):
                change.approved = True
                approved_count += 1
        
        # Resume execution if all pending changes approved
        all_approved = all(c.approved for c in session.pending_changes)
        if all_approved:
            session.phase = AgentPhase.EXECUTING
            self._emit_message('approval_received', {
                'approved': True,
                'count': approved_count
            })
            
            # Execute the pending action that was waiting for approval
            pending_action = session.scratchpad.get('pending_action')
            if pending_action:
                # Execute the approved action
                result = self._execute_action(session, pending_action)
                
                # Reflect on the result
                self._reflect_on_result(session, pending_action, result)
                
                # Clear the pending action
                session.scratchpad.pop('pending_action', None)
                
                # Check if action was complete/abort
                if pending_action.action_type == 'complete':
                    session.phase = AgentPhase.COMPLETE
                    session.end_time = time.time()
                    self._emit_message('session_complete', {
                        'summary': self._generate_summary(session),
                        'filesModified': session.files_modified,
                        'duration': (session.end_time or time.time()) - session.start_time
                    })
                elif pending_action.action_type == 'abort':
                    session.phase = AgentPhase.ERROR
                    session.error = pending_action.params.get('reason', 'Aborted by agent')
                    session.end_time = time.time()
                else:
                    # Continue execution - call continue_execution to resume the loop
                    return self.continue_execution(session_id)
        
        return {
            'success': True,
            'approvedCount': approved_count,
            'allApproved': all_approved
        }
    
    def continue_execution(self, session_id: str) -> Dict[str, Any]:
        """Continue execution loop after approval."""
        session = self.sessions.get(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}
        
        if session.phase != AgentPhase.EXECUTING:
            return {'success': False, 'error': f'Session is in {session.phase.value} phase, cannot continue'}
        
        # Make this the current session
        self.current_session = session
        
        # Continue the execution loop
        iteration = len(session.actions)  # Start from where we left off
        
        while session.phase == AgentPhase.EXECUTING and iteration < self.max_iterations:
            iteration += 1
            
            # Get next action from LLM
            action = self._get_next_action(session)
            
            if action is None:
                break
            
            # Check if approval needed
            if action.action_type in self.require_approval_for:
                session.phase = AgentPhase.AWAITING_APPROVAL
                self._request_approval(session, action)
                session.scratchpad['pending_action'] = action
                # Exit and wait for approval again
                break
            
            # Execute the action
            result = self._execute_action(session, action)
            
            # Check for completion
            if action.action_type == 'complete':
                session.phase = AgentPhase.COMPLETE
                session.end_time = time.time()
                self._emit_message('session_complete', {
                    'summary': self._generate_summary(session),
                    'filesModified': session.files_modified,
                    'duration': (session.end_time or time.time()) - session.start_time
                })
                break
            
            # Check for abort
            if action.action_type == 'abort':
                session.phase = AgentPhase.ERROR
                session.error = action.params.get('reason', 'Aborted by agent')
                session.end_time = time.time()
                break
            
            # Reflect on the result
            self._reflect_on_result(session, action, result)
        
        if iteration >= self.max_iterations:
            session.error = f"Max iterations ({self.max_iterations}) reached"
            session.phase = AgentPhase.ERROR
        
        return {
            'success': True,
            'phase': session.phase.value,
            'actionsCompleted': len(session.actions)
        }
    
    def reject_changes(self, session_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject pending changes and abort session."""
        session = self.sessions.get(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}
        
        session.phase = AgentPhase.ERROR
        session.error = f"Changes rejected: {reason}" if reason else "Changes rejected by user"
        session.end_time = time.time()
        
        self._emit_message('approval_received', {
            'approved': False,
            'reason': reason
        })
        
        return {'success': True}


# Export the coding tools schema for use by Nova
CODING_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "agent_read_file",
            "description": "Read the contents of a file. Use this to examine code before making changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file relative to workspace root"},
                    "startLine": {"type": "integer", "description": "Optional: Start reading from this line (1-indexed)"},
                    "endLine": {"type": "integer", "description": "Optional: Stop reading at this line"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_write_file",
            "description": "Write content to a file. Use this to create or completely replace a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "The complete content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_edit_file",
            "description": "Make surgical edits to a file by finding and replacing exact text. Each edit must match unique text in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string", "description": "Exact text to find (must be unique in file)"},
                                "newText": {"type": "string", "description": "Text to replace with"},
                                "description": {"type": "string", "description": "Human-readable description of the change"}
                            },
                            "required": ["oldText", "newText"]
                        },
                        "description": "Array of edits to apply"
                    }
                },
                "required": ["path", "edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_create_file",
            "description": "Create a new file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path where the file should be created"},
                    "content": {"type": "string", "description": "Content for the new file"}
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
                    "path": {"type": "string", "description": "Path to the file to delete"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_list_directory",
            "description": "List files and directories in a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list", "default": "."},
                    "recursive": {"type": "boolean", "description": "Whether to list recursively", "default": False},
                    "maxDepth": {"type": "integer", "description": "Maximum recursion depth", "default": 2}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_search_files",
            "description": "Search for files by name pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Filename pattern to search for (e.g., '*.tsx', 'Component')"},
                    "directory": {"type": "string", "description": "Directory to search in", "default": "."},
                    "excludePatterns": {"type": "array", "items": {"type": "string"}, "description": "Patterns to exclude"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_search_in_files",
            "description": "Search for text content within files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex to search for"},
                    "directory": {"type": "string", "description": "Directory to search in", "default": "."},
                    "filePattern": {"type": "string", "description": "Only search in files matching this pattern"},
                    "isRegex": {"type": "boolean", "description": "Whether query is a regex", "default": False}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_run_command",
            "description": "Run a shell command. Use sparingly and only for necessary operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                    "cwd": {"type": "string", "description": "Working directory for the command"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agent_complete",
            "description": "Signal that the coding task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of changes made"},
                    "filesModified": {"type": "array", "items": {"type": "string"}, "description": "List of files that were modified"}
                },
                "required": ["summary", "filesModified"]
            }
        }
    }
]


if __name__ == "__main__":
    print("Agentic Coding Orchestrator module loaded.")
    print(f"Available tools: {[t['function']['name'] for t in CODING_TOOLS_SCHEMA]}")

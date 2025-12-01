"""
Nova Integration for Agentic Coding
====================================

This module provides the integration layer between Nova (the orchestrator model)
and the Agentic Coding system. It implements the LLM calling logic using Nova's
existing tool-calling infrastructure.
"""

import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from agentic_coding import (
    AgenticCodingOrchestrator, 
    CodingSession, 
    AgentPhase,
    CODING_TOOLS_SCHEMA
)


@dataclass  
class NovaResponse:
    """Response from Nova model."""
    content: str
    tool_calls: Optional[List[Dict]] = None
    finish_reason: str = "stop"


class NovaCodingOrchestrator(AgenticCodingOrchestrator):
    """
    Extended orchestrator that uses Nova for LLM calls.
    
    This integrates with the existing Nova infrastructure in SolaceOS,
    using the same model loading and inference pipeline.
    """
    
    def __init__(
        self,
        inference_fn: Callable,  # Function to call for inference
        workspace_root: str = ".",
        on_message: Optional[Callable] = None,
        max_iterations: int = 20,
        require_approval_for: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Nova coding orchestrator.
        
        Args:
            inference_fn: Function that takes (messages, tools) and returns response
            workspace_root: Root directory for file operations
            on_message: Callback for streaming messages to UI
            max_iterations: Maximum number of action iterations
            require_approval_for: Action types that require user approval
            system_prompt: Optional system prompt override
        """
        super().__init__(
            model_client=None,  # We use inference_fn instead
            workspace_root=workspace_root,
            on_message=on_message,
            max_iterations=max_iterations,
            require_approval_for=require_approval_for
        )
        
        self.inference_fn = inference_fn
        self.system_prompt = system_prompt or self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for coding tasks."""
        return """You are an expert software engineer AI assistant integrated into SolaceOS.
Your task is to help users with coding tasks by reading, analyzing, and modifying code files.

CAPABILITIES:
- Read and analyze code files
- Make surgical edits to specific parts of files
- Create new files
- Search through codebases
- Run shell commands when necessary
- Plan and execute multi-step coding tasks

GUIDELINES:
1. Always read relevant files before making changes
2. Make minimal, targeted edits - don't rewrite entire files
3. Use the edit_file tool for small changes, write_file only for new content
4. Explain your reasoning clearly
5. Ask for approval before destructive operations
6. Test your changes mentally before applying them

RESPONSE FORMAT:
When deciding on actions, respond with valid JSON:
{
  "reasoning": "Your thought process",
  "action": "action_name",
  "params": {...parameters...}
}

Available actions: read_file, write_file, edit_file, create_file, delete_file, 
list_directory, search_files, search_in_files, run_command, complete, abort
"""
    
    def _call_llm(self, prompt: str, expect_json: bool = False) -> Any:
        """
        Call Nova for inference.
        
        Args:
            prompt: The prompt to send
            expect_json: Whether to expect JSON response
            
        Returns:
            Parsed response (dict if JSON expected, str otherwise)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Add conversation history if we have actions
        if self.current_session and self.current_session.actions:
            # Add context from previous actions
            history_context = self._build_history_context()
            if history_context:
                messages.insert(1, {"role": "assistant", "content": history_context})
        
        try:
            # Call the inference function
            response = self.inference_fn(
                messages=messages,
                tools=CODING_TOOLS_SCHEMA if not expect_json else None,
                max_tokens=2048,
                temperature=0.1  # Low temperature for consistent coding
            )
            
            # Extract content from response
            if isinstance(response, dict):
                content = response.get('content', '') or response.get('text', '')
                tool_calls = response.get('tool_calls', [])
                
                # If we have tool calls, convert to our action format
                if tool_calls and not expect_json:
                    return self._convert_tool_calls(tool_calls)
                
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # Parse JSON if expected
            if expect_json:
                return self._parse_json_response(content)
            
            return content
            
        except Exception as e:
            print(f"[NovaCoding] LLM call failed: {e}")
            if expect_json:
                return {"error": str(e)}
            return f"Error calling model: {e}"
    
    def _build_history_context(self) -> str:
        """Build context from action history."""
        if not self.current_session:
            return ""
        
        lines = ["Previous actions in this session:"]
        for action in self.current_session.actions[-5:]:  # Last 5 actions
            status = "✓" if action.status == "completed" else "✗"
            result_preview = ""
            if action.result:
                if action.result.get('success'):
                    result_preview = f" - Success"
                else:
                    result_preview = f" - Failed: {action.result.get('error', 'Unknown')}"
            lines.append(f"  {status} {action.action_type}({action.params}){result_preview}")
        
        return "\n".join(lines)
    
    def _convert_tool_calls(self, tool_calls: List[Dict]) -> Dict[str, Any]:
        """Convert tool calls format to our action format."""
        if not tool_calls:
            return {"action": "complete", "params": {"summary": "No action needed"}}
        
        # Take the first tool call
        tc = tool_calls[0]
        function_name = tc.get('function', {}).get('name', '')
        
        # Parse arguments
        args_str = tc.get('function', {}).get('arguments', '{}')
        try:
            params = json.loads(args_str) if isinstance(args_str, str) else args_str
        except:
            params = {}
        
        # Map tool names to action types
        action_map = {
            'agent_read_file': 'read_file',
            'agent_write_file': 'write_file',
            'agent_edit_file': 'edit_file',
            'agent_create_file': 'create_file',
            'agent_delete_file': 'delete_file',
            'agent_rename_file': 'rename_file',
            'agent_list_directory': 'list_directory',
            'agent_search_files': 'search_files',
            'agent_search_in_files': 'search_in_files',
            'agent_run_command': 'run_command',
            'agent_complete': 'complete',
            'agent_request_approval': 'request_approval',
        }
        
        action_type = action_map.get(function_name, function_name)
        
        return {
            "action": action_type,
            "params": params,
            "reasoning": f"Using {function_name} tool"
        }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling markdown code blocks."""
        # Clean up the content
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith('```'):
            lines = content.split('\n')
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            content = '\n'.join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            print(f"[NovaCoding] Failed to parse JSON: {e}")
            print(f"[NovaCoding] Content was: {content[:200]}...")
            return {"error": "Failed to parse response", "raw": content[:500]}


def create_nova_coding_orchestrator(
    orchestrator_instance,  # The main orchestrator from orchestrator.py
    workspace_root: str = ".",
    on_message: Optional[Callable] = None
) -> NovaCodingOrchestrator:
    """
    Create a Nova coding orchestrator using the existing orchestrator instance.
    
    Args:
        orchestrator_instance: The main SolaceOS orchestrator
        workspace_root: Root directory for file operations
        on_message: Callback for streaming messages
        
    Returns:
        Configured NovaCodingOrchestrator
    """
    def inference_fn(messages, tools=None, max_tokens=2048, temperature=0.1):
        """Wrapper around the orchestrator's inference."""
        # Use the orchestrator's model to generate response
        # This should work with the already-loaded Nova model
        
        # Format messages for the model
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"<|system|>\n{content}\n"
            elif role == 'user':
                prompt += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                prompt += f"<|assistant|>\n{content}\n"
        
        prompt += "<|assistant|>\n"
        
        # Call the orchestrator's inference
        # This assumes the orchestrator has a method to generate text
        if hasattr(orchestrator_instance, 'generate'):
            response = orchestrator_instance.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools
            )
            return response
        
        # Fallback: try to use the model directly
        if hasattr(orchestrator_instance, 'model') and orchestrator_instance.model:
            response = orchestrator_instance.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return {'content': response}
        
        raise ValueError("No model available for inference")
    
    return NovaCodingOrchestrator(
        inference_fn=inference_fn,
        workspace_root=workspace_root,
        on_message=on_message
    )


# =============================================================================
# STREAMING INTEGRATION
# =============================================================================

class StreamingNovaCodingOrchestrator(NovaCodingOrchestrator):
    """
    Extended orchestrator with streaming support for real-time UI updates.
    """
    
    def __init__(
        self,
        inference_fn: Callable,
        stream_callback: Callable[[str], None],
        **kwargs
    ):
        """
        Initialize with streaming support.
        
        Args:
            inference_fn: Function for inference
            stream_callback: Callback for streaming tokens
            **kwargs: Additional arguments for parent class
        """
        super().__init__(inference_fn=inference_fn, **kwargs)
        self.stream_callback = stream_callback
    
    def _call_llm(self, prompt: str, expect_json: bool = False) -> Any:
        """Call LLM with streaming for reasoning display."""
        # For JSON responses, use non-streaming for reliability
        if expect_json:
            return super()._call_llm(prompt, expect_json=True)
        
        # For reasoning/text responses, stream tokens
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        full_response = ""
        
        try:
            # Call with streaming
            for chunk in self.inference_fn(
                messages=messages,
                stream=True,
                max_tokens=2048,
                temperature=0.1
            ):
                token = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
                full_response += token
                self.stream_callback(token)
            
            return full_response
            
        except Exception as e:
            print(f"[NovaCoding] Streaming failed: {e}")
            # Fallback to non-streaming
            return super()._call_llm(prompt, expect_json=False)


# =============================================================================
# API PROVIDER INTEGRATION
# =============================================================================

def create_api_coding_orchestrator(
    provider: str,
    api_key: str,
    model: str,
    workspace_root: str = ".",
    on_message: Optional[Callable] = None
) -> NovaCodingOrchestrator:
    """
    Create a coding orchestrator using an external API provider.
    
    Supports: openai, anthropic, google, openrouter
    
    Args:
        provider: API provider name
        api_key: API key
        model: Model name/ID
        workspace_root: Root directory
        on_message: Message callback
        
    Returns:
        Configured orchestrator
    """
    
    if provider == 'openai':
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        def inference_fn(messages, tools=None, **kwargs):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=kwargs.get('max_tokens', 2048),
                temperature=kwargs.get('temperature', 0.1)
            )
            msg = response.choices[0].message
            return {
                'content': msg.content or '',
                'tool_calls': [
                    {
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    }
                    for tc in (msg.tool_calls or [])
                ]
            }
    
    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        def inference_fn(messages, tools=None, **kwargs):
            # Convert messages format for Anthropic
            system = ""
            anthropic_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system = msg['content']
                else:
                    anthropic_messages.append(msg)
            
            # Convert tools to Anthropic format
            anthropic_tools = None
            if tools:
                anthropic_tools = [
                    {
                        'name': t['function']['name'],
                        'description': t['function']['description'],
                        'input_schema': t['function']['parameters']
                    }
                    for t in tools
                ]
            
            response = client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 2048),
                system=system,
                messages=anthropic_messages,
                tools=anthropic_tools
            )
            
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == 'text':
                    content += block.text
                elif block.type == 'tool_use':
                    tool_calls.append({
                        'function': {
                            'name': block.name,
                            'arguments': json.dumps(block.input)
                        }
                    })
            
            return {'content': content, 'tool_calls': tool_calls}
    
    elif provider == 'google':
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)
        
        def inference_fn(messages, tools=None, **kwargs):
            # Convert to Gemini format
            prompt = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages
            ])
            
            response = gen_model.generate_content(prompt)
            return {'content': response.text}
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return NovaCodingOrchestrator(
        inference_fn=inference_fn,
        workspace_root=workspace_root,
        on_message=on_message
    )


if __name__ == "__main__":
    # Example: Test with a mock inference function
    def mock_inference(messages, tools=None, **kwargs):
        return {
            'content': json.dumps({
                'reasoning': 'Testing the system',
                'action': 'list_directory',
                'params': {'path': '.'}
            })
        }
    
    orchestrator = NovaCodingOrchestrator(
        inference_fn=mock_inference,
        workspace_root=".",
        on_message=lambda msg: print(f"[MSG] {msg['type']}: {msg.get('data', {})}")
    )
    
    print("Nova Coding Orchestrator initialized.")
    print(f"Available tools: {[t['function']['name'] for t in CODING_TOOLS_SCHEMA]}")

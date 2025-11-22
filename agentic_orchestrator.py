"""
Agentic Orchestrator for SolaceHomeUI
Implements GPT-OSS style multi-step reasoning with:
- Planning phase
- Iterative tool execution
- Reflection and self-correction
- Explicit reasoning traces
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from mcp_server import get_mcp_server, MCPToolResult


class AgentPhase(Enum):
    """Agent execution phases"""
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    RESPONSE = "response"
    COMPLETE = "complete"


@dataclass
class AgentState:
    """Tracks the agent's current state and history"""
    user_query: str
    phase: AgentPhase = AgentPhase.PLANNING
    plan: List[str] = field(default_factory=list)
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)
    gathered_info: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 5  # Reduced from 15 to prevent infinite loops
    is_complete: bool = False
    final_answer: Optional[str] = None
    
    def add_reasoning(self, thought: str):
        """Add a reasoning step to the trace"""
        self.reasoning_trace.append(f"[Iteration {self.iteration_count}] {thought}")
        print(f"💭 {thought}")
    
    def record_tool_execution(self, tool_name: str, args: Dict[str, Any], result: MCPToolResult):
        """Record a tool execution"""
        self.executed_tools.append({
            "tool": tool_name,
            "args": args,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "iteration": self.iteration_count
        })
        
        # Store info in gathered_info for easy access
        if result.success:
            self.gathered_info[f"{tool_name}_{self.iteration_count}"] = result.result
    
    def should_continue(self) -> bool:
        """Check if agent should continue iterating"""
        # Stop if explicitly marked complete
        if self.is_complete or self.phase == AgentPhase.COMPLETE:
            return False
        
        # Stop if hit max iterations
        if self.iteration_count >= self.max_iterations:
            return False
        
        # Early stop if we keep getting the same irrelevant results
        if self.iteration_count >= 3:
            recent_tools = [t["tool"] for t in self.executed_tools[-3:]]
            # If we've searched 3 times in a row, probably not finding what we need
            if recent_tools.count("search_web") >= 3:
                self.add_reasoning("⚠️ Multiple searches yielding same results - stopping early")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "user_query": self.user_query,
            "phase": self.phase.value,
            "plan": self.plan,
            "executed_tools": self.executed_tools,
            "gathered_info": self.gathered_info,
            "reasoning_trace": self.reasoning_trace,
            "iteration_count": self.iteration_count,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer
        }


class AgenticOrchestrator:
    """
    Agentic orchestrator that enables multi-step tool use with planning and reflection.
    Similar to GPT-OSS's approach but model-agnostic.
    """
    
    def __init__(self, model_client):
        """
        Initialize the orchestrator.
        
        Args:
            model_client: The LLM client (Solace, GPT, Claude, etc.) that supports tool calls
        """
        self.model = model_client
        self.mcp_server = get_mcp_server()
        self.state: Optional[AgentState] = None
    
    def run_agent_loop(self, user_query: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main agent loop - runs until completion or max iterations.
        
        Args:
            user_query: The user's question/request
            context: Optional conversation history
            
        Returns:
            Dictionary containing final answer and execution trace
        """
        # Initialize state
        self.state = AgentState(user_query=user_query)
        
        print("=" * 80)
        print("🤖 AGENTIC ORCHESTRATOR ACTIVATED")
        print("=" * 80)
        
        # Phase 1: Planning
        self._planning_phase()
        
        # Phase 2-N: Execution loop
        while self.state.should_continue():
            self.state.iteration_count += 1
            self._execution_phase()
            
            # Check if we should move to response
            if self._should_generate_response():
                break
        
        # Final phase: Generate response
        self._response_phase()
        
        print("=" * 80)
        print(f"✅ AGENT COMPLETE - {self.state.iteration_count} iterations")
        print("=" * 80)
        
        return self._get_final_output()
    
    def _planning_phase(self):
        """Phase 1: Create a plan for answering the query"""
        self.state.phase = AgentPhase.PLANNING
        self.state.add_reasoning("Entering planning phase...")
        
        # Get available tools
        tools = self.mcp_server.list_tools()
        tool_descriptions = "\n".join([
            f"- {tool['name']}: {tool['description']}" 
            for tool in tools
        ])
        
        planning_prompt = f"""You are an AI assistant with access to tools. Analyze this query and create a plan.
        USER QUERY: {self.state.user_query}
        AVAILABLE TOOLS:
        {tool_descriptions}
        Create a step-by-step plan to answer this query. Consider:
        1. What information do I need to answer this?
        2. Which tools would help gather that information?
        3. In what order should I use them?
        4. What might I need to search for iteratively?

        CRITICAL GUIDELINES FOR TOOL USAGE:
        - USE TOOLS ONLY for actions that require external data, system interaction, or real-time information.
        - DO NOT USE TOOLS for:
            * Greetings or small talk (e.g., "How are you?", "Hi")
            * General knowledge questions (e.g., "What is Python?", "Explain OAuth")
            * Creative tasks (e.g., "Write a poem", "Tell me a story")
            * Opinion questions (e.g., "What do you think about AI?")
            * Code review or debugging (e.g., "Can you help me debug this code?")
            * When the user is providing information (e.g., "GPT-4o built this", "Here's my data...")
        - If a step in your plan involves any of the above, use your internal knowledge instead of a tool.

        Output your plan as a JSON array of steps:
        {{"plan": ["step 1", "step 2", ...]}}
        IMPORTANT: Be specific about search queries and tool usage."""

        # Get plan from model
        try:
            plan_response = self._call_model(planning_prompt, expect_json=True)
            
            if isinstance(plan_response, dict) and "plan" in plan_response:
                self.state.plan = plan_response["plan"]
                self.state.add_reasoning(f"Created plan with {len(self.state.plan)} steps")
                for i, step in enumerate(self.state.plan, 1):
                    print(f"  {i}. {step}")
            else:
                # Fallback: simple plan
                self.state.plan = ["Use available tools to gather information", "Synthesize answer"]
                self.state.add_reasoning("Using fallback plan")
        
        except Exception as e:
            print(f"⚠️ Planning failed: {e}")
            self.state.plan = ["Gather information using tools", "Answer question"]
        
        self.state.phase = AgentPhase.EXECUTION
    
    def _execution_phase(self):
        """Phase 2: Execute tools iteratively"""
        self.state.add_reasoning(f"Execution iteration {self.state.iteration_count}")
        
        # Build context from previous executions
        execution_context = self._build_execution_context()
        
        decision_prompt = f"""You are executing a plan to answer a query. Decide your next action.
        USER QUERY: {self.state.user_query}
        YOUR PLAN:
        {self._format_plan()}
        EXECUTION SO FAR:
        {execution_context}
        DECISION:
        Based on what you've learned, what should you do next?
        Options:
        1. Use a tool (specify which tool and arguments)
        2. Search for more information (specify search query)
        3. You have enough information to answer

        CRITICAL RULES:
        1. ASKING vs TELLING:
            - ASKING (Use Tool): "What is X?", "Search for Y", "How does Z work?" (Questions seeking info)
            - TELLING (Do NOT Use Tool): "X was built by Y", "Here's my data: [data]", "I have 3 cats" (Statements providing info)
            - If the user's original query or a past result uses past tense verbs (built, created, fixed, added) or phrases like "Here's", "For context", or contains statistics ("95%"), it's TELLING. Do NOT use a tool.

        2. PARAMETER RULES:
            - ONLY use parameters explicitly listed in the available tools.
            - NEVER invent parameters like `max_results`, `filter`, `language`, etc. If they aren't in the schema, don't use them.
            - When in doubt, call the tool with ONLY the required `query` parameter, and leave optional ones out.

        3. COMMON MISTAKES TO AVOID:
            - DO NOT call a tool for greetings, knowledge questions, creative tasks, opinions, or code review.
            - DO NOT hallucinate parameters.

        Respond with JSON:
        {{"action": "use_tool"|"search"|"ready", "tool": "tool_name", "arguments": {{}}, "reasoning": "why"}}
        THINK CAREFULLY: Do you have enough information to answer comprehensively? Or do you need more?"""

        try:
            decision = self._call_model(decision_prompt, expect_json=True)
            
            if isinstance(decision, dict):
                action = decision.get("action")
                reasoning = decision.get("reasoning", "")
                
                self.state.add_reasoning(f"Decision: {action} - {reasoning}")
                
                if action == "ready":
                    self.state.is_complete = True
                    return
                
                elif action == "search":
                    # Search is just a web_search tool call
                    query = decision.get("arguments", {}).get("query", "")
                    if query:
                        self._execute_tool("search_web", {"query": query})
                
                elif action == "use_tool":
                    tool_name = decision.get("tool")
                    arguments = decision.get("arguments", {})
                    if tool_name:
                        self._execute_tool(tool_name, arguments)
                
                else:
                    self.state.add_reasoning("⚠️ Invalid action, moving to response phase")
                    self.state.is_complete = True
            
        except Exception as e:
            print(f"⚠️ Execution decision failed: {e}")
            self.state.is_complete = True
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Execute a specific tool and record results"""
        # SAFETY RAIL: Validate tool exists before attempting execution
        available_tools = {tool['name'] for tool in self.mcp_server.list_tools()}
        
        if tool_name not in available_tools:
            # Tool doesn't exist - provide helpful feedback
            self.state.add_reasoning(
                f"⚠️ Tool '{tool_name}' not found! Available tools: {', '.join(sorted(available_tools))}"
            )
            
            # Log this for emergence learning
            from tool_state import log_missing_tool_request
            log_missing_tool_request(tool_name, arguments)
            
            # Stop the agentic loop - don't keep trying non-existent tools
            self.state.is_complete = True
            return
        
        self.state.add_reasoning(f"Executing: {tool_name}({arguments})")
        
        result = self.mcp_server.execute_tool(tool_name, arguments)
        self.state.record_tool_execution(tool_name, arguments, result)
        # Special handling for deep_research: extract final_summary for easier access
        if tool_name == "deep_research" and result.success:
            if isinstance(result.result, dict) and "final_summary" in result.result:
                self.state.gathered_info["deep_research_summary"] = result.result["final_summary"]
                self.state.gathered_info["deep_research_sources"] = result.result.get("sources", []) 
        if result.success:
            # Show preview of result
            result_preview = str(result.result)[:200]
            self.state.add_reasoning(f"✓ Got result: {result_preview}...")
        else:
            self.state.add_reasoning(f"✗ Tool failed: {result.error}")
    
    def _should_generate_response(self) -> bool:
        """Determine if we have enough information to respond"""
        # Simple heuristic: if we've executed tools and model said "ready", we're done
        return self.state.is_complete or self.state.iteration_count >= self.state.max_iterations
    
    def _response_phase(self):
        """Final phase: Generate comprehensive response"""
        self.state.phase = AgentPhase.RESPONSE
        self.state.add_reasoning("Generating final response...")
        
        # Compile all gathered information
        info_summary = self._compile_information()
        
        response_prompt = f"""You have gathered information to answer a query. Now provide a comprehensive answer.

USER QUERY: {self.state.user_query}

INFORMATION GATHERED:
{info_summary}

INSTRUCTIONS:
- Provide a clear, comprehensive answer
- Cite specific information from your tool results
- If you found conflicting information, note it
- If you couldn't find certain information, acknowledge it
- Be direct and helpful

YOUR ANSWER:"""

        try:
            final_answer = self._call_model(response_prompt, expect_json=False)
            self.state.final_answer = final_answer
            self.state.phase = AgentPhase.COMPLETE
        
        except Exception as e:
            print(f"⚠️ Response generation failed: {e}")
            self.state.final_answer = "I encountered an error generating my final response."
    
    def _build_execution_context(self) -> str:
        """Build a summary of what's been executed so far"""
        if not self.state.executed_tools:
            return "No tools executed yet."
        
        summary_lines = []
        for execution in self.state.executed_tools[-3:]:  # Last 3 tools
            tool = execution["tool"]
            success = "✓" if execution["success"] else "✗"
            result_preview = str(execution["result"])[:150]
            summary_lines.append(f"{success} {tool}: {result_preview}...")
        
        return "\n".join(summary_lines)
    
    def _compile_information(self) -> str:
        """Compile all gathered information into a summary"""
        
        # Priority 1: Check executed_tools for actual results
        if self.state.executed_tools:
            summary_lines = []
            
            for execution in self.state.executed_tools:
                tool = execution["tool"]
                result = execution.get("result", {})
                
                # Handle different result types
                if isinstance(result, dict):
                    # For deep_research, extract final_summary
                    if tool == "deep_research" and "final_summary" in result:
                        summary_lines.append(f"=== Deep Research Results ===")
                        summary_lines.append(result["final_summary"])
                        
                        # Also include sources
                        if "sources" in result and result["sources"]:
                            summary_lines.append(f"\nSources: {', '.join(result['sources'][:5])}")
                    
                    # For other tools returning dicts
                    elif "result" in result:
                        summary_lines.append(f"=== {tool} Results ===")
                        summary_lines.append(str(result["result"])[:500])
                    else:
                        summary_lines.append(f"=== {tool} Results ===")
                        summary_lines.append(str(result)[:500])
                
                # Handle string results
                elif isinstance(result, str):
                    summary_lines.append(f"=== {tool} Results ===")
                    summary_lines.append(result[:500])
                
                # Handle other types
                else:
                    summary_lines.append(f"=== {tool} Results ===")
                    summary_lines.append(str(result)[:500])
            
            if summary_lines:
                return "\n\n".join(summary_lines)
        
        # Priority 2: Fallback to gathered_info if available
        if self.state.gathered_info:
            summary_lines = []
            for key, value in self.state.gathered_info.items():
                value_preview = str(value)[:500]
                summary_lines.append(f"- {key}: {value_preview}")
            return "\n".join(summary_lines)
        
        # Priority 3: Nothing found
        return "No information gathered."
    
    def _format_plan(self) -> str:
        """Format the plan for display"""
        return "\n".join([f"{i}. {step}" for i, step in enumerate(self.state.plan, 1)])
    
    def _call_model(self, prompt: str, expect_json: bool = False) -> Any:
        """
        Call the underlying model.
        This is a placeholder - you'll need to implement this based on your model client.
        
        Args:
            prompt: The prompt to send
            expect_json: Whether to expect JSON response
            
        Returns:
            Model response (str or dict if JSON)
        """
        # TODO: Implement based on your model client
        # For now, this is a stub that raises NotImplementedError
        raise NotImplementedError(
            "You need to implement _call_model() based on your model client "
            "(llama-cpp, OpenAI API, Anthropic API, etc.)"
        )
    
    def _get_final_output(self) -> Dict[str, Any]:
        """Get the final output including answer and trace"""
        return {
            "answer": self.state.final_answer,
            "reasoning_trace": self.state.reasoning_trace,
            "tools_used": self.state.executed_tools,
            "iterations": self.state.iteration_count,
            "state": self.state.to_dict()
        }


if __name__ == "__main__":
    print("Agentic Orchestrator module loaded.")
    print("To use: create an instance with your model client and call run_agent_loop()")

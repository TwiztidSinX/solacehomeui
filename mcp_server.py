"""
MCP (Model Context Protocol) Server for SolaceHomeUI
Provides standardized tool interface compatible with Claude, GPT, and other models.
"""

import json
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio

# Import existing tools
from tools import TOOL_REGISTRY, TOOLS_SCHEMA


@dataclass
class MCPTool:
    """Standard MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPToolResult:
    """Standard MCP tool execution result"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result
        }
        if self.error:
            data["error"] = self.error
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class MCPServer:
    """
    MCP-compliant server for SolaceHomeUI tools.
    Converts existing tools to MCP format and provides standardized execution.
    """
    
    def __init__(self):
        self.tools = self._convert_tools_to_mcp()
        
    def _convert_tools_to_mcp(self) -> List[MCPTool]:
        """Convert SolaceHomeUI tools to MCP format"""
        mcp_tools = []
        
        for tool_def in TOOLS_SCHEMA:
            func_def = tool_def.get("function", {})
            mcp_tool = MCPTool(
                name=func_def.get("name"),
                description=func_def.get("description"),
                input_schema={
                    "type": "object",
                    "properties": func_def.get("parameters", {}).get("properties", {}),
                    "required": func_def.get("parameters", {}).get("required", [])
                }
            )
            mcp_tools.append(mcp_tool)
        
        return mcp_tools
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all available tools in MCP format"""
        return [tool.to_dict() for tool in self.tools]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Execute a tool and return standardized result.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            MCPToolResult with execution details
        """
        print(f"[MCP] Executing tool: {tool_name} with args: {arguments}")
        
        # Check if tool exists
        if tool_name not in TOOL_REGISTRY:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        # Execute the tool
        try:
            result = TOOL_REGISTRY[tool_name](**arguments)
            
            # Handle special response types (image_generation, media_browser, etc.)
            if isinstance(result, dict) and "type" in result:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result.get("message", ""),
                    metadata=result
                )
            
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result
            )
            
        except Exception as e:
            print(f"[MCP] Tool execution failed: {e}")
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.to_dict()
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "tools_available": len(self.tools),
            "server": "SolaceHomeUI MCP Server v1.0"
        }


# Create global MCP server instance
mcp_server = MCPServer()


def get_mcp_server() -> MCPServer:
    """Get the global MCP server instance"""
    return mcp_server


if __name__ == "__main__":
    # Test the MCP server
    server = get_mcp_server()
    
    print("=== MCP Server Test ===")
    print(f"Available tools: {len(server.list_tools())}")
    print("\nTool List:")
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    
    # Test a simple tool execution
    print("\n=== Testing search_web tool ===")
    result = server.execute_tool("search_web", {"query": "test query"})
    print(f"Success: {result.success}")
    if result.success:
        print(f"Result preview: {str(result.result)[:100]}...")
    else:
        print(f"Error: {result.error}")

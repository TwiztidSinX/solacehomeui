"""
Integration between Agentic Orchestrator and llama-cpp models
Provides the _call_model implementation for llama-cpp based models
"""

import json
import re
from typing import Any, Dict, List, Optional
from llama_cpp import Llama

from agentic_orchestrator import AgenticOrchestrator


class LlamaCppModelClient:
    """
    Wrapper for llama-cpp models that provides a consistent interface
    for the agentic orchestrator
    """
    
    def __init__(self, model: Llama, temperature: float = 0.1, max_tokens: int = 2048):
        """
        Initialize the model client.
        
        Args:
            model: llama-cpp Llama instance
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def call(self, prompt: str, expect_json: bool = False, system_prompt: Optional[str] = None) -> Any:
        """
        Call the model with a prompt.
        
        Args:
            prompt: The prompt to send
            expect_json: Whether to expect JSON response
            system_prompt: Optional system prompt override
            
        Returns:
            Model response (str or dict if JSON)
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=self.temperature if not expect_json else 0.0,  # Lower temp for JSON
                max_tokens=self.max_tokens
            )
            
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            
            if expect_json:
                # Try to extract and parse JSON
                return self._extract_json(content)
            
            return content
        
        except Exception as e:
            print(f"⚠️ Model call failed: {e}")
            if expect_json:
                return {}
            return f"Error: {str(e)}"
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from model response.
        Handles cases where model wraps JSON in markdown or adds extra text.
        
        Args:
            text: Raw model output
            
        Returns:
            Parsed JSON dict or empty dict if parsing fails
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
                print(f"Attempted to parse: {json_str[:200]}...")
        
        # Fallback: try parsing the whole thing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"⚠️ Could not extract JSON from: {text[:200]}...")
            return {}


class LlamaCppAgenticOrchestrator(AgenticOrchestrator):
    """
    Agentic orchestrator specialized for llama-cpp models.
    Implements the _call_model method.
    """
    
    def __init__(self, model: Llama, temperature: float = 0.1, max_tokens: int = 2048):
        """
        Initialize orchestrator with llama-cpp model.
        
        Args:
            model: llama-cpp Llama instance
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_client = LlamaCppModelClient(model, temperature, max_tokens)
        super().__init__(self.model_client)
    
    def _call_model(self, prompt: str, expect_json: bool = False) -> Any:
        """
        Call the llama-cpp model.
        
        Args:
            prompt: The prompt to send
            expect_json: Whether to expect JSON response
            
        Returns:
            Model response (str or dict if JSON)
        """
        return self.model_client.call(prompt, expect_json=expect_json)


def create_agentic_orchestrator_for_solace(model_path: str) -> LlamaCppAgenticOrchestrator:
    """
    Convenience function to create an agentic orchestrator for Solace.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Configured LlamaCppAgenticOrchestrator instance
    """
    print(f"Loading model for agentic orchestration: {model_path}")
    
    model = Llama(
        model_path=model_path,
        n_ctx=8192,  # Larger context for agent loops
        n_gpu_layers=-1,  # Use GPU if available
        verbose=False
    )
    
    print("✓ Model loaded")
    
    return LlamaCppAgenticOrchestrator(
        model=model,
        temperature=0.1,  # Lower temp for more focused reasoning
        max_tokens=2048   # Allow longer responses
    )


# Example usage function
def example_usage():
    """
    Example of how to use the agentic orchestrator with Solace
    """
    import os
    
    # Path to your model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "llama", "your-model.gguf")
    
    # Create orchestrator
    orchestrator = create_agentic_orchestrator_for_solace(MODEL_PATH)
    
    # Run agent loop
    result = orchestrator.run_agent_loop(
        user_query="What's the latest news about AI safety research?"
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result["answer"])
    
    print("\n" + "=" * 80)
    print("REASONING TRACE:")
    print("=" * 80)
    for step in result["reasoning_trace"]:
        print(step)
    
    print(f"\nTotal iterations: {result['iterations']}")
    print(f"Tools used: {len(result['tools_used'])}")


if __name__ == "__main__":
    print("Llama-cpp Agentic Orchestrator Integration")
    print("Use create_agentic_orchestrator_for_solace() to create an instance")
    print("Or run example_usage() for a demo")

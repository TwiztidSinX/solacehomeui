from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import json
import os
import tools

app = FastAPI()

# Allow browser calls from any origin (UI served elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_tool_settings():
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'tool_settings.json')
    if not os.path.exists(settings_path):
        return {}
    with open(settings_path, 'r') as f:
        return json.load(f)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

def _load_searxng_url():
    """
    Pull SearXNG URL from tool_settings.json or nova_settings.json (any casing).
    """
    settings = get_tool_settings()
    if settings.get('searXngUrl'):
        return settings.get('searXngUrl')
    if settings.get('searxngUrl'):
        return settings.get('searxngUrl')
    # Fallback to nova_settings.json if available
    try:
        here = os.path.dirname(__file__)
        candidates = [
            os.path.join(here, 'nova_settings.json'),
            os.path.join(here, '..', 'nova_settings.json'),
        ]
        for p in candidates:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    nova_settings = json.load(f)
                    url = (
                        nova_settings.get('searxngUrl')
                        or nova_settings.get('searXngUrl')
                        or nova_settings.get('searXngURL')
                    )
                    if url:
                        return url
    except Exception:
        pass
    # Last-resort default
    return "http://localhost:8088"
    return None


@app.get("/search")
async def search(query: str, engine: str = None):
    """
    Proxy search to SearXNG; optional engine/category hint (e.g., 'videos').
    """
    searxng_url = _load_searxng_url()

    if not searxng_url:
        raise HTTPException(status_code=500, detail="SearXNG URL not configured. Set searXngUrl in tool_settings.json or searxngUrl in nova_settings.json.")

    try:
        params = {"q": query, "format": "json"}
        if engine:
            params["categories"] = engine

        url = searxng_url.rstrip('/') + '/'
        response = requests.get(url, params=params, timeout=10)
        try:
            response.raise_for_status()
        except Exception as e:
            # Include body for easier debugging
            print(f"[tool_server] SearXNG error at {url} -> {response.status_code if response is not None else 'no response'}; body: {getattr(response, 'text', '')[:400]}")
            raise HTTPException(
                status_code=response.status_code if response is not None else 500,
                detail=f"SearXNG returned {response.status_code if response is not None else 'unknown'} at {url}: {response.text[:500]}"
            ) from e

        results = response.json()

        # Temporary debug logging
        try:
            print(f"[tool_server] SearXNG results count: {len(results.get('results', []))} for '{query}' (engine={engine})")
        except Exception:
            pass
        
        formatted_results = []
        for item in results.get("results", []):
            formatted_results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content")
            })
            
        return {"results": formatted_results}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to SearXNG at {searxng_url}: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from SearXNG.")


@app.get("/tools")
async def list_tools():
    """
    List available tools from tools.py TOOLS_SCHEMA.
    """
    try:
        tool_list = []
        for t in tools.TOOLS_SCHEMA:
            fn = t.get("function", {})
            tool_list.append({
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            })
        return {"tools": tool_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {e}")


@app.post("/tool_call")
async def tool_call(payload: dict):
    """
    Call a tool by name with arguments.
    """
    try:
        tool_name = payload.get("name")
        arguments = payload.get("arguments", {}) or {}
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")

        result = tools.dispatch_tool(tool_name, arguments)
        return {"result": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool call failed: {e}")


@app.post("/register_tool")
async def register_tool(payload: dict):
    """
    Dynamically register a new tool at runtime and persist it.
    Expected payload: {name, description, parameters, code, handler_name?}
    """
    try:
        name = payload.get("name")
        description = payload.get("description", "")
        parameters = payload.get("parameters") or {"type": "object", "properties": {}, "required": []}
        code = payload.get("code")
        handler_name = payload.get("handler_name")

        if not name or not code:
            raise HTTPException(status_code=400, detail="Both 'name' and 'code' are required.")

        tools.register_dynamic_tool(name, description, parameters, code, handler_name)
        return {"status": "ok", "name": name}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Register tool failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

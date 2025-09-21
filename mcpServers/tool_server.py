from fastapi import FastAPI, HTTPException
import uvicorn
import requests
import json
import os

app = FastAPI()

def get_tool_settings():
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'tool_settings.json')
    if not os.path.exists(settings_path):
        return {}
    with open(settings_path, 'r') as f:
        return json.load(f)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/search")
async def search(query: str):
    settings = get_tool_settings()
    searxng_url = settings.get('searXngUrl')

    if not searxng_url:
        raise HTTPException(status_code=500, detail="SearXNG URL not configured.")

    try:
        # Construct the search URL for SearXNG's JSON API
        search_request_url = f"{searxng_url.rstrip('/')}/?q={query}&format=json"
        
        response = requests.get(search_request_url, timeout=10)
        response.raise_for_status()
        
        results = response.json()
        
        # Extract and format the most relevant information
        formatted_results = []
        for item in results.get("results", []):
            formatted_results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content")
            })
            
        return {"results": formatted_results}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to SearXNG: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from SearXNG.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
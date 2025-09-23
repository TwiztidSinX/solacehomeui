from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
import torch
from pydantic import BaseModel
import io

# --- Basic App Setup ---
app = FastAPI()

# --- Pydantic Models for Request Bodies ---
class TTSRequest(BaseModel):
    text: str
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC" # Default model for now
    speaker: str | None = None

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
import torch
from pydantic import BaseModel
import io
import tempfile
import os
from faster_whisper import WhisperModel
from TTS.api import TTS
import subprocess
from llama_cpp import Llama

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Path to the llama.cpp executable (we might need to make this configurable later)
LLAMA_CPP_MAIN_PATH = os.path.join(PROJECT_ROOT, "llama-gpu", "main.exe")

# --- Basic App Setup ---
app = FastAPI()


# --- Pydantic Models for Request Bodies ---
class TTSRequest(BaseModel):
    text: str
    model_name: str = "Kyutai-TTS/tts-0.75b-en-public"
    speaker: str | None = None

# --- Model Loading Logic ---
stt_model = None
tts_models = {} # Cache for loaded TTS models
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_stt_model():
    """Loads the STT model into memory."""
    global stt_model
    if stt_model is None:
        print("Loading STT model (faster-whisper)...")
        try:
            compute_type = "float16" if device == "cuda" else "int8"
            stt_model = WhisperModel("base", device=device, compute_type=compute_type)
            print("STT Model Loaded successfully.")
        except Exception as e:
            print(f"Failed to load STT model: {e}")
            stt_model = None

def load_tts_model(model_name: str):
    """
    Universal TTS model loader. Scans the model directory and uses the
    appropriate library to load it.
    """
    if model_name in tts_models: # Return cached model if already loaded
        return tts_models[model_name]

    print(f"Attempting to load TTS model: {model_name}...")
    model_path = os.path.join(PROJECT_ROOT, "models", "tts_models", model_name)
    
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found: {model_path}")
        tts_models[model_name] = None
        return None

    # --- Dispatcher Logic ---
    files_in_dir = os.listdir(model_path)

    # 1. Coqui TTS / SafeTensors Path
    if 'config.json' in files_in_dir and any(f.endswith('.pth') or f.endswith('.safetensors') for f in files_in_dir):
        print(f"Detected Coqui-style model. Loading with TTS library...")
        try:
            model_instance = TTS(model_path=model_path).to(device)
            tts_models[model_name] = {"type": "coqui", "model": model_instance}
            print(f"Successfully loaded {model_name} via Coqui TTS.")
            return tts_models[model_name]
        except Exception as e:
            print(f"Failed to load Coqui-style model '{model_name}': {e}")
            tts_models[model_name] = None
            return None

    # 2. GGUF Path
    elif any(f.endswith('.gguf') for f in files_in_dir):
        print(f"Detected GGUF model. Storing path...")
        try:
            gguf_file = next(f for f in files_in_dir if f.endswith('.gguf'))
            full_gguf_path = os.path.join(model_path, gguf_file)
            tts_models[model_name] = {"type": "gguf", "model_path": full_gguf_path}
            print(f"Successfully stored path for GGUF model: {full_gguf_path}")
            return tts_models[model_name]
        except Exception as e:
            print(f"Failed to process GGUF model in '{model_name}': {e}")
            tts_models[model_name] = None
            return None

    else:
        print(f"Error: Could not determine model type for '{model_name}'. No config.json or .gguf file found.")
        tts_models[model_name] = None
        return None


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    load_stt_model()
    # Pre-load a default TTS model for faster first response
    load_tts_model("Kyutai-TTS/tts-0.75b-en-public")

@app.post("/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Endpoint to transcribe audio to text.
    Accepts an audio file upload.
    """
    if not stt_model:
        return {"error": "STT model not loaded."}
    
    # faster-whisper works with file paths, so we save the upload to a temporary file.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            content = await audio_file.read()
            tmp_audio_file.write(content)
            tmp_audio_file_path = tmp_audio_file.name
        
        segments, info = stt_model.transcribe(tmp_audio_file_path, beam_size=5)
        
        transcribed_text = "".join(segment.text for segment in segments)
        
        print(f"Transcription successful: {transcribed_text}")
        return {"transcription": transcribed_text.strip()}

    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"error": f"Failed to transcribe audio: {e}"}
    finally:
        # Clean up the temporary file
        if 'tmp_audio_file_path' in locals() and os.path.exists(tmp_audio_file_path):
            os.remove(tmp_audio_file_path)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Endpoint to synthesize speech from text."""
    model_info = load_tts_model(request.model_name)

    if not model_info:
        return {"error": f"TTS model '{request.model_name}' is not available or failed to load."}

    model_type = model_info.get("type")
    
    print(f"Synthesizing speech for text: '{request.text}' with model '{request.model_name}' (type: {model_type})")
    
    try:
        if model_type == "coqui":
            model = model_info.get("model")
            wav_buffer = io.BytesIO()
            model.tts_to_file(text=request.text, speaker=request.speaker, file_path=wav_buffer)
            wav_buffer.seek(0)
            return StreamingResponse(wav_buffer, media_type="audio/wav")
        
        elif model_type == "gguf":
            model_path = model_info.get("model_path")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                tmp_wav_path = tmp_wav_file.name

            command = [
                LLAMA_CPP_MAIN_PATH,
                "-m", model_path,
                "--tts", request.text,
                "-o", tmp_wav_path
            ]
            
            print(f"Executing llama.cpp command: {' '.join(command)}")
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            if not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) == 0:
                raise RuntimeError("llama.cpp command did not produce a valid WAV file.")

            def file_streamer():
                try:
                    with open(tmp_wav_path, "rb") as f:
                        yield from f
                finally:
                    os.remove(tmp_wav_path)
            
            return StreamingResponse(file_streamer(), media_type="audio/wav")
            
        else:
            return {"error": f"Unknown model type '{model_type}' for synthesis."}

    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF TTS synthesis (llama.cpp failed): {e.stderr}")
        return {"error": f"llama.cpp execution failed: {e.stderr}"}
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return {"error": f"Failed to synthesize audio: {e}"}

# --- Main Execution ---
if __name__ == "__main__":
    # This allows running the server directly for testing
    uvicorn.run(app, host="0.0.0.0", port=8880)

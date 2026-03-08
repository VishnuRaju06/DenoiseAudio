import os
import tempfile
import warnings
import torchaudio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference.engine import process_audio, get_model

warnings.filterwarnings("ignore")

app = FastAPI(title="Gunshot Denoising API", version="2.0.0")

# Allow CORS so frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend statically at "/frontend"
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Warm up the model on startup
@app.on_event("startup")
def startup_event():
    get_model()

# ========================================================================
# DENOISING ENDPOINT
# ========================================================================
@app.post("/denoise")
async def denoise_endpoint(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    try:
        # Load audio
        waveform, sr = torchaudio.load(tmp_in_path)
        
        # Run inference
        clean_audio, output_sr = process_audio(waveform, sr)
        
        # Save output temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_out_path = tmp_out.name
            
        torchaudio.save(tmp_out_path, clean_audio, output_sr, format="wav")

        # Cleanup input
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
            
        # Return processed audio
        return FileResponse(
            tmp_out_path, 
            media_type="audio/wav", 
            filename=f"clean_{file.filename}.wav",
            headers={"Content-Disposition": f"attachment; filename=clean_{file.filename}.wav"}
        )

    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Make sure to run from the root directory so imports work
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

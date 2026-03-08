import os
import io
import torch
import torchaudio
import tempfile
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from unet_model import UNet

warnings.filterwarnings("ignore")

app = FastAPI(title="Gunshot Denoising API")

# Allow CORS so our local frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend statically at "/"
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# ========================================================================
# 1. SETUP THE MODEL
# ========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Model on device: {device}")

model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

try:
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

# ========================================================================
# 2. AUDIO PROCESSING FUNCTIONS
# ========================================================================
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 22050
MAX_LENGTH = 256

def audio_to_spectrogram(waveform):
    stft = torch.stft(
        waveform, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        return_complex=True,
        pad_mode='reflect'
    )
    return torch.abs(stft), torch.angle(stft)

def spectrogram_to_audio(magnitude, phase):
    stft = magnitude * torch.exp(1j * phase)
    return torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH)

# ========================================================================
# 3. DENOISING ENDPOINT
# ========================================================================
@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    try:
        # Load audio
        waveform, sr = torchaudio.load(tmp_in_path)
        
        # Resample and format
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = transform(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.to(device)
        
        # Process (Waveform -> Mag/Phase -> Mask -> Reconstruct)
        magnitude, phase = audio_to_spectrogram(waveform)
        input_tensor = magnitude.unsqueeze(0) 
        
        _, _, freq, time = input_tensor.shape
        pad_freq = (16 - freq % 16) % 16
        pad_time = (16 - time % 16) % 16
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_time, 0, pad_freq))

        with torch.no_grad():
            output = model(input_tensor) 
            mask = torch.sigmoid(output)
            clean_magnitude = input_tensor * mask
            
        clean_magnitude = clean_magnitude[:, :, :freq, :time].squeeze(0).squeeze(0) 
        clean_audio = spectrogram_to_audio(clean_magnitude, phase.squeeze(0))
        
        # Save output temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_out_path = tmp_out.name
            
        # Add channel dim back
        clean_audio = clean_audio.unsqueeze(0).cpu()
        torchaudio.save(tmp_out_path, clean_audio, SAMPLE_RATE, format="wav")

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
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)

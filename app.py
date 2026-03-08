import gradio as gr
import torch
import torchaudio
import numpy as np
import warnings
import librosa
from unet_model import UNet

warnings.filterwarnings("ignore")

# ========================================================================
# 1. SETUP THE MODEL
# ========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# We instantiate the U-Net. 
# Depending on your training input, the parameters might optionally be different.
# Typically, Audio U-Nets use n_channels=1 (magnitude spectrogram), n_classes=1 (mask or clean spectrogram).
model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

try:
    # Load the weights
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

# ========================================================================
# 2. AUDIO PROCESSING HIGHLIGHTS
# ========================================================================
# Assuming standard spectrogram processing:
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 22050  # adjust if your model expects a different sample rate
MAX_LENGTH = 256     # standard crop length for spectrograms (e.g. 256 frames). Adjust based on your model input.

def audio_to_spectrogram(waveform):
    # Convert exactly as in training
    stft = torch.stft(
        waveform, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        return_complex=True,
        pad_mode='reflect'
    )
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    return magnitude, phase

def spectrogram_to_audio(magnitude, phase):
    # Reconstruct complex STFT
    stft = magnitude * torch.exp(1j * phase)
    # Inverse STFT
    waveform = torch.istft(
        stft, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH
    )
    return waveform

# ========================================================================
# 3. DEFINE THE DENOISING INFERENCE FUNCTION
# ========================================================================
def pad_or_trim(tensor, max_frames=MAX_LENGTH):
    """ Ensure the spectrogram has a fixed width if required by Unet """
    orig_frames = tensor.shape[-1]
    if orig_frames > max_frames:
        return tensor[..., :max_frames]
    elif orig_frames < max_frames:
        padding = max_frames - orig_frames
        return torch.nn.functional.pad(tensor, (0, padding))
    return tensor

def denoise_audio(audio_path):
    if audio_path is None:
        return None
        
    try:
        # Load the MP3 / WAV audio file
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = transform(waveform)
            
        # Convert to mono if it's stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.to(device)
        
        # 1. Transform audio to magnitude and phase
        magnitude, phase = audio_to_spectrogram(waveform)
        
        # 2. Prepare for U-Net [batch, channels, freq, time]
        input_tensor = magnitude.unsqueeze(0) # add batch dim
        
        # Model might expect dimension multiple of 16 due to max pools
        # So pad the time and freq dimensions so it's safely divisible by 16.
        _, _, freq, time = input_tensor.shape
        pad_freq = (16 - freq % 16) % 16
        pad_time = (16 - time % 16) % 16
        
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_time, 0, pad_freq))

        with torch.no_grad():
            # Pass through the model to get a mask
            output = model(input_tensor) 
            
            # Apply mask to magnitude
            # We assume it outputted a mask in [0, 1] range via sigmoid
            mask = torch.sigmoid(output)
            clean_magnitude = input_tensor * mask
            
        # Undo padding
        clean_magnitude = clean_magnitude[:, :, :freq, :time]
        
        # 3. Reconstruct audio using original noisy phase
        # Squeeze batch and channel dims
        clean_magnitude = clean_magnitude.squeeze(0).squeeze(0) 
        clean_audio = spectrogram_to_audio(clean_magnitude, phase.squeeze(0))
        
        # 4. Format for Gradio
        output_numpy = clean_audio.cpu().numpy().squeeze()
            
        return (SAMPLE_RATE, output_numpy)
        
    except Exception as e:
        print(f"Error during audio processing: {e}")
        return None

# ========================================================================
# 4. CREATE THE GRADIO WEB INTERFACE
# ========================================================================
demo = gr.Interface(
    fn=denoise_audio,
    inputs=gr.Audio(type="filepath", label="Upload Noisy Audio (MP3 or WAV)"),
    outputs=gr.Audio(label="Clean Denoised Audio"),
    title="🔫 Gunshot Audio Denoising",
    description="Upload an MP3 or WAV file containing gunshot noise. The model will process it and output the clean audio.",
    theme="huggingface"
)

if __name__ == "__main__":
    print("Starting the web app...")
    demo.launch(inbrowser=True)

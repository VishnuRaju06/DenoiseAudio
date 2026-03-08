import torch
import torchaudio
import torch.nn.functional as F
from model.unet_model import UNet

N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 22050

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache the model so we don't load it twice
_model = None

def get_model(weights_path="model/best_model.pth"):
    global _model
    if _model is None:
        model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded successfully!")
            _model = model
        except Exception as e:
            print(f"Error loading model weights: {e}")
            _model = model # Still return uninitialized if error for tests
    return _model

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
    # The length might be slightly off due to STFT framing, but this is safe
    return torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH)

def process_audio(waveform, sr):
    model = get_model()
    
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
    input_tensor = F.pad(input_tensor, (0, pad_time, 0, pad_freq))

    with torch.no_grad():
        output = model(input_tensor) 
        mask = torch.sigmoid(output)
        clean_magnitude = input_tensor * mask
        
    clean_magnitude = clean_magnitude[:, :, :freq, :time].squeeze(0).squeeze(0) 
    clean_audio = spectrogram_to_audio(clean_magnitude, phase.squeeze(0))
    
    return clean_audio.unsqueeze(0).cpu(), SAMPLE_RATE

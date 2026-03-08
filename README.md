# Gunshot Denoising AI Web App

A state-of-the-art PyTorch U-Net audio denoising project that targets and removes gunshot noise from MP3 and WAV audio files. 

This project features a custom, premium web application built with a **FastAPI backend** and a beautiful **HTML/CSS/JS frontend** using glassmorphism design.

## Dataset & Training
- **Dataset**: The model was trained on the custom [Gunshot Dataset](https://www.kaggle.com/datasets/vishnu0609/gunshot-dataset) from Kaggle.
- **Training Notebook**: The complete training code and approach can be found in `gunshot-dae-conunet.ipynb` included in this repository.

## Features

- **Custom CNN Architecture**: Utilizes a PyTorch U-Net variant (`unet_model.py`) optimized for audio spectrogram masking.
- **FastAPI Backend**: A blazing-fast Python server that processes STFTs, runs model inference, and reconstructs clean audio.
- **Premium Frontend UX**: A sleek, dynamic drag-and-drop web UI with processing animations and a built-in player.

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch
- Torchaudio
- FastAPI & Uvicorn

### 1. Clone the repository
```bash
git clone https://github.com/VishnuRaju06/DenoiseAudio.git
cd DenoiseAudio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python backend.py
```

### 4. Access the Web App
Open your browser and navigate to:
`http://127.0.0.1:8000/frontend/index.html`

## Model Details
- The model expects an input spectrogram (magnitude and phase split). 
- It processes the magnitude by outputting a mask between `[0, 1]` via a Sigmoid activation.
- The clean audio is reconstructed via Inverse STFT using the masked magnitude and original phase.

*Note: You must have the trained model weights `best_model.pth` in the root directory for the application to function.*

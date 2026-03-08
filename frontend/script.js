// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileNameDisplay = document.getElementById('file-name');
const removeBtn = document.getElementById('remove-btn');
const processBtn = document.getElementById('process-btn');
const loadingState = document.getElementById('loading-state');
const resultArea = document.getElementById('result-area');
const audioOutput = document.getElementById('audio-output');
const downloadBtn = document.getElementById('download-btn');

let selectedFile = null;

// Drag & Drop Handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
        handleFiles(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length) {
        handleFiles(e.target.files[0]);
    }
}

function handleFiles(file) {
    // Basic validation
    if (!file.type.startsWith('audio/')) {
        alert('Please select a valid audio file (MP3, WAV, etc.)');
        return;
    }
    
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    
    // UI Transitions
    dropZone.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    resultArea.classList.add('hidden'); // Hide previous results
    processBtn.disabled = false;
}

// Remove Audio
removeBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    
    dropZone.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    resultArea.classList.add('hidden');
    processBtn.disabled = true;
});

// Process Audio (API Call)
processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI Updates
    processBtn.classList.add('hidden');
    fileInfo.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultArea.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Send to FastAPI Endpoint
        const response = await fetch('/denoise', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        // Handle Audio Blob Response
        const audioBlob = await response.blob();
        if (audioBlob.size === 0) throw new Error("Received empty audio file!");

        const audioUrl = URL.createObjectURL(audioBlob);

        // Update Output Player & Download Link
        audioOutput.src = audioUrl;
        downloadBtn.href = audioUrl;
        downloadBtn.download = 'denoised_audio.wav';

        // Show Success UI
        loadingState.classList.add('hidden');
        resultArea.classList.remove('hidden');
        
        // Show file selection area again so user can upload more
        fileInfo.classList.remove('hidden');
        processBtn.classList.remove('hidden');
        processBtn.disabled = false;
        
    } catch (error) {
        console.error('Error processing audio:', error);
        alert('An error occurred during denoising: ' + error.message);
        
        // Reset UI
        loadingState.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        processBtn.classList.remove('hidden');
        processBtn.disabled = false;
    }
});

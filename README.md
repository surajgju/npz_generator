# NPZ Generator 🕺

A standalone tool for generating expressive 3D motion data (SMPL-X) from audio files and visualizing them interactively.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have the required libraries installed:
```bash
python3 -m pip install streamlit plotly opencv-python torch librosa tqdm smplx trimesh pyrender
```

### 2. Generate Motion
Place your audio files (mp3, wav, m4a, etc.) in the `./input` folder, then run:
```bash
python3 generate_npz.py
```
*This will process the audio and save the resulting motion data to `output/intro_output.npz`.*

---

## 📽️ Visualization Options

### Option A: Interactive Web Visualizer (Recommended)
Explore the 3D model, rotate, zoom, and scrub through frames in your browser.
```bash
streamlit run visualize_web.py
```

### Option B: Render to Video (MP4)
Generate a high-quality video of the motion synced with the original audio.
```bash
python3 render.py
```
*Result will be saved as `output.mp4`.*

---

## ✨ Features & Defaults

- **Standardized Workflow**: `generate_npz.py` now automatically saves to a consistent filename (`output/intro_output.npz`) that both visualizers read by default.
- **Smart Audio Detection**: Supports `.mp3`, `.m4a`, `.flac`, `.ogg`, and `.wav`.
- **Automatic Configuration**: Defaults to `./input` for audio and `./output` for results.
- **Expressive Motion**: Generates full SMPL-X data including body poses, hand movements, and facial expressions.

---

## 🛠️ Advanced Configuration

If you need to customize paths, you can use these arguments with `generate_npz.py`:

- `--audio_folder`: Folder containing audio files (default: `./input`).
- `--save_folder`: Folder for NPZ and MP4 results (default: `./output`).
- `--no_visualization`: Skip the automatic rendering step during generation.
- `--model_folder`: Path to SMPL-X models (default: `./models/`).

---

## 🌩️ Gemini API Live Streaming Integration

In preparation for real-time 3D motion rendering powered by **Gemini Multimodal Live API**, a new extensible pipeline has been provided: `live_streaming_pipeline.py`.

### Overview
This script is designed to accept audio chunks (such as PCM bytes streamed from a WebRTC or WebSocket connection) and generates the SMPL-X coefficients on-the-fly, at faster-than-realtime speeds.

### Usage
Run the simulation to see chunked processing in action:
```bash
python3 live_streaming_pipeline.py --audio input/11_nidal_0_114_114.wav --chunk_size 2.0
```

### Implementing in your Gemini Live Server
You can simply import the core pipeline into your backend Node/Python server where you receive Gemini's Base64/PCM streams:

```python
from live_streaming_pipeline import GeminiLivePipeline

# 1. Initialize once
pipeline = GeminiLivePipeline()

# 2. Inside your Gemini Live Websocket / WebRTC stream receiver...
async def on_audio_chunk_received(pcm_chunk_bytes):
    # Convert incoming PCM bytes to numpy float array
    numpy_audio = ... 
    
    # 3. Generate SMPL-X frames exactly for that audio chunk
    coefficients = pipeline.process_audio_chunk(numpy_audio)
    
    # 4. coefficients["poses"], coefficients["expressions"] are ready!
    # Send them to your client (e.g. Three.js / Streamlit) for immediate playback
    await websocket.send({"poses": coefficients["poses"].tolist()})
```


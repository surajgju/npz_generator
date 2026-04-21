# 🕺 NPZ Generator & Real-Time SMPL-X Streaming Pipeline

[![Gemini Ready](https://img.shields.io/badge/Gemini-Live%20Ready-blue?style=for-the-badge&logo=google-gemini&logoColor=white)](https://ai.google.dev/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Three.js-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

An end-to-end pipeline for generating expressive SMPL-X motion from audio. This repository supports high-fidelity **offline NPZ generation**, **MP4 rendering**, and a **real-time WebSocket pipeline** for 3D avatar streaming (Gemini Live ready).

---

## ✨ Features

- **🎭 Expressive Motion Generation**: Uses the **EMAGE** (Expressive Motion Generation) architecture for synchronized body, face, hand, and global translation from raw audio.
- **⚡ Real-Time Streaming Pipeline**: A low-latency system that processes audio chunks via WebSockets and streams SMPL‑X animation directly to a Three.js viewer.
- **🎙️ Voice Conversation (Native Gemini 3.1 Live)**: Uses the `google-genai` native WebSocket stream for ultra-low latency. Browser Push-to-Talk (`/ws/conversation`) streams user PCM directly to Gemini and routes assistant response audio into the real-time animation pipeline.
- **🧭 Sessionized Streaming (Protocol v2)**: Per-reply stream sessions with reconnect-aware handshake (`anim_subscribe`) using `stream_session_id`, `server_boot_id`, `server_clock_id`, and monotonic `server_time_ms`.
- **🔁 Snapshot Recovery**: Reconnecting clients receive a short snapshot window (2-3s), then jump to live edge without replay burst.
- **🛡️ Drift/Freeze Protection**: Worker tail-lock alignment with timeout and resync request; viewer has explicit stall states (`stalled_hold`, `stalled_ease`, `resyncing`).
- **🧹 Session Lifecycle GC**: Deprecated/idle sessions are evicted by TTL and LRU caps to prevent buffer leaks.
- **📦 Offline NPZ Generation**: Batch process audio files in `./input` to high-fidelity animation coefficients.
- **📊 Advanced Post-Processing**: Features automatic gain control (AGC) for expressions, EMA smoothing for jitter-free motion, and jaw-scaling for crisp lip-sync.
- **🌐 Dual Visualization Suite**:
    - **Offline (Streamlit)**: High-performance local NPZ viewer.
    - **Live (Vite/React)**: Modern Three.js viewer for real-time streaming and interactive debugging.
- **🔗 Gemini 3.1 Live Native Integration**: Optimized engine that bypasses legacy ADK overhead, supporting the latest multi-modal Live API for seamless speech-to-motion updates.

---

## 🚀 Quick Start

### 1. Environment Setup
Install the core Python dependencies (requires PyTorch and SMPL-X):
```bash
python3 -m pip install -r requirements.txt
```

### 2. Frontend Setup
The live viewer resides in the `/frontend` directory. We recommend running it using the Vite development server:
```bash
# Export SMPL-X faces (requires smplx weights in 'models/')
python3 scripts/export_faces.py

# Start frontend
cd frontend && npm install && npm run dev
```
*Note: Exporting faces ensures that the Three.js viewer can render the avatar mesh correctly. The React frontend will run on port 5173 while communicating with the FastAPI backend on port 8000.*

### 3. Generate Motion (Offline)
Generate high-fidelity motion from your audio files in the `./input` folder:
```bash
python3 generate_npz.py --audio_folder ./input --save_folder ./output
```

### 4. Visualize Offline
Prefer a quick local view? Use the Streamlit or MP4 render paths:
- **Streamlit**: `python3 -m streamlit run visualize_web.py`
- **MP4 Render**: `python3 render.py` (Outputs `output.mp4`)

---

## ⚡ Real-Time Streaming Pipeline

Our streaming architecture allows you to live-stream audio to a server and get back animation frames for immediate rendering.

### 1. Start the Live Server
```bash
# Set base FPS via environment variable
STREAM_FPS=20 python3 -m uvicorn server.app:app --reload --port 8000
```

For browser Push-to-Talk conversation, set:
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for ADK/Gemini Live audio
export GEMINI_API_KEY=""

### 2. Open the Live Viewer
The Three.js viewer will be live at: `http://localhost:5173/`. Ensure the FastAPI server is running concurrently.

### 3. Stream Audio (Example Simulator)
Use our utility script to simulate a live audio stream from a file:
```bash
python3 scripts/stream_audio_to_ws.py --audio ./input/romantic_narration.mp3 --chunk 0.5
python3 scripts/stream_audio_to_ws.py --audio ./input/swara_2.mp3 --chunk 0.5
```

### 4. Protocol v2 Reconnect Flow (Implemented)
- Client connects `/ws/anim`, sends:
  - `anim_subscribe { protocol_version: 2, known_boot_id, known_server_clock_id, known_stream_session_id, last_applied_frame }`
- Server responds:
  - `anim_subscribe_ack` with `mode: resume|live_only|reset_required`
- If `resume`, server sends:
  - `anim_snapshot_start`
  - snapshot `anim` frames (`phase: "snapshot"`)
  - `anim_snapshot_end` with precise anchors:
    - `snapshot_end_frame`, `snapshot_end_server_time_ms`
    - `live_head_frame`, `live_head_server_time_ms`
    - `audio_live_edge_frame`, `audio_live_edge_server_time_ms`
- Server then sends live `anim` frames (`phase: "live"`).
- Worker treats snapshot frames as historical context only and does **not** replay snapshot history at accelerated speed.
- Reconnect playhead is seeded from live edge (`max(expected_frame, live_head_frame, audio_live_edge_frame)`), then transitions to normal live playback.

### 5. Audio Channel Metadata
`/ws/audio_out` now includes:
- `stream_session_id` (plus `session_id` fallback during rollout)
- `chunk_id`
- `audio_sample_cursor` (absolute samples in session)
- `server_time_ms`
- `server_boot_id`
- `server_clock_id`

### 6. Conversation Channel (`/ws/conversation`)
- Handshake:
  - `hello` -> `hello_ack { conversation_id, protocol_version, server_boot_id, server_clock_id, server_time_ms }`
- Lifecycle:
  - `listening`
  - `assistant_thinking_start` / `assistant_thinking_end`
  - `assistant_speaking_start { reply_id, stream_session_id }`
  - `assistant_speaking_end { reply_id, stream_session_id }`
  - `interrupted { reply_id, stream_session_id }`
- User control:
  - `ptt_start`, repeated `user_audio { seq, sr, dtype, pcm_b64 }`, `ptt_end`, `interrupt`

---

## 🧠 System Architecture

```mermaid
flowchart TD
    subgraph AudioSource ["Audio Ingestion (Real-Time)"]
        A[Gemini Live API Bridge]
        B[Simulated WAV Streamer]
    end

    subgraph Backend ["Modular Backend (FastAPI)"]
        C["/ws/audio Receiver"]
        D["audio_pipeline.py (Inference)"]
        E["session.py (State & GC)"]
        F["conversation.py (PTT/Gemini)"]
        
        C --> D
        D --> E
        F --> D
        E -->|/ws/anim| G
    end

    subgraph Frontend ["WebGL Frontend (Three.js)"]
        G[Animation Worker (Protocol v2)]
        H[Three.js Live Viewer]
        I[HUD Statistics Panel]
        
        F -->|Binary Floats| G
        G --> H
        H -.-> I
    end

    A --> C
    B --> C
```

---

## 📂 Project Structure

| Directory/File | Description |
| :--- | :--- |
| `frontend/` | React + Three.js application source code. |
| `frontend/dist/` | Frontend build output. |
| `server/` | **Modularized Backend**: `app.py` (coordinator), `session.py` (state), `audio_pipeline.py` (ML/Audio), `conversation.py` (PTT). |
| `emage_utils/` | Core EMAGE model implementation and VQ-VAE utils. |
| `scripts/` | Export utilities and audio streaming simulators. |
| `models/` | SMPL-X and EMAGE model weight storage path. |

---

## 🛠️ Advanced Configuration

Fine-tune your animation quality via CLI or config files:

- **Signal Processing**:
  - `expression_target`: Scales facial PCA coefficients for quiet audio.
  - `expression_smooth_alpha`: Alpha value for EMA smoothing (default: `1.0`).
- **Pipeline Performance**:
  - `overlap_sec`: Controls chunk overlap for seamless motion blending (default: `0.25s`).
  - `STREAM_FPS`: Sets the target animation rate for real-time vertex streaming (default: `20`).
- **Session/Recovery**:
  - `SNAPSHOT_SECONDS` (default: `3.0`)
  - `MAX_SESSIONS` (default: `8`)
  - `SESSION_IDLE_TTL_MS` (default: `45000`)
  - `DEPRECATED_TTL_MS` (default: `15000`)
  - `SESSION_GC_INTERVAL_MS` (default: `5000`)

### Reconnect & Stall Defaults (Implemented)
- `STREAM_FPS=20`
- Worker:
  - `MAX_TAIL_LOCK_MS=3000` (bounded wait for first live frame after snapshot tail alignment)
- Viewer:
  - `STARTUP_SUPPRESS_MS=4000` (suppresses expected startup watchdog noise)
  - `STALL_HOLD_MS=300` then `STALL_EASE_MS=700` (freeze-then-ease behavior)

### Edge-Case Handling (Current)
- Snapshot tail behind audio live edge: initialize at live edge immediately (no tail freeze on missing historical frames).
- No live frames after snapshot: bounded tail lock timeout, then `resync_request`.
- Session mismatch between `/ws/anim` and `/ws/audio_out`: viewer enters resync/reset path.
- Server restart (`server_boot_id` change): client receives reset semantics and reboots playback state.
- Deprecated sessions: retained briefly for resumability, then evicted by session GC.

---

## 🧪 Testing Checklist
1. [ ] Run `python3 generate_npz.py` and verify `output/intro_output.npz`.
2. [ ] Run `python3 render.py` and check `output.mp4`.
3. [ ] Export faces: `python3 scripts/export_faces.py`.
4. [ ] Start the backend server: `STREAM_FPS=20 python3 -m uvicorn server.app:app --reload --port 8000`.
5. [ ] Start the frontend: `cd frontend && npm run dev`.
6. [ ] Open the live viewer at `http://localhost:5173`.
7. [ ] Connect Microphone and verify Gemini 3.1 Live PTT conversation starts high-fidelity motion.

---

## 📜 Credits & References
- **EMAGE**: Expressive Motion Generation from Audio via Latent Cross-Modal Transformer.
- **SMPL-X**: A joint body, face, and hand model for human motion research.
- **Three.js**: The rendering system for the real-time WebGL viewer.

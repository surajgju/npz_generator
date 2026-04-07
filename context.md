# Deep Context: 3D SMPL-X Animation Streaming Pipeline (Advanced)

## 1. Technical Core: The EMAGE Model
The server uses the **EMAGE (Expressive Motion Generation)** architecture, a state-of-the-art framework for creating synchronized body and face motion from audio.
- **Latent Space**: The model leverages multiple pre-trained **VQ-VAE (Vector Quantized Variational Autoencoders)** to encode human motion into discrete latent codes across four regions:
  - **Face**: Focused on expression and lip-sync.
  - **Upper Body**: Gestures and breathing.
  - **Lower Body/Global**: Leg positioning and root translation.
  - **Hands**: Finger and palm articulation.
- **Audio Conditioning**: A cross-modal transformer or diffusion model maps raw audio spectrograms to these latent spatial-temporal tokens.
- **Outputs**:
  - **Pose (165 coeffs)**: 21 body joints (63), global root (3), jaw (3), eyes (6), and 30 hand joints (90) — all in axis-angle format.
  - **Expressions (100 coeffs)**: PCA (Principal Component Analysis) coefficients for the SMPL-X facial mesh.
  - **Translation (3 coeffs)**: Global X/Y/Z root movement.

## 2. Server-Side Retargeting & Signal Processing
Raw model outputs are normalized and mapped in `server/retargeter.py` using parameters from `retarget_map.json`:
- **Dynamic Gain**: An automatic gain control (AGC) mechanism (`expression_target`) scales PCA coefficients based on the rolling average magnitude of the last `N` frames. This ensures expressions remain expressive even for quiet audio.
- **Normalization targets**:
  - `expression_norm_target`: Target standard deviation for PCA coefficients.
  - `jaw_scale`: Independent multiplier for the physical jaw bone rotation to ensure clear mouth articulation.
- **Smoothing (EMA)**: `expression_smooth_alpha` applies an Exponential Moving Average (1.0 = no smoothing, 0.1 = heavy smoothing) to eliminate high-frequency artifacting from the AI inference.
- **Linear Temporal Resampling**: `_resample_frames_linear` handles the transition between `BASE_FPS` (30) and `STREAM_FPS` (variable) using linear interpolation for all pose and expression channels. This ensures fluid motion regardless of the target broadcast rate.
- **Slow-Motion Debugging**: A `SLOW_MOTION_FACTOR` can be applied to stretch the animation in time (e.g., 5.0 for 5x slower) while maintaining a steady broadcast rate, allowing for frame-by-frame irregularity analysis.

## 3. Frontend Architecture & Synchronization
The WebGL frontend (Three.js) is designed for low-latency visual stability.
- **Multi-Threaded Pipeline**:
  - **WebSocket Handler (Main)**: Receives raw PCM audio and JSON animation frames.
  - **Animation Worker (`web/anim_worker.js`)**: Decouples network jitter from the render loop. It maintains a **priority queue** of frames.
  - **Interpolation Engine**: If the server sends 15-30 FPS, the worker performs **Sliding Window Interpolation** to provide smooth 60+ FPS motion.
- **Sync Logic**:
  - `animOffsetSec`: A calculated delay between the server's play-time and the browser's local `performance.now()`.
  - `audioBuf`: The threshold of queued audio data. The animation loop waits for this buffer to reach a stable level (e.g., 0.5s) before starting playback to ensure perfect lip-sync.

## 4. SMPL-X Morph Target Strategy
- **Base vs. Additive**: We use **Additive Blending**. Morph influences are applied on top of the "Base Face" (Neutral Mean).
- **Persistence Layer**: To avoid flickering during network dropouts, the last applied expression is cached. Each new frame is applied as an influence delta or replacement, ensuring the avatar never snaps back to a "blank stare" mid-sentence.
- **Mesh Detection**: The system dynamically identifies morph target indices by scanning the geometry labels (e.g., searching for "Exp000", "Exp010") to maintain compatibility across different exports of the SMPL-X model.

## 5. Protocol Details
- **Signaling**: JSON messages with `type: "chunk"`, containing `chunk_id`, `nbones`, `nmorphs`, and base64 encoded interleaved data.
- **Frame Structure**:
  - `[0:3]`: Root Position (X, Y, Z).
  - `[3:nbones*4]`: Bone Rotations (Quaternions: X, Y, Z, W).
  - `[nbones*4:]`: Morph Target Influences (Floats, 0.0 - N.N).

## 6. Server Architecture & Security
- **Asynchronous Pipeline**: Uses `asyncio.Queue` (10s buffer) to decouple heavy model inference from high-frequency WebSocket broadcasting.
- **WASM Performance**: FastAPI middleware injects **COOP (Cross-Origin-Opener-Policy)** and **COEP (Cross-Origin-Embedder-Policy)** headers. These are required by modern browsers to enable high-performance multi-threading and `SharedArrayBuffer` in the Three.js renderer.
- **Self-Documenting Code**: Detailed docstrings in `app.py`, `retargeter.py`, and `build_retarget_config.py` provide a technical map of the signal processing flow and configuration parameters.

## 7. Current Debug Configuration
- **Isolation Mode**: Body bones and root translations are temporarily disabled in `web/app.js` render loop.
- **Expression Monitoring**: The HUD directly monitors `Exp000` (Jaw PCA Component 1) as a proxy for speech intensity.
- **Temporal Analysis**: Using `SLOW_MOTION_FACTOR > 1.0` combined with linear resampling to verify pose transitions and blending during high-activity segments.

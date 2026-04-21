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

## 2. Environment Configuration & Tuning
The pipeline is highly configurable via `server/.env.local`. All essential parameters for performance, logic, and visual stability are centralized here:
- **Core Settings**: `STREAM_FPS`, `INFERENCE_BATCH_SAMPLES`, `BASE_MOTION_FPS`.
- **Latency & Backlog**: `AUDIO_IN_QUEUE_MAX_CHUNKS` (default `16`), `INFERENCE_IDLE_FLUSH_SEC`.
- **GC & Stability**: `MAX_SESSIONS`, `SESSION_IDLE_TTL_MS`, `SESSION_GC_INTERVAL_MS`.
- **Frontend Sync (Vite)**: `VITE_SESSION_MISMATCH_GRACE_MS`, `VITE_AUDIO_START_BUFFER_SEC`, etc.

The `streamsettings.py` module acts as the single source of truth, loading `.env.local` and resolving these values for all backend components.

## 3. Server-Side Retargeting & Signal Processing
Raw model outputs are normalized and mapped in `server/retargeter.py` using parameters from `retarget_map.json` and global environment defaults:
- **Dynamic Gain**: An automatic gain control (AGC) mechanism (`expression_target`) scales PCA coefficients based on the rolling average magnitude of the last `N` frames. This ensures expressions remain expressive even for quiet audio.
- **Normalization targets**:
  - `expression_norm_target`: Target standard deviation for PCA coefficients.
  - `jaw_scale`: Independent multiplier for the physical jaw bone rotation to ensure clear mouth articulation.
- **Smoothing (EMA)**: `expression_smooth_alpha` applies an Exponential Moving Average (1.0 = no smoothing, 0.1 = heavy smoothing) to eliminate high-frequency artifacting from the AI inference.
- **Temporal Resampling**:
  - Pose channels use quaternion **SLERP** interpolation in `audio_pipeline.py` (axis-angle -> quaternion -> SLERP -> axis-angle) to avoid rotation ghosting artifacts.
  - Expressions and root translation still use `_resample_frames_linear`.
- **Slow-Motion Debugging**: A `SLOW_MOTION_FACTOR` can be applied to stretch the animation in time (e.g., 5.0 for 5x slower) while maintaining a steady broadcast rate, allowing for frame-by-frame irregularity analysis.
- **Inference Ingress Buffering**: `audio_in_queue` is sized by `AUDIO_IN_QUEUE_MAX_CHUNKS` (default `16`) to bound maximum animation latency and prevent backlog accumulation if the inference worker lags.
- **Global ML Bounds**: `EXPRESSION_MAX_ABS`, `MOUTH_MAX_ABS`, and `EYE_BROW_MAX_ABS` provide a safety ceiling for retargeted coefficients, configurable via environment.

## 4. Frontend Architecture & Synchronization
The WebGL frontend (Three.js) is designed for low-latency visual stability and reconnect recovery.
- **Modular Multi-Threaded Pipeline**:
  - **Main Thread UI (`ViewerController.js`)**: Orchestrates rendering, audio playback, HUD metrics, and DOM interactions.
  - **WebSocket / Protocol (`ConversationClient.js`)**: Encapsulates all backend communication for `/ws/conversation` over a unified socket.
  - **Audio Capturing (`AudioCapture.js`)**: Manages microphone `AudioWorkletNode` isolation and Push-to-Talk payloads asynchronously.
  - **Animation Worker (`frontend/src/viewer/anim_worker.js`)**: Handles protocol v2 handshake, frame buffering, interpolation, snapshot ingest, and resync signaling.
- **Worker Playback States**:
  - `snapshot_loading`: Ingest snapshot frames only.
  - `tail_lock_align`: Clamp near live target until live frames arrive.
  - `live_playing`: Normal audio-synced interpolation.
  - `resyncing`: Stop applying frames and request snapshot refresh.
- **Viewer Playback States**:
  - `buffering`, `playing`, `stalled_hold`, `stalled_ease`, `stalled_idle`, `resyncing`.
- **Sync Logic**:
  - `animOffsetSec`: Initial animation offset from worker frame indexes.
  - `audioBuf`: Audio startup/holding threshold.
  - **Decoupled Tick Clock**: Viewer always sends worker `tick` messages even when `audioStarted=false` using a monotonic fallback elapsed clock; this prevents worker freeze during audio/session recovery windows.
  - **Monotonic Time Alignment**: Worker estimates `server_time_ms - performance.now()` with robust smoothing:
    - EMA for normal jitter,
    - fast convergence for moderate drift,
    - hard snap only for large discontinuities.
- **Voice Conversation Path**:
  - Browser Push-to-Talk captures PCM16 audio via `AudioCapture.js` and sends it to `/ws/conversation` via `ConversationClient.js`.
  - Server flow: `conversation.py` (Runtime) -> `audio_pipeline.py` (`GeminiLiveAudioEngine`) -> `google-genai` Live API.
  - **Explicit Handover**: To prevent the Gemini VAD (Voice Activity Detection) from stalling the pipeline on quiet/short inputs, the backend explicitly sends an `end_of_turn=True` flag when the user releases Push-to-Talk.
  - Assistant response chunks are routed to `ingest_audio_chunk` in `audio_pipeline.py`, fanning out to `/ws/audio_out` and the ML inference queue.
  - Frontend flushes audio scheduler on `stream_session_id` change to prevent cross-reply drift.

## 5. SMPL-X Morph Target Strategy
- **Base vs. Additive**: We use **Additive Blending**. Morph influences are applied on top of the "Base Face" (Neutral Mean).
- **Persistence Layer**: To avoid flickering during network dropouts, the last applied expression is cached. Each new frame is applied as an influence delta or replacement, ensuring the avatar never snaps back to a "blank stare" mid-sentence.
- **Mesh Detection**: The system dynamically identifies morph target indices by scanning the geometry labels (e.g., searching for "Exp000", "Exp010") to maintain compatibility across different exports of the SMPL-X model.

## 6. Protocol Details (Current)
- **Protocol Version**: `2` (with legacy v1 fallback for `/ws/anim`).
- **Server Identity**:
  - `server_boot_id`: new UUID each server boot.
  - `server_clock_id`: monotonic clock identity.
  - `server_time_ms`: monotonic milliseconds since server boot.
- **Sessionized Streams**:
  - `stream_session_id` created per active audio/reply stream (`session_id` retained as compatibility alias).
  - `/ws/anim` subscriber sends `anim_subscribe` with known boot/session/frame.
  - Server responds with `anim_subscribe_ack` mode:
    - `resume`, `live_only`, `reset_required`.
- **Conversation Channel** (`/ws/conversation`):
  - `hello` / `hello_ack` with `protocol_version`.
  - explicit lifecycle events:
    - `listening`
    - `assistant_thinking_start`, `assistant_thinking_end`
    - `assistant_speaking_start`, `assistant_speaking_end` (both include `stream_session_id`)
    - `interrupted`
- **Snapshot Envelope**:
  - `anim_snapshot_start`
  - `anim` frames with `phase: "snapshot"`
  - `anim_snapshot_end` with:
    - `snapshot_end_frame`
    - `snapshot_end_server_time_ms`
    - `live_head_frame`
    - `live_head_server_time_ms`
    - `audio_live_edge_frame`
    - `audio_live_edge_server_time_ms`
    - `fps`
- **Live Frames**:
  - `anim` frames with `phase: "live"`.
- **Audio Metadata** (`/ws/audio_out`):
  - `stream_session_id`, `chunk_id`, `audio_sample_cursor`, `server_time_ms`, `server_boot_id`, `server_clock_id`.
- **Control Message**:
  - Client may send `resync_request` when tail lock or drift thresholds are exceeded.
  - Server may send `anim_session_switch` to proactively move anim clients to a new `stream_session_id` before the next live frame burst.
- **Reconnect Rule (Burst Prevention)**:
  - Snapshot frames are never replayed as backlog.
  - On `anim_snapshot_end`, worker targets live edge:
    - `target_frame = max(expected_frame_from_server_time, live_head_frame, audio_live_edge_frame)`.
  - Worker enters bounded `tail_lock_align` and requests resync if live handoff misses timeout.
- **Frame Payload Layout**:
  - `[0:3]`: Root Position (X, Y, Z).
  - `[3:3 + nbones*4]`: Bone Quaternions (X, Y, Z, W).
  - `[3 + nbones*4:]`: Morph influences.

## 7. Modular Server Architecture & Session Lifecycle
The backend is split into specialized modules for scalability and maintainability:
- **`app.py`**: Thin coordinator. Owns FastAPI WebSockets and API routing (Static asset mounting has been removed in favor of React/Vite development server on port 5173).
- **`session.py`**: Owns `SessionState` dataclasses, the shared session registry, and the GC loop.
- **`audio_pipeline.py`**: The "Core Engine". Owns `inference_worker`, `GeminiLiveAudioEngine`, and the animation broadcast loop.
- **`conversation.py`**: Owns `ConversationRuntime`, managing the PTT state machine and assistant interaction logic.

- **Asynchronous Pipeline**:
  - `audio_in_queue`: decouples incoming audio from inference.
  - `anim_queue`: decouples inference from broadcast cadence.
- **Session State (`SessionState`)**:
  - `session_id`, activity timestamps, deprecation timestamp.
  - `generation_epoch` for interrupt-safe stale-drop filtering.
  - rolling `frame_ring` for reconnect snapshots.
  - latest audio cursor/rate/time for live-edge anchors.
- **GC & Leak Prevention**:
  - `session_gc_loop` in `session.py` removes idle/deprecated sessions by TTL.
  - `MAX_SESSIONS` cap ensures memory stability.
- **WASM Performance**:
  - Middleware in `app.py` sets COOP/COEP/CORP for SharedArrayBuffer compatibility.

## 8. Current Debug Configuration
- **Debug Defaults**:
  - `DEBUG` is off by default in viewer and worker paths.
  - high-frequency logs are reduced to periodic summaries.
- **Operational Metrics**:
  - `snapshotDropCount`, `liveDropCount`, `resyncSkipped`.
  - queue length, in/out FPS, play state, audio buffer.
- **Stall/Recovery Behavior**:
  - Viewer re-applies the held last frame during lag to avoid blank/frozen output.
  - hold first (`stalled_hold`), then additive idle overlay (`stalled_idle`) after `IDLE_START_MS`.
  - Worker emits fallback frames on every tick path (latest cached frame or neutral pose) when interpolation targets are missing.
  - startup watchdog is suppressed briefly during reconnect/bootstrap.
  - Worker auto-restarts on `reset_required` after clearing stale session cache.

## 9. Operational Defaults and Edge Cases
- **Defaults**:
  - `STREAM_FPS=30`
  - `SNAPSHOT_SECONDS=3.0` (`SNAPSHOT_FRAMES=ceil(3.0*STREAM_FPS)`)
  - `MAX_TAIL_LOCK_MS=3000`
  - `STARTUP_SUPPRESS_MS=4000`
  - `STALL_HOLD_MS=300`, `STALL_EASE_MS=700`
  - `STALL_IDLE_BLEND_MS=1200`
  - `SESSION_MISMATCH_GRACE_MS=1200`
  - `AUDIO_IN_QUEUE_MAX_CHUNKS=16`
- **Handled Edge Cases**:
  - Snapshot end behind live audio: worker seeds playhead to audio/live edge immediately.
  - No live frame after snapshot: tail lock timeout -> `resync_request`.
  - Anim/audio session mismatch: viewer enters `resyncing` until channels agree; mismatch detection is grace-windowed to avoid transient false positives.
  - Soft session recovery keeps audio pipeline alive; hard resets are reserved for clock/reset-required paths.
  - Server restart (`server_boot_id` change): reset path, local state cleared.
  - Partial/expired historical state: server falls back to `live_only` or `reset_required`.
  - Deprecated session buffers: evicted by TTL/LRU GC to avoid memory growth.

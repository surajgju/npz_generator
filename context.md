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
  - **Embedding Worker (`frontend/src/rag/embeddings/embeddingWorker.js`)**: Runs the Xenova/all-MiniLM-L6-v2 model inside a dedicated background thread, offloading heavy float matrix extractions from the main browser thread to preserve rendering performance.
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
- **Voice Conversation Path & RAG Integration**:
  - Browser Push-to-Talk captures PCM16 audio via `AudioCapture.js` and sends it to `/ws/conversation` via `ConversationClient.js`.
  - Simultaneously, browser `SpeechRecognition` records local audio, generating a live text transcription.
  - Upon release, the client performs a semantic similarity search across IndexedDB, fetches the most relevant historical interactions, compiles them into a bulleted context string, and sends `ptt_end` with this context.
  - Server flow: `conversation.py` (Runtime) -> `audio_pipeline.py` (`GeminiLiveAudioEngine`) -> `google-genai` Live API. The server injects the client-supplied RAG context into the active Gemini system prompt.
  - **Explicit Handover**: To prevent the Gemini VAD (Voice Activity Detection) from stalling the pipeline on quiet/short inputs, the backend explicitly sends an `end_of_turn=True` flag when the user releases Push-to-Talk.
  - Assistant response chunks are routed to `ingest_audio_chunk` in `audio_pipeline.py`, fanning out to `/ws/audio_out` and the ML inference queue. It also yields streamed text parts via `assistant_text` events.
  - Upon full turn completion, the client combines the user's transcript and the assistant's complete response, adding it as a new interaction memory in the IndexedDB.
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
    - `assistant_text` (streaming text chunks)
    - `assistant_text_complete` (final complete text string)
    - `interrupted`
  - client control messages:
    - `ptt_start`
    - `ptt_end` (includes `context` string and `transcript` string)
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

## 10. Architectural Constraints & Known Limitations
- **Streaming Inference vs. Full-Turn Accumulation**:
  - The pipeline **MUST** remain purely streaming (ingesting audio chunks into the inference queue the moment they arrive). 
  - **DO NOT** attempt a "two-phase" or "full-turn" accumulation strategy (waiting for the LLM to finish speaking before running inference). Doing so structurally breaks the `server_time_ms` sync, causing massive animation lag relative to audio playback and destroying the real-time nature of Gemini Live.
- **Queue Limits (`AUDIO_IN_QUEUE_MAX_CHUNKS`)**:
  - Gemini Live streams audio bursts much faster than real-time (~2.2x speed). If inference runs slower than real-time, the `audio_in_queue` will rapidly fill up. If you experience "Audio queue full; dropped oldest chunk" warnings, either the queue limit must be massively expanded (e.g., 512+) or inference must be accelerated.
- **Apple Silicon (MPS) Limitations**:
  - The `EmageAudioModel` relies on several PyTorch operations (like specific `torch.cat` or convolution paths) that are not natively supported by macOS Metal Performance Shaders (MPS). 
  - Without explicit `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variables, running this model on `mps` will result in crashes or severe CPU-fallback bottlenecks (e.g. taking 3000ms+ to process 500ms of audio). 
  - Due to these $O(N^2)$ sequence bottlenecks, pure real-time 30FPS inference on M-series chips is highly constrained and may require smaller batch sizes (e.g. `4800` samples) or model quantization.

## 11. Client-Side Retrieval-Augmented Generation (RAG)
To make conversations context-aware and deeply personalized without server database overhead, the application implements a fully client-side RAG pipeline:
- **Local IndexedDB Database (`indexeddb.js`)**: Built using Dexie.js for persistent, transactional storage of interactions. The schema stores records under the `vectors` table: `id` (UUID), `text` (raw text block), `embedding` (float array), `timestamp` (epoch ms), `importance` (score multiplier), and `accessCount` (frequency tracking).
- **Background Embedding Generation Worker (`embeddingWorker.js`, `embeddingService.js`)**: 
  - Runs a compiled version of `@xenova/transformers` to load the `Xenova/all-MiniLM-L6-v2` feature extraction model.
  - Operates inside a Web Worker to ensure intensive feature calculations do not lock the Three.js rendering main thread.
  - Implements an exponential backoff retry mechanism (2s, 4s, 8s, 16s, 32s up to 5 attempts) in `embeddingMemory.js` with in-memory hashing for immediate cache hits.
- **Client-Side Cosine Similarity Vector Store (`vectorStore.js`, `ragService.js`)**:
  - Performs native floating-point vector dot products and normalization directly inside browser memory.
  - Performs a semantic top-K search against all records stored in IndexedDB.
  - Access count modifiers automatically increment matched memory priorities when cosine similarity scores exceed the `0.3` relevance threshold.
- **PT-to-RAG Unified Feedback Loop**:
  1. **PTT Start**: Browser `SpeechRecognition` begins capture.
  2. **PTT End**: The captured voice transcript is finalized, and a `semanticSearch` is dispatched to retrieve relevant historical matches.
  3. **Context Construction**: Matches with a score above `0.3` are serialized into a bulleted text context string.
  4. **PTT End Event**: The WebSocket sends `type: "ptt_end"`, including both the raw `transcript` and the RAG `context` string.
  5. **Server Prompt Injection**: `conversation.py` intercepts this payload, injecting the context under a designated section in the Gemini system prompt instructions.
  6. **Streaming text feedback**: The server streams `assistant_text` chunks back to the client along with raw audio bytes.
  7. **Memory Creation**: Upon the turn completing, the client packages the user speech and assistant complete response into `User asked: "..." \n Assistant replied: "..."`, invoking `addMemory` to save it locally.
  8. **Self-Statement Capture**: Any user statement of 3 words or more is also saved individually to boost memory granularity.

## 12. Deployment Infrastructure & Premium Aesthetics
- **EC2 Target Architecture (`deploy/`)**:
  - Automated deployment configured in `setup_ec2.sh` for provisioning Python, Node.js, and CUDA dependencies.
  - Configures systemd services `backend.service` (FastAPI streaming engine) and `admin.service` (admin dashboard) for automatic process recovery and logs streaming.
  - Sets up an Nginx server proxy (`nginx.conf`) that handles WebSocket proxying, enabling secure HTTPS/WSS upgrades and serving static build files with high performance.
- **Warm Bronze Styling System (`index.css`)**:
  - Transitioned the entire UI to a curated, high-end warm bronze aesthetic (`--cv-accent: #8b7355`, background gradients).
  - Uses modern typography, fine glassmorphism backdrops (`backdrop-filter: blur(20px)`), and thin gold/bronze inset borders.
  - Implements smooth 3D tilt perspective animations on hover. The floating HUD and conversation panel tilt dynamic and pop out towards the user (`transform: perspective(800px) rotateY(...) rotateX(...) translateZ(10px)`).
  - Central microphone orb features dynamic pulse rings:
    - Glowing forest-green pulses during audio recording (`listening` state).
    - Glowing warm bronze pulses synchronized with audio playback (`speaking` state).
  - Cleaned up the layout by removing utility tabs and loading a rich custom visual `assets/wallpaper.webp` directly as the WebGL Three.js background texture for a state-of-the-art virtual experience.

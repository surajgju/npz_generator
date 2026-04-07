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
The WebGL frontend (Three.js) is designed for low-latency visual stability and reconnect recovery.
- **Multi-Threaded Pipeline**:
  - **Main Thread (`ViewerController.js`)**: Handles rendering, audio scheduling, HUD metrics, stall states, and session checks.
  - **Animation Worker (`frontend/src/viewer/anim_worker.js`)**: Handles protocol v2 handshake, frame buffering, interpolation, snapshot ingest, and resync signaling.
- **Worker Playback States**:
  - `snapshot_loading`: Ingest snapshot frames only.
  - `tail_lock_align`: Clamp near live target until live frames arrive.
  - `live_playing`: Normal audio-synced interpolation.
  - `resyncing`: Stop applying frames and request snapshot refresh.
- **Viewer Playback States**:
  - `buffering`, `playing`, `stalled_hold`, `stalled_ease`, `resyncing`.
- **Sync Logic**:
  - `animOffsetSec`: Initial animation offset from worker frame indexes.
  - `audioBuf`: Audio startup/holding threshold.
  - **Monotonic Time Alignment**: Worker smooths `server_time_ms - performance.now()` with EMA to estimate server now and target live frame.

## 4. SMPL-X Morph Target Strategy
- **Base vs. Additive**: We use **Additive Blending**. Morph influences are applied on top of the "Base Face" (Neutral Mean).
- **Persistence Layer**: To avoid flickering during network dropouts, the last applied expression is cached. Each new frame is applied as an influence delta or replacement, ensuring the avatar never snaps back to a "blank stare" mid-sentence.
- **Mesh Detection**: The system dynamically identifies morph target indices by scanning the geometry labels (e.g., searching for "Exp000", "Exp010") to maintain compatibility across different exports of the SMPL-X model.

## 5. Protocol Details (Current)
- **Protocol Version**: `2` (with legacy v1 fallback for `/ws/anim`).
- **Server Identity**:
  - `server_boot_id`: new UUID each server boot.
  - `server_clock_id`: monotonic clock identity.
  - `server_time_ms`: monotonic milliseconds since server boot.
- **Sessionized Streams**:
  - `session_id` created when `/ws/audio` producer connects.
  - `/ws/anim` subscriber sends `anim_subscribe` with known boot/session/frame.
  - Server responds with `anim_subscribe_ack` mode:
    - `resume`, `live_only`, `reset_required`.
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
  - `session_id`, `chunk_id`, `audio_sample_cursor`, `server_time_ms`, `server_boot_id`.
- **Control Message**:
  - Client may send `resync_request` when tail lock or drift thresholds are exceeded.
- **Reconnect Rule (Burst Prevention)**:
  - Snapshot frames are never replayed as backlog.
  - On `anim_snapshot_end`, worker targets live edge:
    - `target_frame = max(expected_frame_from_server_time, live_head_frame, audio_live_edge_frame)`.
  - Worker enters bounded `tail_lock_align` and requests resync if live handoff misses timeout.
- **Frame Payload Layout**:
  - `[0:3]`: Root Position (X, Y, Z).
  - `[3:3 + nbones*4]`: Bone Quaternions (X, Y, Z, W).
  - `[3 + nbones*4:]`: Morph influences.

## 6. Server Architecture, Session Lifecycle, and Security
- **Asynchronous Pipeline**:
  - `audio_in_queue`: decouples incoming audio from inference.
  - `anim_queue`: decouples inference from broadcast cadence.
- **Session State (`SessionState`)**:
  - `session_id`, activity timestamps, deprecation timestamp.
  - rolling `frame_ring` for reconnect snapshots.
  - latest audio cursor/rate/time for live-edge anchors.
  - producer/subscriber counters.
- **Session Ownership**:
  - New `/ws/audio` producer creates a new active session.
  - Previous active session is deprecated but temporarily resumable.
- **GC & Leak Prevention**:
  - periodic cleanup removes idle/deprecated sessions by TTL.
  - `MAX_SESSIONS` cap with LRU-style eviction for stale sessions.
- **Cache Headers**:
  - HTML: `Cache-Control: no-cache`.
  - fingerprinted assets: `Cache-Control: public,max-age=31536000,immutable`.
- **WASM Performance**:
  - Middleware sets COOP/COEP/CORP for SharedArrayBuffer compatibility.

## 7. Current Debug Configuration
- **Debug Defaults**:
  - `DEBUG` is off by default in viewer and worker paths.
  - high-frequency logs are reduced to periodic summaries.
- **Operational Metrics**:
  - `snapshotDropCount`, `liveDropCount`, `resyncSkipped`.
  - queue length, in/out FPS, play state, audio buffer.
- **Stall/Recovery Behavior**:
  - hold first (`stalled_hold`), then ease (`stalled_ease`) until live frames resume.
  - startup watchdog is suppressed briefly during reconnect/bootstrap.

## 8. Operational Defaults and Edge Cases
- **Defaults**:
  - `STREAM_FPS=20`
  - `SNAPSHOT_SECONDS=3.0` (`SNAPSHOT_FRAMES=ceil(3.0*STREAM_FPS)`)
  - `MAX_TAIL_LOCK_MS=350`
  - `STARTUP_SUPPRESS_MS=4000`
  - `STALL_HOLD_MS=300`, `STALL_EASE_MS=700`
- **Handled Edge Cases**:
  - Snapshot end behind live audio: worker seeds playhead to audio/live edge immediately.
  - No live frame after snapshot: tail lock timeout -> `resync_request`.
  - Anim/audio session mismatch: viewer enters `resyncing` until channels agree.
  - Server restart (`server_boot_id` change): reset path, local state cleared.
  - Partial/expired historical state: server falls back to `live_only` or `reset_required`.
  - Deprecated session buffers: evicted by TTL/LRU GC to avoid memory growth.

import asyncio
import base64
import json
from typing import Set, Optional
import os
import sys

# Ensure repo root and server dir are on sys.path for local imports
SERVER_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from live_streaming_pipeline import LiveMotionGenerator
from retargeter import SmplxRetargeter
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

# Main FastAPI application instance
app = FastAPI()

_VISEME_BUCKETS = ["A", "E", "I", "O", "U", "MBP", "FV", "VELAR"]


def _resample_frames_linear(data: np.ndarray, target_count: int) -> np.ndarray:
    """
    Resample frame sequences in time using linear interpolation.
    This avoids nearest-index frame picking, which can drop in-between motion
    and produce visible pose jumps.
    """
    if data is None:
        return data
    if target_count <= 0:
        raise ValueError(f"target_count must be > 0, got {target_count}")
    arr = np.asarray(data, dtype=np.float32)
    frames = int(arr.shape[0]) if arr.ndim >= 1 else 0
    if frames <= 1 or frames == target_count:
        return arr
    src = np.arange(frames, dtype=np.float32)
    dst = np.linspace(0.0, float(frames - 1), num=target_count, dtype=np.float32)
    flat = arr.reshape(frames, -1)
    out = np.empty((target_count, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst, src, flat[:, col])
    return out.reshape((target_count,) + arr.shape[1:])


def _viseme_bucket_summary(retargeter: SmplxRetargeter, morphs: np.ndarray):
    """
    Groups mouth blendshapes into phonetic buckets (A, E, I, O, U, etc.)
    and calculates energy/peak statistics to monitor speech quality.
    """
    names = list(getattr(retargeter, "morphs", []) or [])
    viseme_map = getattr(retargeter, "_mouth_viseme_map", {}) or {}
    if morphs is None or morphs.size == 0 or not names or not viseme_map:
        return {}, [], []
    morphs_max = np.max(np.abs(morphs), axis=0)
    summary = {}
    for bucket in _VISEME_BUCKETS:
        mapped_names = viseme_map.get(bucket, []) or []
        idxs = [names.index(name) for name in mapped_names if name in names]
        if not idxs:
            continue
        vals = morphs_max[idxs]
        summary[bucket] = {
            "energy": float(np.mean(vals)),
            "peak": float(np.max(vals)),
            "top": [(names[idxs[i]], float(vals[i])) for i in np.argsort(-vals)[:3]],
        }
    top_bucket = sorted(summary.items(), key=lambda item: item[1]["energy"], reverse=True)[:3]
    underactive = [bucket for bucket, data in summary.items() if data["energy"] < 0.05]
    return summary, top_bucket, underactive

@app.middleware("http")
async def add_coop_coep_headers(request, call_next):
    """
    Add security headers required for SharedArrayBuffer and multi-threading in modern browsers.
    Essential for high-performance WASM components in the web front-end.
    """
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response

generator = LiveMotionGenerator(overlap_sec=0.25)
retargeter = SmplxRetargeter(
    os.path.join(os.path.dirname(__file__), "retarget_map.json"),
)

anim_clients: Set[WebSocket] = set()
audio_clients: Set[WebSocket] = set()
STREAM_FPS = int(os.environ.get("STREAM_FPS", "20"))
# Slow motion factor for debugging (e.g., 5.0 for 5x slower playback)
SLOW_MOTION_FACTOR = float(os.environ.get("SLOW_MOTION_FACTOR", "1.0"))
BASE_FPS = 30
# Frame-level queue to avoid bursty delivery; 10s buffer
anim_queue: asyncio.Queue = asyncio.Queue(maxsize=int(STREAM_FPS * 10))
audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
anim_frame_counter = 0
inference_lock = asyncio.Lock()


async def broadcast_anim_loop():
    """
    Continuously pops processed frames from the anim_queue and broadcasts them
    to all connected animation clients at a steady STREAM_FPS.
    """
    global anim_frame_counter
    while True:
        item = await anim_queue.get()
        if item is None:
            logger.debug("Anim broadcast loop received None frame")
            continue
        root_pos, bone_quats, morphs = item
        if bone_quats.ndim != 2:
            logger.warning("Anim broadcast loop received invalid bone shape: %s", getattr(bone_quats, "shape", None))
            continue
        header = {
            "type": "anim",
            "frame": anim_frame_counter,
            "fps": STREAM_FPS,
            "nbones": int(bone_quats.shape[0]),
            "nmorphs": int(morphs.shape[0]),
            "dtype": "f32",
        }
        payload = np.concatenate([root_pos, bone_quats.reshape(-1), morphs.astype(np.float32)], axis=0).astype(np.float32, copy=False).tobytes()
        dead = []
        for ws in anim_clients:
            try:
                await ws.send_text(json.dumps(header))
                await ws.send_bytes(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            anim_clients.discard(ws)
        anim_frame_counter += 1
        await asyncio.sleep(1 / STREAM_FPS)

async def inference_worker():
    """
    Background worker that performs motion generation from audio.
    Consumes raw audio chunks, runs the AI model, retargets the output to the GLB skeleton,
    and pushes the resulting animation frames to the broadcast queue.
    """
    while True:
        audio_np, sr, chunk_id = await audio_in_queue.get()
        try:
            async with inference_lock:
                coeffs = await asyncio.to_thread(_process_chunk, audio_np, sr, chunk_id)
        except Exception as exc:
            logger.exception("Inference failed for chunk %s: %s", chunk_id, exc)
            continue
        if coeffs is None:
            logger.warning("No coeffs generated for chunk %s (audio_len=%d)", chunk_id, audio_np.shape[0])
            continue
        if chunk_id is not None and str(chunk_id) == "0":
            retargeter.reset_root_offset()
        poses = coeffs["poses"]
        expressions = coeffs["expressions"]
        trans = coeffs["trans"]
        frames_count = int(poses.shape[0])
        # Resample animation frames to match STREAM_FPS, potentially applying slow motion.
        # Use linear interpolation so intermediate motion is preserved.
        if (STREAM_FPS != BASE_FPS or SLOW_MOTION_FACTOR != 1.0) and frames_count > 1:
            target_count = max(1, int(round(frames_count * (STREAM_FPS / BASE_FPS) * SLOW_MOTION_FACTOR)))
            poses = _resample_frames_linear(poses, target_count)
            expressions = _resample_frames_linear(expressions, target_count)
            trans = _resample_frames_linear(trans, target_count)
            frames_count = int(poses.shape[0])
        root_pos, bone_quats, morphs = retargeter.retarget(poses, expressions, trans)
        if frames_count > 0:
            try:
                pose_mag = float(np.mean(np.linalg.norm(poses.reshape(frames_count, -1, 3), axis=2)))
            except Exception:
                pose_mag = 0.0
            expr_mean = float(np.mean(expressions)) if expressions is not None else 0.0
            morph_mean = float(np.mean(morphs)) if morphs is not None else 0.0
            root_min = root_pos.min(axis=0) if root_pos is not None else np.zeros(3)
            root_max = root_pos.max(axis=0) if root_pos is not None else np.zeros(3)
            logger.info(
                "Anim stats chunk=%s frames=%d pose_mag=%.4f expr_mean=%.4f morph_mean=%.4f root_min=%s root_max=%s",
                chunk_id,
                frames_count,
                pose_mag,
                expr_mean,
                morph_mean,
                np.round(root_min, 4).tolist(),
                np.round(root_max, 4).tolist(),
            )
            logger.info(
                "Face gain chunk=%s expr_abs_max=%.4f gain=%.3f morph_abs_max=%.4f jaw_mag=%.4f",
                chunk_id,
                getattr(retargeter, "last_expr_abs_max", 0.0),
                getattr(retargeter, "last_expr_gain", 1.0),
                getattr(retargeter, "last_morph_abs_max", 0.0),
                getattr(retargeter, "last_jaw_mag", 0.0),
            )
            logger.info(
                "Face health chunk=%s jaw_raw=%.4f jaw_post=%.4f mouth_abs_max=%.4f mouth_energy=%.4f clip_count=%d fallback=%.2f",
                chunk_id,
                getattr(retargeter, "last_jaw_raw_mag", 0.0),
                getattr(retargeter, "last_jaw_mag", 0.0),
                getattr(retargeter, "last_mouth_abs_max", 0.0),
                getattr(retargeter, "last_mouth_energy", 0.0),
                getattr(retargeter, "last_clip_count", 0),
                getattr(retargeter, "last_fallback_gain", 1.0),
            )
            logger.info(
                "Face fallback chunk=%s fallback_gain=%.2f",
                chunk_id,
                getattr(retargeter, "last_fallback_gain", 1.0),
            )
            # Extra face diagnostics: top morphs
            if morphs is not None and morphs.size > 0:
                morphs_max = morphs.max(axis=0)
                top_idx = np.argsort(-morphs_max)[:5]
                top = [(retargeter.morphs[i], float(morphs_max[i])) for i in top_idx]
                mouth_names = getattr(retargeter, "_mouth_morphs", []) or []
                mouth_idx = [retargeter.morphs.index(name) for name in mouth_names if name in retargeter.morphs]
                mouth_top = []
                if mouth_idx:
                    mouth_vals = morphs_max[mouth_idx]
                    mouth_top_idx = np.argsort(-mouth_vals)[:5]
                    mouth_top = [(retargeter.morphs[mouth_idx[i]], float(mouth_vals[i])) for i in mouth_top_idx]
                logger.info(
                    "Face stats chunk=%s top=%s",
                    chunk_id,
                    top,
                )
                logger.info(
                    "Face mouth stats chunk=%s mouth_targets=%s mouth_top=%s",
                    chunk_id,
                    mouth_names,
                    mouth_top,
                )
            logger.info(
                "Face viseme prompt chunk=%s labels=%s mouth_targets=%s",
                chunk_id,
                _VISEME_BUCKETS,
                mouth_names,
            )
            viseme_summary, top_bucket, underactive = _viseme_bucket_summary(retargeter, morphs)
            if viseme_summary:
                logger.info(
                    "Face viseme summary chunk=%s top=%s",
                    chunk_id,
                    [(name, round(data["energy"], 4), round(data["peak"], 4)) for name, data in top_bucket],
                )
                logger.info(
                    "Face viseme buckets chunk=%s summary=%s",
                    chunk_id,
                    {
                        name: {
                            "energy": round(data["energy"], 4),
                            "peak": round(data["peak"], 4),
                            "top": data["top"],
                        }
                        for name, data in viseme_summary.items()
                    },
                )
                if underactive:
                    logger.warning(
                        "Face semantic mismatch chunk=%s underactive=%s",
                        chunk_id,
                        underactive,
                    )
        dropped = 0
        for i in range(frames_count):
            try:
                anim_queue.put_nowait((root_pos[i], bone_quats[i], morphs[i]))
            except asyncio.QueueFull:
                try:
                    _ = anim_queue.get_nowait()
                    dropped += 1
                    anim_queue.put_nowait((root_pos[i], bone_quats[i], morphs[i]))
                except Exception:
                    break
        if dropped > 0:
            logger.warning("Dropped %d anim frames due to full queue", dropped)
        logger.info("Enqueued %d anim frames (queue=%d fps=%d)", frames_count, anim_queue.qsize(), STREAM_FPS)


@app.on_event("startup")
async def _startup():
    """Initialize background tasks when the server starts."""
    app.state.broadcast_task = asyncio.create_task(broadcast_anim_loop())
    app.state.infer_task = asyncio.create_task(inference_worker())


def _process_chunk(audio_np: np.ndarray, sr: int, chunk_id: Optional[str]) -> Optional[dict]:
    """Helper to run the heavy LiveMotionGenerator in a separate thread to keep the event loop responsive."""
    coeffs = generator.process_audio_chunk(audio_np, sr, chunk_id)
    if not coeffs:
        return None
    return coeffs


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Primary WebSocket for receiving incoming audio streams from Gemini/Audio source.
    Decodes base64 audio and routes it to both listening clients and the inference worker.
    """
    await websocket.accept()
    logger.info("Audio WS connected: %s", websocket.client)
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            audio_b64 = data.get("audio_b64")
            if not audio_b64:
                await websocket.send_text(json.dumps({"type": "error", "error": "audio_b64 missing"}))
                continue
            sr = int(data.get("sr", 16000))
            dtype = str(data.get("dtype", "int16")).lower()
            chunk_id = data.get("chunk_id")
            logger.debug("Audio chunk recv id=%s sr=%d dtype=%s b64=%d", chunk_id, sr, dtype, len(audio_b64))
            audio_bytes = base64.b64decode(audio_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32) if dtype == "float32" else np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Broadcast audio to listeners first (lowest latency)
            if audio_clients:
                header = {
                    "type": "audio",
                    "chunk_id": chunk_id,
                    "sr": sr,
                    "channels": 1,
                    "dtype": "int16",
                    "samples": int(len(audio_bytes) // 2),
                }
                dead_audio = []
                for ws in audio_clients:
                    try:
                        await ws.send_text(json.dumps(header))
                        await ws.send_bytes(audio_bytes)
                    except Exception:
                        dead_audio.append(ws)
                for ws in dead_audio:
                    audio_clients.discard(ws)
                logger.info("Audio broadcast chunk_id=%s samples=%d clients=%d", chunk_id, int(len(audio_bytes) // 2), len(audio_clients))
            # Ack immediately to avoid upstream stalls
            await websocket.send_text(json.dumps({"type": "ack", "chunk_id": chunk_id}))
            try:
                audio_in_queue.put_nowait((audio_np, sr, chunk_id))
            except asyncio.QueueFull:
                try:
                    _ = audio_in_queue.get_nowait()
                    audio_in_queue.put_nowait((audio_np, sr, chunk_id))
                    logger.warning("Audio queue full; dropped oldest chunk %s", chunk_id)
                except Exception:
                    logger.error("Audio queue overflow; dropping chunk %s", chunk_id)
    except WebSocketDisconnect:
        logger.info("Audio WS disconnected: %s", websocket.client)

@app.websocket("/ws/audio_out")
async def ws_audio_out(websocket: WebSocket):
    """Secondary WebSocket to stream the incoming audio to web clients for playback."""
    await websocket.accept()
    audio_clients.add(websocket)
    logger.info("Audio OUT WS connected: %s (clients=%d)", websocket.client, len(audio_clients))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        audio_clients.discard(websocket)
        logger.info("Audio OUT WS disconnected: %s (clients=%d)", websocket.client, len(audio_clients))


@app.websocket("/ws/anim")
async def ws_anim(websocket: WebSocket):
    """WebSocket for streaming retargeted animation frames (pose + morphs) to the 3D renderer."""
    await websocket.accept()
    anim_clients.add(websocket)
    logger.info("Anim WS connected: %s (clients=%d)", websocket.client, len(anim_clients))
    try:
        await websocket.send_text(json.dumps(retargeter.anim_init_header(STREAM_FPS)))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        anim_clients.discard(websocket)
        logger.info("Anim WS disconnected: %s (clients=%d)", websocket.client, len(anim_clients))


# Mount static site last so WebSocket routes take precedence.
app.mount("/", StaticFiles(directory="web", html=True), name="web")

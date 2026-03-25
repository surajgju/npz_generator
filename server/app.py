import asyncio
import base64
import json
from typing import Set, Optional
import os
import sys

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from live_streaming_pipeline import LiveMotionGenerator, SmplxVertexStreamer
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI()

@app.middleware("http")
async def add_coop_coep_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response

generator = LiveMotionGenerator(overlap_sec=0.25)
vertex_streamer = SmplxVertexStreamer()

clients: Set[WebSocket] = set()
audio_clients: Set[WebSocket] = set()
STREAM_FPS = int(os.environ.get("STREAM_FPS", "20"))
BASE_FPS = 30
# Frame-level queue to avoid bursty delivery; 10s buffer
queue: asyncio.Queue = asyncio.Queue(maxsize=int(STREAM_FPS * 10))
audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
frame_counter = 0
inference_lock = asyncio.Lock()


async def broadcast_loop():
    global frame_counter
    while True:
        item = await queue.get()
        if item is None:
            logger.debug("Broadcast loop received None frame")
            continue
        try:
            q_frame, vmin, scale = item
        except Exception:
            logger.warning("Broadcast loop received invalid payload: %s", type(item))
            continue
        if q_frame.ndim != 2:
            logger.warning("Broadcast loop received invalid frame shape: %s", getattr(q_frame, "shape", None))
            continue
        logger.debug("Broadcasting frame %d to %d clients (queue=%d)", frame_counter, len(clients), queue.qsize())
        header = {
            "type": "verts",
            "frame": frame_counter,
            "nverts": int(q_frame.shape[0]),
            "fps": STREAM_FPS,
            "dtype": "int16",
            "quant": "minmax",
            "min": vmin.tolist(),
            "scale": scale.tolist(),
        }
        payload = q_frame.astype(np.int16, copy=False).tobytes()
        dead = []
        for ws in clients:
            try:
                await ws.send_text(json.dumps(header))
                await ws.send_bytes(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
        frame_counter += 1
        await asyncio.sleep(1 / STREAM_FPS)

async def inference_worker():
    while True:
        audio_np, sr, chunk_id = await audio_in_queue.get()
        try:
            async with inference_lock:
                verts = await asyncio.to_thread(_process_chunk, audio_np, sr, chunk_id)
        except Exception as exc:
            logger.exception("Inference failed for chunk %s: %s", chunk_id, exc)
            continue
        if verts is None:
            logger.warning("No verts generated for chunk %s (audio_len=%d)", chunk_id, audio_np.shape[0])
            continue
        frames_count = int(verts.shape[0])
        if STREAM_FPS != BASE_FPS and frames_count > 1:
            target_count = max(1, int(round(frames_count * STREAM_FPS / BASE_FPS)))
            if target_count <= 1:
                idx = np.array([0], dtype=np.int32)
            else:
                step = (frames_count - 1) / float(target_count - 1)
                idx = np.round(np.arange(target_count) * step).astype(np.int32)
            verts = verts[idx]
            frames_count = int(verts.shape[0])
        verts = verts.astype(np.float32, copy=False)
        vmin = verts.min(axis=(0, 1))
        vmax = verts.max(axis=(0, 1))
        vrange = vmax - vmin
        vmin = vmin - vrange * 0.02
        vmax = vmax + vrange * 0.02
        scale = (vmax - vmin) / 65535.0
        scale = np.where(scale == 0, 1e-6, scale)
        logger.info("Chunk quant: id=%s vmin=%s vmax=%s scale=%s", chunk_id, vmin.tolist(), vmax.tolist(), scale.tolist())
        dropped = 0
        for i in range(frames_count):
            try:
                q = np.rint((verts[i] - vmin) / scale)
                q = np.clip(q, 0, 65535).astype(np.int32) - 32768
                q = q.astype(np.int16)
                queue.put_nowait((q, vmin, scale))
            except asyncio.QueueFull:
                try:
                    _ = queue.get_nowait()
                    dropped += 1
                    q = np.rint((verts[i] - vmin) / scale)
                    q = np.clip(q, 0, 65535).astype(np.int32) - 32768
                    q = q.astype(np.int16)
                    queue.put_nowait((q, vmin, scale))
                except Exception:
                    break
        if dropped > 0:
            logger.warning("Dropped %d frames due to full queue", dropped)
        logger.info("Enqueued %d frames (queue=%d fps=%d)", frames_count, queue.qsize(), STREAM_FPS)


@app.on_event("startup")
async def _startup():
    app.state.broadcast_task = asyncio.create_task(broadcast_loop())
    app.state.infer_task = asyncio.create_task(inference_worker())


def _process_chunk(audio_np: np.ndarray, sr: int, chunk_id: Optional[str]) -> Optional[np.ndarray]:
    coeffs = generator.process_audio_chunk(audio_np, sr, chunk_id)
    if not coeffs:
        return None
    verts = vertex_streamer.vertices_from_coeffs(
        coeffs["poses"],
        coeffs["expressions"],
        coeffs["trans"],
        coeffs.get("betas"),
    )
    return verts


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
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
    await websocket.accept()
    audio_clients.add(websocket)
    logger.info("Audio OUT WS connected: %s (clients=%d)", websocket.client, len(audio_clients))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        audio_clients.discard(websocket)
        logger.info("Audio OUT WS disconnected: %s (clients=%d)", websocket.client, len(audio_clients))


@app.websocket("/ws/verts")
async def ws_verts(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    logger.info("Verts WS connected: %s (clients=%d)", websocket.client, len(clients))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
        logger.info("Verts WS disconnected: %s (clients=%d)", websocket.client, len(clients))


# Mount static site last so WebSocket routes take precedence.
app.mount("/", StaticFiles(directory="web", html=True), name="web")

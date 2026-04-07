import asyncio
import base64
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Ensure repo root and server dir are on sys.path for local imports
SERVER_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from live_streaming_pipeline import LiveMotionGenerator
from npz_logging import setup_logging
from retargeter import SmplxRetargeter

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

PROTOCOL_VERSION = 2
SERVER_BOOT_ID = str(uuid.uuid4())
SERVER_CLOCK_ID = "monotonic-ms-v1"
BOOT_MONO_NS = time.monotonic_ns()

_VISEME_BUCKETS = ["A", "E", "I", "O", "U", "MBP", "FV", "VELAR"]


def _server_time_ms() -> int:
    return int((time.monotonic_ns() - BOOT_MONO_NS) / 1_000_000)


def _payload_bytes(root_pos: np.ndarray, bone_quats: np.ndarray, morphs: np.ndarray) -> bytes:
    return (
        np.concatenate([root_pos, bone_quats.reshape(-1), morphs.astype(np.float32)], axis=0)
        .astype(np.float32, copy=False)
        .tobytes()
    )


def _resample_frames_linear(data: np.ndarray, target_count: int) -> np.ndarray:
    """Resample frame sequences in time using linear interpolation."""
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
async def add_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    elif re.search(r"\.[0-9a-fA-F]{8,}\.", path):
        response.headers["Cache-Control"] = "public,max-age=31536000,immutable"
    return response


generator = LiveMotionGenerator(overlap_sec=0.25)
retargeter = SmplxRetargeter(os.path.join(os.path.dirname(__file__), "retarget_map.json"))

STREAM_FPS = int(os.environ.get("STREAM_FPS", "20"))
SLOW_MOTION_FACTOR = float(os.environ.get("SLOW_MOTION_FACTOR", "1.0"))
BASE_FPS = 30

SNAPSHOT_SECONDS = float(os.environ.get("SNAPSHOT_SECONDS", "3.0"))
SNAPSHOT_FRAMES = int(math.ceil(SNAPSHOT_SECONDS * STREAM_FPS))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
SESSION_IDLE_TTL_MS = int(os.environ.get("SESSION_IDLE_TTL_MS", "45000"))
DEPRECATED_TTL_MS = int(os.environ.get("DEPRECATED_TTL_MS", "15000"))
SESSION_GC_INTERVAL_MS = int(os.environ.get("SESSION_GC_INTERVAL_MS", "5000"))

anim_queue: asyncio.Queue = asyncio.Queue(maxsize=int(STREAM_FPS * 10))
audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
inference_lock = asyncio.Lock()

anim_clients: Set[WebSocket] = set()
anim_client_protocol: Dict[WebSocket, int] = {}
anim_client_session: Dict[WebSocket, Optional[str]] = {}
audio_clients: Set[WebSocket] = set()


@dataclass
class SessionFrame:
    frame: int
    server_time_ms: int
    root_pos: np.ndarray
    bone_quats: np.ndarray
    morphs: np.ndarray


@dataclass
class SessionState:
    session_id: str
    created_ms: int
    last_activity_ms: int
    deprecated_at_ms: Optional[int] = None
    frame_ring: Deque[SessionFrame] = field(default_factory=lambda: deque(maxlen=SNAPSHOT_FRAMES))
    latest_frame: int = -1
    latest_audio_cursor: int = 0
    latest_audio_sr: int = 16000
    latest_audio_server_time_ms: int = 0
    producer_connected: bool = False
    anim_subscriber_count: int = 0


sessions_lock = asyncio.Lock()
sessions: Dict[str, SessionState] = {}
active_session_id: Optional[str] = None


def _audio_frame_from_cursor(samples: int, sr: int) -> int:
    if sr <= 0:
        return 0
    return int(round((samples / float(sr)) * STREAM_FPS))


def _drain_queue_nowait(q: asyncio.Queue) -> int:
    drained = 0
    while True:
        try:
            _ = q.get_nowait()
            drained += 1
        except Exception:
            break
    return drained


async def _create_audio_session() -> str:
    global active_session_id
    now_ms = _server_time_ms()
    sid = str(uuid.uuid4())
    async with sessions_lock:
        protocol2_clients = 0
        if active_session_id and active_session_id in sessions:
            prev = sessions[active_session_id]
            prev.deprecated_at_ms = now_ms
            prev.producer_connected = False
            prev.anim_subscriber_count = 0
        for ws in list(anim_clients):
            if anim_client_protocol.get(ws, 1) >= 2:
                protocol2_clients += 1
                anim_client_session[ws] = sid
        sessions[sid] = SessionState(
            session_id=sid,
            created_ms=now_ms,
            last_activity_ms=now_ms,
            latest_audio_server_time_ms=now_ms,
            producer_connected=True,
            anim_subscriber_count=protocol2_clients,
        )
        active_session_id = sid
    # Switching streams: clear queued old work so new session starts clean.
    drained_anim = _drain_queue_nowait(anim_queue)
    drained_audio = _drain_queue_nowait(audio_in_queue)
    logger.info(
        "Created session %s (boot=%s). Drained anim=%d audio=%d",
        sid,
        SERVER_BOOT_ID,
        drained_anim,
        drained_audio,
    )
    return sid


async def _session_gc_loop():
    global active_session_id
    while True:
        await asyncio.sleep(SESSION_GC_INTERVAL_MS / 1000.0)
        now_ms = _server_time_ms()
        async with sessions_lock:
            removable: List[str] = []
            for sid, st in sessions.items():
                idle_ms = now_ms - st.last_activity_ms
                deprecated_ms = (now_ms - st.deprecated_at_ms) if st.deprecated_at_ms is not None else None
                if (
                    idle_ms > SESSION_IDLE_TTL_MS
                    and not st.producer_connected
                    and st.anim_subscriber_count <= 0
                ):
                    removable.append(sid)
                    continue
                if st.deprecated_at_ms is not None and deprecated_ms is not None and deprecated_ms > DEPRECATED_TTL_MS:
                    removable.append(sid)
            if len(sessions) - len(removable) > MAX_SESSIONS:
                candidates = sorted(
                    (
                        st
                        for sid, st in sessions.items()
                        if sid not in removable and not st.producer_connected and st.anim_subscriber_count <= 0
                    ),
                    key=lambda st: st.last_activity_ms,
                )
                overflow = (len(sessions) - len(removable)) - MAX_SESSIONS
                removable.extend([st.session_id for st in candidates[:overflow]])
            for sid in removable:
                if sid in sessions:
                    del sessions[sid]
                    if sid == active_session_id:
                        active_session_id = None
            if removable:
                logger.info("Session GC evicted=%s active=%s", removable, active_session_id)


async def _session_snapshot(session_id: str) -> Tuple[Optional[SessionState], List[SessionFrame]]:
    async with sessions_lock:
        st = sessions.get(session_id)
        if st is None:
            return None, []
        st.last_activity_ms = _server_time_ms()
        frames = list(st.frame_ring)
        snap = SessionState(
            session_id=st.session_id,
            created_ms=st.created_ms,
            last_activity_ms=st.last_activity_ms,
            deprecated_at_ms=st.deprecated_at_ms,
            frame_ring=deque(maxlen=SNAPSHOT_FRAMES),
            latest_frame=st.latest_frame,
            latest_audio_cursor=st.latest_audio_cursor,
            latest_audio_sr=st.latest_audio_sr,
            latest_audio_server_time_ms=st.latest_audio_server_time_ms,
            producer_connected=st.producer_connected,
            anim_subscriber_count=st.anim_subscriber_count,
        )
        return snap, frames


async def _send_snapshot(ws: WebSocket, session_id: str) -> None:
    st, frames = await _session_snapshot(session_id)
    if st is None:
        return
    if frames:
        start_frame = frames[0].frame
        end_frame = frames[-1].frame
    else:
        start_frame = -1
        end_frame = -1
    await ws.send_text(
        json.dumps(
            {
                "type": "anim_snapshot_start",
                "session_id": session_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": STREAM_FPS,
            }
        )
    )
    for fr in frames:
        header = {
            "type": "anim",
            "session_id": session_id,
            "phase": "snapshot",
            "frame": fr.frame,
            "fps": STREAM_FPS,
            "server_time_ms": fr.server_time_ms,
            "nbones": int(fr.bone_quats.shape[0]),
            "nmorphs": int(fr.morphs.shape[0]),
            "dtype": "f32",
        }
        await ws.send_text(json.dumps(header))
        await ws.send_bytes(_payload_bytes(fr.root_pos, fr.bone_quats, fr.morphs))
    live_head_frame = st.latest_frame
    live_head_server_time_ms = frames[-1].server_time_ms if frames else _server_time_ms()
    audio_live_edge_frame = _audio_frame_from_cursor(st.latest_audio_cursor, st.latest_audio_sr)
    await ws.send_text(
        json.dumps(
            {
                "type": "anim_snapshot_end",
                "session_id": session_id,
                "snapshot_end_frame": end_frame,
                "snapshot_end_server_time_ms": live_head_server_time_ms,
                "live_head_frame": live_head_frame,
                "live_head_server_time_ms": live_head_server_time_ms,
                "audio_live_edge_frame": audio_live_edge_frame,
                "audio_live_edge_server_time_ms": st.latest_audio_server_time_ms,
                "fps": STREAM_FPS,
            }
        )
    )


async def broadcast_anim_loop():
    while True:
        item = await anim_queue.get()
        if item is None:
            continue
        session_id, root_pos, bone_quats, morphs = item
        if bone_quats.ndim != 2:
            logger.warning("Anim broadcast invalid bone shape: %s", getattr(bone_quats, "shape", None))
            continue
        now_ms = _server_time_ms()
        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            frame_id = st.latest_frame + 1
            st.latest_frame = frame_id
            st.last_activity_ms = now_ms
            st.frame_ring.append(
                SessionFrame(
                    frame=frame_id,
                    server_time_ms=now_ms,
                    root_pos=root_pos,
                    bone_quats=bone_quats,
                    morphs=morphs,
                )
            )
            active_sid = active_session_id
        if session_id != active_sid:
            await asyncio.sleep(1 / STREAM_FPS)
            continue
        payload = _payload_bytes(root_pos, bone_quats, morphs)
        dead: List[WebSocket] = []
        for ws in list(anim_clients):
            try:
                header = {
                    "type": "anim",
                    "session_id": session_id,
                    "phase": "live",
                    "frame": frame_id,
                    "fps": STREAM_FPS,
                    "server_time_ms": now_ms,
                    "nbones": int(bone_quats.shape[0]),
                    "nmorphs": int(morphs.shape[0]),
                    "dtype": "f32",
                }
                await ws.send_text(json.dumps(header))
                await ws.send_bytes(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            anim_clients.discard(ws)
            anim_client_protocol.pop(ws, None)
            sid = anim_client_session.pop(ws, None)
            if sid is None:
                continue
            async with sessions_lock:
                st = sessions.get(sid)
                if st:
                    st.anim_subscriber_count = max(0, st.anim_subscriber_count - 1)
        await asyncio.sleep(1 / STREAM_FPS)


async def inference_worker():
    while True:
        audio_np, sr, chunk_id, session_id = await audio_in_queue.get()
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
                "Anim stats session=%s chunk=%s frames=%d pose_mag=%.4f expr_mean=%.4f morph_mean=%.4f root_min=%s root_max=%s",
                session_id,
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
                logger.info("Face stats chunk=%s top=%s", chunk_id, top)
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
                    logger.warning("Face semantic mismatch chunk=%s underactive=%s", chunk_id, underactive)
        dropped = 0
        for i in range(frames_count):
            try:
                anim_queue.put_nowait((session_id, root_pos[i], bone_quats[i], morphs[i]))
            except asyncio.QueueFull:
                try:
                    _ = anim_queue.get_nowait()
                    dropped += 1
                    anim_queue.put_nowait((session_id, root_pos[i], bone_quats[i], morphs[i]))
                except Exception:
                    break
        if dropped > 0:
            logger.warning("Dropped %d anim frames due to full queue (session=%s)", dropped, session_id)
        logger.info(
            "Enqueued %d anim frames (session=%s queue=%d fps=%d)",
            frames_count,
            session_id,
            anim_queue.qsize(),
            STREAM_FPS,
        )


@app.on_event("startup")
async def _startup():
    logger.info(
        "Server startup boot=%s clock=%s protocol=%d fps=%d",
        SERVER_BOOT_ID,
        SERVER_CLOCK_ID,
        PROTOCOL_VERSION,
        STREAM_FPS,
    )
    app.state.broadcast_task = asyncio.create_task(broadcast_anim_loop())
    app.state.infer_task = asyncio.create_task(inference_worker())
    app.state.gc_task = asyncio.create_task(_session_gc_loop())


def _process_chunk(audio_np: np.ndarray, sr: int, chunk_id: Optional[str]) -> Optional[dict]:
    coeffs = generator.process_audio_chunk(audio_np, sr, chunk_id)
    if not coeffs:
        return None
    return coeffs


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    session_id = await _create_audio_session()
    logger.info("Audio WS connected: %s session=%s", websocket.client, session_id)
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
            logger.debug("Audio chunk recv session=%s id=%s sr=%d dtype=%s", session_id, chunk_id, sr, dtype)
            audio_bytes = base64.b64decode(audio_b64)
            if dtype == "float32":
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sample_count = int(audio_np.shape[0])
            now_ms = _server_time_ms()
            async with sessions_lock:
                st = sessions.get(session_id)
                if st:
                    st.last_activity_ms = now_ms
                    st.latest_audio_sr = sr
                    st.latest_audio_cursor += sample_count
                    st.latest_audio_server_time_ms = now_ms
            audio_cursor = 0
            async with sessions_lock:
                st = sessions.get(session_id)
                if st:
                    audio_cursor = st.latest_audio_cursor
            if audio_clients:
                header = {
                    "type": "audio",
                    "session_id": session_id,
                    "chunk_id": chunk_id,
                    "sr": sr,
                    "channels": 1,
                    "dtype": "int16",
                    "samples": sample_count,
                    "audio_sample_cursor": audio_cursor,
                    "server_time_ms": now_ms,
                    "server_boot_id": SERVER_BOOT_ID,
                }
                dead_audio = []
                for ws in list(audio_clients):
                    try:
                        await ws.send_text(json.dumps(header))
                        await ws.send_bytes(audio_bytes)
                    except Exception:
                        dead_audio.append(ws)
                for ws in dead_audio:
                    audio_clients.discard(ws)
                logger.info(
                    "Audio broadcast session=%s chunk_id=%s samples=%d clients=%d",
                    session_id,
                    chunk_id,
                    sample_count,
                    len(audio_clients),
                )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ack",
                        "chunk_id": chunk_id,
                        "session_id": session_id,
                        "server_boot_id": SERVER_BOOT_ID,
                        "server_time_ms": now_ms,
                    }
                )
            )
            try:
                audio_in_queue.put_nowait((audio_np, sr, chunk_id, session_id))
            except asyncio.QueueFull:
                try:
                    old = audio_in_queue.get_nowait()
                    audio_in_queue.put_nowait((audio_np, sr, chunk_id, session_id))
                    logger.warning("Audio queue full; dropped oldest chunk %s", old[2] if len(old) > 2 else "?")
                except Exception:
                    logger.error("Audio queue overflow; dropping chunk %s", chunk_id)
    except WebSocketDisconnect:
        async with sessions_lock:
            st = sessions.get(session_id)
            if st:
                st.producer_connected = False
                st.last_activity_ms = _server_time_ms()
        logger.info("Audio WS disconnected: %s session=%s", websocket.client, session_id)


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


@app.websocket("/ws/anim")
async def ws_anim(websocket: WebSocket):
    await websocket.accept()
    logger.info("Anim WS connected: %s", websocket.client)
    protocol = 1
    subscribed_session: Optional[str] = None
    try:
        first_text: Optional[str] = None
        try:
            first_text = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
        except asyncio.TimeoutError:
            first_text = None
        subscribe_msg: Dict[str, Any] = {}
        if first_text:
            try:
                subscribe_msg = json.loads(first_text)
            except Exception:
                subscribe_msg = {}
        if subscribe_msg.get("type") == "anim_subscribe" and int(subscribe_msg.get("protocol_version", 2)) >= 2:
            protocol = int(subscribe_msg.get("protocol_version", 2))
            known_boot_id = subscribe_msg.get("known_boot_id")
            known_session_id = subscribe_msg.get("known_session_id")
            async with sessions_lock:
                active_sid = active_session_id
            if known_boot_id and known_boot_id != SERVER_BOOT_ID:
                mode = "reset_required"
            elif active_sid is None:
                mode = "live_only"
            elif known_session_id and known_session_id == active_sid:
                mode = "resume"
            else:
                mode = "live_only"
            subscribed_session = active_sid
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "anim_subscribe_ack",
                        "protocol_version": PROTOCOL_VERSION,
                        "mode": mode,
                        "server_boot_id": SERVER_BOOT_ID,
                        "server_clock_id": SERVER_CLOCK_ID,
                        "session_id": active_sid,
                        "stream_fps": STREAM_FPS,
                        "server_time_ms": _server_time_ms(),
                    }
                )
            )
            if mode == "reset_required":
                await websocket.close()
                return
            await websocket.send_text(
                json.dumps(
                    {
                        **retargeter.anim_init_header(STREAM_FPS),
                        "protocol_version": PROTOCOL_VERSION,
                        "server_boot_id": SERVER_BOOT_ID,
                        "server_clock_id": SERVER_CLOCK_ID,
                        "session_id": active_sid,
                    }
                )
            )
            if mode == "resume" and active_sid:
                await _send_snapshot(websocket, active_sid)
        else:
            protocol = 1
            await websocket.send_text(json.dumps(retargeter.anim_init_header(STREAM_FPS)))
        anim_clients.add(websocket)
        anim_client_protocol[websocket] = protocol
        anim_client_session[websocket] = subscribed_session
        if subscribed_session:
            async with sessions_lock:
                st = sessions.get(subscribed_session)
                if st:
                    st.anim_subscriber_count += 1
                    st.last_activity_ms = _server_time_ms()
        while True:
            text = await websocket.receive_text()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except Exception:
                continue
            if msg.get("type") == "resync_request":
                req_session = msg.get("session_id")
                async with sessions_lock:
                    active_sid = active_session_id
                if active_sid and req_session == active_sid:
                    logger.info("Resync request accepted session=%s reason=%s", req_session, msg.get("reason"))
                    await _send_snapshot(websocket, active_sid)
                else:
                    logger.info(
                        "Resync request ignored req_session=%s active=%s reason=%s",
                        req_session,
                        active_sid,
                        msg.get("reason"),
                    )
    except WebSocketDisconnect:
        anim_clients.discard(websocket)
        anim_client_protocol.pop(websocket, None)
        sid = anim_client_session.pop(websocket, None)
        if sid:
            async with sessions_lock:
                st = sessions.get(sid)
                if st:
                    st.anim_subscriber_count = max(0, st.anim_subscriber_count - 1)
                    st.last_activity_ms = _server_time_ms()
        logger.info("Anim WS disconnected: %s (clients=%d)", websocket.client, len(anim_clients))


# Mount static site last so WebSocket routes take precedence.
app.mount("/", StaticFiles(directory="web", html=True), name="web")

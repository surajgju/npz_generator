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
CONVERSATION_PROTOCOL_VERSION = 1
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
CONVERSATION_AUDIO_SR = int(os.environ.get("CONVERSATION_AUDIO_SR", os.environ.get("CONVERSATION_STT_SR", "16000")))
CONVERSATION_PCM_CHUNK_MS = int(os.environ.get("CONVERSATION_PCM_CHUNK_MS", "20"))
CONVERSATION_PTT_MAX_SEC = float(os.environ.get("CONVERSATION_PTT_MAX_SEC", "20"))

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
    next_frame_index: int = 0
    latest_audio_cursor: int = 0
    latest_audio_sr: int = 16000
    latest_audio_server_time_ms: int = 0
    producer_connected: bool = False
    anim_subscriber_count: int = 0
    generation_epoch: int = 0
    conversation_id: Optional[str] = None
    reply_id: Optional[str] = None


sessions_lock = asyncio.Lock()
sessions: Dict[str, SessionState] = {}
active_session_id: Optional[str] = None
epoch_drop_count = 0


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


def _resolve_stream_session_id(payload: Dict[str, Any]) -> Optional[str]:
    return (
        payload.get("stream_session_id")
        or payload.get("streamsessionid")
        or payload.get("session_id")
    )


def _pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


async def _broadcast_audio_control(action: str, stream_session_id: str, reason: str) -> None:
    if not audio_clients:
        return
    header = {
        "type": "audio_control",
        "action": action,
        "stream_session_id": stream_session_id,
        "session_id": stream_session_id,
        "reason": reason,
        "server_time_ms": _server_time_ms(),
        "server_boot_id": SERVER_BOOT_ID,
        "server_clock_id": SERVER_CLOCK_ID,
    }
    dead_audio: List[WebSocket] = []
    for ws in list(audio_clients):
        try:
            await ws.send_text(json.dumps(header))
        except Exception:
            dead_audio.append(ws)
    for ws in dead_audio:
        audio_clients.discard(ws)


async def ingest_audio_chunk(
    *,
    audio_np: np.ndarray,
    sr: int,
    chunk_id: Optional[str],
    stream_session_id: str,
    generation_epoch: int,
    source: str,
    conversation_id: Optional[str],
    reply_id: Optional[str],
) -> None:
    global epoch_drop_count
    sample_count = int(audio_np.shape[0])
    now_ms = _server_time_ms()
    async with sessions_lock:
        st = sessions.get(stream_session_id)
        if st is None:
            logger.warning("Dropping audio chunk for missing stream session=%s", stream_session_id)
            return
        if generation_epoch != st.generation_epoch:
            epoch_drop_count += 1
            logger.info(
                "Dropping stale audio chunk session=%s epoch=%d active_epoch=%d",
                stream_session_id,
                generation_epoch,
                st.generation_epoch,
            )
            return
        st.last_activity_ms = now_ms
        st.latest_audio_sr = sr
        st.latest_audio_cursor += sample_count
        st.latest_audio_server_time_ms = now_ms
        if conversation_id:
            st.conversation_id = conversation_id
        if reply_id:
            st.reply_id = reply_id
        audio_cursor = st.latest_audio_cursor
    audio_int16 = np.clip(audio_np * 32768.0, -32768, 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    if audio_clients:
        header = {
            "type": "audio",
            "stream_session_id": stream_session_id,
            "session_id": stream_session_id,
            "conversation_id": conversation_id,
            "reply_id": reply_id,
            "chunk_id": chunk_id,
            "sr": sr,
            "channels": 1,
            "dtype": "int16",
            "samples": sample_count,
            "audio_sample_cursor": audio_cursor,
            "generation_epoch": generation_epoch,
            "server_time_ms": now_ms,
            "server_boot_id": SERVER_BOOT_ID,
            "server_clock_id": SERVER_CLOCK_ID,
        }
        dead_audio: List[WebSocket] = []
        for ws in list(audio_clients):
            try:
                await ws.send_text(json.dumps(header))
                await ws.send_bytes(audio_bytes)
            except Exception:
                dead_audio.append(ws)
        for ws in dead_audio:
            audio_clients.discard(ws)
        logger.info(
            "Audio broadcast stream=%s chunk_id=%s samples=%d clients=%d source=%s",
            stream_session_id,
            chunk_id,
            sample_count,
            len(audio_clients),
            source,
        )
    try:
        audio_in_queue.put_nowait(
            (
                audio_np,
                sr,
                chunk_id,
                stream_session_id,
                generation_epoch,
                source,
                conversation_id,
                reply_id,
            )
        )
    except asyncio.QueueFull:
        try:
            old = audio_in_queue.get_nowait()
            audio_in_queue.put_nowait(
                (
                    audio_np,
                    sr,
                    chunk_id,
                    stream_session_id,
                    generation_epoch,
                    source,
                    conversation_id,
                    reply_id,
                )
            )
            logger.warning("Audio queue full; dropped oldest chunk %s", old[2] if len(old) > 2 else "?")
        except Exception:
            logger.error("Audio queue overflow; dropping chunk %s", chunk_id)


async def _create_audio_session(
    conversation_id: Optional[str] = None,
    reply_id: Optional[str] = None,
) -> str:
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
            next_frame_index=0,
            producer_connected=True,
            anim_subscriber_count=protocol2_clients,
            generation_epoch=0,
            conversation_id=conversation_id,
            reply_id=reply_id,
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
            next_frame_index=st.next_frame_index,
            latest_audio_cursor=st.latest_audio_cursor,
            latest_audio_sr=st.latest_audio_sr,
            latest_audio_server_time_ms=st.latest_audio_server_time_ms,
            producer_connected=st.producer_connected,
            anim_subscriber_count=st.anim_subscriber_count,
            generation_epoch=st.generation_epoch,
            conversation_id=st.conversation_id,
            reply_id=st.reply_id,
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
                "stream_session_id": session_id,
                "session_id": session_id,
                "server_clock_id": SERVER_CLOCK_ID,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": STREAM_FPS,
            }
        )
    )
    for fr in frames:
        header = {
            "type": "anim",
            "stream_session_id": session_id,
            "session_id": session_id,
            "phase": "snapshot",
            "frame": fr.frame,
            "fps": STREAM_FPS,
            "server_time_ms": fr.server_time_ms,
            "server_clock_id": SERVER_CLOCK_ID,
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
                "stream_session_id": session_id,
                "session_id": session_id,
                "server_clock_id": SERVER_CLOCK_ID,
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
    global epoch_drop_count
    while True:
        item = await anim_queue.get()
        if item is None:
            continue
        if len(item) >= 6:
            session_id, generation_epoch, frame_id_hint, root_pos, bone_quats, morphs = item
        else:
            session_id, root_pos, bone_quats, morphs = item
            generation_epoch = 0
            frame_id_hint = None
        if bone_quats.ndim != 2:
            logger.warning("Anim broadcast invalid bone shape: %s", getattr(bone_quats, "shape", None))
            continue
        now_ms = _server_time_ms()
        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            if generation_epoch != st.generation_epoch:
                epoch_drop_count += 1
                continue
            next_frame = st.latest_frame + 1
            frame_id = int(frame_id_hint) if frame_id_hint is not None else next_frame
            if frame_id <= st.latest_frame:
                # Reject out-of-order/duplicate frames per stream session.
                continue
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
                    "stream_session_id": session_id,
                    "session_id": session_id,
                    "phase": "live",
                    "frame": frame_id,
                    "fps": STREAM_FPS,
                    "server_time_ms": now_ms,
                    "server_clock_id": SERVER_CLOCK_ID,
                    "generation_epoch": generation_epoch,
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
    global epoch_drop_count
    while True:
        item = await audio_in_queue.get()
        if len(item) >= 8:
            (
                audio_np,
                sr,
                chunk_id,
                session_id,
                generation_epoch,
                source,
                conversation_id,
                reply_id,
            ) = item
        else:
            audio_np, sr, chunk_id, session_id = item
            generation_epoch = 0
            source = "legacy"
            conversation_id = None
            reply_id = None
        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            if generation_epoch != st.generation_epoch:
                epoch_drop_count += 1
                logger.info(
                    "Skipping stale inference chunk session=%s epoch=%d active_epoch=%d source=%s",
                    session_id,
                    generation_epoch,
                    st.generation_epoch,
                    source,
                )
                continue
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
        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            if generation_epoch != st.generation_epoch:
                epoch_drop_count += 1
                continue
            start_frame_idx = st.next_frame_index
            st.next_frame_index += frames_count
        for i in range(frames_count):
            frame_index = start_frame_idx + i
            try:
                anim_queue.put_nowait(
                    (
                        session_id,
                        generation_epoch,
                        frame_index,
                        root_pos[i],
                        bone_quats[i],
                        morphs[i],
                    )
                )
            except asyncio.QueueFull:
                try:
                    _ = anim_queue.get_nowait()
                    dropped += 1
                    anim_queue.put_nowait(
                        (
                            session_id,
                            generation_epoch,
                            frame_index,
                            root_pos[i],
                            bone_quats[i],
                            morphs[i],
                        )
                    )
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


class GoogleAdkLiveAudioEngine:
    def __init__(self) -> None:
        self.model = os.environ.get(
            "GOOGLE_ADK_MODEL",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        )
        self.system_instruction = os.environ.get(
            "GOOGLE_ADK_SYSTEM_PROMPT",
            "You are a concise helpful voice assistant.",
        )
        self._runner = None
        self._session_service = None
        self._types = None
        self._run_config_cls = None
        self._streaming_mode = None
        self._live_request_queue_cls = None

    def _ensure_runtime(self) -> None:
        if self._runner is not None:
            return
        try:
            from google.adk.agents import Agent  # type: ignore
            from google.adk.agents.live_request_queue import LiveRequestQueue  # type: ignore
            from google.adk.agents.run_config import RunConfig, StreamingMode  # type: ignore
            from google.adk.runners import Runner  # type: ignore
            from google.adk.sessions import InMemorySessionService  # type: ignore
            from google.genai import types  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing ADK/Gemini dependencies. Install with: "
                "python3 -m pip install google-adk google-genai"
            ) from exc
        self._types = types
        self._run_config_cls = RunConfig
        self._streaming_mode = StreamingMode
        self._live_request_queue_cls = LiveRequestQueue
        self._session_service = InMemorySessionService()
        agent = Agent(
            name="voice_assistant",
            model=self.model,
            instruction=self.system_instruction,
        )
        self._runner = Runner(
            app_name="npz_generator",
            agent=agent,
            session_service=self._session_service,
        )

    async def _ensure_session(self, conversation_id: str) -> None:
        assert self._session_service is not None
        session = await self._session_service.get_session(
            app_name="npz_generator",
            user_id=conversation_id,
            session_id=conversation_id,
        )
        if not session:
            await self._session_service.create_session(
                app_name="npz_generator",
                user_id=conversation_id,
                session_id=conversation_id,
            )

    @staticmethod
    def _parse_pcm_rate_from_mime(mime_type: str) -> int:
        m = re.search(r"rate=(\d+)", mime_type or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 24000
        return 24000

    @staticmethod
    def _decode_base64url(data: str) -> bytes:
        raw = data.strip().replace("-", "+").replace("_", "/")
        while len(raw) % 4:
            raw += "="
        return base64.b64decode(raw)

    @staticmethod
    def _resample_int16_linear(audio_i16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if audio_i16.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
            return audio_i16
        src = audio_i16.astype(np.float32) / 32768.0
        dst_count = max(1, int(round(src.shape[0] * (dst_sr / float(src_sr)))))
        dst_index = np.linspace(0, src.shape[0] - 1, num=dst_count, dtype=np.float32)
        src_index = np.arange(src.shape[0], dtype=np.float32)
        resampled = np.interp(dst_index, src_index, src)
        return np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    def _build_run_config(self):
        assert self._run_config_cls is not None
        assert self._streaming_mode is not None
        assert self._types is not None
        kwargs = {
            "streaming_mode": self._streaming_mode.BIDI,
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": None,
            "output_audio_transcription": None,
        }
        if hasattr(self._types, "SessionResumptionConfig"):
            kwargs["session_resumption"] = self._types.SessionResumptionConfig()
        try:
            return self._run_config_cls(**kwargs)
        except TypeError:
            kwargs.pop("session_resumption", None)
            return self._run_config_cls(**kwargs)

    @staticmethod
    def _event_to_dict(event: Any) -> Dict[str, Any]:
        if isinstance(event, dict):
            return event
        if hasattr(event, "model_dump"):
            try:
                return event.model_dump(exclude_none=True, by_alias=True)
            except Exception:
                return event.model_dump(exclude_none=True)
        return {}

    def _extract_audio_chunks(self, event: Any) -> List[Tuple[bytes, int]]:
        payload = self._event_to_dict(event)
        content = payload.get("content") or {}
        parts = content.get("parts") or []
        out: List[Tuple[bytes, int]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            blob = part.get("inlineData") or part.get("inline_data")
            if not isinstance(blob, dict):
                continue
            mime_type = blob.get("mimeType") or blob.get("mime_type") or ""
            if not str(mime_type).startswith("audio/pcm"):
                continue
            data = blob.get("data")
            if isinstance(data, str):
                chunk = self._decode_base64url(data)
            elif isinstance(data, (bytes, bytearray)):
                chunk = bytes(data)
            else:
                continue
            if not chunk:
                continue
            out.append((chunk, self._parse_pcm_rate_from_mime(str(mime_type))))
        return out

    async def stream_audio_events(
        self,
        *,
        conversation_id: str,
        input_pcm_bytes: bytes,
        input_sr: int,
    ):
        self._ensure_runtime()
        await self._ensure_session(conversation_id)
        assert self._live_request_queue_cls is not None
        assert self._types is not None
        assert self._runner is not None
        live_request_queue = self._live_request_queue_cls()
        run_config = self._build_run_config()

        async def _upstream() -> None:
            chunk_bytes = max(320, int((input_sr * 0.02)) * 2)
            for i in range(0, len(input_pcm_bytes), chunk_bytes):
                if i >= len(input_pcm_bytes):
                    break
                chunk = input_pcm_bytes[i : i + chunk_bytes]
                if not chunk:
                    continue
                audio_blob = self._types.Blob(
                    mime_type=f"audio/pcm;rate={input_sr}",
                    data=chunk,
                )
                live_request_queue.send_realtime(audio_blob)
                await asyncio.sleep(0)
            live_request_queue.close()

        upstream_task = asyncio.create_task(_upstream())
        try:
            async for event in self._runner.run_live(
                user_id=conversation_id,
                session_id=conversation_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                for chunk_bytes, src_sr in self._extract_audio_chunks(event):
                    yield chunk_bytes, src_sr
        finally:
            live_request_queue.close()
            if not upstream_task.done():
                upstream_task.cancel()
            try:
                await upstream_task
            except Exception:
                pass


class ConversationRuntime:
    def __init__(self, websocket: WebSocket, conversation_id: str) -> None:
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.adk = GoogleAdkLiveAudioEngine()
        self._listening = False
        self._user_audio_chunks: List[bytes] = []
        self._user_audio_sr: int = CONVERSATION_AUDIO_SR
        self._reply_task: Optional[asyncio.Task] = None
        self.current_stream_session_id: Optional[str] = None
        self.current_reply_id: Optional[str] = None
        self.current_generation_epoch: int = 0

    async def _send(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("conversation_id", self.conversation_id)
        payload.setdefault("server_boot_id", SERVER_BOOT_ID)
        payload.setdefault("server_clock_id", SERVER_CLOCK_ID)
        payload.setdefault("server_time_ms", _server_time_ms())
        await self.websocket.send_text(json.dumps(payload))

    async def send_hello_ack(self) -> None:
        await self._send(
            {
                "type": "hello_ack",
                "conversation_id": self.conversation_id,
                "protocol_version": CONVERSATION_PROTOCOL_VERSION,
            }
        )

    async def handle_ptt_start(self) -> None:
        self._listening = True
        self._user_audio_chunks = []
        self._user_audio_sr = CONVERSATION_AUDIO_SR
        await self._send({"type": "listening"})

    async def handle_user_audio(self, payload: Dict[str, Any]) -> None:
        if not self._listening:
            return
        b64 = payload.get("pcm_b64") or payload.get("pcmb64") or payload.get("audio_b64")
        if not b64:
            return
        raw = base64.b64decode(b64)
        sr = int(payload.get("sr", CONVERSATION_AUDIO_SR))
        if sr > 0:
            self._user_audio_sr = sr
        self._user_audio_chunks.append(raw)
        max_bytes = int(CONVERSATION_PTT_MAX_SEC * self._user_audio_sr * 2)
        current = sum(len(c) for c in self._user_audio_chunks)
        if current > max_bytes:
            # Keep the tail if client overruns configured max PTT duration.
            joined = b"".join(self._user_audio_chunks)
            self._user_audio_chunks = [joined[-max_bytes:]]

    async def handle_ptt_end(self) -> None:
        if not self._listening:
            return
        self._listening = False
        if self._reply_task and not self._reply_task.done():
            await self.interrupt(reason="new_turn")
        pcm_bytes = b"".join(self._user_audio_chunks)
        self._user_audio_chunks = []
        self._reply_task = asyncio.create_task(self._run_turn(pcm_bytes, self._user_audio_sr))

    async def interrupt(self, reason: str = "interrupt") -> None:
        global epoch_drop_count
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
        if self.current_stream_session_id:
            sid = self.current_stream_session_id
            now_ms = _server_time_ms()
            async with sessions_lock:
                st = sessions.get(sid)
                if st:
                    st.generation_epoch += 1
                    st.deprecated_at_ms = now_ms
                    st.last_activity_ms = now_ms
                    st.producer_connected = False
                    self.current_generation_epoch = st.generation_epoch
            epoch_drop_count += 1
            await _broadcast_audio_control("stop", sid, reason)
        await self._send(
            {
                "type": "interrupted",
                "reply_id": self.current_reply_id,
                "stream_session_id": self.current_stream_session_id,
                "reason": reason,
            }
        )

    async def _run_turn(self, pcm_bytes: bytes, input_sr: int) -> None:
        if not pcm_bytes:
            return
        stream_session_id: Optional[str] = None
        reply_id: Optional[str] = None
        generation_epoch = 0
        emitted_thinking_end = False
        emitted_speaking_start = False
        output_sample_accum = np.zeros((0,), dtype=np.int16)
        out_chunk_samples = max(1, int((CONVERSATION_AUDIO_SR * CONVERSATION_PCM_CHUNK_MS) / 1000))
        chunk_idx = 0
        stale_stream = False
        try:
            await self._send({"type": "assistant_thinking_start"})
            reply_id = str(uuid.uuid4())
            async for raw_chunk, chunk_sr in self.adk.stream_audio_events(
                conversation_id=self.conversation_id,
                input_pcm_bytes=pcm_bytes,
                input_sr=input_sr,
            ):
                audio_i16 = np.frombuffer(raw_chunk, dtype=np.int16)
                if audio_i16.size == 0:
                    continue
                if chunk_sr != CONVERSATION_AUDIO_SR:
                    audio_i16 = self.adk._resample_int16_linear(audio_i16, chunk_sr, CONVERSATION_AUDIO_SR)
                if audio_i16.size == 0:
                    continue
                if output_sample_accum.size == 0:
                    output_sample_accum = audio_i16
                else:
                    output_sample_accum = np.concatenate([output_sample_accum, audio_i16], axis=0)
                while output_sample_accum.size >= out_chunk_samples:
                    chunk_i16 = output_sample_accum[:out_chunk_samples]
                    output_sample_accum = output_sample_accum[out_chunk_samples:]
                    if stream_session_id is None:
                        stream_session_id = await _create_audio_session(
                            conversation_id=self.conversation_id,
                            reply_id=reply_id,
                        )
                        async with sessions_lock:
                            st = sessions.get(stream_session_id)
                            generation_epoch = st.generation_epoch if st else 0
                        self.current_reply_id = reply_id
                        self.current_stream_session_id = stream_session_id
                        self.current_generation_epoch = generation_epoch
                    async with sessions_lock:
                        st = sessions.get(stream_session_id)
                        if st is None or st.generation_epoch != generation_epoch:
                            stale_stream = True
                            break
                    await ingest_audio_chunk(
                        audio_np=(chunk_i16.astype(np.float32) / 32768.0),
                        sr=CONVERSATION_AUDIO_SR,
                        chunk_id=str(chunk_idx),
                        stream_session_id=stream_session_id,
                        generation_epoch=generation_epoch,
                        source="assistant_adk_live",
                        conversation_id=self.conversation_id,
                        reply_id=reply_id,
                    )
                    if not emitted_thinking_end:
                        emitted_thinking_end = True
                        await self._send({"type": "assistant_thinking_end"})
                    if not emitted_speaking_start:
                        emitted_speaking_start = True
                        await self._send(
                            {
                                "type": "assistant_speaking_start",
                                "reply_id": reply_id,
                                "stream_session_id": stream_session_id,
                            }
                        )
                    chunk_idx += 1
                if stale_stream:
                    break
            if (
                stream_session_id is not None
                and output_sample_accum.size > 0
                and not stale_stream
            ):
                await ingest_audio_chunk(
                    audio_np=(output_sample_accum.astype(np.float32) / 32768.0),
                    sr=CONVERSATION_AUDIO_SR,
                    chunk_id=str(chunk_idx),
                    stream_session_id=stream_session_id,
                    generation_epoch=generation_epoch,
                    source="assistant_adk_live",
                    conversation_id=self.conversation_id,
                    reply_id=reply_id,
                )
                if not emitted_thinking_end:
                    emitted_thinking_end = True
                    await self._send({"type": "assistant_thinking_end"})
                if not emitted_speaking_start:
                    emitted_speaking_start = True
                    await self._send(
                        {
                            "type": "assistant_speaking_start",
                            "reply_id": reply_id,
                            "stream_session_id": stream_session_id,
                        }
                    )
            if not emitted_thinking_end:
                await self._send({"type": "assistant_thinking_end"})
            if emitted_speaking_start and stream_session_id is not None:
                await self._send(
                    {
                        "type": "assistant_speaking_end",
                        "reply_id": reply_id,
                        "stream_session_id": stream_session_id,
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Conversation turn failed conversation=%s: %s", self.conversation_id, exc)
            await self._send({"type": "error", "message": str(exc)})
            try:
                await self._send({"type": "assistant_thinking_end"})
            except Exception:
                pass
        finally:
            if stream_session_id:
                async with sessions_lock:
                    st = sessions.get(stream_session_id)
                    if st:
                        st.producer_connected = False
                        st.last_activity_ms = _server_time_ms()

    async def close(self) -> None:
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()


@app.on_event("startup")
async def _startup():
    logger.info(
        "Server startup boot=%s clock=%s protocol=%d fps=%d",
        SERVER_BOOT_ID,
        SERVER_CLOCK_ID,
        PROTOCOL_VERSION,
        STREAM_FPS,
    )
    _log_conversation_dependency_health()
    app.state.broadcast_task = asyncio.create_task(broadcast_anim_loop())
    app.state.infer_task = asyncio.create_task(inference_worker())
    app.state.gc_task = asyncio.create_task(_session_gc_loop())


def _process_chunk(audio_np: np.ndarray, sr: int, chunk_id: Optional[str]) -> Optional[dict]:
    coeffs = generator.process_audio_chunk(audio_np, sr, chunk_id)
    if not coeffs:
        return None
    return coeffs


def _log_conversation_dependency_health() -> None:
    missing: List[str] = []
    try:
        from google.adk.agents import Agent  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        missing.append("google-adk")
    try:
        from google import genai  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        missing.append("google-genai")
    if missing:
        logger.warning(
            "Conversation dependencies missing: %s. Install into active interpreter with: "
            "python3 -m pip install %s",
            missing,
            " ".join(missing),
        )


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
                audio_np = _pcm16_bytes_to_float32(audio_bytes)
            async with sessions_lock:
                st = sessions.get(session_id)
                generation_epoch = st.generation_epoch if st else 0
            await ingest_audio_chunk(
                audio_np=audio_np,
                sr=sr,
                chunk_id=chunk_id,
                stream_session_id=session_id,
                generation_epoch=generation_epoch,
                source="simulator",
                conversation_id=None,
                reply_id=None,
            )
            now_ms = _server_time_ms()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ack",
                        "chunk_id": chunk_id,
                        "stream_session_id": session_id,
                        "session_id": session_id,
                        "generation_epoch": generation_epoch,
                        "server_boot_id": SERVER_BOOT_ID,
                        "server_clock_id": SERVER_CLOCK_ID,
                        "server_time_ms": now_ms,
                    }
                )
            )
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


@app.websocket("/ws/conversation")
async def ws_conversation(websocket: WebSocket):
    await websocket.accept()
    conversation_id = str(uuid.uuid4())
    runtime = ConversationRuntime(websocket, conversation_id)
    logger.info("Conversation WS connected: %s conversation=%s", websocket.client, conversation_id)
    await runtime.send_hello_ack()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "hello":
                await runtime.send_hello_ack()
            elif mtype == "ptt_start":
                await runtime.handle_ptt_start()
            elif mtype == "user_audio":
                await runtime.handle_user_audio(msg)
            elif mtype == "ptt_end":
                await runtime.handle_ptt_end()
            elif mtype == "interrupt":
                await runtime.interrupt(reason="interrupt")
            elif mtype == "ping":
                await runtime._send({"type": "pong", "ts": msg.get("ts")})
    except WebSocketDisconnect:
        logger.info("Conversation WS disconnected: %s conversation=%s", websocket.client, conversation_id)
    finally:
        await runtime.close()


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
            known_clock_id = subscribe_msg.get("known_server_clock_id")
            known_session_id = (
                subscribe_msg.get("known_stream_session_id")
                or subscribe_msg.get("known_session_id")
            )
            async with sessions_lock:
                active_sid = active_session_id
            if known_boot_id and known_boot_id != SERVER_BOOT_ID:
                mode = "reset_required"
            elif known_clock_id and known_clock_id != SERVER_CLOCK_ID:
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
                        "stream_session_id": active_sid,
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
                        "stream_session_id": active_sid,
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
                req_session = _resolve_stream_session_id(msg)
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

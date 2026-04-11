"""FastAPI application entry point for the NPZ Generator streaming server.

This module is intentionally thin. It owns:
  - App-level constants (boot ID, protocol versions, clock)
  - HTTP middleware (CORS / COEP headers, cache-control)
  - WebSocket route handlers (/ws/audio, /ws/audio_out, /ws/conversation, /ws/anim)
  - Startup lifecycle hook

All domain logic lives in:
  - session.py          — session state and GC
  - audio_pipeline.py   — inference, anim broadcast, Gemini engine, audio ingest
  - conversation.py     — ConversationRuntime / PTT / turn management
"""
import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
import sys
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Suppress known environment/runtime warnings not actionable inside app logic.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*", category=Warning)
warnings.filterwarnings("ignore", message=r".*enable_nested_tensor is True.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Python version 3\.9 past its end of life.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*non-supported Python version.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings:.*response_modalities.*", category=UserWarning)

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure repo root and server dir are on sys.path for local imports.
SERVER_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from live_streaming_pipeline import LiveMotionGenerator
from npz_logging import setup_logging
from retargeter import SmplxRetargeter
from . import session as session_state
from .session import (
    STREAM_FPS,
    SNAPSHOT_FRAMES,
    SessionFrame,
    sessions,
    sessions_lock,
    anim_clients,
    anim_client_protocol,
    anim_client_session,
    audio_clients,
    create_audio_session,
    session_snapshot,
    session_gc_loop,
    _audio_frame_from_cursor,
)
from .audio_pipeline import (
    anim_queue,
    audio_in_queue,
    broadcast_anim_loop,
    inference_worker,
    ingest_audio_chunk,
    pcm16_bytes_to_float32,
    GeminiLiveAudioEngine,
)
from .conversation import ConversationRuntime, CONVERSATION_AUDIO_SR

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App-level constants
# ---------------------------------------------------------------------------

app = FastAPI()

PROTOCOL_VERSION: int = 2
CONVERSATION_PROTOCOL_VERSION: int = 1
SERVER_BOOT_ID: str = str(uuid.uuid4())
SERVER_CLOCK_ID: str = "monotonic-ms-v1"
_BOOT_MONO_NS: int = time.monotonic_ns()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _server_time_ms() -> int:
    return int((time.monotonic_ns() - _BOOT_MONO_NS) / 1_000_000)


# ---------------------------------------------------------------------------
# Shared ML models (initialised once at import time)
# ---------------------------------------------------------------------------

generator = LiveMotionGenerator(overlap_sec=0.25)
retargeter = SmplxRetargeter(os.path.join(SERVER_DIR, "retarget_map.json"))

# ---------------------------------------------------------------------------
# HTTP middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    elif re.search(r"[.-][A-Za-z0-9_-]{8,}\.", path):
        response.headers["Cache-Control"] = "public,max-age=31536000,immutable"
    return response

# ---------------------------------------------------------------------------
# Snapshot helper (used by ws_anim)
# ---------------------------------------------------------------------------

def _payload_bytes(root_pos: np.ndarray, bone_quats: np.ndarray, morphs: np.ndarray) -> bytes:
    return (
        np.concatenate([root_pos, bone_quats.reshape(-1), morphs.astype(np.float32)], axis=0)
        .astype(np.float32, copy=False)
        .tobytes()
    )


async def _send_snapshot(ws: WebSocket, session_id: str) -> None:
    st, frames = await session_snapshot(session_id, _server_time_ms)
    if st is None:
        return
    start_frame = frames[0].frame if frames else -1
    end_frame = frames[-1].frame if frames else -1
    await ws.send_text(json.dumps({
        "type": "anim_snapshot_start",
        "stream_session_id": session_id,
        "session_id": session_id,
        "server_clock_id": SERVER_CLOCK_ID,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": STREAM_FPS,
    }))
    for fr in frames:
        await ws.send_text(json.dumps({
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
        }))
        await ws.send_bytes(_payload_bytes(fr.root_pos, fr.bone_quats, fr.morphs))
    live_head_server_time_ms = frames[-1].server_time_ms if frames else _server_time_ms()
    audio_live_edge_frame = _audio_frame_from_cursor(st.latest_audio_cursor, st.latest_audio_sr)
    await ws.send_text(json.dumps({
        "type": "anim_snapshot_end",
        "stream_session_id": session_id,
        "session_id": session_id,
        "server_clock_id": SERVER_CLOCK_ID,
        "snapshot_end_frame": end_frame,
        "snapshot_end_server_time_ms": live_head_server_time_ms,
        "live_head_frame": st.latest_frame,
        "live_head_server_time_ms": live_head_server_time_ms,
        "audio_live_edge_frame": audio_live_edge_frame,
        "audio_live_edge_server_time_ms": st.latest_audio_server_time_ms,
        "fps": STREAM_FPS,
    }))


def _resolve_stream_session_id(payload: Dict[str, Any]) -> Optional[str]:
    return (
        payload.get("stream_session_id")
        or payload.get("streamsessionid")
        or payload.get("session_id")
    )

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _log_conversation_dependency_health() -> None:
    missing: List[str] = []
    if importlib.util.find_spec("google.genai") is None:
        missing.append("google-genai")
    if missing:
        logger.warning(
            "Conversation dependencies missing: %s. Install with: python3 -m pip install %s",
            missing, " ".join(missing),
        )


@app.on_event("startup")
async def _startup():
    logger.info(
        "Server startup boot=%s clock=%s protocol=%d fps=%d",
        SERVER_BOOT_ID, SERVER_CLOCK_ID, PROTOCOL_VERSION, STREAM_FPS,
    )
    _log_conversation_dependency_health()
    app.state.broadcast_task = asyncio.create_task(
        broadcast_anim_loop(SERVER_CLOCK_ID, _server_time_ms)
    )
    app.state.infer_task = asyncio.create_task(
        inference_worker(generator, retargeter)
    )
    app.state.gc_task = asyncio.create_task(
        session_gc_loop(_server_time_ms)
    )

# ---------------------------------------------------------------------------
# Helper: create_audio_session bound to server constants
# ---------------------------------------------------------------------------

async def _create_audio_session(
    conversation_id: Optional[str] = None,
    reply_id: Optional[str] = None,
) -> str:
    return await create_audio_session(
        server_boot_id=SERVER_BOOT_ID,
        anim_queue=anim_queue,
        audio_in_queue=audio_in_queue,
        server_time_fn=_server_time_ms,
        conversation_id=conversation_id,
        reply_id=reply_id,
    )

# ---------------------------------------------------------------------------
# WebSocket routes
# ---------------------------------------------------------------------------

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
            audio_bytes = base64.b64decode(audio_b64)
            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.float32)
                if dtype == "float32"
                else pcm16_bytes_to_float32(audio_bytes)
            )
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
                server_boot_id=SERVER_BOOT_ID,
                server_clock_id=SERVER_CLOCK_ID,
                server_time_fn=_server_time_ms,
            )
            await websocket.send_text(json.dumps({
                "type": "ack",
                "chunk_id": chunk_id,
                "stream_session_id": session_id,
                "session_id": session_id,
                "generation_epoch": generation_epoch,
                "server_boot_id": SERVER_BOOT_ID,
                "server_clock_id": SERVER_CLOCK_ID,
                "server_time_ms": _server_time_ms(),
            }))
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
    runtime = ConversationRuntime(
        websocket=websocket,
        conversation_id=conversation_id,
        server_boot_id=SERVER_BOOT_ID,
        server_clock_id=SERVER_CLOCK_ID,
        server_time_fn=_server_time_ms,
        conversation_protocol_version=CONVERSATION_PROTOCOL_VERSION,
        create_audio_session_fn=_create_audio_session,
    )
    logger.info("Conversation WS connected: %s conversation=%s", websocket.client, conversation_id)
    # await runtime.send_hello_ack() # Removed redundant call: loop handles "hello" message.
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
        logger.info(
            "Conversation WS disconnected: %s conversation=%s",
            websocket.client, conversation_id,
        )
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

        if (
            subscribe_msg.get("type") == "anim_subscribe"
            and int(subscribe_msg.get("protocol_version", 2)) >= 2
        ):
            protocol = int(subscribe_msg.get("protocol_version", 2))
            known_boot_id = subscribe_msg.get("known_boot_id")
            known_clock_id = subscribe_msg.get("known_server_clock_id")
            known_session_id = (
                subscribe_msg.get("known_stream_session_id")
                or subscribe_msg.get("known_session_id")
            )
            async with sessions_lock:
                active_sid = session_state.active_session_id

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
            await websocket.send_text(json.dumps({
                "type": "anim_subscribe_ack",
                "protocol_version": PROTOCOL_VERSION,
                "mode": mode,
                "server_boot_id": SERVER_BOOT_ID,
                "server_clock_id": SERVER_CLOCK_ID,
                "stream_session_id": active_sid,
                "session_id": active_sid,
                "stream_fps": STREAM_FPS,
                "server_time_ms": _server_time_ms(),
            }))
            if mode == "reset_required":
                await websocket.close()
                return
            await websocket.send_text(json.dumps({
                **retargeter.anim_init_header(STREAM_FPS),
                "protocol_version": PROTOCOL_VERSION,
                "server_boot_id": SERVER_BOOT_ID,
                "server_clock_id": SERVER_CLOCK_ID,
                "stream_session_id": active_sid,
                "session_id": active_sid,
            }))
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
                    active_sid = session_state.active_session_id
                if active_sid and req_session == active_sid:
                    logger.info(
                        "Resync request accepted session=%s reason=%s",
                        req_session, msg.get("reason"),
                    )
                    await _send_snapshot(websocket, active_sid)
                else:
                    logger.info(
                        "Resync request ignored req_session=%s active=%s reason=%s",
                        req_session, active_sid, msg.get("reason"),
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
# app.mount("/", StaticFiles(directory="web", html=True), name="web")

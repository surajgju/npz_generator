
"""Session state management for the NPZ Generator streaming server.

Owns SessionState, SessionFrame, the sessions dict, and all CRUD helpers.
Imported by audio_pipeline.py, conversation.py, and app.py.
"""
import asyncio
import json
import logging
import math
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import sys
import numpy as np
from fastapi import WebSocket

# ---------------------------------------------------------------------------
# Config (read once at import time so every module shares the same values)
# ---------------------------------------------------------------------------

STREAM_FPS: int = int(os.environ.get("STREAM_FPS", "20"))
SLOW_MOTION_FACTOR: float = float(os.environ.get("SLOW_MOTION_FACTOR", "1.0"))
BASE_FPS: int = 30
SNAPSHOT_SECONDS: float = float(os.environ.get("SNAPSHOT_SECONDS", "3.0"))
SNAPSHOT_FRAMES: int = int(math.ceil(SNAPSHOT_SECONDS * STREAM_FPS))
MAX_SESSIONS: int = int(os.environ.get("MAX_SESSIONS", "8"))
SESSION_IDLE_TTL_MS: int = int(os.environ.get("SESSION_IDLE_TTL_MS", "45000"))
DEPRECATED_TTL_MS: int = int(os.environ.get("DEPRECATED_TTL_MS", "15000"))
SESSION_GC_INTERVAL_MS: int = int(os.environ.get("SESSION_GC_INTERVAL_MS", "5000"))

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    frame_ring: Deque[SessionFrame] = field(
        default_factory=lambda: deque(maxlen=SNAPSHOT_FRAMES)
    )
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


# ---------------------------------------------------------------------------
# Shared mutable state (single source of truth)
# ---------------------------------------------------------------------------

sessions_lock: asyncio.Lock = asyncio.Lock()
sessions: Dict[str, SessionState] = {}
active_session_id: Optional[str] = None

# Anim client tracking lives here because session GC needs it.
anim_clients: "set[WebSocket]" = set()
anim_client_protocol: Dict[WebSocket, int] = {}
anim_client_session: Dict[WebSocket, Optional[str]] = {}
audio_clients: "set[WebSocket]" = set()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audio_frame_from_cursor(samples: int, sr: int) -> int:
    if sr <= 0:
        return 0
    return int(round((samples / float(sr)) * STREAM_FPS))


def _drain_queue_nowait(q: asyncio.Queue) -> int:
    drained = 0
    while True:
        try:
            q.get_nowait()
            drained += 1
        except Exception:
            break
    return drained


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def create_audio_session(
    server_boot_id: str,
    anim_queue: asyncio.Queue,
    audio_in_queue: asyncio.Queue,
    server_time_fn,
    conversation_id: Optional[str] = None,
    reply_id: Optional[str] = None,
) -> str:
    global active_session_id
    now_ms = server_time_fn()
    sid = str(uuid.uuid4())
    protocol2_ws: List[WebSocket] = []
    async with sessions_lock:
        if active_session_id and active_session_id in sessions:
            prev = sessions[active_session_id]
            prev.deprecated_at_ms = now_ms
            prev.producer_connected = False
            prev.anim_subscriber_count = 0
        for ws in list(anim_clients):
            if anim_client_protocol.get(ws, 1) >= 2:
                protocol2_ws.append(ws)
                anim_client_session[ws] = sid
        sessions[sid] = SessionState(
            session_id=sid,
            created_ms=now_ms,
            last_activity_ms=now_ms,
            latest_audio_server_time_ms=now_ms,
            next_frame_index=0,
            producer_connected=True,
            anim_subscriber_count=len(protocol2_ws),
            generation_epoch=0,
            conversation_id=conversation_id,
            reply_id=reply_id,
        )
        active_session_id = sid
    drained_anim = _drain_queue_nowait(anim_queue)
    drained_audio = _drain_queue_nowait(audio_in_queue)
    logger.info(
        "Created session %s (boot=%s). Drained anim=%d audio=%d",
        sid, server_boot_id, drained_anim, drained_audio,
    )
    if protocol2_ws:
        switch_msg = json.dumps(
            {
                "type": "anim_session_switch",
                "stream_session_id": sid,
                "session_id": sid,
                "server_boot_id": server_boot_id,
                "server_time_ms": server_time_fn(),
            }
        )
        dead: List[WebSocket] = []
        for ws in protocol2_ws:
            try:
                await ws.send_text(switch_msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with sessions_lock:
                for ws in dead:
                    anim_clients.discard(ws)
                    anim_client_protocol.pop(ws, None)
                    old_sid = anim_client_session.pop(ws, None)
                    if old_sid and old_sid in sessions:
                        sessions[old_sid].anim_subscriber_count = max(
                            0, sessions[old_sid].anim_subscriber_count - 1
                        )
    return sid


async def session_snapshot(
    session_id: str, server_time_fn
) -> Tuple[Optional[SessionState], List[SessionFrame]]:
    async with sessions_lock:
        st = sessions.get(session_id)
        if st is None:
            return None, []
        st.last_activity_ms = server_time_fn()
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


async def session_gc_loop(server_time_fn) -> None:
    """Periodically evict idle / deprecated sessions."""
    global active_session_id
    import logging
    logger = logging.getLogger(__name__)
    while True:
        await asyncio.sleep(SESSION_GC_INTERVAL_MS / 1000.0)
        now_ms = server_time_fn()
        async with sessions_lock:
            removable: List[str] = []
            for sid, st in sessions.items():
                idle_ms = now_ms - st.last_activity_ms
                deprecated_ms = (
                    (now_ms - st.deprecated_at_ms)
                    if st.deprecated_at_ms is not None
                    else None
                )
                if (
                    idle_ms > SESSION_IDLE_TTL_MS
                    and not st.producer_connected
                    and st.anim_subscriber_count <= 0
                ):
                    removable.append(sid)
                    continue
                if (
                    st.deprecated_at_ms is not None
                    and deprecated_ms is not None
                    and deprecated_ms > DEPRECATED_TTL_MS
                ):
                    removable.append(sid)
            if len(sessions) - len(removable) > MAX_SESSIONS:
                candidates = sorted(
                    (
                        st
                        for sid, st in sessions.items()
                        if sid not in removable
                        and not st.producer_connected
                        and st.anim_subscriber_count <= 0
                    ),
                    key=lambda st: st.last_activity_ms,
                )
                overflow = (len(sessions) - len(removable)) - MAX_SESSIONS
                removable.extend(st.session_id for st in candidates[:overflow])
            for sid in removable:
                if sid in sessions:
                    del sessions[sid]
                    if sid == active_session_id:
                        active_session_id = None
            if removable:
                logger.info(
                    "Session GC evicted=%s active=%s", removable, active_session_id
                )

            total_mem = 0
            for st in sessions.values():
                total_mem += estimate_session_memory(st)
            # Warn if > 200 MB (adjust threshold)
            if total_mem > 200 * 1024 * 1024:
                logger.warning(
                    "High snapshot memory: %.2f MB across %d sessions",
                    total_mem / (1024*1024), len(sessions)
                )   


def estimate_frame_memory(frame: SessionFrame) -> int:
    """Approximate memory usage of a single SessionFrame in bytes."""
    total = sys.getsizeof(frame)  # dataclass overhead
    total += frame.root_pos.nbytes
    total += frame.bone_quats.nbytes
    total += frame.morphs.nbytes
    return total

def estimate_session_memory(st: SessionState) -> int:
    total = sys.getsizeof(st)
    total += sys.getsizeof(st.frame_ring)
    for frame in st.frame_ring:
        total += estimate_frame_memory(frame)
    # Add other numpy arrays if any
    return total

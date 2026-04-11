"""Audio pipeline for the NPZ Generator streaming server.

Owns:
  - GeminiLiveAudioEngine  — native Gemini Live API client
  - ingest_audio_chunk     — validates and fans audio out to clients + worker queue
  - broadcast_audio_control— control messages to audio_out clients
  - broadcast_anim_loop    — dequeue retargeted frames and stream to anim clients
  - inference_worker       — runs SMPL-X model inference, pushes frames to anim_queue

Shared state queues live here and are imported by app.py for startup.
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket

from . import session as session_state
from .session import (
    STREAM_FPS,
    SLOW_MOTION_FACTOR,
    BASE_FPS,
    SessionFrame,
    SessionState,
    anim_clients,
    anim_client_protocol,
    anim_client_session,
    audio_clients,
    sessions,
    sessions_lock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared queues and concurrency primitives
# ---------------------------------------------------------------------------

anim_queue: asyncio.Queue = asyncio.Queue(maxsize=int(STREAM_FPS * 10))
audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
inference_lock: asyncio.Lock = asyncio.Lock()
epoch_drop_count: int = 0

# ---------------------------------------------------------------------------
# Gemini Live Audio Engine
# ---------------------------------------------------------------------------

try:
    from google.genai import types as _genai_types  # type: ignore
except ImportError:
    _genai_types = None  # type: ignore


class GeminiLiveAudioEngine:
    """Streams PCM audio to Gemini Live API and yields PCM audio response chunks.

    Uses the native google-genai AsyncSession (send_realtime_input + receive)
    directly, bypassing the ADK runner which sent the deprecated media_chunks
    format. Supports Gemini 2.5 and 3.1 Live models.
    """

    DEFAULT_MODEL_CANDIDATES: List[str] = [
        "gemini-3.1-flash-live-preview",
        "gemini-2.0-flash-live-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
    ]

    def __init__(self) -> None:
        self.system_instruction: str = os.environ.get(
            "GOOGLE_ADK_SYSTEM_PROMPT",
            "You are a concise helpful voice assistant.",
        )
        self.model_candidates: List[str] = self._build_model_candidates()
        self.model: str = self.model_candidates[0]
        logger.info(
            "Gemini Live engine initialised model_candidates=%s",
            self.model_candidates,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _normalize_model(name: str) -> str:
        value = (name or "").strip()
        return value[len("models/"):] if value.startswith("models/") else value

    def _build_model_candidates(self) -> List[str]:
        env_model = self._normalize_model(os.environ.get("GOOGLE_ADK_MODEL", ""))
        env_extra = [
            self._normalize_model(m)
            for m in os.environ.get("GOOGLE_ADK_MODEL_CANDIDATES", "").split(",")
            if m.strip()
        ]
        seen: List[str] = []
        for m in [env_model, *env_extra, *self.DEFAULT_MODEL_CANDIDATES]:
            if m and m not in seen:
                seen.append(m)
        return seen or list(self.DEFAULT_MODEL_CANDIDATES)

    @staticmethod
    def _make_genai_client():
        try:
            from google import genai  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing google-genai. Install with: pip install google-genai"
            ) from exc
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        return genai.Client(api_key=api_key)

    @staticmethod
    def _build_live_config(system_instruction: str):
        types = _genai_types
        if types is None:
            raise RuntimeError("google-genai is not installed. Run: pip install google-genai")

        # Build speech / voice config only when the types are available.
        voice_name = os.environ.get("GEMINI_LIVE_VOICE", "Zephyr")
        try:
            speech_cfg = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            )
        except Exception:
            speech_cfg = None

        kwargs = dict(response_modalities=["AUDIO"])
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if speech_cfg is not None:
            kwargs["speech_config"] = speech_cfg

        return types.LiveConnectConfig(**kwargs)

    @staticmethod
    def _parse_pcm_rate(mime_type: str) -> int:
        m = re.search(r"rate=(\d+)", mime_type or "")
        return int(m.group(1)) if m else 24000

    @staticmethod
    def _resample_int16_linear(
        audio_i16: np.ndarray, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        if audio_i16.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
            return audio_i16
        src = audio_i16.astype(np.float32) / 32768.0
        dst_count = max(1, int(round(src.shape[0] * (dst_sr / float(src_sr)))))
        dst_index = np.linspace(0, src.shape[0] - 1, num=dst_count, dtype=np.float32)
        src_index = np.arange(src.shape[0], dtype=np.float32)
        resampled = np.interp(dst_index, src_index, src)
        return np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

    @staticmethod
    def _is_model_unsupported_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "not found" in msg
            or "not supported for bidigeneratecontent" in msg
            or ("policy violation" in msg and "model" in msg)
        )

    @staticmethod
    def _is_api_key_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "api key" in msg or "unauthenticated" in msg or "permission_denied" in msg

    @staticmethod
    def _truthy_attr(obj: Any, names: List[str]) -> bool:
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, bool):
                if value:
                    return True
                continue
            if value in (1, "1", "true", "True", "TRUE"):
                return True
        return False

    @classmethod
    def _is_turn_complete_response(cls, response: Any) -> bool:
        """Detect end-of-turn markers across SDK field naming variants."""
        names = [
            "turn_complete",
            "turnComplete",
            "generation_complete",
            "generationComplete",
            "interrupted",
        ]
        if cls._truthy_attr(response, names):
            return True
        server_content = getattr(response, "server_content", None)
        if server_content is not None and cls._truthy_attr(server_content, names):
            return True
        return False

    # Keep the old name as an alias so ConversationRuntime._run_turn still works
    _is_invalid_api_key_error = _is_api_key_error
    _is_model_not_supported_error = _is_model_unsupported_error

    @classmethod
    def user_message_for_exception(cls, exc: Exception) -> str:
        msg = str(exc)
        lower = msg.lower()
        if "reported as leaked" in lower:
            return (
                "Gemini API key is disabled (reported as leaked). "
                "Create a new key in Google AI Studio and set GEMINI_API_KEY."
            )
        if cls._is_api_key_error(exc):
            return (
                "Gemini authentication failed. Verify GEMINI_API_KEY "
                "and ensure Live API access is enabled."
            )
        if cls._is_model_unsupported_error(exc):
            return (
                "Configured Gemini Live model is not available. "
                "Set GOOGLE_ADK_MODEL to a supported live audio model."
            )
        return msg

    # ------------------------------------------------------------------ streaming

    async def stream_audio_events(
        self,
        *,
        conversation_id: str,
        input_pcm_bytes: bytes,
        input_sr: int,
    ):
        """Connect to Gemini Live, stream audio via send_realtime_input, yield (pcm_bytes, sample_rate) tuples.

        Uses send_realtime_input (matches the useLiveAPI reference pattern) which is
        the correct API for real-time audio streaming as of google-genai >=0.8.
        Falls back to the legacy session.send() path when send_realtime_input is
        unavailable on the session object.
        """
        client = self._make_genai_client()
        live_config = self._build_live_config(self.system_instruction)
        # ~20 ms chunks at the input sample rate (2 bytes per int16 sample)
        chunk_size = max(320, int(input_sr * 0.02) * 2)
        mime_type = f"audio/pcm;rate={input_sr}"
        idle_timeout_sec = float(os.environ.get("GEMINI_LIVE_RECEIVE_IDLE_TIMEOUT_SEC", "8"))
        max_turn_sec = float(os.environ.get("GEMINI_LIVE_MAX_TURN_SEC", "45"))
        last_exc: Optional[Exception] = None

        for idx, model_name in enumerate(self.model_candidates):
            self.model = model_name
            try:
                async with client.aio.live.connect(
                    model=model_name, config=live_config
                ) as session:
                    logger.info(
                        "Gemini Live connected model=%s conversation=%s",
                        model_name,
                        conversation_id,
                    )

                    # ── Send audio ───────────────────────────────────────────
                    # Prefer send_realtime_input (matches useLiveAPI reference):
                    #   session.send_realtime_input(audio={"data": b64, "mime_type": mime})
                    # Fall back to session.send() for older SDK versions.
                    use_realtime_input = callable(getattr(session, "send_realtime_input", None))

                    if use_realtime_input:
                        import base64 as _base64
                        for offset in range(0, len(input_pcm_bytes), chunk_size):
                            chunk = input_pcm_bytes[offset: offset + chunk_size]
                            if not chunk:
                                continue
                            b64_chunk = _base64.b64encode(chunk).decode("ascii")
                            await session.send_realtime_input(
                                audio={"data": b64_chunk, "mime_type": mime_type}
                            )
                        # Signal end-of-turn with empty bytes + end_of_turn=True.
                        # Passing a dict to session.send() raises ValueError in
                        # all current google-genai SDK versions.
                        await session.send(input=b"", end_of_turn=True)
                        logger.debug(
                            "Gemini Live audio sent via send_realtime_input model=%s bytes=%d",
                            model_name, len(input_pcm_bytes),
                        )
                    else:
                        # Legacy path: send raw PCM bytes
                        for offset in range(0, len(input_pcm_bytes), chunk_size):
                            chunk = input_pcm_bytes[offset: offset + chunk_size]
                            if not chunk:
                                continue
                            await session.send(input=chunk, end_of_turn=False)
                        await session.send(input=b"", end_of_turn=True)
                        logger.debug(
                            "Gemini Live audio sent via legacy session.send model=%s bytes=%d",
                            model_name, len(input_pcm_bytes),
                        )

                    # ── Receive loop ─────────────────────────────────────────
                    recv_iter = session.receive().__aiter__()
                    turn_start = asyncio.get_running_loop().time()
                    interrupted = False

                    while True:
                        elapsed = asyncio.get_running_loop().time() - turn_start
                        if elapsed >= max_turn_sec:
                            logger.warning(
                                "Gemini Live turn timeout model=%s conversation=%s elapsed=%.2fs",
                                model_name,
                                conversation_id,
                                elapsed,
                            )
                            break
                        timeout = max(0.1, min(idle_timeout_sec, max_turn_sec - elapsed))
                        try:
                            response = await asyncio.wait_for(recv_iter.__anext__(), timeout=timeout)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Gemini Live receive idle timeout model=%s conversation=%s timeout=%.2fs",
                                model_name,
                                conversation_id,
                                timeout,
                            )
                            break

                        # ── Handle interrupted flag (mirrors useLiveAPI reference) ──
                        server_content = getattr(response, "server_content", None)
                        if server_content is not None:
                            if getattr(server_content, "interrupted", False):
                                logger.info(
                                    "Gemini Live interrupted model=%s conversation=%s",
                                    model_name, conversation_id,
                                )
                                interrupted = True
                                break

                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn is not None:
                                for part in getattr(model_turn, "parts", None) or []:
                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data is None:
                                        continue
                                    mime = getattr(inline_data, "mime_type", "") or ""
                                    if not mime.startswith("audio/pcm"):
                                        continue
                                    data = getattr(inline_data, "data", None)
                                    if isinstance(data, (bytes, bytearray)) and data:
                                        yield bytes(data), self._parse_pcm_rate(mime)

                        if self._is_turn_complete_response(response):
                            logger.debug(
                                "Gemini Live turn complete model=%s conversation=%s",
                                model_name,
                                conversation_id,
                            )
                            break

                    if interrupted:
                        break
                return  # success

            except Exception as exc:
                last_exc = exc
                has_next = idx + 1 < len(self.model_candidates)
                if has_next and self._is_model_unsupported_error(exc):
                    logger.warning(
                        "Gemini Live model %s failed (%s); retrying with %s",
                        model_name,
                        str(exc).splitlines()[0],
                        self.model_candidates[idx + 1],
                    )
                    continue
                raise

        if last_exc is not None:
            raise last_exc


# ---------------------------------------------------------------------------
# Audio broadcast helpers
# ---------------------------------------------------------------------------

def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


async def broadcast_audio_control(
    action: str,
    stream_session_id: str,
    reason: str,
    server_boot_id: str,
    server_clock_id: str,
    server_time_fn,
) -> None:
    if not audio_clients:
        return
    header = {
        "type": "audio_control",
        "action": action,
        "stream_session_id": stream_session_id,
        "session_id": stream_session_id,
        "reason": reason,
        "server_time_ms": server_time_fn(),
        "server_boot_id": server_boot_id,
        "server_clock_id": server_clock_id,
    }
    dead: List[WebSocket] = []
    for ws in list(audio_clients):
        try:
            await ws.send_text(json.dumps(header))
        except Exception:
            dead.append(ws)
    for ws in dead:
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
    server_boot_id: str,
    server_clock_id: str,
    server_time_fn,
) -> None:
    global epoch_drop_count
    sample_count = int(audio_np.shape[0])
    now_ms = server_time_fn()
    async with sessions_lock:
        st = sessions.get(stream_session_id)
        if st is None:
            logger.warning(
                "Dropping audio chunk for missing stream session=%s", stream_session_id
            )
            return
        if generation_epoch != st.generation_epoch:
            epoch_drop_count += 1
            logger.info(
                "Dropping stale audio chunk session=%s epoch=%d active_epoch=%d",
                stream_session_id, generation_epoch, st.generation_epoch,
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
            "server_boot_id": server_boot_id,
            "server_clock_id": server_clock_id,
        }
        dead: List[WebSocket] = []
        for ws in list(audio_clients):
            try:
                await ws.send_text(json.dumps(header))
                await ws.send_bytes(audio_bytes)
            except Exception:
                dead.append(ws)
        for ws in dead:
            audio_clients.discard(ws)
        logger.info(
            "Audio broadcast stream=%s chunk_id=%s samples=%d clients=%d source=%s",
            stream_session_id, chunk_id, sample_count, len(audio_clients), source,
        )

    try:
        audio_in_queue.put_nowait((
            audio_np, sr, chunk_id, stream_session_id,
            generation_epoch, source, conversation_id, reply_id,
        ))
    except asyncio.QueueFull:
        try:
            old = audio_in_queue.get_nowait()
            audio_in_queue.put_nowait((
                audio_np, sr, chunk_id, stream_session_id,
                generation_epoch, source, conversation_id, reply_id,
            ))
            logger.warning(
                "Audio queue full; dropped oldest chunk %s",
                old[2] if len(old) > 2 else "?",
            )
        except Exception:
            logger.error("Audio queue overflow; dropping chunk %s", chunk_id)


# ---------------------------------------------------------------------------
# broadcast_anim_loop — dequeue and fan out retargeted frames
# ---------------------------------------------------------------------------

def _payload_bytes(
    root_pos: np.ndarray, bone_quats: np.ndarray, morphs: np.ndarray
) -> bytes:
    return (
        np.concatenate(
            [root_pos, bone_quats.reshape(-1), morphs.astype(np.float32)], axis=0
        )
        .astype(np.float32, copy=False)
        .tobytes()
    )


async def broadcast_anim_loop(server_clock_id: str, server_time_fn) -> None:
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
            logger.warning(
                "Anim broadcast invalid bone shape: %s",
                getattr(bone_quats, "shape", None),
            )
            continue
        now_ms = server_time_fn()
        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            if generation_epoch != st.generation_epoch:
                epoch_drop_count += 1
                continue
            frame_id = (
                int(frame_id_hint) if frame_id_hint is not None else st.latest_frame + 1
            )
            if frame_id <= st.latest_frame:
                continue  # reject out-of-order / duplicate
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
            active_sid = session_state.active_session_id

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
                    "server_clock_id": server_clock_id,
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
                s = sessions.get(sid)
                if s:
                    s.anim_subscriber_count = max(0, s.anim_subscriber_count - 1)
        await asyncio.sleep(1 / STREAM_FPS)


# ---------------------------------------------------------------------------
# inference_worker — runs SMPL-X model, pushes frames to anim_queue
# ---------------------------------------------------------------------------

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


async def inference_worker(generator, retargeter) -> None:
    """Dequeue audio chunks, run inference, push retargeted frames to anim_queue."""
    global epoch_drop_count
    while True:
        item = await audio_in_queue.get()
        audio_np, sr, chunk_id, session_id, generation_epoch, source, conversation_id, reply_id = item

        async with sessions_lock:
            st = sessions.get(session_id)
            if st is None:
                continue
            if generation_epoch != st.generation_epoch:
                epoch_drop_count += 1
                logger.info(
                    "Skipping stale inference chunk session=%s epoch=%d active_epoch=%d source=%s",
                    session_id, generation_epoch, st.generation_epoch, source,
                )
                continue

        try:
            async with inference_lock:
                coeffs = await asyncio.to_thread(
                    lambda: generator.process_audio_chunk(audio_np, sr, chunk_id)
                )
        except Exception as exc:
            logger.exception("Inference failed for chunk %s: %s", chunk_id, exc)
            continue

        if not coeffs:
            logger.warning(
                "No coeffs generated for chunk %s (audio_len=%d)",
                chunk_id, audio_np.shape[0],
            )
            continue

        if chunk_id is not None and str(chunk_id) == "0":
            retargeter.reset_root_offset()

        poses = coeffs["poses"]
        expressions = coeffs["expressions"]
        trans = coeffs["trans"]
        frames_count = int(poses.shape[0])

        if (STREAM_FPS != BASE_FPS or SLOW_MOTION_FACTOR != 1.0) and frames_count > 1:
            target_count = max(
                1,
                int(round(frames_count * (STREAM_FPS / BASE_FPS) * SLOW_MOTION_FACTOR)),
            )
            poses = _resample_frames_linear(poses, target_count)
            expressions = _resample_frames_linear(expressions, target_count)
            trans = _resample_frames_linear(trans, target_count)
            frames_count = int(poses.shape[0])

        root_pos, bone_quats, morphs = retargeter.retarget(poses, expressions, trans)

        if frames_count > 0:
            _log_inference_stats(
                retargeter, chunk_id, session_id, frames_count,
                poses, expressions, morphs, root_pos,
            )

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
                anim_queue.put_nowait((
                    session_id, generation_epoch, frame_index,
                    root_pos[i], bone_quats[i], morphs[i],
                ))
            except asyncio.QueueFull:
                try:
                    anim_queue.get_nowait()
                    dropped += 1
                    anim_queue.put_nowait((
                        session_id, generation_epoch, frame_index,
                        root_pos[i], bone_quats[i], morphs[i],
                    ))
                except Exception:
                    break

        if dropped > 0:
            logger.warning(
                "Dropped %d anim frames due to full queue (session=%s)", dropped, session_id
            )
        logger.info(
            "Enqueued %d anim frames (session=%s queue=%d fps=%d)",
            frames_count, session_id, anim_queue.qsize(), STREAM_FPS,
        )


def _log_inference_stats(
    retargeter, chunk_id, session_id, frames_count,
    poses, expressions, morphs, root_pos,
) -> None:
    try:
        pose_mag = float(np.mean(np.linalg.norm(poses.reshape(frames_count, -1, 3), axis=2)))
    except Exception:
        pose_mag = 0.0
    expr_mean = float(np.mean(expressions)) if expressions is not None else 0.0
    morph_mean = float(np.mean(morphs)) if morphs is not None else 0.0
    root_min = root_pos.min(axis=0) if root_pos is not None else np.zeros(3)
    root_max = root_pos.max(axis=0) if root_pos is not None else np.zeros(3)
    logger.info(
        "Anim stats session=%s chunk=%s frames=%d pose_mag=%.4f "
        "expr_mean=%.4f morph_mean=%.4f root_min=%s root_max=%s",
        session_id, chunk_id, frames_count, pose_mag, expr_mean, morph_mean,
        np.round(root_min, 4).tolist(), np.round(root_max, 4).tolist(),
    )
    logger.info(
        "Face gain chunk=%s expr_abs_max=%.4f gain=%.3f morph_abs_max=%.4f jaw_mag=%.4f",
        chunk_id,
        getattr(retargeter, "last_expr_abs_max", 0.0),
        getattr(retargeter, "last_expr_gain", 1.0),
        getattr(retargeter, "last_morph_abs_max", 0.0),
        getattr(retargeter, "last_jaw_mag", 0.0),
    )
    if morphs is not None and morphs.size > 0:
        morphs_max = morphs.max(axis=0)
        top_idx = np.argsort(-morphs_max)[:5]
        top = [(retargeter.morphs[i], float(morphs_max[i])) for i in top_idx]
        mouth_names = getattr(retargeter, "_mouth_morphs", []) or []
        mouth_idx = [
            retargeter.morphs.index(n) for n in mouth_names if n in retargeter.morphs
        ]
        mouth_top = []
        if mouth_idx:
            mouth_vals = morphs_max[mouth_idx]
            mouth_top_i = np.argsort(-mouth_vals)[:5]
            mouth_top = [
                (retargeter.morphs[mouth_idx[i]], float(mouth_vals[i]))
                for i in mouth_top_i
            ]
        logger.info("Face stats chunk=%s top=%s", chunk_id, top)
        logger.info(
            "Face mouth stats chunk=%s mouth_targets=%s mouth_top=%s",
            chunk_id, mouth_names, mouth_top,
        )

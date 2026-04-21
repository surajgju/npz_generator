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
import importlib
import json
import logging
import multiprocessing as mp
import os
import re
import queue as queue_module
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket
from streamsettings import (
    DEFAULT_OVERLAP_SEC,
    LIVE_IDLE_FLUSH_MS,
    LIVE_INFERENCE_BATCH_SAMPLES,
)

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

FLUSH_REASON_BATCH_COMPLETE = "batch_complete"
FLUSH_REASON_IDLE_TIMEOUT = "idle_timeout"

# ---------------------------------------------------------------------------
# Shared queues and concurrency primitives
# ---------------------------------------------------------------------------

anim_queue: asyncio.Queue = asyncio.Queue(maxsize=int(STREAM_FPS * 10))
AUDIO_IN_QUEUE_MAX_CHUNKS: int = int(os.environ.get("AUDIO_IN_QUEUE_MAX_CHUNKS", "16"))
audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_IN_QUEUE_MAX_CHUNKS)
epoch_drop_count: int = 0


@dataclass
class InferenceProcessService:
    input_queue: Any
    output_queue: Any
    control_queue: Any
    stop_event: Any
    ready_event: Any
    process: mp.Process
    batch_samples: int
    overlap_sec: float
    retarget_map_path: str
    startup_timeout_sec: float

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
            genai = importlib.import_module("google.genai")  # type: ignore
        except (ModuleNotFoundError, ImportError) as exc:
            raise RuntimeError(
                "Missing google-genai in the active Python environment. "
                "Install with: pip install google-genai"
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
    def _build_live_config(system_instruction: str, *, enable_speech: bool = True):
        types = _genai_types
        if types is None:
            raise RuntimeError("google-genai is not installed. Run: pip install google-genai")

        # Build speech / voice config only when the types are available.
        voice_name = os.environ.get("GEMINI_LIVE_VOICE", "Aoede")
        try:
            speech_cfg = (
                types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )
                if enable_speech
                else None
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
    def _is_invalid_request_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "invalid argument" in msg
            or "invalid frame payload data" in msg
            or "1007" in msg
        )

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
        if "cannot import name 'genai' from 'google'" in lower or "no module named 'google.genai'" in lower:
            return (
                "google-genai is not installed in the Python interpreter running the server. "
                "Activate your project venv and start uvicorn with that interpreter."
            )
        if "missing google-genai in the active python environment" in lower:
            return (
                "google-genai is not installed in the Python interpreter running the server. "
                "Activate your project venv and start uvicorn with that interpreter."
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
        Falls back to send_client_content for older/limited session objects.
        """
        client = self._make_genai_client()
        live_config_with_speech = self._build_live_config(
            self.system_instruction, enable_speech=True
        )
        live_config_without_speech = self._build_live_config(
            self.system_instruction, enable_speech=False
        )
        live_config_variants = [("speech:on", live_config_with_speech)]
        if (
            live_config_with_speech.model_dump(mode="json", exclude_none=True)
            != live_config_without_speech.model_dump(mode="json", exclude_none=True)
        ):
            live_config_variants.append(("speech:off", live_config_without_speech))
        # ~20 ms chunks at the input sample rate (2 bytes per int16 sample)
        chunk_size = max(320, int(input_sr * 0.02) * 2)
        mime_type = f"audio/pcm;rate={input_sr}"
        idle_timeout_sec = float(os.environ.get("GEMINI_LIVE_RECEIVE_IDLE_TIMEOUT_SEC", "8"))
        max_turn_sec = float(os.environ.get("GEMINI_LIVE_MAX_TURN_SEC", "45"))
        last_exc: Optional[Exception] = None

        for idx, model_name in enumerate(self.model_candidates):
            self.model = model_name
            has_next_model = idx + 1 < len(self.model_candidates)

            for cfg_idx, (cfg_name, live_config) in enumerate(live_config_variants):
                has_next_cfg = cfg_idx + 1 < len(live_config_variants)
                try:
                    async with client.aio.live.connect(
                        model=model_name, config=live_config
                    ) as session:
                        logger.info(
                            "Gemini Live connected model=%s config=%s conversation=%s",
                            model_name,
                            cfg_name,
                            conversation_id,
                        )

                        # Realtime input is the supported path for streamed microphone PCM.
                        # Send buffered user audio in ~20 ms chunks and explicitly close the
                        # audio stream to trigger deterministic generation after PTT ends.
                        sent_chunks = 0
                        if callable(getattr(session, "send_realtime_input", None)):
                            for start in range(0, len(input_pcm_bytes), chunk_size):
                                chunk = input_pcm_bytes[start : start + chunk_size]
                                if not chunk:
                                    continue
                                await session.send_realtime_input(
                                    audio=_genai_types.Blob(data=chunk, mime_type=mime_type)
                                )
                                sent_chunks += 1
                            await session.send_realtime_input(audio_stream_end=True)
                        else:
                            # Backward-compat fallback for older SDK sessions.
                            await session.send_client_content(
                                turns=[
                                    _genai_types.Content(
                                        role="user",
                                        parts=[
                                            _genai_types.Part.from_bytes(
                                                data=input_pcm_bytes,
                                                mime_type=mime_type,
                                            )
                                        ],
                                    )
                                ],
                                turn_complete=True,
                            )
                            sent_chunks = max(
                                1, (len(input_pcm_bytes) + chunk_size - 1) // max(1, chunk_size)
                            )

                        logger.debug(
                            "Gemini Live audio sent model=%s config=%s bytes=%d sr=%d chunks=%d",
                            model_name,
                            cfg_name,
                            len(input_pcm_bytes),
                            input_sr,
                            sent_chunks,
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

                    if has_next_cfg and self._is_invalid_request_error(exc):
                        logger.warning(
                            "Gemini Live invalid request model=%s config=%s (%s); retrying with %s",
                            model_name,
                            cfg_name,
                            str(exc).splitlines()[0],
                            live_config_variants[cfg_idx + 1][0],
                        )
                        continue

                    if has_next_model and self._is_model_unsupported_error(exc):
                        logger.warning(
                            "Gemini Live model %s failed (%s); retrying with %s",
                            model_name,
                            str(exc).splitlines()[0],
                            self.model_candidates[idx + 1],
                        )
                        break

                    if has_next_model and self._is_invalid_request_error(exc):
                        logger.warning(
                            "Gemini Live model %s invalid request (%s); retrying with %s",
                            model_name,
                            str(exc).splitlines()[0],
                            self.model_candidates[idx + 1],
                        )
                        break
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

    enqueue_monotonic_ns = time.monotonic_ns()
    try:
        audio_in_queue.put_nowait((
            audio_np, sr, chunk_id, stream_session_id,
            generation_epoch, source, conversation_id, reply_id, enqueue_monotonic_ns,
        ))
    except asyncio.QueueFull:
        try:
            old = audio_in_queue.get_nowait()
            enqueue_monotonic_ns = time.monotonic_ns()
            audio_in_queue.put_nowait((
                audio_np, sr, chunk_id, stream_session_id,
                generation_epoch, source, conversation_id, reply_id, enqueue_monotonic_ns,
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
        timing = None
        if len(item) >= 7:
            session_id, generation_epoch, frame_id_hint, root_pos, bone_quats, morphs, timing = item
        elif len(item) >= 6:
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
            st.next_frame_index = max(st.next_frame_index, frame_id + 1)
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
                if timing:
                    header["timing"] = timing
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


def _reset_generator_stream_state(generator) -> None:
    """Reset rolling state carried across chunks in LiveMotionGenerator."""
    for attr in (
        "_prev_overlap",
        "_prev_last_pose",
        "_prev_last_expr",
        "_prev_last_trans",
    ):
        if hasattr(generator, attr):
            setattr(generator, attr, None)


def _min_samples_for_generator_call(generator) -> int:
    """Compute a safe minimum audio sample count for one model inference call."""
    try:
        sr = int(getattr(generator, "sr", 16000)) or 16000
        pose_fps = int(getattr(generator, "pose_fps", 30)) or 30
        cfg = getattr(getattr(generator, "model", None), "cfg", None)
        pre_frames = int(getattr(cfg, "seed_frames", 4)) if cfg is not None else 4
        # modeling_emage_audio.inference needs total_len > 2*seed_frames to avoid empty cats.
        min_frames = max(1, (2 * pre_frames) + 1)
        min_samples = int(np.ceil((min_frames * sr) / float(pose_fps)))
        return max(1, min_samples)
    except Exception:
        return 4800


def _coalesce_chunk_ids(chunk_ids: List[Optional[str]]) -> Optional[str]:
    values = [str(cid) for cid in chunk_ids if cid is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return f"{values[0]}-{values[-1]}"

def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle to (..., 4) quaternion [x,y,z,w]."""
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angles
    small = angles < 1e-8
    axis = np.zeros_like(axis_angle)
    axis[~small[..., 0]] = axis_angle[~small[..., 0]] / angles[~small[..., 0]]
    sin_half = np.sin(half)
    quat = np.concatenate([axis * sin_half, np.cos(half)], axis=-1)
    quat[small[..., 0]] = np.array([0, 0, 0, 1], dtype=quat.dtype)
    return quat

def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert (..., 4) quaternion [x,y,z,w] to (..., 3) axis-angle."""
    angle = 2 * np.arccos(np.clip(quat[..., 3], -1, 1))
    sin_half = np.sin(angle * 0.5)
    axis = np.zeros_like(quat[..., :3])
    mask = sin_half > 1e-8
    axis[mask] = quat[mask][..., :3] / sin_half[mask][..., np.newaxis]
    return axis * angle[..., np.newaxis]

def slerp_quat(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (...,4)."""
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    neg_mask = dot < 0.0
    q2 = np.where(neg_mask, -q2, q2)
    dot = np.where(neg_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near_linear = sin_theta < 1e-6
    safe_sin_theta = np.where(near_linear, 1.0, sin_theta)

    w1 = np.sin((1.0 - t) * theta) / safe_sin_theta
    w2 = np.sin(t * theta) / safe_sin_theta

    slerp = (w1 * q1) + (w2 * q2)
    linear = ((1.0 - t) * q1) + (t * q2)
    out = np.where(near_linear, linear, slerp)

    # Keep unit quaternions to avoid numerical drift.
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.where(norm > 1e-8, norm, 1.0)



def _safe_queue_size(queue_obj: Any) -> int:
    try:
        return int(queue_obj.qsize())
    except Exception:
        return -1


def _put_mp_queue_with_drop(queue_obj: Any, item: Any) -> bool:
    try:
        queue_obj.put_nowait(item)
        return True
    except queue_module.Full:
        try:
            queue_obj.get_nowait()
        except Exception:
            pass
        try:
            queue_obj.put_nowait(item)
            return True
        except Exception:
            return False


def _put_mp_queue_blocking(queue_obj: Any, item: Any, stop_event: Any, timeout: float = 0.1) -> bool:
    while not stop_event.is_set():
        try:
            queue_obj.put(item, timeout=timeout)
            return True
        except queue_module.Full:
            continue
        except Exception:
            return False
    return False


def _enqueue_anim_frame(item: Any) -> bool:
    try:
        anim_queue.put_nowait(item)
        return True
    except asyncio.QueueFull:
        try:
            anim_queue.get_nowait()
            anim_queue.put_nowait(item)
            return True
        except Exception:
            return False


def _bootstrap_batch_audio(batch_audio: np.ndarray, min_required_samples: int) -> np.ndarray:
    if batch_audio.size >= min_required_samples:
        return batch_audio
    pad = min_required_samples - int(batch_audio.size)
    if pad <= 0:
        return batch_audio
    if batch_audio.size <= 0:
        return np.zeros((min_required_samples,), dtype=np.float32)
    return np.pad(batch_audio, (pad, 0), mode="edge")


def _drain_mp_queue_nowait(queue_obj: Any) -> int:
    drained = 0
    while True:
        try:
            queue_obj.get_nowait()
            drained += 1
        except queue_module.Empty:
            return drained
        except Exception:
            return drained


def _clear_pending_session_state(
    pending_chunks: Dict[str, deque],
    pending_samples: Dict[str, int],
    pending_sr: Dict[str, int],
    pending_epoch: Dict[str, int],
    seen_first_batch: Dict[str, bool],
    session_next_frame_index: Dict[str, int],
    last_diag_log: Dict[str, float],
    last_backend_summary_log: Dict[str, float],
    pending_last_chunk_at: Dict[str, float],
    *,
    session_id: Optional[str] = None,
) -> None:
    if session_id is None:
        pending_chunks.clear()
        pending_samples.clear()
        pending_sr.clear()
        pending_epoch.clear()
        seen_first_batch.clear()
        session_next_frame_index.clear()
        last_diag_log.clear()
        last_backend_summary_log.clear()
        pending_last_chunk_at.clear()
        return
    pending_chunks.pop(session_id, None)
    pending_samples.pop(session_id, None)
    pending_sr.pop(session_id, None)
    pending_epoch.pop(session_id, None)
    seen_first_batch.pop(session_id, None)
    session_next_frame_index.pop(session_id, None)
    last_diag_log.pop(session_id, None)
    last_backend_summary_log.pop(session_id, None)
    pending_last_chunk_at.pop(session_id, None)


def _drain_inference_control_queue(
    control_queue: Any,
    pending_chunks: Dict[str, deque],
    pending_samples: Dict[str, int],
    pending_sr: Dict[str, int],
    pending_epoch: Dict[str, int],
    seen_first_batch: Dict[str, bool],
    session_next_frame_index: Dict[str, int],
    last_diag_log: Dict[str, float],
    last_backend_summary_log: Dict[str, float],
    pending_last_chunk_at: Dict[str, float],
    reset_token: int,
) -> int:
    while True:
        try:
            message = control_queue.get_nowait()
        except queue_module.Empty:
            return reset_token
        except Exception:
            return reset_token
        if not isinstance(message, dict):
            continue
        msg_type = message.get("type")
        if msg_type == "reset_all":
            _clear_pending_session_state(
                pending_chunks,
                pending_samples,
                pending_sr,
                pending_epoch,
                seen_first_batch,
                session_next_frame_index,
                last_diag_log,
                last_backend_summary_log,
                pending_last_chunk_at,
            )
            reset_token += 1
        elif msg_type == "drop_session":
            _clear_pending_session_state(
                pending_chunks,
                pending_samples,
                pending_sr,
                pending_epoch,
                seen_first_batch,
                session_next_frame_index,
                last_diag_log,
                last_backend_summary_log,
                pending_last_chunk_at,
                session_id=message.get("session_id"),
            )
            reset_token += 1


def start_inference_service(
    retarget_map_path: str,
    *,
    overlap_sec: float = DEFAULT_OVERLAP_SEC,
    batch_samples: Optional[int] = None,
    startup_timeout_sec: Optional[float] = None,
) -> InferenceProcessService:
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue(maxsize=AUDIO_IN_QUEUE_MAX_CHUNKS)
    output_queue = ctx.Queue(maxsize=max(32, STREAM_FPS * 6))
    control_queue = ctx.Queue(maxsize=8)
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    configured_batch_samples = max(1, LIVE_INFERENCE_BATCH_SAMPLES)
    service_batch_samples = max(1, batch_samples or configured_batch_samples)
    process = ctx.Process(
        target=_inference_process_main,
        args=(
            input_queue,
            output_queue,
            control_queue,
            stop_event,
            ready_event,
            retarget_map_path,
            overlap_sec,
            service_batch_samples,
        ),
        name="npz-inference",
        daemon=True,
    )
    timeout = startup_timeout_sec
    if timeout is None:
        try:
            timeout = float(os.environ.get("INFERENCE_STARTUP_TIMEOUT_SEC", "300"))
        except ValueError:
            timeout = 300.0
    process.start()
    logger.info(
        "Inference process starting pid=%s batch_samples=%d overlap_sec=%.3f input_q=%d output_q=%d",
        process.pid,
        service_batch_samples,
        overlap_sec,
        AUDIO_IN_QUEUE_MAX_CHUNKS,
        max(32, STREAM_FPS * 6),
    )
    return InferenceProcessService(
        input_queue=input_queue,
        output_queue=output_queue,
        control_queue=control_queue,
        stop_event=stop_event,
        ready_event=ready_event,
        process=process,
        batch_samples=service_batch_samples,
        overlap_sec=overlap_sec,
        retarget_map_path=retarget_map_path,
        startup_timeout_sec=float(timeout),
    )


async def wait_for_inference_service_ready(
    service: InferenceProcessService,
    timeout_sec: Optional[float] = None,
) -> None:
    timeout = float(service.startup_timeout_sec if timeout_sec is None else timeout_sec)
    deadline = time.monotonic() + timeout
    while True:
        if service.ready_event.is_set():
            if service.process.exitcode is not None and service.process.exitcode != 0:
                raise RuntimeError(
                    f"Inference process exited during startup with code {service.process.exitcode}"
                )
            return
        if service.process.exitcode is not None:
            raise RuntimeError(
                f"Inference process failed during startup with code {service.process.exitcode}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(0.2, remaining))
    raise TimeoutError(
        f"Inference process did not become ready within {timeout:.1f}s"
    )


async def stop_inference_service(service: Optional[InferenceProcessService]) -> None:
    if service is None:
        return
    try:
        service.stop_event.set()
    except Exception:
        pass
    try:
        _put_mp_queue_with_drop(service.input_queue, None)
    except Exception:
        pass

    def _join() -> None:
        try:
            service.process.join(timeout=5.0)
        except Exception:
            pass
        if service.process.is_alive():
            try:
                service.process.terminate()
            except Exception:
                pass
            try:
                service.process.join(timeout=5.0)
            except Exception:
                pass
        if service.process.is_alive():
            try:
                service.process.kill()
            except Exception:
                pass
            try:
                service.process.join(timeout=5.0)
            except Exception:
                pass

    await asyncio.to_thread(_join)
    for queue_obj in (service.input_queue, service.output_queue, service.control_queue):
        try:
            queue_obj.close()
        except Exception:
            pass
        try:
            queue_obj.cancel_join_thread()
        except Exception:
            pass


async def reset_inference_service_session(
    service: Optional[InferenceProcessService],
    *,
    reason: str = "session_switch",
) -> None:
    if service is None:
        return

    def _reset() -> tuple[int, int, bool]:
        drained_input = _drain_mp_queue_nowait(service.input_queue)
        drained_output = _drain_mp_queue_nowait(service.output_queue)
        sent = _put_mp_queue_with_drop(
            service.control_queue,
            {
                "type": "reset_all",
                "reason": reason,
                "server_time_ms": int(time.monotonic_ns() / 1_000_000),
            },
        )
        return drained_input, drained_output, sent

    drained_input, drained_output, sent = await asyncio.to_thread(_reset)
    logger.info(
        "Inference service reset reason=%s drained_input=%d drained_output=%d control_sent=%s",
        reason,
        drained_input,
        drained_output,
        sent,
    )


def _inference_process_main(
    input_queue: Any,
    output_queue: Any,
    control_queue: Any,
    stop_event: Any,
    ready_event: Any,
    retarget_map_path: str,
    overlap_sec: float,
    batch_samples: int,
) -> None:
    from npz_logging import setup_logging
    from live_streaming_pipeline import LiveMotionGenerator
    from .retargeter import SmplxRetargeter

    setup_logging()
    logger.info(
        "Inference process booting pid=%s batch_samples=%d overlap_sec=%.3f",
        os.getpid(),
        batch_samples,
        overlap_sec,
    )
    try:
        generator = LiveMotionGenerator(overlap_sec=overlap_sec)
        retargeter = SmplxRetargeter(retarget_map_path)
        min_required_samples = _min_samples_for_generator_call(generator)
        logger.info(
            "Inference process ready pid=%s min_required_samples=%d batch_samples=%d",
            os.getpid(),
            min_required_samples,
            batch_samples,
        )
        ready_event.set()

        flush_idle_sec = max(0.0, float(LIVE_IDLE_FLUSH_MS) / 1000.0)

        pending_chunks: Dict[str, deque] = defaultdict(deque)
        pending_samples: Dict[str, int] = defaultdict(int)
        pending_sr: Dict[str, int] = {}
        pending_epoch: Dict[str, int] = {}
        seen_first_batch: Dict[str, bool] = defaultdict(lambda: False)
        session_next_frame_index: Dict[str, int] = defaultdict(int)
        last_diag_log: Dict[str, float] = {}
        last_backend_summary_log: Dict[str, float] = {}
        pending_last_chunk_at: Dict[str, float] = {}
        reset_token = 0

        def drain_control() -> None:
            nonlocal reset_token
            reset_token = _drain_inference_control_queue(
                control_queue,
                pending_chunks,
                pending_samples,
                pending_sr,
                pending_epoch,
                seen_first_batch,
                session_next_frame_index,
                last_diag_log,
                last_backend_summary_log,
                pending_last_chunk_at,
                reset_token,
            )

        def run_pending_batch(
            session_id: str,
            target_samples: int,
            *,
            flush_reason: Optional[str] = None,
        ) -> bool:
            normalized_flush_reason = flush_reason or FLUSH_REASON_BATCH_COMPLETE
            available = int(pending_samples.get(session_id, 0))
            if available <= 0:
                _clear_pending_session_state(
                    pending_chunks,
                    pending_samples,
                    pending_sr,
                    pending_epoch,
                    seen_first_batch,
                    session_next_frame_index,
                    last_diag_log,
                    last_backend_summary_log,
                    pending_last_chunk_at,
                    session_id=session_id,
                )
                return False

            remaining = min(max(1, int(target_samples)), available)
            consumed_ids: List[Optional[str]] = []
            parts: List[np.ndarray] = []
            earliest_enqueue_ns: Optional[int] = None

            while remaining > 0 and pending_chunks[session_id]:
                cid, arr, queued_at_ns = pending_chunks[session_id][0]
                take = min(remaining, int(arr.shape[0]))
                if take <= 0:
                    pending_chunks[session_id].popleft()
                    continue
                parts.append(arr[:take])
                consumed_ids.append(cid)
                if earliest_enqueue_ns is None or queued_at_ns < earliest_enqueue_ns:
                    earliest_enqueue_ns = queued_at_ns
                pending_samples[session_id] -= take
                remaining -= take
                if take == int(arr.shape[0]):
                    pending_chunks[session_id].popleft()
                else:
                    pending_chunks[session_id][0] = (cid, arr[take:], queued_at_ns)

            if not parts:
                return False

            if pending_samples.get(session_id, 0) <= 0 and not pending_chunks.get(session_id):
                pending_last_chunk_at.pop(session_id, None)

            batch_audio = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            batch_chunk_id = _coalesce_chunk_ids(consumed_ids)
            batch_audio_for_model = batch_audio
            batch_reset_token = reset_token
            batch_sr = pending_sr.get(session_id, 16000)
            batch_started_ns = time.monotonic_ns()
            inputqueuewait_ms = 0.0
            if earliest_enqueue_ns is not None:
                inputqueuewait_ms = max(0.0, (batch_started_ns - earliest_enqueue_ns) / 1_000_000.0)

            if not seen_first_batch[session_id]:
                _reset_generator_stream_state(generator)
                seen_first_batch[session_id] = True
                if batch_audio.size < min_required_samples:
                    batch_audio_for_model = _bootstrap_batch_audio(batch_audio, min_required_samples)

            try:
                generator_started_ns = time.perf_counter_ns()
                coeffs = generator.process_audio_chunk(
                    batch_audio_for_model,
                    batch_sr,
                    batch_chunk_id,
                    overlap_source_audio=batch_audio,
                )
                generator_elapsed_ms = (time.perf_counter_ns() - generator_started_ns) / 1_000_000.0
            except Exception as exc:
                logger.exception(
                    "Inference crashed for session=%s chunk=%s (batched_samples=%d flush_reason=%s): %s",
                    session_id,
                    batch_chunk_id,
                    int(batch_audio.shape[0]),
                    normalized_flush_reason,
                    exc,
                )
                _clear_pending_session_state(
                    pending_chunks,
                    pending_samples,
                    pending_sr,
                    pending_epoch,
                    seen_first_batch,
                    session_next_frame_index,
                    last_diag_log,
                    last_backend_summary_log,
                    pending_last_chunk_at,
                    session_id=session_id,
                )
                return False

            drain_control()
            if reset_token != batch_reset_token:
                logger.info(
                    "Dropping computed batch after reset session=%s chunk=%s",
                    session_id,
                    batch_chunk_id,
                )
                return False

            if not coeffs:
                logger.warning(
                    "No coeffs generated for chunk %s (audio_len=%d flush_reason=%s)",
                    batch_chunk_id,
                    batch_audio.shape[0],
                    normalized_flush_reason,
                )
                return False

            if batch_chunk_id is not None and str(batch_chunk_id).startswith("0"):
                retargeter.reset_root_offset()

            poses = coeffs["poses"]
            expressions = coeffs["expressions"]
            trans = coeffs["trans"]
            frames_count = int(poses.shape[0])
            generator_timings = getattr(generator, "last_process_timings", {}) or {}
            infer_ms = float(generator_timings.get("infer_ms", generator_elapsed_ms))
            resample_ms = float(generator_timings.get("resample_ms", 0.0))

            if (STREAM_FPS != BASE_FPS or SLOW_MOTION_FACTOR != 1.0) and frames_count > 1:
                stream_resample_started_ns = time.perf_counter_ns()
                target_count = max(
                    1,
                    int(round(frames_count * (STREAM_FPS / BASE_FPS) * SLOW_MOTION_FACTOR)),
                )
                expressions = _resample_frames_linear(expressions, target_count)
                trans = _resample_frames_linear(trans, target_count)

                joint_count = poses.shape[1] // 3
                poses_aa = poses.reshape(frames_count, joint_count, 3)
                poses_quat = axis_angle_to_quat(poses_aa)

                src_idx = np.linspace(0, frames_count - 1, target_count)
                new_quats = []
                for i in range(target_count):
                    low = int(np.floor(src_idx[i]))
                    high = min(low + 1, frames_count - 1)
                    t = src_idx[i] - low
                    q_low = poses_quat[low]
                    q_high = poses_quat[high]
                    q_interp = slerp_quat(q_low, q_high, t)
                    new_quats.append(q_interp)
                new_poses_quat = np.stack(new_quats, axis=0)
                new_poses_aa = quat_to_axis_angle(new_poses_quat)
                poses = new_poses_aa.reshape(target_count, -1)
                frames_count = target_count
                resample_ms += (time.perf_counter_ns() - stream_resample_started_ns) / 1_000_000.0

            retarget_started_ns = time.perf_counter_ns()
            root_pos, bone_quats, morphs = retargeter.retarget(poses, expressions, trans)
            retarget_ms = (time.perf_counter_ns() - retarget_started_ns) / 1_000_000.0

            if frames_count > 0:
                _log_inference_stats(
                    retargeter,
                    batch_chunk_id,
                    session_id,
                    frames_count,
                    poses,
                    expressions,
                    morphs,
                    root_pos,
                )

            frame_start = session_next_frame_index[session_id]
            session_next_frame_index[session_id] += frames_count

            timing = {
                "inputqueuewait_ms": round(inputqueuewait_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "resample_ms": round(resample_ms, 3),
                "retarget_ms": round(retarget_ms, 3),
                "outputqueuewait_ms": 0.0,
                "batch_samples": int(batch_audio.shape[0]),
                "frames_count": int(frames_count),
                "flush_reason": normalized_flush_reason,
            }
            enqueued_frames = 0
            outputqueuewait_ms = 0.0
            for i in range(frames_count):
                if (i % 8) == 0:
                    drain_control()
                    if reset_token != batch_reset_token:
                        logger.info(
                            "Stopping frame enqueue after reset session=%s chunk=%s enqueued=%d/%d",
                            session_id,
                            batch_chunk_id,
                            enqueued_frames,
                            frames_count,
                        )
                        break
                frame_index = frame_start + i
                enqueue_started_ns = time.perf_counter_ns()
                frame_timing = dict(timing)
                frame_timing["outputqueuewait_ms"] = round(outputqueuewait_ms, 3)
                if not _put_mp_queue_blocking(
                    output_queue,
                    (
                        session_id,
                        pending_epoch.get(session_id, 0),
                        frame_index,
                        root_pos[i],
                        bone_quats[i],
                        morphs[i],
                        frame_timing,
                    ),
                    stop_event,
                ):
                    break
                outputqueuewait_ms += (time.perf_counter_ns() - enqueue_started_ns) / 1_000_000.0
                enqueued_frames += 1

            timing["outputqueuewait_ms"] = round(outputqueuewait_ms, 3)

            logger.info(
                "Enqueued %d anim frames (session=%s queue=%d fps=%d chunk=%s flush_reason=%s)",
                enqueued_frames,
                session_id,
                _safe_queue_size(output_queue),
                STREAM_FPS,
                batch_chunk_id,
                normalized_flush_reason,
            )
            now = time.time()
            last_summary = last_backend_summary_log.get(session_id, 0.0)
            if normalized_flush_reason == FLUSH_REASON_IDLE_TIMEOUT or (now - last_summary) >= 10.0:
                logger.info(
                    "Backend timing session=%s chunk=%s pending_chunks=%d pending_samples=%d inputqueuewait_ms=%.3f infer_ms=%.3f resample_ms=%.3f retarget_ms=%.3f outputqueuewait_ms=%.3f batch_samples=%d frames_count=%d flush_reason=%s",
                    session_id,
                    batch_chunk_id,
                    len(pending_chunks.get(session_id, ())),
                    int(pending_samples.get(session_id, 0)),
                    timing["inputqueuewait_ms"],
                    timing["infer_ms"],
                    timing["resample_ms"],
                    timing["retarget_ms"],
                    timing["outputqueuewait_ms"],
                    timing["batch_samples"],
                    timing["frames_count"],
                    timing["flush_reason"],
                )
                last_backend_summary_log[session_id] = now
            return enqueued_frames > 0

        while not stop_event.is_set():
            drain_control()

            if flush_idle_sec > 0:
                now = time.time()
                idle_sessions = [
                    sid
                    for sid, last_chunk_at in pending_last_chunk_at.items()
                    if pending_samples.get(sid, 0) > 0
                    and pending_samples.get(sid, 0) < batch_samples
                    and (now - last_chunk_at) >= flush_idle_sec
                ]
                for sid in idle_sessions:
                    if stop_event.is_set():
                        break
                    run_pending_batch(
                        sid,
                        pending_samples.get(sid, 0),
                        flush_reason=FLUSH_REASON_IDLE_TIMEOUT,
                    )

            try:
                item = input_queue.get(timeout=0.1)
            except queue_module.Empty:
                continue
            except Exception as exc:
                logger.exception("Fatal error getting from inference input queue: %s", exc)
                continue

            if item is None:
                if stop_event.is_set():
                    break
                continue

            if len(item) >= 9:
                audio_np, sr, chunk_id, session_id, generation_epoch, source, conversation_id, reply_id, enqueue_monotonic_ns = item
            else:
                audio_np, sr, chunk_id, session_id, generation_epoch, source, conversation_id, reply_id = item
                enqueue_monotonic_ns = time.monotonic_ns()

            prev_epoch = pending_epoch.get(session_id)
            if prev_epoch is not None and prev_epoch != generation_epoch:
                _clear_pending_session_state(
                    pending_chunks,
                    pending_samples,
                    pending_sr,
                    pending_epoch,
                    seen_first_batch,
                    session_next_frame_index,
                    last_diag_log,
                    last_backend_summary_log,
                    pending_last_chunk_at,
                    session_id=session_id,
                )
            pending_epoch[session_id] = generation_epoch
            pending_sr[session_id] = int(sr)

            audio_arr = np.asarray(audio_np, dtype=np.float32)
            if audio_arr.size == 0:
                continue

            pending_chunks[session_id].append((chunk_id, audio_arr, int(enqueue_monotonic_ns)))
            pending_samples[session_id] += int(audio_arr.shape[0])
            pending_last_chunk_at[session_id] = time.time()

            now = time.time()
            last = last_diag_log.get(session_id, 0)
            if now - last > 30:
                logger.info(
                    "Inference pending session=%s samples=%d chunks=%d batch_samples=%d",
                    session_id,
                    pending_samples[session_id],
                    len(pending_chunks[session_id]),
                    batch_samples,
                )
                last_diag_log[session_id] = now

            while pending_samples[session_id] >= batch_samples and not stop_event.is_set():
                if not run_pending_batch(
                    session_id,
                    batch_samples,
                    flush_reason=FLUSH_REASON_BATCH_COMPLETE,
                ):
                    break

        logger.info("Inference process exiting pid=%s", os.getpid())
    except Exception:
        logger.exception("Inference process crashed")
        raise


async def _feed_audio_to_inference_process(service: InferenceProcessService) -> None:
    while not service.stop_event.is_set():
        item = await audio_in_queue.get()
        if item is None:
            continue
        if service.process.exitcode is not None:
            raise RuntimeError(
                f"Inference process exited with code {service.process.exitcode} while audio was queued"
            )
        ok = await asyncio.to_thread(_put_mp_queue_with_drop, service.input_queue, item)
        if not ok:
            logger.warning(
                "Inference input queue full; dropped oldest chunk session=%s chunk=%s",
                item[3] if len(item) > 3 else "?",
                item[2] if len(item) > 2 else "?",
            )


async def _drain_inference_output(service: InferenceProcessService) -> None:
    while not service.stop_event.is_set():
        try:
            item = await asyncio.to_thread(service.output_queue.get, True, 0.1)
        except queue_module.Empty:
            if service.process.exitcode is not None:
                if service.stop_event.is_set():
                    return
                raise RuntimeError(
                    f"Inference process exited with code {service.process.exitcode}"
                )
            continue
        except Exception as exc:
            logger.exception("Fatal error getting from inference output queue: %s", exc)
            continue

        if item is None:
            continue
        _enqueue_anim_frame(item)


async def inference_worker(service: InferenceProcessService) -> None:
    """Bridge audio chunks to the spawned inference process and forward its frames."""
    await wait_for_inference_service_ready(service)
    audio_task = asyncio.create_task(_feed_audio_to_inference_process(service))
    output_task = asyncio.create_task(_drain_inference_output(service))
    try:
        await asyncio.gather(audio_task, output_task)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Inference bridge stopped due to error")
        raise
    finally:
        for task in (audio_task, output_task):
            task.cancel()
        await asyncio.gather(audio_task, output_task, return_exceptions=True)
        await stop_inference_service(service)

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

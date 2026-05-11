"""Conversation runtime for the NPZ Generator streaming server.

Owns the ConversationRuntime class which manages a single /ws/conversation
WebSocket session: push-to-talk, audio buffering, Gemini Live turn execution,
and assistant lifecycle events.
"""
import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket

from .audio_pipeline import GeminiLiveAudioEngine, ingest_audio_chunk
from .session import sessions, sessions_lock
from .tenant_service import resolve_tenant_config, get_rag_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONVERSATION_AUDIO_SR: int = int(
    os.environ.get(
        "CONVERSATION_AUDIO_SR",
        os.environ.get("CONVERSATION_STT_SR", "16000"),
    )
)
CONVERSATION_PCM_CHUNK_MS: int = int(os.environ.get("CONVERSATION_PCM_CHUNK_MS", "20"))
CONVERSATION_PTT_MAX_SEC: float = float(os.environ.get("CONVERSATION_PTT_MAX_SEC", "20"))
CONVERSATION_PTT_GUARD_SEC: float = float(
    os.environ.get("CONVERSATION_PTT_GUARD_SEC", str(CONVERSATION_PTT_MAX_SEC + 1.0))
)


# ---------------------------------------------------------------------------
# RAG Tool Definition
# ---------------------------------------------------------------------------

RAG_TOOL_DEFINITION = {
    "function_declarations": [
        {
            "name": "query_knowledge_base",
            "description": "Search the internal knowledge base for facts, policies, and documentation to answer user questions accurately.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The specific topic or question to search for."
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

# ---------------------------------------------------------------------------
# ConversationRuntime
# ---------------------------------------------------------------------------

class ConversationRuntime:
    """Manages one /ws/conversation WebSocket session end-to-end."""

    def __init__(
        self,
        websocket: WebSocket,
        conversation_id: str,
        server_boot_id: str,
        server_clock_id: str,
        server_time_fn,
        conversation_protocol_version: int,
        create_audio_session_fn,
        servingid: str = "dev"
    ) -> None:
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.servingid = servingid
        self._server_boot_id = server_boot_id
        self._server_clock_id = server_clock_id
        self._server_time_fn = server_time_fn
        self._conversation_protocol_version = conversation_protocol_version
        self._create_audio_session = create_audio_session_fn

        self.adk = GeminiLiveAudioEngine()
        self._listening: bool = False
        self._user_audio_chunks: List[bytes] = []
        self._user_audio_sr: int = CONVERSATION_AUDIO_SR
        self._reply_task: Optional[asyncio.Task] = None
        self._ptt_guard_task: Optional[asyncio.Task] = None
        self.current_stream_session_id: Optional[str] = None
        self.current_reply_id: Optional[str] = None
        self.current_generation_epoch: int = 0

    async def _send(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("conversation_id", self.conversation_id)
        payload.setdefault("server_boot_id", self._server_boot_id)
        payload.setdefault("server_clock_id", self._server_clock_id)
        payload.setdefault("server_time_ms", self._server_time_fn())
        await self.websocket.send_text(json.dumps(payload))

    async def send_hello_ack(self) -> None:
        await self._send(
            {
                "type": "hello_ack",
                "conversation_id": self.conversation_id,
                "protocol_version": self._conversation_protocol_version,
            }
        )

    def _cancel_ptt_guard(self) -> None:
        if self._ptt_guard_task and not self._ptt_guard_task.done():
            self._ptt_guard_task.cancel()
        self._ptt_guard_task = None

    async def _ptt_guard(self) -> None:
        try:
            await asyncio.sleep(max(0.25, CONVERSATION_PTT_GUARD_SEC))
            if not self._listening:
                return
            logger.warning(
                "PTT guard timeout conversation=%s after %.2fs; forcing ptt_end",
                self.conversation_id,
                CONVERSATION_PTT_GUARD_SEC,
            )
            await self.handle_ptt_end()
        except asyncio.CancelledError:
            return

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes tool calls requested by Gemini."""
        if name == "query_knowledge_base":
            query = args.get("query", "")
            logger.info("RAG Tool triggered: sid=%s query='%s'", self.servingid, query)
            context = await get_rag_context(self.servingid, query)
            return {"result": context if context else "No specific information found in the knowledge base."}
        return {"error": f"Unknown tool: {name}"}

    async def handle_ptt_start(self) -> None:
        if self._reply_task and not self._reply_task.done():
            await self.interrupt(reason="user_started_speaking")
        self._cancel_ptt_guard()
        self._listening = True
        self._user_audio_chunks = []
        self._user_audio_sr = CONVERSATION_AUDIO_SR
        self._ptt_guard_task = asyncio.create_task(self._ptt_guard())
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
            joined = b"".join(self._user_audio_chunks)
            self._user_audio_chunks = [joined[-max_bytes:]]

    async def handle_ptt_end(self) -> None:
        self._cancel_ptt_guard()
        if not self._listening:
            return
        self._listening = False
        if not self._user_audio_chunks:
            return
        audio_bytes = b"".join(self._user_audio_chunks)
        self._user_audio_chunks = []
        total_samples = len(audio_bytes) // 2
        duration_sec = total_samples / max(1, self._user_audio_sr)
        logger.info(
            "Conversation PTT end: received %d bytes of audio (%.2f sec at sr=%d)",
            len(audio_bytes),
            duration_sec,
            self._user_audio_sr,
        )
        if self._reply_task and not self._reply_task.done():
            await self.interrupt(reason="new_ptt")
        self._reply_task = asyncio.create_task(
            self._run_turn(audio_bytes, self._user_audio_sr)
        )

    async def interrupt(self, reason: str = "interrupt") -> None:
        self._cancel_ptt_guard()
        self._listening = False
        self._user_audio_chunks = []
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reply_task = None
        if self.current_stream_session_id:
            async with sessions_lock:
                st = sessions.get(self.current_stream_session_id)
                if st:
                    st.generation_epoch += 1
                    st.producer_connected = False
        await self._send(
            {
                "type": "interrupted",
                "stream_session_id": self.current_stream_session_id,
                "session_id": self.current_stream_session_id,
                "reply_id": self.current_reply_id,
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
        emitted_speaking_end = False
        output_sample_accum = np.zeros((0,), dtype=np.int16)
        out_chunk_samples = max(
            1, int((CONVERSATION_AUDIO_SR * CONVERSATION_PCM_CHUNK_MS) / 1000)
        )
        chunk_idx = 0
        stale_stream = False

        try:
            await self._send({"type": "assistant_thinking_start"})
            reply_id = str(uuid.uuid4())

            # Resolve tenant-scoped prompt and RAG context
            tenant_config = resolve_tenant_config(self.servingid)
            base_prompt = tenant_config.get("system_prompt") if tenant_config else "You are a concise helpful voice assistant."
            
            # Simple RAG: query is the last PTT (could be expanded with STT later)
            # For now, we use a generic query or could just use the base prompt.
            # Ideally we'd have the STT of what the user just said.
            # But the Live API handles STT internally.
            # As a workaround, we can provide general context or skip RAG if no query.
            rag_context = await get_rag_context(self.servingid, "general info")
            
            final_prompt = base_prompt
            # We don't prepend RAG context anymore, we let Gemini call the tool!

            async for raw_chunk, chunk_sr in self.adk.stream_audio_events(
                conversation_id=self.conversation_id,
                input_pcm_bytes=pcm_bytes,
                input_sr=input_sr,
                system_instruction=final_prompt,
                tools=[RAG_TOOL_DEFINITION],
                tool_handler=self.handle_tool_call
            ):
                audio_i16 = np.frombuffer(raw_chunk, dtype=np.int16)
                if audio_i16.size == 0:
                    continue
                if chunk_sr != CONVERSATION_AUDIO_SR:
                    audio_i16 = self.adk._resample_int16_linear(
                        audio_i16, chunk_sr, CONVERSATION_AUDIO_SR
                    )
                if audio_i16.size == 0:
                    continue

                output_sample_accum = (
                    audio_i16
                    if output_sample_accum.size == 0
                    else np.concatenate([output_sample_accum, audio_i16])
                )

                while output_sample_accum.size >= out_chunk_samples:
                    chunk_i16 = output_sample_accum[:out_chunk_samples]
                    output_sample_accum = output_sample_accum[out_chunk_samples:]

                    if stream_session_id is None:
                        stream_session_id = await self._create_audio_session(
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
                        source="assistant_gemini_live",
                        conversation_id=self.conversation_id,
                        reply_id=reply_id,
                        server_boot_id=self._server_boot_id,
                        server_clock_id=self._server_clock_id,
                        server_time_fn=self._server_time_fn,
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

            # Flush remainder
            if stream_session_id is not None and output_sample_accum.size > 0 and not stale_stream:
                await ingest_audio_chunk(
                    audio_np=(output_sample_accum.astype(np.float32) / 32768.0),
                    sr=CONVERSATION_AUDIO_SR,
                    chunk_id=str(chunk_idx),
                    stream_session_id=stream_session_id,
                    generation_epoch=generation_epoch,
                    source="assistant_gemini_live",
                    conversation_id=self.conversation_id,
                    reply_id=reply_id,
                    server_boot_id=self._server_boot_id,
                    server_clock_id=self._server_clock_id,
                    server_time_fn=self._server_time_fn,
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
                emitted_thinking_end = True
            if emitted_speaking_start and stream_session_id is not None:
                await self._send(
                    {
                        "type": "assistant_speaking_end",
                        "reply_id": reply_id,
                        "stream_session_id": stream_session_id,
                        "session_id": stream_session_id,
                    }
                )
                emitted_speaking_end = True

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (
                self.adk._is_api_key_error(exc)
                or self.adk._is_model_unsupported_error(exc)
            ):
                logger.warning(
                    "Conversation turn failed conversation=%s: %s",
                    self.conversation_id, exc,
                )
            else:
                logger.exception(
                    "Conversation turn failed conversation=%s: %s",
                    self.conversation_id, exc,
                )
            await self._send(
                {"type": "error", "message": self.adk.user_message_for_exception(exc)}
            )
            try:
                if not emitted_thinking_end:
                    await self._send({"type": "assistant_thinking_end"})
                    emitted_thinking_end = True
            except Exception:
                pass
        finally:
            try:
                if not emitted_thinking_end:
                    await self._send({"type": "assistant_thinking_end"})
                    emitted_thinking_end = True
                if (
                    emitted_speaking_start
                    and not emitted_speaking_end
                    and stream_session_id is not None
                ):
                    await self._send(
                        {
                            "type": "assistant_speaking_end",
                            "reply_id": reply_id,
                            "stream_session_id": stream_session_id,
                            "session_id": stream_session_id,
                        }
                    )
                    emitted_speaking_end = True
            except Exception:
                pass
            if stream_session_id:
                async with sessions_lock:
                    st = sessions.get(stream_session_id)
                    if st:
                        st.producer_connected = False
                        st.last_activity_ms = self._server_time_fn()
            if self.current_reply_id == reply_id:
                self.current_reply_id = None

    async def close(self) -> None:
        self._cancel_ptt_guard()
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reply_task = None

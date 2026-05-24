import asyncio
import contextlib
import json
import os
import queue
import sys
import threading
import time
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from server import audio_pipeline, session as session_state
from server.session import SessionState
from live_streaming_pipeline import LiveMotionGenerator
from streamsettings import BASE_MOTION_FPS, DEFAULT_OVERLAP_SEC


def _wait_until(predicate, timeout=1.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class _FakeGeneratorCfg:
    seed_frames = 1


class _FakeGeneratorModel:
    cfg = _FakeGeneratorCfg()

    def inference(self, *args, **kwargs):
        return {}


class _FakeLiveMotionGenerator:
    calls = []

    def __init__(self, overlap_sec=DEFAULT_OVERLAP_SEC):
        self.sr = 16000
        self.pose_fps = BASE_MOTION_FPS
        self.overlap_sec = float(overlap_sec)
        self.model = _FakeGeneratorModel()
        self._prev_overlap = None
        self._prev_last_pose = None
        self._prev_last_expr = None
        self._prev_last_trans = None
        self.last_process_timings = {
            "infer_ms": 12.5,
            "resample_ms": 1.25,
        }

    def process_audio_chunk(self, audio_chunk, sample_rate, chunk_id=None, overlap_source_audio=None):
        self.__class__.calls.append(
            {
                "audio_chunk": np.array(audio_chunk, copy=True),
                "sample_rate": int(sample_rate),
                "chunk_id": chunk_id,
                "overlap_source_audio": None
                if overlap_source_audio is None
                else np.array(overlap_source_audio, copy=True),
            }
        )
        frames = 2
        return {
            "poses": np.zeros((frames, 165), dtype=np.float32),
            "expressions": np.zeros((frames, 8), dtype=np.float32),
            "trans": np.zeros((frames, 3), dtype=np.float32),
        }


class _FakeRetargeter:
    def __init__(self, *args, **kwargs):
        self.morphs = ["Exp000", "Exp001", "Exp002"]
        self._mouth_morphs = []
        self.last_expr_abs_max = 0.0
        self.last_expr_gain = 1.0
        self.last_morph_abs_max = 0.0
        self.last_jaw_mag = 0.0

    def reset_root_offset(self):
        return None

    def retarget(self, poses, expressions, trans):
        frames = int(poses.shape[0])
        return (
            np.zeros((frames, 3), dtype=np.float32),
            np.zeros((frames, 1, 4), dtype=np.float32),
            np.zeros((frames, len(self.morphs)), dtype=np.float32),
        )


class _InferenceProcessHarness:
    def __init__(self, *, flush_idle_sec=0.05, batch_samples=2400):
        self.flush_idle_sec = flush_idle_sec
        self.batch_samples = batch_samples
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.control_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self._stack = ExitStack()
        self.thread = None

    @property
    def service(self):
        return SimpleNamespace(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            control_queue=self.control_queue,
        )

    def start(self):
        fake_live_module = types.ModuleType("live_streaming_pipeline")
        fake_live_module.LiveMotionGenerator = _FakeLiveMotionGenerator
        fake_retargeter_module = types.ModuleType("server.retargeter")
        fake_retargeter_module.SmplxRetargeter = _FakeRetargeter

        _FakeLiveMotionGenerator.calls.clear()
        self._stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "live_streaming_pipeline": fake_live_module,
                    "server.retargeter": fake_retargeter_module,
                },
            )
        )
        self._stack.enter_context(
            patch.dict(
                os.environ,
                {"INFERENCE_IDLE_FLUSH_SEC": str(self.flush_idle_sec)},
                clear=False,
            )
        )
        self.thread = threading.Thread(
            target=audio_pipeline._inference_process_main,
            args=(
                self.input_queue,
                self.output_queue,
                self.control_queue,
                self.stop_event,
                self.ready_event,
                "unused-retarget-map.json",
                0.25,
                self.batch_samples,
            ),
            daemon=True,
            name="test-inference-process",
        )
        self.thread.start()
        if not self.ready_event.wait(timeout=2.0):
            self.stop()
            raise RuntimeError("Fake inference process did not become ready")
        return self

    def stop(self):
        self.stop_event.set()
        try:
            self.input_queue.put_nowait(None)
        except Exception:
            pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._stack.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def put_audio(self, *, session_id="session-a", samples=1200, chunk_id="1", generation_epoch=0):
        self.input_queue.put(
            (
                np.ones((samples,), dtype=np.float32),
                16000,
                chunk_id,
                session_id,
                generation_epoch,
                "unit_test",
                None,
                None,
                time.monotonic_ns(),
            )
        )


class StreamingFixRegressionTests(unittest.TestCase):
    def test_reset_inference_service_session_drains_main_queues_and_sends_reset(self):
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        control_queue = queue.Queue()
        input_queue.put(("old-audio",))
        output_queue.put(("old-frame",))
        service = SimpleNamespace(
            input_queue=input_queue,
            output_queue=output_queue,
            control_queue=control_queue,
        )

        asyncio.run(audio_pipeline.reset_inference_service_session(service, reason="unit_test"))

        self.assertTrue(input_queue.empty())
        self.assertTrue(output_queue.empty())
        control = control_queue.get_nowait()
        self.assertEqual(control["type"], "reset_all")
        self.assertEqual(control["reason"], "unit_test")
        self.assertIn("server_time_ms", control)

    def test_child_reset_clears_pending_buffers_before_idle_flush(self):
        with _InferenceProcessHarness(flush_idle_sec=0.2) as harness:
            harness.put_audio(samples=1200, chunk_id="7")
            self.assertTrue(
                _wait_until(lambda: harness.input_queue.empty(), timeout=1.0),
                "child thread never consumed the queued audio chunk",
            )

            asyncio.run(
                audio_pipeline.reset_inference_service_session(
                    harness.service,
                    reason="test_child_reset",
                )
            )

            time.sleep(0.3)
            self.assertTrue(harness.output_queue.empty())
            self.assertEqual(_FakeLiveMotionGenerator.calls, [])

    def test_partial_batch_idle_flush_emits_frames(self):
        with self.assertLogs(audio_pipeline.logger.name, level="INFO") as logs:
            with _InferenceProcessHarness(flush_idle_sec=0.05) as harness:
                harness.put_audio(samples=1200, chunk_id="11")
                frame = harness.output_queue.get(timeout=1.0)

        self.assertEqual(frame[0], "session-a")
        self.assertGreaterEqual(frame[2], 0)
        self.assertIn("inputqueuewait_ms", frame[6])
        self.assertEqual(frame[6]["flush_reason"], "idle_timeout")
        self.assertEqual(frame[6]["batch_samples"], 1200)
        self.assertEqual(len(_FakeLiveMotionGenerator.calls), 1)
        self.assertTrue(
            any("flush_reason=idle_timeout" in message for message in logs.output),
            logs.output,
        )
        self.assertTrue(
            any("inputqueuewait_ms=" in message and "infer_ms=" in message for message in logs.output),
            logs.output,
        )

    def test_overlap_source_audio_preserves_real_history_after_bootstrap_padding(self):
        class _FakeCfg:
            lf = 0
            cf = 0
            lu = 0
            cu = 0
            lh = 0
            ch = 0
            ll = 0
            cl = 0

        class _FakeModel:
            cfg = _FakeCfg()

            def inference(self, *args, **kwargs):
                return {}

        class _FakeMotionVQ:
            def decode(self, **kwargs):
                frames = 12
                return {
                    "motion_axis_angle": torch.zeros((1, frames, 165), dtype=torch.float32),
                    "expression": torch.zeros((1, frames, 100), dtype=torch.float32),
                    "trans": torch.zeros((1, frames, 3), dtype=torch.float32),
                }

        generator = LiveMotionGenerator.__new__(LiveMotionGenerator)
        generator.device = torch.device("cpu")
        generator.sr = 8
        generator.pose_fps = BASE_MOTION_FPS
        generator.speaker_id = torch.zeros(1, 1).long()
        generator.overlap_sec = DEFAULT_OVERLAP_SEC
        generator._prev_overlap = None
        generator._prev_last_pose = None
        generator._prev_last_expr = None
        generator._prev_last_trans = None
        generator.target_fps = BASE_MOTION_FPS
        generator.model = _FakeModel()
        generator.motion_vq = _FakeMotionVQ()

        first_real_audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        first_padded_audio = np.array([9.0, 9.0, 3.0], dtype=np.float32)
        second_real_audio = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        generator.process_audio_chunk(
            first_padded_audio,
            8,
            chunk_id="first",
            overlap_source_audio=first_real_audio,
        )
        np.testing.assert_array_equal(
            generator._prev_overlap,
            np.array([2.0, 3.0], dtype=np.float32),
        )

        generator.process_audio_chunk(
            second_real_audio,
            8,
            chunk_id="second",
            overlap_source_audio=second_real_audio,
        )
        np.testing.assert_array_equal(
            generator._prev_overlap,
            np.array([5.0, 6.0], dtype=np.float32),
        )


class _RecordingWebSocket:
    def __init__(self):
        self.headers = []
        self.payloads = []
        self.frame_sent = asyncio.Event()

    async def send_text(self, message):
        self.headers.append(json.loads(message))

    async def send_bytes(self, payload):
        self.payloads.append(payload)
        self.frame_sent.set()


class BroadcastLoopRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._real_sleep = asyncio.sleep
        self._original_anim_queue = audio_pipeline.anim_queue
        audio_pipeline.anim_queue = asyncio.Queue()
        async with session_state.sessions_lock:
            session_state.sessions.clear()
            session_state.active_session_ids.clear()
        session_state.anim_clients.clear()
        session_state.anim_client_protocol.clear()
        session_state.anim_client_session.clear()

    async def asyncTearDown(self):
        self._drain_anim_queue()
        audio_pipeline.anim_queue = self._original_anim_queue
        async with session_state.sessions_lock:
            session_state.sessions.clear()
            session_state.active_session_ids.clear()
        session_state.anim_clients.clear()
        session_state.anim_client_protocol.clear()
        session_state.anim_client_session.clear()

    def _drain_anim_queue(self):
        while True:
            try:
                audio_pipeline.anim_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def test_inactive_session_frames_are_skipped_without_extra_sleep(self):
        ws = _RecordingWebSocket()
        async with session_state.sessions_lock:
            session_state.sessions["session-a"] = SessionState(
                session_id="session-a",
                created_ms=0,
                last_activity_ms=0,
                client_id="test-client",
            )
            session_state.sessions["session-b"] = SessionState(
                session_id="session-b",
                created_ms=0,
                last_activity_ms=0,
                client_id="test-client",
            )
            session_state.active_session_ids["test-client"] = "session-b"

        session_state.anim_clients["test-client"].add(ws)
        session_state.anim_client_protocol[ws] = 2
        session_state.anim_client_session[ws] = "session-b"

        sleep_calls = []

        async def fake_sleep(delay, result=None):
            sleep_calls.append(delay)
            await self._real_sleep(0)
            return result

        task = asyncio.create_task(
            audio_pipeline.broadcast_anim_loop("test-clock", lambda: 123)
        )
        try:
            with patch.object(audio_pipeline.asyncio, "sleep", fake_sleep):
                await audio_pipeline.anim_queue.put(
                    (
                        "session-a",
                        0,
                        0,
                        np.zeros((3,), dtype=np.float32),
                        np.zeros((1, 4), dtype=np.float32),
                        np.zeros((1,), dtype=np.float32),
                    )
                )
                await audio_pipeline.anim_queue.put(
                    (
                        "session-b",
                        0,
                        0,
                        np.zeros((3,), dtype=np.float32),
                        np.zeros((1, 4), dtype=np.float32),
                        np.zeros((1,), dtype=np.float32),
                        {
                            "inputqueuewait_ms": 1.0,
                            "infer_ms": 2.0,
                            "resample_ms": 3.0,
                            "retarget_ms": 4.0,
                            "outputqueuewait_ms": 5.0,
                            "batch_samples": 1200,
                            "frames_count": 2,
                            "flush_reason": "batch_complete",
                        },
                    )
                )

                await asyncio.wait_for(ws.frame_sent.wait(), timeout=1.0)
                await self._real_sleep(0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(len(ws.headers), 1)
        self.assertEqual(ws.headers[0]["session_id"], "session-b")
        self.assertEqual(ws.headers[0]["timing"]["flush_reason"], "batch_complete")
        self.assertEqual(ws.headers[0]["timing"]["infer_ms"], 2.0)
        self.assertEqual(len(ws.payloads), 1)
        self.assertEqual(sleep_calls, [1 / audio_pipeline.STREAM_FPS])


if __name__ == "__main__":
    unittest.main()

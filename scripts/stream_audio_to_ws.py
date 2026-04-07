import argparse
import asyncio
import base64
import json
import time
import os
import sys

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import librosa
import numpy as np
import websockets
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


async def stream_audio(audio_path: str, ws_url: str, chunk_sec: float):
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_size = int(sr * chunk_sec)
    async with websockets.connect(ws_url, max_size=2**24) as ws:
        chunk_id = 0
        start_t = time.perf_counter()
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == 0:
                break
            pcm = (chunk * 32767.0).astype(np.int16).tobytes()
            payload = {
                "chunk_id": f"{chunk_id}",
                "sr": sr,
                "dtype": "int16",
                "audio_b64": base64.b64encode(pcm).decode("utf-8"),
            }
            await ws.send(json.dumps(payload))
            ack = await ws.recv()
            logger.info("Sent chunk %d (%d samples). Ack: %s", chunk_id, len(chunk), ack)
            chunk_id += 1
            # Keep sender in real-time pace to avoid overfilling server queues.
            target_elapsed = chunk_id * chunk_sec
            sleep_for = target_elapsed - (time.perf_counter() - start_t)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ws", default="ws://localhost:8000/ws/audio")
    parser.add_argument("--chunk", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(stream_audio(args.audio, args.ws, args.chunk))


if __name__ == "__main__":
    main()

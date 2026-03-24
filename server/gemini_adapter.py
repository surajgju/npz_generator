import base64
from typing import Optional


class GeminiAudioBridge:
    """
    Stub adapter for Gemini Multimodal Live API audio frames.
    This class formats audio chunks for the /ws/audio endpoint.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def build_audio_payload(self, audio_pcm: bytes, chunk_id: Optional[str] = None) -> dict:
        """
        Accepts raw PCM16 bytes and returns payload for /ws/audio.
        """
        return {
            "chunk_id": chunk_id,
            "sr": self.sample_rate,
            "dtype": "int16",
            "audio_b64": base64.b64encode(audio_pcm).decode("utf-8"),
        }

    # TODO: Integrate Gemini Live SDK callback and forward frames to /ws/audio.

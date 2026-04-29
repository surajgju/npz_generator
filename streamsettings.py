"""Shared streaming and rendering defaults for the NPZ generator.

Values are resolved once per process from the environment so server,
generator, and offline tools stay aligned without duplicated literals.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Union

try:
    from dotenv import load_dotenv
    # Look for .env.local in the current dir or 'server' dir
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")
    elif os.path.exists("server/.env.local"):
        load_dotenv("server/.env.local")
except ImportError:
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


BASE_MOTION_FPS: int = _env_int("BASE_MOTION_FPS", 30)
STREAM_FPS: int = _env_int("STREAM_FPS", 20)
DEFAULT_OVERLAP_SEC: float = _env_float("DEFAULT_OVERLAP_SEC", 0.25)
DEFAULT_INFERENCE_BATCH_SAMPLES: int = _env_int("INFERENCE_BATCH_SAMPLES", 2400)
DEFAULT_RENDER_FPS: int = _env_int("DEFAULT_RENDER_FPS", BASE_MOTION_FPS)
LIVE_INFERENCE_BATCH_SAMPLES: int = _env_int("LIVE_INFERENCE_BATCH_SAMPLES", 8000)
LIVE_IDLE_FLUSH_MS: float = _env_float("LIVE_IDLE_FLUSH_MS", 150.0)
EXPRESSION_MAX_ABS: float = _env_float("EXPRESSION_MAX_ABS", 0.85)
MOUTH_MAX_ABS: float = _env_float("MOUTH_MAX_ABS", 0.70)
EYE_BROW_MAX_ABS: float = _env_float("EYE_BROW_MAX_ABS", 0.55)

# --- Retargeter Advanced Defaults ---
EXPRESSION_NORM_MIN: float = _env_float("EXPRESSION_NORM_MIN", 0.2)
EXPRESSION_NORM_MAX: float = _env_float("EXPRESSION_NORM_MAX", 2.5)
EXPRESSION_TARGET: float = _env_float("EXPRESSION_TARGET", 0.35)
EXPRESSION_GAIN_MIN: float = _env_float("EXPRESSION_GAIN_MIN", 0.3)
EXPRESSION_GAIN_MAX: float = _env_float("EXPRESSION_GAIN_MAX", 3.0)
EXPRESSION_OFFSET_STRENGTH: float = _env_float("EXPRESSION_OFFSET_STRENGTH", 0.5)
EXPRESSION_FALLBACK_GAIN: float = _env_float("EXPRESSION_FALLBACK_GAIN", 3.0)
EXPRESSION_FALLBACK_THRESH: float = _env_float("EXPRESSION_FALLBACK_THRESH", 0.05)

# --- App Constants ---
ALLOWED_ORIGINS: str = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

# Backward-compatible aliases for modules/tests that still refer to these names.
BASE_FPS: int = BASE_MOTION_FPS

_logged_settings: bool = False


def resolved_stream_settings() -> Dict[str, Union[int, float]]:
    return {
        "base_motion_fps": BASE_MOTION_FPS,
        "stream_fps": STREAM_FPS,
        "default_overlap_sec": DEFAULT_OVERLAP_SEC,
        "default_inference_batch_samples": DEFAULT_INFERENCE_BATCH_SAMPLES,
        "default_render_fps": DEFAULT_RENDER_FPS,
        "live_inference_batch_samples": LIVE_INFERENCE_BATCH_SAMPLES,
        "live_idle_flush_ms": LIVE_IDLE_FLUSH_MS,
        "expression_max_abs": EXPRESSION_MAX_ABS,
        "mouth_max_abs": MOUTH_MAX_ABS,
        "eye_brow_max_abs": EYE_BROW_MAX_ABS,
    }


def log_resolved_stream_settings(logger: logging.Logger, *, prefix: str = "Resolved stream settings") -> None:
    global _logged_settings
    if _logged_settings:
        return
    _logged_settings = True
    settings = resolved_stream_settings()
    logger.info(
        "%s base_motion_fps=%d stream_fps=%d default_overlap_sec=%.3f default_inference_batch_samples=%d default_render_fps=%d live_inference_batch_samples=%d live_idle_flush_ms=%.1f expression_max_abs=%.3f mouth_max_abs=%.3f eye_brow_max_abs=%.3f",
        prefix,
        settings["base_motion_fps"],
        settings["stream_fps"],
        settings["default_overlap_sec"],
        settings["default_inference_batch_samples"],
        settings["default_render_fps"],
        settings["live_inference_batch_samples"],
        settings["live_idle_flush_ms"],
        settings["expression_max_abs"],
        settings["mouth_max_abs"],
        settings["eye_brow_max_abs"],
    )

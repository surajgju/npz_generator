"""Shared streaming and rendering defaults for the NPZ generator.

Values are resolved once per process from the environment so server,
generator, and offline tools stay aligned without duplicated literals.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Union


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
    }


def log_resolved_stream_settings(logger: logging.Logger, *, prefix: str = "Resolved stream settings") -> None:
    global _logged_settings
    if _logged_settings:
        return
    _logged_settings = True
    settings = resolved_stream_settings()
    logger.info(
        "%s base_motion_fps=%d stream_fps=%d default_overlap_sec=%.3f default_inference_batch_samples=%d default_render_fps=%d",
        prefix,
        settings["base_motion_fps"],
        settings["stream_fps"],
        settings["default_overlap_sec"],
        settings["default_inference_batch_samples"],
        settings["default_render_fps"],
    )

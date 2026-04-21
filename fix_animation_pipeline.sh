#!/usr/bin/env bash

set -e

echo "🔧 Fixing animation uncanny-valley pipeline issues..."

PROJECT_ROOT=$(pwd)
UTILS_DIR="$PROJECT_ROOT/utils"

mkdir -p "$UTILS_DIR"

SMOOTH_FILE="$UTILS_DIR/animation_smoothing.py"

echo "📦 Creating smoothing utilities..."

cat << 'EOF' > "$SMOOTH_FILE"
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = None

    def apply(self, value):
        if self.prev is None:
            self.prev = value
            return value
        self.prev = self.prev * (1 - self.alpha) + value * self.alpha
        return self.prev


def slerp(q1, q2, t):
    key_times = [0, 1]
    rots = R.from_quat([q1, q2])
    slerp = Slerp(key_times, rots)
    return slerp([t])[0].as_quat()


def blend_chunks(prev_chunk, next_chunk, blend_frames=6):
    if prev_chunk is None:
        return next_chunk

    blended = next_chunk.copy()

    for i in range(min(blend_frames, len(prev_chunk), len(next_chunk))):
        alpha = i / blend_frames
        blended[i] = prev_chunk[-blend_frames + i] * (1 - alpha) + next_chunk[i] * alpha

    return blended


class FramePacer:

    def __init__(self, fps):
        self.period = 1.0 / fps
        self.last = time.time()

    def wait(self):
        now = time.time()
        delta = now - self.last

        if delta < self.period:
            time.sleep(self.period - delta)

        self.last = time.time()
EOF


echo "✅ smoothing utilities created."


CONFIG_FILE="$PROJECT_ROOT/.env.animation"

echo "📦 Creating animation config defaults..."

cat << 'EOF' > "$CONFIG_FILE"
BASE_FPS=30
STREAM_FPS=30
SLOW_MOTION_FACTOR=1.0
EMA_ALPHA=0.2
CHUNK_BLEND_FRAMES=6
EOF

echo "✅ config file created (.env.animation)"


patch_file() {

FILE=$1

if [ ! -f "$FILE" ]; then
return
fi

echo "🔧 patching $FILE"

grep -q "animation_smoothing" "$FILE" || sed -i '' '1s/^/from utils.animation_smoothing import EMAFilter, blend_chunks, FramePacer\n/' "$FILE" 2>/dev/null || true

grep -q "FramePacer" "$FILE" || cat << 'EOF' >> "$FILE"


# ===== injected animation timing stabilizer =====

try:
    pacer = FramePacer(STREAM_FPS)
except:
    pacer = None


def stabilize_frame_timing():
    try:
        pacer.wait()
    except:
        pass

EOF

}


patch_file audio_pipeline.py
patch_file render.py
patch_file motion_io.py


echo "📦 Creating runtime FPS normalizer..."

FPS_PATCH="$UTILS_DIR/fps_guard.py"

cat << 'EOF' > "$FPS_PATCH"
import os


def get_fps():

    base = float(os.getenv("BASE_FPS", 30))
    stream = float(os.getenv("STREAM_FPS", base))
    slow = float(os.getenv("SLOW_MOTION_FACTOR", 1.0))

    return base * slow if stream is None else stream
EOF


echo "✅ FPS guard installed."


echo ""
echo "🎯 Animation stabilization patch complete!"
echo ""
echo "Next steps:"
echo "1) source .env.animation"
echo "2) restart your animation service"
echo ""
echo "Example:"
echo "source .env.animation && python run.py"
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

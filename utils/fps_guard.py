import os


def get_fps():

    base = float(os.getenv("BASE_FPS", 30))
    stream = float(os.getenv("STREAM_FPS", base))
    slow = float(os.getenv("SLOW_MOTION_FACTOR", 1.0))

    return base * slow if stream is None else stream

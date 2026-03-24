import logging
import os
import sys

_CONFIGURED = False


def setup_logging():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _CONFIGURED = True


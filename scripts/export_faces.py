import json
import os
import sys

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import smplx
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    model = smplx.create(
        model_path="models",
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_expression_coeffs=100,
    )
    faces = model.faces.tolist()
    out_dir = os.path.join(ROOT_DIR, "frontend", "public")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "faces.json")
    with open(out_path, "w") as f:
        json.dump(faces, f)
    logger.info("Wrote %s with %d faces", out_path, len(faces))


if __name__ == "__main__":
    main()

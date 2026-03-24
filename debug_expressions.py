"""
debug_expressions.py
--------------------
Renders a close-up face-tracking MP4 video of the full animation
plus a PNG overview grid of N evenly-spaced sample frames.
"""

import os
import time
import numpy as np
import torch
import smplx
import trimesh
import pyrender
import imageio
from PIL import Image, ImageDraw, ImageFont
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR    = "expr_debug"
VIDEO_PATH    = os.path.join(OUTPUT_DIR, "face_expressions.mp4")
GRID_PATH     = os.path.join(OUTPUT_DIR, "expressions_overview.png")
N_SAMPLES     = 12       # frames to include in the PNG grid
GRID_COLS     = 4
IMG_SIZE      = 512
FPS           = 30
# ─────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────
data        = np.load("output/intro_output.npz")
poses       = data["poses"]             # (T, 165)
trans       = data["trans"]             # (T, 3)
expressions = data["expressions"]       # (T, 100)
betas       = data["betas"][:10]
T           = poses.shape[0]
logger.info("Total frames: %d", T)

# ── SMPL-X BATCHED INFERENCE (all frames at once) ─────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

model = smplx.create(
    model_path="models",
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    num_expression_coeffs=100,
    batch_size=T
).to(device)

poses_t = torch.tensor(poses,       dtype=torch.float32).to(device)
trans_t = torch.tensor(trans,       dtype=torch.float32).to(device)
expr_t  = torch.tensor(expressions, dtype=torch.float32).to(device)
beta_t  = torch.tensor(betas,       dtype=torch.float32).unsqueeze(0).expand(T, -1).to(device)

logger.info("Running batched SMPL-X inference...")
t0 = time.time()
with torch.no_grad():
    output = model(
        betas=beta_t,
        global_orient=poses_t[:, :3],
        body_pose=poses_t[:, 3:66],
        jaw_pose=poses_t[:, 66:69],
        leye_pose=poses_t[:, 69:72],
        reye_pose=poses_t[:, 72:75],
        left_hand_pose=poses_t[:, 75:120],
        right_hand_pose=poses_t[:, 120:165],
        expression=expr_t,
        transl=trans_t
    )

all_verts = output.vertices.cpu().numpy()   # (T, V, 3)
faces     = model.faces
logger.info("Inference done in %.1fs", time.time() - t0)

# ── PYRENDER SCENE ────────────────────────────────────────────────────────
scene    = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
camera   = pyrender.PerspectiveCamera(yfov=np.pi / 5.5)
renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)

def render_frame(verts):
    """Render one close-up frame with the camera tracking the head."""
    scene.clear()

    tm = trimesh.Trimesh(verts, faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tm))

    # Head centroid: top-5% vertices by Y = skull region → shift down 0.10 m for face
    y_thresh   = np.percentile(verts[:, 1], 95)
    hc         = verts[verts[:, 1] >= y_thresh].mean(axis=0)
    face_y     = hc[1] - 0.10     # shift from skull-top to eye/mouth level

    cam_pose = np.eye(4, dtype=np.float64)
    cam_pose[:3, 3] = [hc[0], face_y, hc[2] + 0.45]
    scene.add(camera, pose=cam_pose)

    def lp(tx, ty, tz):
        p = np.eye(4, dtype=np.float64); p[:3, 3] = [tx, ty, tz]; return p

    scene.add(pyrender.DirectionalLight(color=np.ones(3),              intensity=4.0), pose=lp(hc[0]+0.6, face_y+0.3, hc[2]+0.3))
    scene.add(pyrender.DirectionalLight(color=np.ones(3),              intensity=1.5), pose=lp(hc[0]-0.6, face_y+0.1, hc[2]+0.3))
    scene.add(pyrender.DirectionalLight(color=np.array([0.8,0.9,1.0]), intensity=2.0), pose=lp(hc[0],     face_y+0.2, hc[2]-0.5))

    color, _ = renderer.render(scene)
    return color

# ── FULL VIDEO ────────────────────────────────────────────────────────────
logger.info("Rendering %d frames -> %s", T, VIDEO_PATH)
t1 = time.time()

writer = imageio.get_writer(VIDEO_PATH, fps=FPS, format="FFMPEG",
                             codec="libx264", pixelformat="yuv420p", quality=8)
for i in range(T):
    if i % 100 == 0:
        elapsed = time.time() - t1
        fps_now = i / elapsed if elapsed > 0 else 0
        eta     = (T - i) / fps_now if fps_now > 0 else 0
        logger.info("Frame %d/%d | %.1f fps | ETA %.0fs", i, T, fps_now, eta)

    frame = render_frame(all_verts[i])
    writer.append_data(frame)

writer.close()
logger.info("Video saved in %.1fs -> %s", time.time() - t1, VIDEO_PATH)

# ── PNG OVERVIEW GRID ─────────────────────────────────────────────────────
logger.info("Building %d-frame overview grid...", N_SAMPLES)
sample_idx = np.linspace(0, T - 1, N_SAMPLES, dtype=int)
grid_frames = [render_frame(all_verts[i]) for i in sample_idx]

GRID_ROWS = (N_SAMPLES + GRID_COLS - 1) // GRID_COLS
LABEL_H   = 24
grid      = Image.new("RGB", (GRID_COLS * IMG_SIZE, GRID_ROWS * (IMG_SIZE + LABEL_H)), (30, 30, 30))
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    font = ImageFont.load_default()

for j, (fi, img_arr) in enumerate(zip(sample_idx, grid_frames)):
    col  = j % GRID_COLS
    row  = j // GRID_COLS
    cell = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(cell)
    draw.rectangle([0, IMG_SIZE - LABEL_H, IMG_SIZE, IMG_SIZE], fill=(20, 20, 20))
    draw.text((6, IMG_SIZE - LABEL_H + 3), f"Frame {fi}", fill=(220, 220, 220), font=font)
    grid.paste(cell, (col * IMG_SIZE, row * (IMG_SIZE + LABEL_H)))

grid.save(GRID_PATH)
logger.info("Grid saved -> %s", GRID_PATH)
logger.info("All done. Video: %s | Grid: %s", VIDEO_PATH, GRID_PATH)

import time
import numpy as np
import torch
import smplx
import trimesh
import pyrender
import imageio
from npz_logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

# -----------------------------
# LOAD MOTION DATA
# -----------------------------
data = np.load("output/intro_output.npz")

poses = data["poses"]            # (T, 165)
trans = data["trans"]            # (T, 3)
expressions = data["expressions"]  # (T, 100)
betas = data["betas"][:10]       # use first 10

T = poses.shape[0]

# -----------------------------
# LOAD SMPL-X MODEL
# -----------------------------
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

# -------------------------------------------------------
# PRE-COMPUTE ALL VERTICES IN ONE BATCHED FORWARD PASS
# -------------------------------------------------------
logger.info("Running batched SMPL-X inference for %d frames...", T)
t0 = time.time()

poses_t  = torch.tensor(poses, dtype=torch.float32).to(device)         # (T, 165)
trans_t  = torch.tensor(trans, dtype=torch.float32).to(device)         # (T, 3)
expr_t   = torch.tensor(expressions, dtype=torch.float32).to(device)   # (T, 100)
beta_t   = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1).to(device)  # (T, 10)

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

all_vertices = output.vertices.cpu().numpy()  # (T, V, 3)
faces = model.faces
logger.info("Batched inference done in %.1fs", time.time() - t0)

# -----------------------------
# SETUP RENDERER
# -----------------------------
scene    = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
camera   = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
renderer = pyrender.OffscreenRenderer(512, 512)

# Lights are DirectionalLight — direction comes from their pose, not position.
# We create them once and re-add each frame with offsets relative to the body.
key_light  = pyrender.DirectionalLight(color=np.ones(3),              intensity=4.0)
fill_light = pyrender.DirectionalLight(color=np.ones(3),              intensity=1.5)
rim_light  = pyrender.DirectionalLight(color=np.array([0.8, 0.9, 1.0]), intensity=2.0)

def make_pose(tx, ty, tz):
    """Translation-only 4x4 pose matrix."""
    p = np.eye(4, dtype=np.float64)
    p[:3, 3] = [tx, ty, tz]
    return p

frames = []

# -----------------------------
# RENDER LOOP  (body-tracking camera)
# -----------------------------
logger.info("Rendering frames...")
t1 = time.time()

for i in range(T):
    if i % 100 == 0:
        elapsed = time.time() - t1
        fps_so_far = i / elapsed if elapsed > 0 else 0
        eta = (T - i) / fps_so_far if fps_so_far > 0 else 0
        logger.info("Frame %d/%d | %.1f fps | ETA %.0fs", i, T, fps_so_far, eta)

    scene.clear()

    # --- Mesh ---
    tm = trimesh.Trimesh(all_vertices[i], faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tm))

    # --- Body root position (XZ from trans, Y fixed at mid-body height) ---
    bx, _, bz = trans[i]          # body root x / z from motion data
    body_y    = 1.0               # fixed mid-body height in world space

    # Camera: in front of the body (+Z), centred on X, at mid-body height
    scene.add(camera, pose=make_pose(bx, body_y, bz + 3.0))

    # Three-point lights anchored to the body position
    scene.add(key_light,  pose=make_pose(bx + 1.0, body_y + 0.8, bz + 2.5))  # key: upper-left
    scene.add(fill_light, pose=make_pose(bx - 1.0, body_y + 0.5, bz + 2.0))  # fill: upper-right
    scene.add(rim_light,  pose=make_pose(bx,        body_y + 0.5, bz - 2.0))  # rim: behind

    color, _ = renderer.render(scene)
    frames.append(color)

logger.info("Render loop done in %.1fs", time.time() - t1)

# -----------------------------
# SAVE VIDEO  (MP4 via ffmpeg — much faster than GIF)
# -----------------------------
writer = imageio.get_writer("output.mp4", fps=30, format="FFMPEG", codec="libx264", pixelformat="yuv420p", quality=8)
for frame in frames:
    writer.append_data(frame)
writer.close()

logger.info("Done. Saved as output.mp4")

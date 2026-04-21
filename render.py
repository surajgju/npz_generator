import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

from npz_logging import setup_logging
from server.retargeter import SmplxRetargeter
from streamsettings import DEFAULT_RENDER_FPS
import logging

setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_GLB_PATH = "frontend/public/assets/head.glb"
DEFAULT_NPZ_PATH = "output/intro_output.npz"
DEFAULT_RETARGET_PATH = "server/retarget_map.json"
DEFAULT_OUTPUT_PATH = "output.mp4"


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for [x, y, z, w] quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _decompose_local_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 into translation, quaternion, scale."""
    t = mat[:3, 3].astype(np.float32)
    rs = mat[:3, :3].astype(np.float32)

    sx = float(np.linalg.norm(rs[:, 0]))
    sy = float(np.linalg.norm(rs[:, 1]))
    sz = float(np.linalg.norm(rs[:, 2]))
    scale = np.array([sx, sy, sz], dtype=np.float32)

    safe_scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    rot = rs @ np.diag(1.0 / safe_scale)

    if np.linalg.det(rot) < 0:
        scale[0] *= -1.0
        rot[:, 0] *= -1.0

    quat = R.from_matrix(rot).as_quat().astype(np.float32)
    return t, quat, scale


def _compose_local_matrix(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    rot = R.from_quat(q).as_matrix().astype(np.float32)
    mat[:3, :3] = rot @ np.diag(s)
    mat[:3, 3] = t
    return mat


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Stable look-at matrix for pyrender."""
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    f = target - eye
    fn = np.linalg.norm(f)
    if fn < 1e-8:
        f = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        f = f / fn

    s = np.cross(f, up)
    sn = np.linalg.norm(s)
    if sn < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        s = np.cross(f, up)
        sn = np.linalg.norm(s)
    if sn < 1e-8:
        s = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        s = s / sn

    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def _apply_mesh_weights(meshes: List[pyrender.Mesh], weights: np.ndarray):
    for mesh in meshes:
        mesh.weights = weights


def _should_hide(node_name: str, mesh_name: Optional[str], pattern: Optional[re.Pattern]) -> bool:
    if pattern is None:
        return False
    text = f"{node_name} {mesh_name or ''}".lower()
    return pattern.search(text) is not None


@dataclass
class RenderConfig:
    npz_path: str
    glb_path: str
    retarget_map: str
    output_path: str
    width: int
    height: int
    fps: int
    camera_mode: str
    camera_smooth: float
    camera_distance: Optional[float]
    camera_y_offset: float
    root_scale: float
    expression_clip: float
    start_frame: int
    end_frame: Optional[int]
    max_frames: Optional[int]
    hide_regex: str
    background: Tuple[float, float, float]
    frames_dir: Optional[str]
    dry_run: bool


class OfflineRigRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self.tm_scene = trimesh.load(cfg.glb_path, force="scene")
        if not isinstance(self.tm_scene, trimesh.Scene):
            raise ValueError(f"Expected scene glb, got type={type(self.tm_scene)}")

        self.exclude_pattern = re.compile(cfg.hide_regex, re.IGNORECASE) if cfg.hide_regex else None

        self.scene = pyrender.Scene(
            bg_color=[cfg.background[0], cfg.background[1], cfg.background[2], 1.0],
            ambient_light=[0.32, 0.32, 0.32, 1.0],
        )
        self.node_map: Dict[str, pyrender.Node] = {}
        self.rest_trs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.bones: Dict[str, pyrender.Node] = {}
        self.render_meshes: List[pyrender.Mesh] = []
        self._build_scene_graph()

        self.retargeter = SmplxRetargeter(cfg.retarget_map)
        self._bind_bones()

        self.camera = None
        self.renderer = None
        self.camera_node = None
        self.key_node = None
        self.fill_node = None
        self.rim_node = None
        if not cfg.dry_run:
            self._setup_camera_and_lights()

        bmin, bmax = self.tm_scene.bounds
        self.base_center = ((bmin + bmax) * 0.5).astype(np.float32)
        self.extents = (bmax - bmin).astype(np.float32)
        self.base_height = float(self.extents[1]) if self.extents.shape[0] >= 2 else 1.6

        self.camera_eye: Optional[np.ndarray] = None
        self.camera_target: Optional[np.ndarray] = None

    def _build_scene_graph(self):
        py_meshes: Dict[str, pyrender.Mesh] = {}
        for geom_name, geom in self.tm_scene.geometry.items():
            material = None
            if hasattr(geom.visual, "material"):
                mat = geom.visual.material
                img = getattr(mat, "baseColorTexture", getattr(mat, "image", None))
                if img is not None:
                    texture = pyrender.Texture(source=img, source_channels="RGB")
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorTexture=texture,
                        roughnessFactor=0.85,
                        metallicFactor=0.02,
                    )
            py_meshes[geom_name] = pyrender.Mesh.from_trimesh(geom, material=material)

        parents = self.tm_scene.graph.transforms.parents
        edge_data = self.tm_scene.graph.transforms.edge_data

        for node_name in self.tm_scene.graph.nodes:
            world, geom_name = self.tm_scene.graph.get(node_name)
            world = np.asarray(world, dtype=np.float32)
            parent = parents.get(node_name)
            if parent is None:
                local = world
            else:
                edge = edge_data.get((parent, node_name))
                if edge is not None and "matrix" in edge:
                    local = np.asarray(edge["matrix"], dtype=np.float32)
                else:
                    parent_world, _ = self.tm_scene.graph.get(parent)
                    parent_world = np.asarray(parent_world, dtype=np.float32)
                    local = np.linalg.inv(parent_world) @ world
            self.rest_trs[node_name] = _decompose_local_matrix(local)

            mesh = None
            if geom_name is not None and not _should_hide(node_name, geom_name, self.exclude_pattern):
                mesh = py_meshes[geom_name]

            py_node = pyrender.Node(name=node_name, matrix=local, mesh=mesh)
            self.node_map[node_name] = py_node

        for node_name in self.tm_scene.graph.nodes:
            parent = parents.get(node_name)
            if parent:
                self.scene.add_node(self.node_map[node_name], parent_node=self.node_map[parent])
            else:
                self.scene.add_node(self.node_map[node_name])

        seen = set()
        for node in self.node_map.values():
            if node.mesh is None:
                continue
            mid = id(node.mesh)
            if mid in seen:
                continue
            seen.add(mid)
            self.render_meshes.append(node.mesh)

        logger.info(
            "Scene graph loaded: nodes=%d meshes=%d geometries=%d",
            len(self.node_map),
            len(self.render_meshes),
            len(self.tm_scene.geometry),
        )

    def _setup_camera_and_lights(self):
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.2)
        self.renderer = pyrender.OffscreenRenderer(self.cfg.width, self.cfg.height)
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4, dtype=np.float32))

        self.key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.5)
        self.fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.2)
        self.rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.key_node = self.scene.add(self.key_light, pose=np.eye(4, dtype=np.float32))
        self.fill_node = self.scene.add(self.fill_light, pose=np.eye(4, dtype=np.float32))
        self.rim_node = self.scene.add(self.rim_light, pose=np.eye(4, dtype=np.float32))

    def _bind_bones(self):
        missing = []
        for bone_name in self.retargeter.bones:
            node = self.node_map.get(bone_name)
            if node is None:
                missing.append(bone_name)
            else:
                self.bones[bone_name] = node
        if missing:
            logger.warning("Missing %d bones in GLB (sample=%s)", len(missing), missing[:8])
        logger.info("Bone bindings: %d/%d", len(self.bones), len(self.retargeter.bones))

    def _camera_offsets(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.camera_mode == "face":
            distance = self.cfg.camera_distance or max(0.8, self.base_height * 0.95)
            target_off = np.array([0.0, self.base_height * 0.82, 0.0], dtype=np.float32)
            eye_off = np.array([0.02, self.base_height * 0.08 + self.cfg.camera_y_offset, distance], dtype=np.float32)
            return eye_off, target_off

        if self.cfg.camera_mode == "body":
            distance = self.cfg.camera_distance or max(2.5, self.base_height * 2.0)
            target_off = np.array([0.0, self.base_height * 0.52, 0.0], dtype=np.float32)
            eye_off = np.array([0.0, self.base_height * 0.18 + self.cfg.camera_y_offset, distance], dtype=np.float32)
            return eye_off, target_off

        # full
        distance = self.cfg.camera_distance or max(3.2, self.base_height * 2.5)
        target_off = np.array([0.0, self.base_height * 0.48, 0.0], dtype=np.float32)
        eye_off = np.array([0.0, self.base_height * 0.22 + self.cfg.camera_y_offset, distance], dtype=np.float32)
        return eye_off, target_off

    def _update_camera_and_lights(self, root: np.ndarray):
        if self.camera_node is None:
            return
        eye_off, target_off = self._camera_offsets()
        follow = np.array([root[0], 0.0, root[2]], dtype=np.float32)
        target = self.base_center + target_off + follow
        eye = target + eye_off

        alpha = float(np.clip(self.cfg.camera_smooth, 0.0, 1.0))
        if self.camera_eye is None:
            self.camera_eye = eye.copy()
            self.camera_target = target.copy()
        else:
            self.camera_eye = self.camera_eye * (1.0 - alpha) + eye * alpha
            self.camera_target = self.camera_target * (1.0 - alpha) + target * alpha

        cam_pose = _look_at(self.camera_eye, self.camera_target)
        self.scene.set_pose(self.camera_node, cam_pose)

        key_pos = self.camera_target + np.array([1.8, 1.4, 1.4], dtype=np.float32)
        fill_pos = self.camera_target + np.array([-2.2, 0.9, 1.1], dtype=np.float32)
        rim_pos = self.camera_target + np.array([0.0, 1.6, -2.0], dtype=np.float32)

        self.scene.set_pose(self.key_node, _look_at(key_pos, self.camera_target))
        self.scene.set_pose(self.fill_node, _look_at(fill_pos, self.camera_target))
        self.scene.set_pose(self.rim_node, _look_at(rim_pos, self.camera_target))

    def _apply_frame(self, frame_idx: int, root_pos: np.ndarray, bone_quats: np.ndarray, morphs: np.ndarray):
        clip = self.cfg.expression_clip
        if clip > 0:
            morphs = np.clip(morphs, -clip, clip)
        _apply_mesh_weights(self.render_meshes, morphs.astype(np.float32))

        root_bone = self.retargeter.root_bone
        for i, bone_name in enumerate(self.retargeter.bones):
            node = self.bones.get(bone_name)
            if node is None:
                continue

            rest_t, rest_q, rest_s = self.rest_trs[bone_name]
            q = _quat_mul(rest_q, bone_quats[i])
            t = rest_t.copy()
            if bone_name == root_bone:
                t = t + (root_pos * float(self.cfg.root_scale))

            self.scene.set_pose(node, _compose_local_matrix(t, q, rest_s))

        self._update_camera_and_lights(root_pos)

    def render(self):
        motion = np.load(self.cfg.npz_path)
        poses = motion["poses"]
        expressions = motion["expressions"]
        trans = motion["trans"]

        logger.info(
            "Loaded NPZ: frames=%d poses=%s expressions=%s trans=%s",
            poses.shape[0],
            poses.shape,
            expressions.shape,
            trans.shape,
        )

        self.retargeter.reset_root_offset()
        root_all, bone_all, morph_all = self.retargeter.retarget(poses, expressions, trans)

        total = root_all.shape[0]
        start = max(0, self.cfg.start_frame)
        end = total if self.cfg.end_frame is None else min(total, self.cfg.end_frame)
        if end <= start:
            raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total}")

        indices = list(range(start, end))
        if self.cfg.max_frames is not None:
            indices = indices[: max(0, self.cfg.max_frames)]

        if self.cfg.frames_dir:
            os.makedirs(self.cfg.frames_dir, exist_ok=True)

        if self.cfg.dry_run:
            first = indices[0]
            self._apply_frame(first, root_all[first], bone_all[first], morph_all[first])
            logger.info(
                "Dry run complete: frames=%d range=[%d,%d) bones=%d morphs=%d",
                len(indices),
                start,
                end,
                bone_all.shape[1],
                morph_all.shape[1],
            )
            return

        writer = imageio.get_writer(
            self.cfg.output_path,
            fps=self.cfg.fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )

        logger.info(
            "Rendering %d frames -> %s (%dx%d @ %dfps, camera=%s)",
            len(indices),
            self.cfg.output_path,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            self.cfg.camera_mode,
        )

        t0 = time.time()
        for j, i in enumerate(indices):
            self._apply_frame(i, root_all[i], bone_all[i], morph_all[i])
            color, _ = self.renderer.render(self.scene)
            writer.append_data(color)

            if self.cfg.frames_dir:
                frame_path = os.path.join(self.cfg.frames_dir, f"frame_{j:05d}.png")
                imageio.imwrite(frame_path, color)

            if j % 50 == 0 or j == len(indices) - 1:
                elapsed = max(1e-6, time.time() - t0)
                fps_now = (j + 1) / elapsed
                logger.info("Frame %d/%d (src=%d) | %.2f fps", j + 1, len(indices), i, fps_now)

        writer.close()
        self.renderer.delete()
        logger.info("Done in %.2fs -> %s", time.time() - t0, self.cfg.output_path)


def parse_args() -> RenderConfig:
    parser = argparse.ArgumentParser(description="Render NPZ coefficients onto rigged GLB and export MP4.")
    parser.add_argument("--npz", default=DEFAULT_NPZ_PATH, help="Input NPZ file path.")
    parser.add_argument("--glb", default=DEFAULT_GLB_PATH, help="Input GLB rig path.")
    parser.add_argument("--retarget", default=DEFAULT_RETARGET_PATH, help="Retarget map JSON path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output MP4 path.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS, help="Output FPS.")
    parser.add_argument("--camera", choices=["full", "body", "face"], default="body", help="Camera framing mode.")
    parser.add_argument("--camera-smooth", type=float, default=0.2, help="0-1 camera smoothing factor.")
    parser.add_argument("--camera-distance", type=float, default=None, help="Override camera distance.")
    parser.add_argument("--camera-y-offset", type=float, default=0.0, help="Extra camera Y offset.")
    parser.add_argument("--root-scale", type=float, default=1.0, help="Scale root translation for framing/stability.")
    parser.add_argument("--expression-clip", type=float, default=3.0, help="Clamp morph weights to [-clip, clip].")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument("--max-frames", type=int, default=None, help="Render at most N frames.")
    parser.add_argument(
        "--hide-regex",
        default=r"(icosphere|helper|marker|gizmo)",
        help="Regex to skip helper node/mesh names.",
    )
    parser.add_argument(
        "--background",
        nargs=3,
        type=float,
        default=(0.06, 0.06, 0.07),
        metavar=("R", "G", "B"),
        help="Background RGB in 0..1.",
    )
    parser.add_argument("--frames-dir", default=None, help="Optional directory to also save PNG frames.")
    parser.add_argument("--dry-run", action="store_true", help="Validate retarget/scene pipeline without rendering.")

    args = parser.parse_args()

    return RenderConfig(
        npz_path=args.npz,
        glb_path=args.glb,
        retarget_map=args.retarget,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_mode=args.camera,
        camera_smooth=args.camera_smooth,
        camera_distance=args.camera_distance,
        camera_y_offset=args.camera_y_offset,
        root_scale=args.root_scale,
        expression_clip=args.expression_clip,
        start_frame=args.start,
        end_frame=args.end,
        max_frames=args.max_frames,
        hide_regex=args.hide_regex,
        background=tuple(float(v) for v in args.background),
        frames_dir=args.frames_dir,
        dry_run=args.dry_run,
    )


def main():
    cfg = parse_args()
    renderer = OfflineRigRenderer(cfg)
    renderer.render()


if __name__ == "__main__":
    main()

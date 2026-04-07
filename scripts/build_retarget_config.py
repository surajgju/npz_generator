import json
import os
import re
import struct
from typing import Tuple, Optional, List
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
GLB_PATH = os.path.join(ROOT, "web", "assets", "head.glb")
OUT_PATH = os.path.join(ROOT, "server", "retarget_map.json")
NPZ_PATH = os.path.join(ROOT, "output", "intro_output.npz")


def read_glb(path: str) -> Tuple[dict, Optional[bytes]]:
    with open(path, "rb") as f:
        header = f.read(12)
        magic, version, length = struct.unpack("<4sII", header)
        if magic != b"glTF":
            raise ValueError("Not a GLB file")
        chunk_header = f.read(8)
        chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
        json_bytes = f.read(chunk_length)
        gltf = json.loads(json_bytes.decode("utf-8"))
        bin_bytes = None
        chunk_header = f.read(8)
        if len(chunk_header) == 8:
            chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
            if chunk_type == b"BIN\0":
                bin_bytes = f.read(chunk_length)
        return gltf, bin_bytes


def collect_bones(gltf: dict) -> list:
    skins = gltf.get("skins", [])
    if not skins:
        raise ValueError("No skins found in GLB")
    joints = skins[0].get("joints", [])
    nodes = gltf.get("nodes", [])
    bones = []
    for j in joints:
        name = nodes[j].get("name") if j < len(nodes) else None
        if not name:
            raise ValueError(f"Missing bone name for joint index {j}")
        bones.append(name)
    return bones


def collect_morphs(gltf: dict) -> list:
    meshes = gltf.get("meshes", [])
    morphs = []
    for mesh in meshes:
        names = mesh.get("extras", {}).get("targetNames") or []
        if names:
            morphs.extend(names)
        for prim in mesh.get("primitives", []):
            names = prim.get("extras", {}).get("targetNames") or []
            if names:
                morphs.extend(names)
    def collect_prefix(prefix: str, limit: Optional[int] = None) -> list:
        out = []
        pattern = rf"{prefix}\d+"
        for name in morphs:
            if re.fullmatch(pattern, name) is not None:
                out.append(name)
        if not out:
            return []
        ordered = sorted(set(out), key=lambda n: int(n[len(prefix):]))
        if limit is not None:
            ordered = [n for n in ordered if int(n[len(prefix):]) < limit]
        return ordered
    exp_names = collect_prefix("Exp", limit=100)
    if exp_names:
        return exp_names
    shape_names = collect_prefix("Shape", limit=100)
    return shape_names


def load_expression_norm_stats(path: str) -> Tuple[List[float], List[float]]:
    if not os.path.exists(path):
        raise SystemExit(f"Missing calibration NPZ at {path}")
    with np.load(path) as data:
        if "expressions" not in data:
            raise SystemExit(f"NPZ missing expressions: {path}")
        expr = data["expressions"]
    if expr.ndim != 2:
        raise SystemExit(f"Unexpected expressions shape: {expr.shape}")
    mean = np.mean(expr, axis=0).astype(np.float32)
    demeaned = expr - mean.reshape(1, -1)
    p95 = np.percentile(np.abs(demeaned), 95, axis=0).astype(np.float32)
    return mean.tolist(), p95.tolist()


def read_accessor(gltf: dict, bin_bytes: bytes, accessor_index: int) -> np.ndarray:
    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    if accessor_index >= len(accessors):
        raise ValueError(f"Accessor index out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    count = int(accessor.get("count", 0))
    component_type = int(accessor.get("componentType", 0))
    type_name = accessor.get("type", "SCALAR")
    comp_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    if component_type not in comp_map:
        raise ValueError(f"Unsupported component type: {component_type}")
    dtype = comp_map[component_type]
    comps = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }.get(type_name, 1)
    total = count * comps
    raw = None
    buffer_view_index = accessor.get("bufferView")
    if buffer_view_index is not None:
        if buffer_view_index >= len(buffer_views):
            raise ValueError(f"BufferView index out of range: {buffer_view_index}")
        buffer_view = buffer_views[buffer_view_index]
        byte_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        raw = np.frombuffer(bin_bytes, dtype=dtype, count=total, offset=byte_offset).copy()
    else:
        raw = np.zeros((total,), dtype=dtype)

    sparse = accessor.get("sparse")
    if sparse:
        scount = int(sparse.get("count", 0))
        if scount > 0:
            idx_info = sparse.get("indices", {})
            val_info = sparse.get("values", {})
            idx_view = idx_info.get("bufferView")
            val_view = val_info.get("bufferView")
            if idx_view is not None and val_view is not None:
                if idx_view >= len(buffer_views) or val_view >= len(buffer_views):
                    raise ValueError("Sparse bufferView index out of range")
                idx_bv = buffer_views[idx_view]
                val_bv = buffer_views[val_view]
                idx_offset = int(idx_bv.get("byteOffset", 0)) + int(idx_info.get("byteOffset", 0))
                val_offset = int(val_bv.get("byteOffset", 0)) + int(val_info.get("byteOffset", 0))
                idx_dtype = comp_map.get(int(idx_info.get("componentType", 0)))
                if idx_dtype is None:
                    raise ValueError("Unsupported sparse indices component type")
                idx_raw = np.frombuffer(bin_bytes, dtype=idx_dtype, count=scount, offset=idx_offset)
                val_raw = np.frombuffer(bin_bytes, dtype=dtype, count=scount * comps, offset=val_offset)
                if comps > 1:
                    raw = raw.reshape((count, comps))
                    val_raw = val_raw.reshape((scount, comps))
                    raw[idx_raw] = val_raw
                else:
                    raw[idx_raw] = val_raw

    if comps > 1:
        raw = raw.reshape((count, comps))
    return raw


def node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float32).reshape(4, 4)
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    # Build TRS matrix.
    x, y, z, w = r
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    mat = rot.copy()
    mat[0, :3] *= s[0]
    mat[1, :3] *= s[1]
    mat[2, :3] *= s[2]
    mat[:3, 3] = t
    return mat


def compute_world_matrices(gltf: dict) -> List[np.ndarray]:
    nodes = gltf.get("nodes", [])
    children = {i: set() for i in range(len(nodes))}
    parents = {i: None for i in range(len(nodes))}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            parents[c] = i
            children[i].add(c)
    world = [None for _ in nodes]
    def resolve(idx: int) -> np.ndarray:
        if world[idx] is not None:
            return world[idx]
        local = node_local_matrix(nodes[idx])
        parent = parents[idx]
        if parent is None:
            world[idx] = local
        else:
            world[idx] = resolve(parent) @ local
        return world[idx]
    for i in range(len(nodes)):
        resolve(i)
    return world


def find_eye_centers(gltf: dict, world: Optional[List[np.ndarray]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    nodes = gltf.get("nodes", [])
    if world is None:
        world = compute_world_matrices(gltf)
    left = None
    right = None
    for i, node in enumerate(nodes):
        name = (node.get("name") or "").lower()
        if "left_eye" in name and left is None:
            left = world[i][:3, 3]
        if "right_eye" in name and right is None:
            right = world[i][:3, 3]
    return left, right


def find_jaw_center(gltf: dict, world: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
    nodes = gltf.get("nodes", [])
    if world is None:
        world = compute_world_matrices(gltf)
    jaw = None
    for i, node in enumerate(nodes):
        name = (node.get("name") or "").lower()
        if name == "jaw":
            jaw = world[i][:3, 3]
            break
    if jaw is not None:
        return jaw
    for i, node in enumerate(nodes):
        name = (node.get("name") or "").lower()
        if "jaw" in name:
            return world[i][:3, 3]
    return None


def get_mesh_world_matrix(gltf: dict, world: List[np.ndarray], mesh_index: int = 0) -> np.ndarray:
    nodes = gltf.get("nodes", [])
    for i, node in enumerate(nodes):
        if node.get("mesh") == mesh_index:
            return world[i]
    return np.eye(4, dtype=np.float32)


def detect_blink_morphs(gltf: dict, bin_bytes: Optional[bytes], morphs: List[str]) -> Tuple[List[str], List[str]]:
    if bin_bytes is None:
        return [], []
    meshes = gltf.get("meshes", [])
    if not meshes:
        return [], []
    prim = meshes[0].get("primitives", [None])[0]
    if prim is None:
        return [], []
    pos_accessor = prim.get("attributes", {}).get("POSITION")
    if pos_accessor is None:
        return [], []
    base_pos = read_accessor(gltf, bin_bytes, pos_accessor).astype(np.float32)
    world = compute_world_matrices(gltf)
    mesh_world = get_mesh_world_matrix(gltf, world, mesh_index=0)
    try:
        mesh_inv = np.linalg.inv(mesh_world)
    except np.linalg.LinAlgError:
        mesh_inv = np.eye(4, dtype=np.float32)
    left_eye, right_eye = find_eye_centers(gltf, world)
    if left_eye is None or right_eye is None:
        return [], []
    left_eye = (mesh_inv @ np.array([left_eye[0], left_eye[1], left_eye[2], 1.0], dtype=np.float32))[:3]
    right_eye = (mesh_inv @ np.array([right_eye[0], right_eye[1], right_eye[2], 1.0], dtype=np.float32))[:3]
    radius = 0.035
    left_mask = np.linalg.norm(base_pos - left_eye.reshape(1, 3), axis=1) <= radius
    right_mask = np.linalg.norm(base_pos - right_eye.reshape(1, 3), axis=1) <= radius
    if not np.any(left_mask) or not np.any(right_mask):
        return [], []
    targets = prim.get("targets", [])
    names = meshes[0].get("extras", {}).get("targetNames") or []
    if not names:
        names = prim.get("extras", {}).get("targetNames") or []
    valid = set(morphs)
    scored = []
    for i, target in enumerate(targets):
        if i >= len(names):
            continue
        name = names[i]
        if name not in valid:
            continue
        targ_pos = target.get("POSITION")
        if targ_pos is None:
            continue
        try:
            delta = read_accessor(gltf, bin_bytes, targ_pos).astype(np.float32)
            mag = np.linalg.norm(delta, axis=1)
            scored.append((name, float(np.mean(mag[left_mask])), float(np.mean(mag[right_mask]))))
        except Exception:
            continue
    if not scored:
        return [], []
    left_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    right_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
    left_names = [name for name, _, _ in left_sorted[:2]]
    right_names = [name for name, _, _ in right_sorted[:2]]
    return left_names, right_names


def detect_mouth_morphs(gltf: dict, bin_bytes: Optional[bytes], morphs: List[str]) -> List[str]:
    if bin_bytes is None:
        return []
    meshes = gltf.get("meshes", [])
    if not meshes:
        return []
    prim = meshes[0].get("primitives", [None])[0]
    if prim is None:
        return []
    pos_accessor = prim.get("attributes", {}).get("POSITION")
    if pos_accessor is None:
        return []
    base_pos = read_accessor(gltf, bin_bytes, pos_accessor).astype(np.float32)
    world = compute_world_matrices(gltf)
    mesh_world = get_mesh_world_matrix(gltf, world, mesh_index=0)
    try:
        mesh_inv = np.linalg.inv(mesh_world)
    except np.linalg.LinAlgError:
        mesh_inv = np.eye(4, dtype=np.float32)
    jaw_center = find_jaw_center(gltf, world)
    if jaw_center is None:
        return []
    jaw_center = (mesh_inv @ np.array([jaw_center[0], jaw_center[1], jaw_center[2], 1.0], dtype=np.float32))[:3]
    base_min = base_pos.min(axis=0)
    base_max = base_pos.max(axis=0)
    head_height = float(max(1e-6, base_max[1] - base_min[1]))
    radius = max(0.03, head_height * 0.08)
    center = jaw_center.copy()
    center[1] += head_height * 0.02
    mouth_mask = np.linalg.norm(base_pos - center.reshape(1, 3), axis=1) <= radius
    if not np.any(mouth_mask):
        return []
    targets = prim.get("targets", [])
    names = meshes[0].get("extras", {}).get("targetNames") or []
    if not names:
        names = prim.get("extras", {}).get("targetNames") or []
    valid = set(morphs)
    scored = []
    for i, target in enumerate(targets):
        if i >= len(names):
            continue
        name = names[i]
        if name not in valid:
            continue
        targ_pos = target.get("POSITION")
        if targ_pos is None:
            continue
        try:
            delta = read_accessor(gltf, bin_bytes, targ_pos).astype(np.float32)
            mag = np.linalg.norm(delta, axis=1)
            scored.append((name, float(np.mean(mag[mouth_mask]))))
        except Exception:
            continue
    if not scored:
        return []
    top_k = min(8, len(scored))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_k]]


def main():
    if not os.path.exists(GLB_PATH):
        raise SystemExit(f"Missing GLB at {GLB_PATH}")
    gltf, bin_bytes = read_glb(GLB_PATH)
    bones = collect_bones(gltf)
    morphs = collect_morphs(gltf)
    expression_norm_mean, expression_norm_p95 = load_expression_norm_stats(NPZ_PATH)
    blink_left, blink_right = detect_blink_morphs(gltf, bin_bytes, morphs)
    mouth_morphs = detect_mouth_morphs(gltf, bin_bytes, morphs)

    root_bone = "pelvis" if "pelvis" in bones else ("root" if "root" in bones else bones[0])
    # Explicit SMPL-X joint -> GLB bone mapping (vertex-group names).
    bone_map = {name: name for name in bones}
    # Keep map only for SMPL-X joints we know.
    smplx_joints = {
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "leye",
        "reye",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    }
    # Handle eye bone naming in this GLB.
    if "left_eye_smplhf" in bones:
        bone_map["leye"] = "left_eye_smplhf"
    if "right_eye_smplhf" in bones:
        bone_map["reye"] = "right_eye_smplhf"
    # Remove any non-SMPL-X entries.
    bone_map = {k: v for k, v in bone_map.items() if k in smplx_joints}

    config = {
        "bones": bones,                       # List of all bone names in the target GLB
        "morphs": morphs,                     # List of all morph target (blendshape) names in the target GLB
        "root_bone": root_bone,               # The bone used as the translation root (usually pelvis)
        "bone_map": bone_map,                 # Mapping from SMPL-X joint names to GLB bone names
        "rest_align_quat": {},                # Legacy alignment quaternions
        "rest_local_quat": {},                # Current basis alignment for retargeting rotations
        "global_align_quat": [0.0, 0.0, 0.0, 1.0], # Global rotation to align SMPL-X space to GLB space
        "buffer_seconds": 3.0,                # How many seconds of animation to buffer on the client
        "blink_morphs_left": blink_left,       # Detected blendshapes involved in left eye blink
        "blink_morphs_right": blink_right,     # Detected blendshapes involved in right eye blink
        "blink_strength": 0.6,                # Target weight for blink animation
        "blink_interval_sec": [3.0, 6.0],     # Random range for time between blinks
        "blink_duration_sec": 0.12,           # Duration of a single blink
        "saccade_interval_sec": [1.0, 3.0],   # Random range for time between small eye movements
        "saccade_yaw_deg": 2.0,               # Max yaw magnitude for eye saccades
        "saccade_pitch_deg": 1.0,             # Max pitch magnitude for eye saccades
        "expression_norm_mean": expression_norm_mean, # Mean expression values from calibration
        "expression_norm_p95": expression_norm_p95,   # 95th percentile values for normalization
        "expression_norm_target": 0.35,       # Target weight for normalized expressions
        "expression_norm_min": 0.2,           # Minimum allowed scaling factor for normalization
        "expression_norm_max": 3.0,           # Maximum allowed scaling factor for normalization
        "expression_smooth_alpha": 0.35,      # Alpha for EMA smoothing of expressions (low = smoother)
        "expression_scale": 2.0,              # Global multiplier for all expression blendshapes
        "expression_clip": 1.2,               # Maximum absolute weight for any blendshape
        "expression_target": 0.35,            # Target average weight if normalization is not used
        "expression_gain_min": 0.3,           # Min gain for dynamic expression scaling
        "expression_gain_max": 2.0,           # Max gain for dynamic expression scaling
        "expression_offset_strength": 1.0,    # How much of the calibration mean to subtract (0 to 1)
        "expression_fallback_gain": 1.5,      # Gain applied if expression signal is very weak
        "expression_fallback_thresh": 0.05,   # Threshold below which fallback gain is applied
        "jaw_scale": 1.0,                     # Multiplier for raw jaw rotation
        "mouth_morphs": mouth_morphs,         # Detected blendshapes involved in mouth/speech
        "mouth_safety": {                     # Safeguards to prevent mouth/jaw clipping or artifacts
            "jaw_close_thresh": 0.12,         # Threshold to force jaw closed
            "mouth_energy_thresh": 0.25,      # Min energy required to allow mouth movement
            "reduce_factor": 0.55,            # How much to reduce mouth morphs if energy is low
        },
        "client_morph_smooth_alpha": 0.6,     # Alpha for smoothing blendshapes on the client-side
    }

    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote retarget map to {OUT_PATH}")
    print(f"bones={len(bones)} morphs={len(morphs)} root_bone={root_bone}")


if __name__ == "__main__":
    main()

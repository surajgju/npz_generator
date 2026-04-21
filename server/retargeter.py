import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from streamsettings import (
    EYE_BROW_MAX_ABS,
    EXPRESSION_MAX_ABS,
    MOUTH_MAX_ABS,
    EXPRESSION_NORM_MIN,
    EXPRESSION_NORM_MAX,
    EXPRESSION_TARGET,
    EXPRESSION_GAIN_MIN,
    EXPRESSION_GAIN_MAX,
    EXPRESSION_OFFSET_STRENGTH,
    EXPRESSION_FALLBACK_GAIN,
    EXPRESSION_FALLBACK_THRESH,
)

logger = logging.getLogger(__name__)

SMPLX_JOINT_NAMES = [
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
    # left hand
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
    # right hand
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
]

ALIASES = {
    # SMPL-X eye names to SMPL-X exported GLB bone names.
    "leye": "left_eye_smplhf",
    "reye": "right_eye_smplhf",
}


def _normalize(name: str) -> str:
    """Normalize a string by converting to lowercase and removing non-alphanumeric characters."""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (T,3) to quaternion (T,4) in [x,y,z,w]."""
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angles
    small = angles < 1e-8
    axis = np.zeros_like(axis_angle)
    axis[~small[..., 0]] = axis_angle[~small[..., 0]] / angles[~small[..., 0]]
    sin_half = np.sin(half)
    quat = np.concatenate([axis * sin_half, np.cos(half)], axis=-1)
    # For tiny angles, return identity
    if np.any(small):
        quat[small[..., 0]] = np.array([0, 0, 0, 1], dtype=quat.dtype)
    return quat


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Perform Hamilton product of two quaternions (q = q1 * q2).
    Expects quaternions in [x, y, z, w] format with shape (..., 4).
    """
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x2, y2, z2, w2 = np.split(q2, 4, axis=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.concatenate([x, y, z, w], axis=-1)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion (negates the vector part [x, y, z])."""
    qc = q.copy()
    qc[..., :3] *= -1.0
    return qc


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q. q: (...,4), v: (...,3)."""
    qvec = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


class SmplxRetargeter:
    """
    Handles retargeting of SMPL-X pose, expression, and translation data to a target GLB model.
    It manages bone mapping, coordinate system alignments, and expression/viseme processing.
    """
    def __init__(self, config_path: str, morph_path: Optional[str] = None):
        """
        Initialize the retargeter with a JSON configuration file.
        
        Args:
            config_path: Path to the retarget_map.json defining bone/morph relationships.
            morph_path: Optional path to additional morph mapping data.
        """
        self.config_path = config_path
        self.morph_path = morph_path
        self.bones: List[str] = []
        self.morphs: List[str] = []
        self.root_bone: Optional[str] = None
        self.rest_align_quat: Dict[str, np.ndarray] = {}
        self.rest_local_quat: Dict[str, np.ndarray] = {}
        self._bone_map: Dict[int, int] = {}
        self._root_offset: Optional[np.ndarray] = None
        self._logged_root_offset = False
        self._root_bone_idx: Optional[int] = None
        self._global_align_quat = None
        self._zero_global_orient: bool = False
        self._direct_expr = False
        self._expr_indices: List[int] = []
        self._max_expr_index = -1
        self._expression_scale = 1.0
        self._expression_clip = None
        self._expression_max_abs = float(EXPRESSION_MAX_ABS)
        self._expression_norm_p95 = None
        self._expression_norm_mean = None
        self._expression_norm_target = 1.0
        self._expression_norm_min = float(EXPRESSION_NORM_MIN)
        self._expression_norm_max = float(EXPRESSION_NORM_MAX)
        self._expression_norm_scales = None
        self._logged_norm_scales = False
        self._logged_norm_morph = False
        self._expression_offset = None
        self._logged_expression_offset = False
        self._expression_smooth_alpha = None
        self._expression_smooth_prev = None
        self._expression_target = float(EXPRESSION_TARGET)
        self._expression_gain_min = float(EXPRESSION_GAIN_MIN)
        self._expression_gain_max = float(EXPRESSION_GAIN_MAX)
        self._expression_offset_strength = float(EXPRESSION_OFFSET_STRENGTH)
        self._expression_fallback_gain = float(EXPRESSION_FALLBACK_GAIN)
        self._expression_fallback_thresh = float(EXPRESSION_FALLBACK_THRESH)
        self._jaw_scale = 1.0
        self._jaw_idx = SMPLX_JOINT_NAMES.index("jaw")
        self._buffer_seconds = None
        self._blink_morphs_left = []
        self._blink_morphs_right = []
        self._blink_strength = 0.6
        self._blink_interval_sec = (3.0, 6.0)
        self._blink_duration_sec = 0.12
        self._saccade_interval_sec = (1.0, 3.0)
        self._saccade_yaw_deg = 2.0
        self._saccade_pitch_deg = 1.0
        self._mouth_morphs = []
        self._mouth_morph_indices: List[int] = []
        self._upper_face_morph_indices: List[int] = []
        self._mouth_viseme_map = {}
        self._mouth_gain = 1.0
        self._mouth_clip = None
        self._mouth_max_abs = float(MOUTH_MAX_ABS)
        self._eye_brow_max_abs = float(EYE_BROW_MAX_ABS)
        self._mouth_smooth_alpha = None
        self._mouth_safety = {
            "jaw_close_thresh": 0.12,
            "mouth_energy_thresh": 0.25,
            "reduce_factor": 0.55,
        }
        self._client_morph_smooth_alpha = None
        self.last_expr_abs_max = 0.0
        self.last_expr_gain = 1.0
        self.last_morph_abs_max = 0.0
        self.last_mouth_abs_max = 0.0
        self.last_mouth_energy = 0.0
        self.last_mouth_gain = 1.0
        self.last_clip_count = 0
        self.last_jaw_raw_mag = 0.0
        self.last_jaw_mag = 0.0
        self.last_fallback_gain = 1.0
        self._load_config()
        self._prepare_direct_expression_map()
        self._prepare_mouth_indices()
        self._prepare_upper_face_indices()
        if self.root_bone and self.root_bone in self.bones:
            self._root_bone_idx = self.bones.index(self.root_bone)

        # Global alignment (default: identity). Overridden via retarget_map.json key: global_align_quat.
        self._global_align_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _load_config(self):
        """
        Load and parse the retarget configuration JSON.
        Sets up internal parameters for scaling, smoothing, and normalization of pose/expressions.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"retarget map not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            data = json.load(f)
        self.bones = data.get("bones", [])
        self.morphs = data.get("morphs", [])
        self.root_bone = data.get("root_bone")
        rest = data.get("rest_align_quat", {})
        for k, v in rest.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape == (4,):
                self.rest_align_quat[k] = arr
        logger.info("Retarget map: rest_align_quat=%d", len(self.rest_align_quat))
        rest_local = data.get("rest_local_quat", {})
        for k, v in rest_local.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape == (4,):
                self.rest_local_quat[k] = arr
        if self.rest_local_quat:
            logger.info("Retarget map: rest_local_quat=%d", len(self.rest_local_quat))
        user_map = data.get("bone_map", {}) or {}
        self._build_bone_map(user_map)
        if "global_align_quat" in data:
            try:
                ga = np.asarray(data.get("global_align_quat"), dtype=np.float32)
                if ga.shape == (4,):
                    self._global_align_quat = ga
            except Exception:
                pass
        if "zero_global_orient" in data:
            try:
                self._zero_global_orient = bool(data.get("zero_global_orient", False))
            except Exception:
                self._zero_global_orient = False
        if "expression_scale" in data:
            try:
                self._expression_scale = float(data.get("expression_scale", 1.0))
            except Exception:
                self._expression_scale = 1.0
        if "expression_clip" in data:
            try:
                clip = float(data.get("expression_clip"))
                if clip > 0:
                    self._expression_clip = clip
            except Exception:
                self._expression_clip = None
        if "expression_norm_p95" in data:
            try:
                p95 = data.get("expression_norm_p95")
                if isinstance(p95, list) and p95:
                    self._expression_norm_p95 = np.array(p95, dtype=np.float32)
            except Exception:
                self._expression_norm_p95 = None
        if "expression_norm_mean" in data:
            try:
                mean = data.get("expression_norm_mean")
                if isinstance(mean, list) and mean:
                    self._expression_norm_mean = np.array(mean, dtype=np.float32)
            except Exception:
                self._expression_norm_mean = None
        if "expression_norm_target" in data:
            try:
                self._expression_norm_target = float(data.get("expression_norm_target", 1.0))
            except Exception:
                self._expression_norm_target = 1.0
        if "expression_norm_min" in data:
            try:
                self._expression_norm_min = float(data.get("expression_norm_min", 0.2))
            except Exception:
                self._expression_norm_min = 0.2
        if "expression_norm_max" in data:
            try:
                self._expression_norm_max = float(data.get("expression_norm_max", 2.5))
            except Exception:
                self._expression_norm_max = 2.5
        if "expression_smooth_alpha" in data:
            try:
                alpha = float(data.get("expression_smooth_alpha"))
                if alpha > 0:
                    self._expression_smooth_alpha = max(0.0, min(1.0, alpha))
            except Exception:
                self._expression_smooth_alpha = None
        if "expression_target" in data:
            try:
                self._expression_target = float(data.get("expression_target", 0.35))
            except Exception:
                self._expression_target = 0.35
        if "expression_gain_min" in data:
            try:
                self._expression_gain_min = float(data.get("expression_gain_min", 0.3))
            except Exception:
                self._expression_gain_min = 0.3
        if "expression_gain_max" in data:
            try:
                self._expression_gain_max = float(data.get("expression_gain_max", 3.0))
            except Exception:
                self._expression_gain_max = 3.0
        if "expression_offset_strength" in data:
            try:
                self._expression_offset_strength = float(data.get("expression_offset_strength", 0.5))
            except Exception:
                self._expression_offset_strength = 0.5
        if "expression_fallback_gain" in data:
            try:
                self._expression_fallback_gain = float(data.get("expression_fallback_gain", 3.0))
            except Exception:
                self._expression_fallback_gain = 3.0
        if "expression_fallback_thresh" in data:
            try:
                self._expression_fallback_thresh = float(data.get("expression_fallback_thresh", 0.05))
            except Exception:
                self._expression_fallback_thresh = 0.05
        if "jaw_scale" in data:
            try:
                self._jaw_scale = float(data.get("jaw_scale", 1.0))
            except Exception:
                self._jaw_scale = 1.0
        if "mouth_gain" in data:
            try:
                self._mouth_gain = float(data.get("mouth_gain", 1.0))
            except Exception:
                self._mouth_gain = 1.0
        if "mouth_clip" in data:
            try:
                clip = float(data.get("mouth_clip"))
                if clip > 0:
                    self._mouth_clip = clip
            except Exception:
                self._mouth_clip = None
        if "mouth_smooth_alpha" in data:
            try:
                alpha = float(data.get("mouth_smooth_alpha"))
                if alpha > 0:
                    self._mouth_smooth_alpha = max(0.0, min(1.0, alpha))
            except Exception:
                self._mouth_smooth_alpha = None
        if "buffer_seconds" in data:
            try:
                self._buffer_seconds = float(data.get("buffer_seconds"))
            except Exception:
                self._buffer_seconds = None
        if "blink_morphs_left" in data:
            vals = data.get("blink_morphs_left") or []
            if isinstance(vals, list):
                self._blink_morphs_left = vals
        if "blink_morphs_right" in data:
            vals = data.get("blink_morphs_right") or []
            if isinstance(vals, list):
                self._blink_morphs_right = vals
        if "blink_strength" in data:
            try:
                self._blink_strength = float(data.get("blink_strength"))
            except Exception:
                self._blink_strength = 0.6
        if "blink_interval_sec" in data:
            vals = data.get("blink_interval_sec") or []
            if isinstance(vals, list) and len(vals) == 2:
                self._blink_interval_sec = (float(vals[0]), float(vals[1]))
        if "blink_duration_sec" in data:
            try:
                self._blink_duration_sec = float(data.get("blink_duration_sec"))
            except Exception:
                self._blink_duration_sec = 0.12
        if "saccade_interval_sec" in data:
            vals = data.get("saccade_interval_sec") or []
            if isinstance(vals, list) and len(vals) == 2:
                self._saccade_interval_sec = (float(vals[0]), float(vals[1]))
        if "saccade_yaw_deg" in data:
            try:
                self._saccade_yaw_deg = float(data.get("saccade_yaw_deg"))
            except Exception:
                self._saccade_yaw_deg = 2.0
        if "saccade_pitch_deg" in data:
            try:
                self._saccade_pitch_deg = float(data.get("saccade_pitch_deg"))
            except Exception:
                self._saccade_pitch_deg = 1.0
        if "mouth_morphs" in data:
            vals = data.get("mouth_morphs") or []
            if isinstance(vals, list):
                self._mouth_morphs = vals
        if "mouth_viseme_map" in data:
            vals = data.get("mouth_viseme_map") or {}
            if isinstance(vals, dict):
                self._mouth_viseme_map = {str(k): list(v) for k, v in vals.items() if isinstance(v, list)}
        if "mouth_safety" in data:
            vals = data.get("mouth_safety") or {}
            if isinstance(vals, dict):
                self._mouth_safety.update(vals)
        if "client_morph_smooth_alpha" in data:
            try:
                alpha = float(data.get("client_morph_smooth_alpha"))
                if alpha > 0:
                    self._client_morph_smooth_alpha = max(0.0, min(1.0, alpha))
            except Exception:
                self._client_morph_smooth_alpha = None

    def _build_bone_map(self, user_map: Dict[str, str]):
        """
        Map SMPL-X joint indices to target GLB bone indices.
        Uses a combination of explicit user mapping, aliases, and exact name matching.
        """
        bone_index = {_normalize(name): i for i, name in enumerate(self.bones)}
        smplx_index = {name: i for i, name in enumerate(SMPLX_JOINT_NAMES)}
        # normalize user_map to smplx name -> gltf bone name
        normalized_user_map: Dict[str, str] = {}
        for k, v in user_map.items():
            normalized_user_map[_normalize(k)] = v
        for smplx_name, smplx_idx in smplx_index.items():
            target_name = None
            # user-provided mapping
            key_norm = _normalize(smplx_name)
            if key_norm in normalized_user_map:
                target_name = normalized_user_map[key_norm]
            # alias fallback
            if target_name is None and smplx_name in ALIASES:
                target_name = ALIASES[smplx_name]
            # exact match fallback
            if target_name is None:
                if key_norm in bone_index:
                    target_name = self.bones[bone_index[key_norm]]
            if target_name is None:
                continue
            target_norm = _normalize(target_name)
            if target_norm in bone_index:
                self._bone_map[smplx_idx] = bone_index[target_norm]
        mapped = len(self._bone_map)
        missing = [name for name, idx in smplx_index.items() if idx not in self._bone_map]
        if mapped == 0:
            logger.warning("Retarget map: no SMPL-X joints mapped to glTF bones. Check bone names.")
        logger.info(
            "Retarget map: gltf bones=%d, smplx mapped=%d/%d",
            len(self.bones),
            mapped,
            len(SMPLX_JOINT_NAMES),
        )
        if missing:
            logger.info("Retarget map: missing joints (sample)=%s", missing[:8])

    def _prepare_direct_expression_map(self):
        """Enable direct Exp###/Shape### mapping when morphs are SMPL-X expression blendshapes."""
        if not self.morphs:
            return
        exp_indices: List[int] = []
        prefix = None
        for name in self.morphs:
            m = re.fullmatch(r"(Exp|Shape)(\d+)", name)
            if not m:
                return
            if prefix is None:
                prefix = m.group(1)
            elif m.group(1) != prefix:
                return
            exp_indices.append(int(m.group(2)))
        if not exp_indices:
            return
        self._direct_expr = True
        self._expr_indices = exp_indices
        self._max_expr_index = max(exp_indices)
        logger.info(
            "Direct expression mapping enabled (%s###): morphs=%d max_index=%d",
            prefix,
            len(exp_indices),
            self._max_expr_index,
        )

    def _prepare_mouth_indices(self):
        if not self.morphs or not self._mouth_morphs:
            return
        indices = []
        for name in self._mouth_morphs:
            try:
                idx = self.morphs.index(name)
            except ValueError:
                continue
            indices.append(idx)
        self._mouth_morph_indices = indices
        if indices:
            logger.info(
                "Mouth morphs mapped: count=%d names=%s",
                len(indices),
                self._mouth_morphs,
            )

    def _prepare_upper_face_indices(self):
        if not self.morphs:
            self._upper_face_morph_indices = []
            return
        mouth_index_set = set(self._mouth_morph_indices)
        self._upper_face_morph_indices = [
            idx for idx in range(len(self.morphs)) if idx not in mouth_index_set
        ]

    def reset_root_offset(self):
        """Reset the root translation offset and expression history. Used when starting a new stream."""
        self._root_offset = None
        self._logged_root_offset = False
        self._expression_offset = None
        self._logged_expression_offset = False
        self._expression_smooth_prev = None
        self.last_expr_abs_max = 0.0
        self.last_expr_gain = 1.0
        self.last_morph_abs_max = 0.0
        self.last_mouth_abs_max = 0.0
        self.last_mouth_energy = 0.0
        self.last_mouth_gain = 1.0
        self.last_clip_count = 0
        self.last_jaw_raw_mag = 0.0
        self.last_jaw_mag = 0.0
        self.last_fallback_gain = 1.0

    def anim_init_header(self, fps: int) -> Dict[str, object]:
        """
        Generate the character description header for 'anim_init' messages.
        Provides the client with bone/morph lists and configuration for playback.
        """
        header = {
            "type": "anim_init",
            "fps": fps,
            "bones": self.bones,
            "morphs": self.morphs,
            "dtype": "f32",
            "hasRoot": True,
            "space": "local",
            "rootBone": self.root_bone,
        }
        if self._buffer_seconds is not None:
            header["bufferSeconds"] = self._buffer_seconds
        if self._blink_morphs_left or self._blink_morphs_right:
            header["blink"] = {
                "left": self._blink_morphs_left,
                "right": self._blink_morphs_right,
                "strength": self._blink_strength,
                "intervalSec": list(self._blink_interval_sec),
                "durationSec": self._blink_duration_sec,
            }
        header["saccade"] = {
            "intervalSec": list(self._saccade_interval_sec),
            "yawDeg": self._saccade_yaw_deg,
            "pitchDeg": self._saccade_pitch_deg,
        }
        if self._mouth_morphs:
            header["mouth"] = {
                "morphs": self._mouth_morphs,
                "visemeMap": self._mouth_viseme_map,
                "jawCloseThresh": float(self._mouth_safety.get("jaw_close_thresh", 0.12)),
                "mouthEnergyThresh": float(self._mouth_safety.get("mouth_energy_thresh", 0.25)),
                "reduceFactor": float(self._mouth_safety.get("reduce_factor", 0.55)),
                "gain": float(self._mouth_gain),
                "clip": float(min(self._mouth_max_abs, self._mouth_clip))
                if self._mouth_clip is not None
                else float(self._mouth_max_abs),
                "smoothAlpha": None if self._mouth_smooth_alpha is None else float(self._mouth_smooth_alpha),
            }
        if self._client_morph_smooth_alpha is not None:
            header["morphSmoothAlpha"] = float(self._client_morph_smooth_alpha)
        return header

    def retarget(
        self,
        poses: np.ndarray,
        expressions: Optional[np.ndarray],
        trans: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert SMPL-X data into the target model's coordinate frame and format.
        
        Args:
            poses: (T, 165 or 156) array of joint axis-angles.
            expressions: (T, 10 or 100) array of SMPL-X expression coefficients.
            trans: (T, 3) array of root translations.
            
        Returns:
            Tuple containing:
            - root_pos: (T, 3) root translation offsets.
            - bone_quats: (T, N_BONES, 4) quaternions for all character bones.
            - morphs: (T, N_MORPHS) weights for blendshapes.
        """
        if poses.ndim != 2 or poses.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected poses shape {poses.shape}")
        t = poses.shape[0]
        joint_count = poses.shape[1] // 3
        joints = poses.reshape(t, joint_count, 3)
        nbones = len(self.bones)
        bone_quats = np.zeros((t, nbones, 4), dtype=np.float32)
        bone_quats[..., 3] = 1.0
        for smplx_idx, bone_idx in self._bone_map.items():
            if smplx_idx >= joint_count or bone_idx >= nbones:
                continue
            # Zero out the global orient (pelvis/joint-0) to prevent EMAGE's training-space
            # body orientation from tilting the GLB avatar backward.
            if self._zero_global_orient and smplx_idx == 0:
                bone_quats[:, bone_idx] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                continue
            axis_angle = joints[:, smplx_idx]
            raw_mag = float(np.mean(np.linalg.norm(axis_angle, axis=1))) if axis_angle.size else 0.0
            if smplx_idx == self._jaw_idx and self._jaw_scale != 1.0:
                axis_angle = axis_angle * self._jaw_scale
            if smplx_idx == self._jaw_idx:
                try:
                    self.last_jaw_raw_mag = raw_mag
                    self.last_jaw_mag = float(np.mean(np.linalg.norm(axis_angle, axis=1)))
                except Exception:
                    self.last_jaw_raw_mag = 0.0
                    self.last_jaw_mag = 0.0
            q = _axis_angle_to_quat(axis_angle)
            bone_name = self.bones[bone_idx]
            # First map SMPL-X rotation into glTF parent axes via global align.
            if self._global_align_quat is not None:
                ga = np.broadcast_to(self._global_align_quat, q.shape).astype(np.float32)
                q = _quat_mul(ga, _quat_mul(q, _quat_conj(ga)))

            # Convert parent-axes rotation into glTF bone local axes using rest pose basis.
            rest_local = self.rest_local_quat.get(bone_name)
            if rest_local is not None:
                rest_q = np.broadcast_to(rest_local, q.shape).astype(np.float32)
                q = _quat_mul(_quat_conj(rest_q), _quat_mul(q, rest_q))
            else:
                # Fallback: older rest_align_quat behavior.
                align = self.rest_align_quat.get(bone_name)
                if align is not None:
                    align_q = np.broadcast_to(align, q.shape)
                    q = _quat_mul(align_q, _quat_mul(q, _quat_conj(align_q)))
            bone_quats[:, bone_idx] = q.astype(np.float32)
        root_pos = np.zeros((t, 3), dtype=np.float32)
        if trans is not None and trans.shape[1] == 3:
            trans_f = trans.astype(np.float32)
            if self._root_offset is None:
                self._root_offset = trans_f[0].copy()
                if not self._logged_root_offset:
                    logger.info("Root offset set: %s", np.round(self._root_offset, 4).tolist())
                    self._logged_root_offset = True
            root_pos = trans_f - self._root_offset
            # Apply global orientation to root translation as well.
            if self._global_align_quat is not None:
                ga = np.broadcast_to(self._global_align_quat, (t, 4)).astype(np.float32)
                root_pos = _quat_rotate_vec(ga, root_pos)
            # lock Y, keep XZ
            root_pos[:, 1] = 0.0
        nmorphs = len(self.morphs)
        morphs = np.zeros((t, nmorphs), dtype=np.float32)
        used_norm = False
        if expressions is not None and self._direct_expr:
            expr = expressions.astype(np.float32)
            if self._expression_offset is None:
                if (
                    self._expression_norm_mean is not None
                    and self._expression_norm_mean.shape[0] == expr.shape[1]
                ):
                    self._expression_offset = self._expression_norm_mean
                    if not self._logged_expression_offset:
                        max_abs = float(np.max(np.abs(self._expression_offset))) if self._expression_offset.size else 0.0
                        logger.info("Expression offset set (config): max_abs=%.4f", max_abs)
                        self._logged_expression_offset = True
                else:
                    self._expression_offset = np.mean(expr, axis=0)
                    if not self._logged_expression_offset:
                        max_abs = float(np.max(np.abs(self._expression_offset))) if self._expression_offset.size else 0.0
                        logger.info("Expression offset set: max_abs=%.4f", max_abs)
                        self._logged_expression_offset = True
            if self._expression_offset is not None:
                expr = expr - (self._expression_offset * float(self._expression_offset_strength))
            exp_dim = expr.shape[1] if expr.ndim == 2 else 0
            if self._expression_norm_p95 is not None and exp_dim > 0:
                p95 = self._expression_norm_p95
                if p95.shape[0] == exp_dim:
                    if self._expression_norm_scales is None or self._expression_norm_scales.shape[0] != exp_dim:
                        denom = np.maximum(p95, 1e-6)
                        scales = self._expression_norm_target / denom
                        scales = np.clip(scales, self._expression_norm_min, self._expression_norm_max)
                        self._expression_norm_scales = scales.astype(np.float32)
                        if not self._logged_norm_scales:
                            logger.info(
                                "Expression norm scales: min=%.3f mean=%.3f max=%.3f",
                                float(scales.min()),
                                float(scales.mean()),
                                float(scales.max()),
                            )
                            self._logged_norm_scales = True
                    expr = expr * self._expression_norm_scales.reshape(1, -1)
                    used_norm = True
            if not used_norm:
                try:
                    self.last_expr_abs_max = float(np.max(np.abs(expr))) if expr.size else 0.0
                except Exception:
                    self.last_expr_abs_max = 0.0
                gain = 1.0
                if self.last_expr_abs_max > 1e-6:
                    gain = self._expression_target / self.last_expr_abs_max
                if gain < self._expression_gain_min:
                    gain = self._expression_gain_min
                if gain > self._expression_gain_max:
                    gain = self._expression_gain_max
                self.last_expr_gain = float(gain)
                expr = expr * float(gain)
            else:
                self.last_expr_gain = 1.0
            expr = expr * float(self._expression_scale)
            if (
                self._expression_smooth_alpha is not None
                and expr.ndim == 2
                and expr.shape[0] > 0
                and expr.shape[1] > 0
            ):
                alpha = float(self._expression_smooth_alpha)
                prev = self._expression_smooth_prev
                if prev is None or prev.shape[0] != expr.shape[1]:
                    prev = expr[0].copy()
                smoothed = np.empty_like(expr)
                for i in range(expr.shape[0]):
                    prev = alpha * expr[i] + (1.0 - alpha) * prev
                    smoothed[i] = prev
                self._expression_smooth_prev = prev
                expr = smoothed
            try:
                self.last_expr_abs_max = float(np.max(np.abs(expr))) if expr.size else 0.0
            except Exception:
                self.last_expr_abs_max = 0.0
            if expr.ndim == 2 and expr.shape[1] > 0:
                for i, exp_idx in enumerate(self._expr_indices):
                    if exp_idx < exp_dim:
                        morphs[:, i] = expr[:, exp_idx]
            if self._mouth_morph_indices:
                mouth_vals = morphs[:, self._mouth_morph_indices]
                if mouth_vals.size:
                    self.last_mouth_abs_max = float(np.max(np.abs(mouth_vals)))
                    self.last_mouth_energy = float(np.mean(np.abs(mouth_vals)))
        else:
            self.last_mouth_abs_max = 0.0
            self.last_mouth_energy = 0.0
        try:
            self.last_morph_abs_max = float(np.max(np.abs(morphs))) if morphs.size else 0.0
        except Exception:
            self.last_morph_abs_max = 0.0
        # If morph signal is too small, apply fallback gain (then clip).
        self.last_fallback_gain = 1.0
        if morphs.size and self.last_morph_abs_max < float(self._expression_fallback_thresh):
            gain = float(self._expression_fallback_gain)
            morphs = morphs * gain
            self.last_fallback_gain = gain
            try:
                self.last_morph_abs_max = float(np.max(np.abs(morphs)))
            except Exception:
                self.last_morph_abs_max = 0.0
        clip_count = 0
        if self._mouth_morph_indices and morphs.size:
            mouth_vals = morphs[:, self._mouth_morph_indices]
            if mouth_vals.size:
                mouth_gain = float(self._mouth_gain)
                pre_gain_abs_max = float(np.max(np.abs(mouth_vals)))
                if pre_gain_abs_max > 1e-6:
                    target_gain = (self._mouth_max_abs * 0.95) / pre_gain_abs_max
                    mouth_gain = max(0.0, min(mouth_gain, target_gain))
                self.last_mouth_gain = float(mouth_gain)
                if mouth_gain != 1.0:
                    morphs[:, self._mouth_morph_indices] *= mouth_gain
                mouth_clip = float(self._mouth_max_abs)
                if self._mouth_clip is not None:
                    mouth_clip = min(mouth_clip, float(self._mouth_clip))
                clipped = np.clip(
                    morphs[:, self._mouth_morph_indices],
                    -mouth_clip,
                    mouth_clip,
                )
                clip_count += int(
                    np.count_nonzero(
                        np.abs(morphs[:, self._mouth_morph_indices] - clipped) > 1e-6
                    )
                )
                morphs[:, self._mouth_morph_indices] = clipped
                self.last_mouth_abs_max = float(
                    np.max(np.abs(morphs[:, self._mouth_morph_indices]))
                )
                self.last_mouth_energy = float(
                    np.mean(np.abs(morphs[:, self._mouth_morph_indices]))
                )
        else:
            self.last_mouth_gain = 1.0
        if self._upper_face_morph_indices and morphs.size:
            clipped = np.clip(
                morphs[:, self._upper_face_morph_indices],
                -self._eye_brow_max_abs,
                self._eye_brow_max_abs,
            )
            clip_count += int(
                np.count_nonzero(
                    np.abs(morphs[:, self._upper_face_morph_indices] - clipped) > 1e-6
                )
            )
            morphs[:, self._upper_face_morph_indices] = clipped
        expression_clip = float(self._expression_max_abs)
        if self._expression_clip is not None:
            expression_clip = min(expression_clip, float(self._expression_clip))
        if morphs.size:
            clipped = np.clip(morphs, -expression_clip, expression_clip)
            clip_count += int(np.count_nonzero(np.abs(morphs - clipped) > 1e-6))
            morphs = clipped
        self.last_clip_count = clip_count
        try:
            self.last_morph_abs_max = float(np.max(np.abs(morphs))) if morphs.size else 0.0
        except Exception:
            self.last_morph_abs_max = 0.0
        if self._mouth_morph_indices and morphs.size:
            self.last_mouth_abs_max = float(np.max(np.abs(morphs[:, self._mouth_morph_indices])))
            self.last_mouth_energy = float(np.mean(np.abs(morphs[:, self._mouth_morph_indices])))
        if used_norm and not self._logged_norm_morph and morphs.size:
            logger.info("Expression norm morph_abs_max=%.4f", float(np.max(np.abs(morphs))))
            self._logged_norm_morph = True
        return root_pos, bone_quats, morphs

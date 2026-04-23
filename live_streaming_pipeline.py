import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from typing import Optional, Dict
import logging
from npz_logging import setup_logging
from streamsettings import BASE_MOTION_FPS, DEFAULT_OVERLAP_SEC

setup_logging()
logger = logging.getLogger(__name__)

# Import models
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel

# For simulating streaming audio playback
try:
    import sounddevice as sd
except ImportError:
    sd = None

class LiveMotionGenerator:
    """
    Streaming-ready generator that accepts audio chunks and returns SMPL-X coefficients.
    Models are loaded once and kept on device to avoid cold starts.
    """
    def __init__(self, device=None, model_folder="./models/", overlap_sec: float = DEFAULT_OVERLAP_SEC):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Initializing LiveMotionGenerator on %s...", self.device)

        face_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(self.device)
        upper_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(self.device)
        lower_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(self.device)
        hands_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(self.device)
        global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(self.device)

        self.motion_vq = EmageVQModel(
            face_model=face_motion_vq,
            upper_model=upper_motion_vq,
            lower_model=lower_motion_vq,
            hands_model=hands_motion_vq,
            global_model=global_motion_ae,
        ).to(self.device)
        self.motion_vq.eval()

        self.model = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(self.device)
        self.model.eval()

        self.sr = self.model.cfg.audio_sr
        self.pose_fps = self.model.cfg.pose_fps
        self.speaker_id = torch.zeros(1, 1).long().to(self.device)
        self.overlap_sec = float(overlap_sec)
        self._prev_overlap: Optional[np.ndarray] = None
        self._prev_last_pose: Optional[np.ndarray] = None
        self._prev_last_expr: Optional[np.ndarray] = None
        self._prev_last_trans: Optional[np.ndarray] = None
        self.target_fps = BASE_MOTION_FPS
        self.last_process_timings = {
            "infer_ms": 0.0,
            "resample_ms": 0.0,
        }

        logger.info("Generator ready. Audio SR: %s, Pose FPS: %s", self.sr, self.pose_fps)

    def _prepare_audio(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio_chunk is None or len(audio_chunk) == 0:
            return np.array([], dtype=np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        if audio_chunk.dtype != np.float32:
            if np.issubdtype(audio_chunk.dtype, np.integer):
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_chunk = audio_chunk.astype(np.float32)
        if sample_rate != self.sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=self.sr)
        return audio_chunk

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        chunk_id: Optional[str] = None,
        overlap_source_audio: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Convert a short audio chunk into SMPL-X coefficients.
        Returns dict with poses/expressions/trans at the configured base motion FPS.
        """
        audio_chunk = self._prepare_audio(audio_chunk, sample_rate)
        if len(audio_chunk) == 0:
            logger.warning("Chunk %s: empty audio after prepare", chunk_id)
            return None
        if overlap_source_audio is None:
            overlap_source = audio_chunk
        else:
            overlap_source = self._prepare_audio(overlap_source_audio, sample_rate)
            if len(overlap_source) == 0:
                overlap_source = audio_chunk

        overlap_samples = int(self.overlap_sec * self.sr)
        has_prev = self._prev_overlap is not None and len(self._prev_overlap) > 0
        if has_prev:
            audio_chunk = np.concatenate([self._prev_overlap, audio_chunk], axis=0)
            overlap_history = np.concatenate([self._prev_overlap, overlap_source], axis=0)
        else:
            overlap_history = overlap_source
        logger.debug(
            "Chunk %s: audio_len=%.3fs (samples=%d) overlap_sec=%.3f has_prev=%s",
            chunk_id,
            len(audio_chunk) / float(self.sr),
            len(audio_chunk),
            self.overlap_sec,
            has_prev,
        )

        if overlap_samples > 0 and len(overlap_history) >= overlap_samples:
            self._prev_overlap = overlap_history[-overlap_samples:].copy()
        else:
            self._prev_overlap = overlap_history.copy()

        audio_ts = torch.from_numpy(audio_chunk).float().to(self.device).unsqueeze(0)

        infer_started_ns = time.perf_counter_ns()
        with torch.no_grad():
            trans_zero = torch.zeros(1, 1, 3).to(self.device)
            latent_dict = self.model.inference(audio_ts, self.speaker_id, self.motion_vq, masked_motion=None, mask=None)

            face_latent = latent_dict["rec_face"] if self.model.cfg.lf > 0 and self.model.cfg.cf == 0 else None
            upper_latent = latent_dict["rec_upper"] if self.model.cfg.lu > 0 and self.model.cfg.cu == 0 else None
            hands_latent = latent_dict["rec_hands"] if self.model.cfg.lh > 0 and self.model.cfg.ch == 0 else None
            lower_latent = latent_dict["rec_lower"] if self.model.cfg.ll > 0 and self.model.cfg.cl == 0 else None

            face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if self.model.cfg.cf > 0 else None
            upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if self.model.cfg.cu > 0 else None
            hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if self.model.cfg.ch > 0 else None
            lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if self.model.cfg.cl > 0 else None

            all_pred = self.motion_vq.decode(
                face_latent=face_latent,
                upper_latent=upper_latent,
                lower_latent=lower_latent,
                hands_latent=hands_latent,
                face_index=face_index,
                upper_index=upper_index,
                lower_index=lower_index,
                hands_index=hands_index,
                get_global_motion=True,
                ref_trans=trans_zero[:, 0],
            )
        infer_ms = (time.perf_counter_ns() - infer_started_ns) / 1_000_000.0

        motion_pred = all_pred["motion_axis_angle"]
        t = motion_pred.shape[1]
        motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
        face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
        trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)

        resample_started_ns = time.perf_counter_ns()
        upsample = BASE_MOTION_FPS // self.pose_fps
        if upsample > 1:
            from emage_utils.motion_io import time_upsample_numpy
            motion_pred = time_upsample_numpy(motion_pred, upsample)
            face_pred = time_upsample_numpy(face_pred, upsample)
            trans_pred = time_upsample_numpy(trans_pred, upsample)
        resample_ms = (time.perf_counter_ns() - resample_started_ns) / 1_000_000.0
        self.last_process_timings = {
            "infer_ms": infer_ms,
            "resample_ms": resample_ms,
        }

        overlap_frames = int(round(self.overlap_sec * self.target_fps))
        if has_prev and overlap_frames > 0 and overlap_frames < motion_pred.shape[0]:
            motion_pred = motion_pred[overlap_frames:]
            face_pred = face_pred[overlap_frames:]
            trans_pred = trans_pred[overlap_frames:]

        # Boundary continuity: preserve the last emitted pose at chunk joins, then
        # quickly fade to the newly predicted chunk to avoid visual popping.
        if has_prev and self._prev_last_pose is not None and motion_pred.shape[0] > 0:
            k = min(max(2, overlap_frames), motion_pred.shape[0])
            if k > 0:
                if k > 1:
                    alpha = np.linspace(0.0, 1.0, k, endpoint=True, dtype=np.float32).reshape(-1, 1)
                else:
                    alpha = np.array([[1.0]], dtype=np.float32)
                motion_pred[:k] = self._prev_last_pose.reshape(1, -1) * (1.0 - alpha) + motion_pred[:k] * alpha
                face_pred[:k] = self._prev_last_expr.reshape(1, -1) * (1.0 - alpha) + face_pred[:k] * alpha
                trans_pred[:k] = self._prev_last_trans.reshape(1, -1) * (1.0 - alpha) + trans_pred[:k] * alpha

        if motion_pred.shape[0] > 0:
            self._prev_last_pose = motion_pred[-1].copy()
            self._prev_last_expr = face_pred[-1].copy()
            self._prev_last_trans = trans_pred[-1].copy()

        logger.debug(
            "Chunk %s: output frames=%d overlap_frames=%d",
            chunk_id,
            int(motion_pred.shape[0]),
            overlap_frames,
        )

        return {
            "poses": motion_pred,
            "expressions": face_pred,
            "trans": trans_pred,
            "betas": np.zeros(10, dtype=np.float32),
        }


class SmplxVertexStreamer:
    """Compute SMPL-X vertices from coefficients on the server."""
    def __init__(self, device=None, model_path="models", gender="neutral"):
        import smplx
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_expression_coeffs=100,
        ).to(self.device)
        self.faces = self.model.faces

    def vertices_from_coeffs(self, poses: np.ndarray, expressions: np.ndarray, trans: np.ndarray, betas: Optional[np.ndarray] = None) -> np.ndarray:
        if betas is None:
            betas = np.zeros(10, dtype=np.float32)
        T = poses.shape[0]
        poses_t = torch.tensor(poses, dtype=torch.float32, device=self.device)
        trans_t = torch.tensor(trans, dtype=torch.float32, device=self.device)
        expr_t = torch.tensor(expressions, dtype=torch.float32, device=self.device)
        beta_t = torch.tensor(betas, dtype=torch.float32, device=self.device).unsqueeze(0).expand(T, -1)
        with torch.no_grad():
            output = self.model(
                betas=beta_t,
                global_orient=poses_t[:, :3],
                body_pose=poses_t[:, 3:66],
                jaw_pose=poses_t[:, 66:69],
                leye_pose=poses_t[:, 69:72],
                reye_pose=poses_t[:, 72:75],
                left_hand_pose=poses_t[:, 75:120],
                right_hand_pose=poses_t[:, 120:165],
                expression=expr_t,
                transl=trans_t,
            )
        return output.vertices.detach().cpu().numpy()


# Backward-compatible alias for older imports
GeminiLivePipeline = LiveMotionGenerator


def simulate_live_stream(audio_path, chunk_duration_sec=1.0):
    """
    Simulates a live streaming environment by reading an audio file, 
    processing it in chunks, and queuing the generated coefficients.
    """
    logger.info("--- Starting Streaming Simulation for %s ---", audio_path)
    pipeline = LiveMotionGenerator()
    
    # Load entire audio to simulate incoming network stream
    audio_full, sr = librosa.load(audio_path, sr=pipeline.sr)
    chunk_size = int(sr * chunk_duration_sec)
    
    logger.info("Loaded audio: %.2fs. Chunking by %.2fs.", len(audio_full) / sr, chunk_duration_sec)
    
    # Stats tracking
    total_frames = 0
    start_time = time.time()
    
    # To demonstrate live playback alongside generation
    if sd is not None:
        logger.info("Sounddevice detected. Simulating synchronized playback...")
        # Since sounddevice plays async, we write chunks to an async output stream
        stream = sd.OutputStream(samplerate=sr, channels=1)
        stream.start()

    chunk_idx = 0
    for i in range(0, len(audio_full), chunk_size):
        chunk = audio_full[i:i + chunk_size]
        
        # Pad short chunks at the end
        if len(chunk) < chunk_size:
             chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
             
        # 1. GENERATION: Parse audio chunk -> NPZ coefficients
        gen_start = time.time()
        coefficients = pipeline.process_audio_chunk(chunk, pipeline.sr, chunk_idx)
        gen_time = time.time() - gen_start
        
        num_frames = coefficients['poses'].shape[0]
        total_frames += num_frames
        
        logger.info("Chunk %03d: %d frames in %.3fs (RT factor %.2fx)", chunk_idx, num_frames, gen_time, chunk_duration_sec / gen_time)
        
        # 2. PLAYBACK (if sounddevice is installed to simulate "live" sync)
        if sd is not None:
            # We block slightly here just to simulate playing the chunk
            # In a real async environment like Gemini API, audio would play
            # via a separate thread while coefficients stream to the 3D renderer.
            stream.write(np.expand_dims(chunk, axis=1))

        # 3. RENDER / FORWARDING
        # Here we would send 'coefficients' dict via WebSockets to a web 
        # renderer or append to a live pyrender visualization.
        # For demonstration, we simply yield or process the coefficients.
        
        chunk_idx += 1

    if sd is not None:
        stream.stop()
        stream.close()
        
    logger.info("--- Simulation Complete ---")
    logger.info("Total processed frames: %d", total_frames)
    logger.info("Total time elapsed: %.2fs", time.time() - start_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="./input/audio1.wav")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Simulate chunk length in seconds")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
         logger.warning("Default audio %s not found. Provide a valid file via --audio.", args.audio)
    else:
         simulate_live_stream(args.audio, args.chunk_size)

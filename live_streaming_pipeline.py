import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from collections import deque

# Import models
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel

# For simulating streaming audio playback
try:
    import sounddevice as sd
except ImportError:
    sd = None

class GeminiLivePipeline:
    """
    A unified, streaming-ready pipeline designed to accept audio chunks 
    (e.g., PCM/Base64 from Gemini Multimodal Live API) and generate 
    SMPL-X coefficients on the fly in chunks.
    """
    def __init__(self, device=None, model_folder="./models/"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing GeminiLivePipeline on {self.device}...")
        
        # Load VQ Models
        face_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(self.device)
        upper_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(self.device)
        lower_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(self.device)
        hands_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(self.device)
        global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(self.device)
        
        self.motion_vq = EmageVQModel(
            face_model=face_motion_vq, upper_model=upper_motion_vq,
            lower_model=lower_motion_vq, hands_model=hands_motion_vq,
            global_model=global_motion_ae
        ).to(self.device)
        self.motion_vq.eval()

        # Load Audio Model
        self.model = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(self.device)
        self.model.eval()
        
        self.sr = self.model.cfg.audio_sr
        self.pose_fps = self.model.cfg.pose_fps
        self.speaker_id = torch.zeros(1, 1).long().to(self.device)
        
        print(f"Pipeline ready. Expected Audio SR: {self.sr}, Output Pose FPS: {self.pose_fps}")

    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process a chunk of audio (numpy array) into SMPL-X coefficients.
        Args:
            audio_chunk: 1D numpy array of audio PCM data sampled at self.sr (16000)
        Returns:
            dict containing:
                - poses: (T, 165)
                - expressions: (T, 100)
                - trans: (T, 3)
                - betas: (10,)
        """
        if len(audio_chunk) == 0:
            return None
            
        # Convert to tensor and add batch/channel dims
        audio_ts = torch.from_numpy(audio_chunk).float().to(self.device).unsqueeze(0)
        
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
                face_latent=face_latent, upper_latent=upper_latent, 
                lower_latent=lower_latent, hands_latent=hands_latent,
                face_index=face_index, upper_index=upper_index, 
                lower_index=lower_index, hands_index=hands_index,
                get_global_motion=True, ref_trans=trans_zero[:,0]
            )
            
        # Extract the shapes and sequence length
        motion_pred = all_pred["motion_axis_angle"]
        t = motion_pred.shape[1]
        
        motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
        face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
        trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)
        
        # Upsample logic (if model output is lower FPS than target 30fps)
        upsample = 30 // self.pose_fps
        if upsample > 1:
            from emage_utils.motion_io import time_upsample_numpy
            motion_pred = time_upsample_numpy(motion_pred, upsample)
            face_pred = time_upsample_numpy(face_pred, upsample)
            trans_pred = time_upsample_numpy(trans_pred, upsample)
            
        return {
            "poses": motion_pred,
            "expressions": face_pred,
            "trans": trans_pred,
            "betas": np.zeros(10) # default neutral betas
        }


def simulate_live_stream(audio_path, chunk_duration_sec=1.0):
    """
    Simulates a live streaming environment by reading an audio file, 
    processing it in chunks, and queuing the generated coefficients.
    """
    print(f"\n--- Starting Streaming Simulation for {audio_path} ---")
    pipeline = GeminiLivePipeline()
    
    # Load entire audio to simulate incoming network stream
    audio_full, sr = librosa.load(audio_path, sr=pipeline.sr)
    chunk_size = int(sr * chunk_duration_sec)
    
    print(f"Loaded audio: {len(audio_full)/sr:.2f}s. Chunking by {chunk_duration_sec}s.")
    
    # Stats tracking
    total_frames = 0
    start_time = time.time()
    
    # To demonstrate live playback alongside generation
    if sd is not None:
        print("\nSounddevice detected! Simulating synchronized playback...")
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
        coefficients = pipeline.process_audio_chunk(chunk)
        gen_time = time.time() - gen_start
        
        num_frames = coefficients['poses'].shape[0]
        total_frames += num_frames
        
        print(f"[Chunk {chunk_idx:03d}] Generated {num_frames} frames in {gen_time:.3f}s. (Realtime factor: {chunk_duration_sec/gen_time:.2f}x)")
        
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
        
    print(f"\n--- Simulation Complete ---")
    print(f"Total processed frames: {total_frames}")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="./input/audio1.wav")
    parser.add_argument("--chunk_size", type=float, default=2.0, help="Simulate chunk length in seconds")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
         print(f"Warning: Default audio {args.audio} not found. Please provide a valid file via --audio.")
    else:
         simulate_live_stream(args.audio, args.chunk_size)

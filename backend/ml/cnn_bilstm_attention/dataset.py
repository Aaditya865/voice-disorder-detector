import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_SAMPLES = 32768  # 2.048 seconds at 16000 Hz
TARGET_FRAMES = 128     # Fixed time frames


def compute_log_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Computes a standardized Log-Mel Spectrogram of shape (1, 128, 128).
    
    Steps:
    1. Ensure 1D mono float32.
    2. Pad (circular wrap) or crop audio to exactly TARGET_SAMPLES (32,768 samples).
    3. Compute Mel Spectrogram (n_mels=128, n_fft=1024, hop_length=256).
    4. Convert to dB scale (power_to_db).
    5. Crop time frames to TARGET_FRAMES (128).
    6. Instance normalization (zero-mean, unit-variance).
    """
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Standardize length
    if len(audio) < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="wrap")
    else:
        # Center crop if longer than target
        start_idx = (len(audio) - TARGET_SAMPLES) // 2
        audio = audio[start_idx : start_idx + TARGET_SAMPLES]

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Crop to exactly TARGET_FRAMES
    log_mel = log_mel[:, :TARGET_FRAMES]

    # Normalize to zero mean and unit variance
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    norm_mel = (log_mel - mean) / (std + 1e-6)

    # Return shape (1, 128, 128) float32
    return np.expand_dims(norm_mel.astype(np.float32), axis=0)


def precompute_and_cache_spectrograms(
    metadata_rows: List[Dict[str, str]],
    base_dir: Path,
    cache_path: Path
) -> Dict[str, np.ndarray]:
    """
    Precomputes Log-Mel Spectrograms for metadata rows and caches them to disk.
    """
    cache = {}
    if cache_path.exists():
        try:
            cache = joblib.load(cache_path)
            logger.info(f"Loaded existing spectrogram cache with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load spectrogram cache ({e}), recomputing.")
            cache = {}

    missing_paths = [r["processed_file"] for r in metadata_rows if r["processed_file"] not in cache]

    if missing_paths:
        logger.info(f"Generating Log-Mel Spectrograms for {len(missing_paths)} recordings...")
        for idx, rel_p in enumerate(missing_paths, 1):
            full_p = base_dir / rel_p
            audio, sr = sf.read(str(full_p))
            spec = compute_log_mel_spectrogram(audio, sr=16000)
            cache[rel_p] = spec

            if idx % 100 == 0 or idx == len(missing_paths):
                logger.info(f"  [{idx}/{len(missing_paths)}] spectrograms generated.")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path)
        logger.info(f"Spectrogram cache saved with {len(cache)} total entries.")

    return cache


class LaryngealSpectrogramDataset(Dataset):
    """
    PyTorch Dataset yielding (spectrogram, label, speaker_id, file_path).
    """

    def __init__(
        self,
        metadata_rows: List[Dict[str, str]],
        spectrogram_cache: Dict[str, np.ndarray],
        class_to_idx: Dict[str, int]
    ):
        self.rows = metadata_rows
        self.cache = spectrogram_cache
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.rows[idx]
        rel_p = row["processed_file"]
        spec_np = self.cache[rel_p]  # shape (1, 128, 128)
        label_idx = self.class_to_idx[row["class"]]
        speaker_id = row["speaker_id"]

        spec_tensor = torch.from_numpy(spec_np).float()
        return spec_tensor, label_idx, speaker_id, rel_p

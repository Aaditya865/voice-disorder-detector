import io
from pathlib import Path
from typing import Tuple, Union, Dict, Any, List
import numpy as np
import soundfile as sf
import librosa

from ml.config import SAMPLE_RATE, MIN_DURATION_SECONDS, MAX_DURATION_SECONDS, SUPPORTED_EXTENSIONS


class AudioPreprocessor:
    """
    Standardized audio preprocessing pipeline for Wav2Vec 2.0.
    Ensures identical processing between model training and real-time inference.
    """

    def __init__(self, target_sr: int = SAMPLE_RATE, max_duration: float = MAX_DURATION_SECONDS):
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.max_samples = int(target_sr * max_duration)
        self.min_samples = int(target_sr * MIN_DURATION_SECONDS)

    def load_and_preprocess(self, audio_source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an audio file or bytes buffer, converts to mono, resamples to target_sr,
        normalizes amplitude, and returns the preprocessed waveform with metadata.
        """
        if isinstance(audio_source, (str, Path)):
            path = Path(audio_source)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported audio format: {path.suffix}. Supported: {SUPPORTED_EXTENSIONS}")
            # Load with soundfile or librosa
            try:
                audio, orig_sr = sf.read(str(path))
            except Exception:
                audio, orig_sr = librosa.load(str(path), sr=None, mono=False)
        elif isinstance(audio_source, (bytes, io.BytesIO)):
            bio = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
            bio.seek(0)
            try:
                audio, orig_sr = sf.read(bio)
            except Exception:
                bio.seek(0)
                audio, orig_sr = librosa.load(bio, sr=None, mono=False)
        else:
            raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

        # Convert to floating point 32-bit
        audio = audio.astype(np.float32)

        # Convert multi-channel (stereo) to mono
        if audio.ndim > 1:
            # If shape is (channels, samples) or (samples, channels)
            if audio.shape[0] < audio.shape[1]:
                audio = np.mean(audio, axis=0)
            else:
                audio = np.mean(audio, axis=1)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Resample to 16,000 Hz if needed
        if orig_sr != self.target_sr:
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=self.target_sr)

        # Check duration
        duration = len(audio) / self.target_sr
        if len(audio) < self.min_samples:
            raise ValueError(
                f"Audio duration ({duration:.2f}s) is too short. Minimum required is {MIN_DURATION_SECONDS}s."
            )

        # Peak normalization to prevent clipping and balance energy
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak

        # Truncate if exceeds max duration to preserve memory and consistency
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]

        metadata = {
            "original_sample_rate": int(orig_sr),
            "target_sample_rate": int(self.target_sr),
            "duration_seconds": round(float(duration), 3),
            "processed_samples": int(len(audio)),
            "channels": 1
        }

        return audio, metadata

    def extract_waveform_envelope(self, audio: np.ndarray, num_points: int = 150) -> List[float]:
        """
        Extracts downsampled peak envelope for real waveform visualization on the frontend.
        Calculates true RMS/peak per segment from actual audio samples.
        """
        if len(audio) == 0:
            return [0.0] * num_points
        step = max(1, len(audio) // num_points)
        envelope = []
        for i in range(num_points):
            start = i * step
            end = min(len(audio), start + step)
            if start >= len(audio):
                envelope.append(0.0)
            else:
                chunk = audio[start:end]
                # Use max absolute amplitude in chunk
                val = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
                envelope.append(round(val, 4))
        return envelope

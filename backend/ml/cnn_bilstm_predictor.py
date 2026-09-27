"""
CNN-BiLSTM Attention Inference Wrapper
=======================================
Provides single-file, real-time prediction using the pre-trained
2D CNN + BiLSTM + Attention checkpoints stored in backend/models/cnn_bilstm_attention/.

Pipeline:
  audio (bytes / ndarray)
    → AudioPreprocessor  (resample to 16 kHz, mono, peak-normalize)
    → _extract_windows   (overlapping 2.048-s windows for soft voting)
    → compute_log_mel_spectrogram (128×128 log-Mel per window)
    → AudioCNNBilstmAttention forward pass
    → softmax + soft-vote across windows
    → calibrated confidence dict
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import io

import numpy as np
import torch
import torch.nn.functional as F

from ml.cnn_bilstm_attention.model import AudioCNNBilstmAttention
from ml.cnn_bilstm_attention.dataset import compute_log_mel_spectrogram
from ml.config import TARGET_CLASSES, SAMPLE_RATE

logger = logging.getLogger(__name__)

# ── Checkpoint selection ─────────────────────────────────────────────────────
# We default to Config_A (Baseline Light) as it is the best generalization
# checkpoint from the final held-out test evaluation.
# Set AUGMENTED=True to use the augmented training variant instead.
AUGMENTED = True

_BASE = Path(__file__).resolve().parent.parent.parent / "backend" / "models"
CHECKPOINT_PATH: Path = (
    _BASE / "cnn_bilstm_attention_augmented" / "Config_A_Augmented_best.pt"
    if AUGMENTED
    else _BASE / "cnn_bilstm_attention" / "Config_A_Baseline_Light_best.pt"
)

# ── Model hyper-parameters (must match Config A used during training) ────────
MODEL_KWARGS = dict(
    n_classes=6,
    cnn_channels=[32, 64, 128],
    lstm_hidden=64,
    lstm_layers=1,
    dense_dropout=0.3,
    cnn_dropout=0.1,
)

# Sorted class order used during training (alphabetical = label-encoder order)
CLASSES = ["Laryngitis", "Laryngozele", "Normal", "Vocal_fold_paralysis", "Vocal_fold_polyp", "Vox_senilis"]

# ── Windowing for soft voting ─────────────────────────────────────────────────
# The CNN was trained on 2.048-s windows (TARGET_SAMPLES=32768 @ 16 kHz).
# We stride at 1.024 s (50% overlap) to capture the full sustained vowel.
CNN_WINDOW_SAMPLES: int = 32768          # 2.048 s
CNN_HOP_SAMPLES:    int = 16384          # 1.024 s (50 % overlap)
CNN_MIN_WIN_SAMPLES: int = int(SAMPLE_RATE * 0.5)   # skip very short tails


def _cnn_extract_windows(audio: np.ndarray) -> List[np.ndarray]:
    """Splits 1-D audio into overlapping 2.048-s windows for CNN soft voting."""
    n = len(audio)
    if n <= CNN_WINDOW_SAMPLES:
        pad = CNN_WINDOW_SAMPLES - n
        return [np.pad(audio, (0, pad), mode="wrap")]

    windows, start = [], 0
    while start + CNN_MIN_WIN_SAMPLES <= n:
        chunk = audio[start: start + CNN_WINDOW_SAMPLES]
        if len(chunk) < CNN_WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, CNN_WINDOW_SAMPLES - len(chunk)), mode="wrap")
        windows.append(chunk)
        start += CNN_HOP_SAMPLES

    return windows or [np.pad(audio[:CNN_WINDOW_SAMPLES],
                               (0, max(0, CNN_WINDOW_SAMPLES - len(audio))),
                               mode="wrap")]


class CnnBilstmPredictor:
    """
    Real-time inference wrapper for the 2D CNN + BiLSTM + Attention model.
    Singleton pattern — call CnnBilstmPredictor.get_instance() to reuse.
    """

    _instance: Optional["CnnBilstmPredictor"] = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AudioCNNBilstmAttention] = None
        self.is_loaded: bool = False
        self._load()

    @classmethod
    def get_instance(cls) -> "CnnBilstmPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> bool:
        """Loads the CNN checkpoint from disk."""
        if not CHECKPOINT_PATH.exists():
            logger.warning(f"CNN checkpoint not found: {CHECKPOINT_PATH}")
            self.is_loaded = False
            return False
        try:
            self.model = AudioCNNBilstmAttention(**MODEL_KWARGS)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            # Training script wraps weights inside 'model_state_dict' key
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
                # Log any useful metadata from checkpoint
                if "best_val_loss" in checkpoint:
                    logger.info(f"Checkpoint val_loss={checkpoint['best_val_loss']:.4f}, epoch={checkpoint.get('best_epoch','?')}")
            else:
                state = checkpoint  # flat state_dict
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"CNN-BiLSTM loaded from {CHECKPOINT_PATH.name} on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load CNN checkpoint: {e}")
            self.is_loaded = False
            return False

    def predict(self, waveform: np.ndarray) -> Tuple[str, float, Dict[str, float], int]:
        """
        Run CNN soft-voting inference on a pre-processed mono 16 kHz waveform.

        Returns:
            predicted_class (str)
            confidence      (float, 0-1)
            class_probs     (dict  class → probability)
            num_windows     (int)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("CNN model is not loaded.")

        windows = _cnn_extract_windows(waveform)
        prob_vectors: List[np.ndarray] = []

        with torch.no_grad():
            for win in windows:
                spec = compute_log_mel_spectrogram(win, sr=SAMPLE_RATE)   # (1,128,128)
                x = torch.from_numpy(spec).unsqueeze(0).float().to(self.device)  # (1,1,128,128)
                logits, _ = self.model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (6,)
                prob_vectors.append(probs)

        avg_probs = np.mean(prob_vectors, axis=0)  # soft vote
        pred_idx  = int(np.argmax(avg_probs))
        predicted_class = CLASSES[pred_idx]
        confidence      = float(avg_probs[pred_idx])
        class_probs     = {c: round(float(p), 4) for c, p in zip(CLASSES, avg_probs)}

        return predicted_class, confidence, class_probs, len(windows)

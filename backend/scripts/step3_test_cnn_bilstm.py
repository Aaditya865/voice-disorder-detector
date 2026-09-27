"""
STEP 3 VERIFICATION: CNN-BiLSTM-Attention vs Random Forest
Run from the project root:
  python backend/scripts/step3_test_cnn_bilstm.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier
from ml.config import MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH, SAMPLE_RATE
from ml.predictor import CONFIDENCE_TEMPERATURE, _extract_windows, WINDOW_SAMPLES, HOP_SAMPLES
from ml.cnn_bilstm_predictor import CnnBilstmPredictor

print("Loading models...", flush=True)
preprocessor = AudioPreprocessor()

# RF
extractor = Wav2VecFeatureExtractor.get_instance()
clf = VoiceDisorderClassifier()
rf_ok = clf.load(MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH)
print(f"  Random Forest    : {'OK' if rf_ok else 'NOT FOUND'}")

# CNN
cnn = CnnBilstmPredictor.get_instance()
print(f"  CNN-BiLSTM-Attn  : {'OK' if cnn.is_loaded else 'NOT FOUND'}")

if not rf_ok and not cnn.is_loaded:
    print("[ERROR] Neither model loaded. Exiting.")
    sys.exit(1)

sample_dir = BASE_DIR / "sample_recordings"
wav_files = sorted(sample_dir.glob("*.wav"))

print(f"\n{'File':<42}  {'RF+T (fallback)':>16}  {'CNN-BiLSTM':>12}  {'Winner?':>9}")
print("-" * 95)

rf_classes = clf.classes_ if rf_ok else []

for wav_path in wav_files:
    waveform, meta = preprocessor.load_and_preprocess(str(wav_path))

    # ── RF + Steps 1+2 ───────────────────────────────────────────────────────
    rf_label, rf_conf = "N/A", 0.0
    if rf_ok:
        windows = _extract_windows(waveform)
        vecs = []
        for win in windows:
            feats = extractor.extract_features(win)
            raw = clf.predict_proba(feats, temperature=1.0)[0]
            vecs.append(np.array([raw.get(c, 0.0) for c in rf_classes]))
        avg = np.mean(vecs, axis=0)
        log_a = np.log(np.clip(avg, 1e-9, 1.0)) / CONFIDENCE_TEMPERATURE
        log_a -= log_a.max()
        cal = np.exp(log_a) / np.exp(log_a).sum()
        rf_label = rf_classes[np.argmax(cal)]
        rf_conf  = float(cal.max()) * 100

    # ── CNN-BiLSTM ───────────────────────────────────────────────────────────
    cnn_label, cnn_conf = "N/A", 0.0
    if cnn.is_loaded:
        cnn_label, raw_conf, _, _ = cnn.predict(waveform)
        cnn_conf = raw_conf * 100

    agree = "AGREE" if rf_label == cnn_label else "DIFFER"
    winner = "CNN" if cnn_conf > rf_conf else "RF "

    print(f"{wav_path.name:<42}  {rf_label:<16} {rf_conf:>5.1f}%  {cnn_label:<12} {cnn_conf:>5.1f}%  [{winner}] {agree}")

print("\n[DONE] Step 3 complete.")
print("  The predictor now uses CNN-BiLSTM-Attention as the primary model.")
print("  Random Forest (Steps 1+2) remains as automatic fallback.")

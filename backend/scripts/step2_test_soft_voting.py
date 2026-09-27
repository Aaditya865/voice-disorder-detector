"""
STEP 2 VERIFICATION: Multi-Window Soft Voting vs Single-Pass
Run from the project root:
  python backend/scripts/step2_test_soft_voting.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier
from ml.config import MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH, SAMPLE_RATE
from ml.predictor import (
    CONFIDENCE_TEMPERATURE, WINDOW_SAMPLES, HOP_SAMPLES, _extract_windows
)

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
preprocessor = AudioPreprocessor()
extractor = Wav2VecFeatureExtractor.get_instance()
clf = VoiceDisorderClassifier()
ok = clf.load(MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH)
if not ok:
    print("[ERROR] Model not found.")
    sys.exit(1)
print(f"[OK] Classes: {clf.classes_}")

sample_dir = BASE_DIR / "sample_recordings"
wav_files = sorted(sample_dir.glob("*.wav"))
if not wav_files:
    print(f"[ERROR] No WAV files found in {sample_dir}")
    sys.exit(1)

classes = clf.classes_
print(f"\n{'File':<42} {'Windows':>7}  {'Single-pass':>12}  {'Soft-voted':>12}  {'Gain':>6}")
print("-" * 90)

for wav_path in wav_files:
    waveform, meta = preprocessor.load_and_preprocess(str(wav_path))

    # ── Single-pass (old method) ──────────────────────────────────────────────
    feats = extractor.extract_features(waveform)
    sp_raw = clf.predict_proba(feats, temperature=1.0)[0]
    sp_vec = np.array([sp_raw.get(c, 0.0) for c in classes])
    # apply temperature
    log_p = np.log(np.clip(sp_vec, 1e-9, 1.0)) / CONFIDENCE_TEMPERATURE
    log_p -= log_p.max()
    sp_cal = np.exp(log_p) / np.exp(log_p).sum()
    sp_winner = classes[np.argmax(sp_cal)]
    sp_conf   = float(sp_cal.max()) * 100

    # ── Multi-window soft voting (new method) ─────────────────────────────────
    windows = _extract_windows(waveform)
    prob_vecs = []
    for win in windows:
        wf = extractor.extract_features(win)
        pd = clf.predict_proba(wf, temperature=1.0)[0]
        prob_vecs.append(np.array([pd.get(c, 0.0) for c in classes]))

    avg_vec = np.mean(prob_vecs, axis=0)
    log_a = np.log(np.clip(avg_vec, 1e-9, 1.0)) / CONFIDENCE_TEMPERATURE
    log_a -= log_a.max()
    sv_cal = np.exp(log_a) / np.exp(log_a).sum()
    sv_winner = classes[np.argmax(sv_cal)]
    sv_conf   = float(sv_cal.max()) * 100

    gain = sv_conf - sp_conf
    same = "(same)" if sv_winner == sp_winner else "(CHANGED!)"

    print(f"{wav_path.name:<42} {len(windows):>7}  {sp_conf:>10.1f}%  {sv_conf:>10.1f}%  {gain:>+5.1f}%  {same}")

print(f"\n[DONE] Step 2 complete. Multi-window soft voting active.")
print(f"   Window size : {WINDOW_SAMPLES/SAMPLE_RATE:.1f}s | Hop: {HOP_SAMPLES/SAMPLE_RATE:.1f}s (50% overlap)")
print(f"   Temperature : T={CONFIDENCE_TEMPERATURE}")

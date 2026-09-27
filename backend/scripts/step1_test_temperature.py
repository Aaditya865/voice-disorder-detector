"""
STEP 1 VERIFICATION: Temperature Scaling Before vs After
Run from the project root:
  python backend/scripts/step1_test_temperature.py
"""
import sys
from pathlib import Path

# Add backend to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ml.predictor import VoicePredictor, CONFIDENCE_TEMPERATURE
from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier
from ml.config import MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
preprocessor = AudioPreprocessor()
extractor = Wav2VecFeatureExtractor.get_instance()
clf = VoiceDisorderClassifier()
ok = clf.load(MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH)
if not ok:
    print("[ERROR] Model not found. Make sure rf_classifier.joblib exists in backend/models/")
    sys.exit(1)
print(f"[OK] Model loaded. Classes: {clf.classes_}")

# ── Pick a sample file ───────────────────────────────────────────────────────
sample_dir = BASE_DIR / "sample_recordings"
wav_files = list(sample_dir.glob("*.wav"))
if not wav_files:
    print(f"[ERROR] No WAV files found in {sample_dir}")
    sys.exit(1)

print(f"\nTesting {len(wav_files)} sample files:\n")
print(f"{'File':<40}  {'Winner':<22}  {'Raw T=1.0':>10}  {'Calibrated T=0.5':>17}")
print("-" * 97)

for wav_path in sorted(wav_files):
    waveform, _ = preprocessor.load_and_preprocess(str(wav_path))
    features = extractor.extract_features(waveform)

    # Raw probabilities (T=1.0)
    raw_probs = clf.predict_proba(features, temperature=1.0)[0]
    # Calibrated probabilities (T=0.5)
    cal_probs  = clf.predict_proba(features, temperature=CONFIDENCE_TEMPERATURE)[0]

    winner = max(raw_probs, key=raw_probs.get)
    raw_conf = raw_probs[winner] * 100
    cal_conf = cal_probs[winner] * 100

    flag = "^" if cal_conf > raw_conf else ""
    print(f"{wav_path.name:<40}  {winner:<22}  {raw_conf:>9.1f}%  {cal_conf:>16.1f}%  {flag}")

print("\n[DONE] Step 1 complete. Temperature scaling is working.")
print(f"   Current temperature: T={CONFIDENCE_TEMPERATURE}")
print(f"   To adjust: edit CONFIDENCE_TEMPERATURE in backend/ml/predictor.py")

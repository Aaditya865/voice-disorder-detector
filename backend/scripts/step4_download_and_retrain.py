"""
STEP 4+5: Download SVD Dataset from Zenodo + Retrain with Augmentation
=======================================================================
Run from the project root:
  python backend/scripts/step4_download_and_retrain.py

What this does:
  1. Downloads the Saarbrücken Voice Database (SVD) from Zenodo
     — 6 pathology classes, up to 20 speakers each (except Laryngozele: 3 max)
     — Converts .nsp files → 16 kHz .wav files
     — Saves into dataset/<ClassName>/speaker_<id>/<session_id>-<vowel>.wav
     — Updates metadata.csv with speaker/session/split info

  2. Applies audio augmentation to the training set (3x expansion):
     — Pitch shift: +1 semitone
     — Pitch shift: -1 semitone
     — Time stretch: 0.95x (slightly slower)
     — Gaussian noise injection

  3. Retrains the CNN-BiLSTM-Attention model on the augmented dataset
     — Saves best checkpoint to backend/models/cnn_bilstm_attention_augmented/

Usage tip: This takes 30-90 minutes depending on internet speed.
           Progress is printed to console.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Download dataset from Zenodo
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Downloading SVD dataset from Zenodo")
print("=" * 60)

download_script = BACKEND_DIR / "scripts" / "prepare_svd_option_b_fast.py"
result = subprocess.run(
    [sys.executable, str(download_script)],
    cwd=str(BASE_DIR),
    capture_output=False
)
if result.returncode != 0:
    print("[ERROR] Download failed. Check your internet connection.")
    sys.exit(1)

print("\n[OK] Dataset downloaded successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Run augmented CNN-BiLSTM training
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: Running augmented CNN-BiLSTM training")
print("=" * 60)

augment_script = BACKEND_DIR / "scripts" / "run_augmented_experiment.py"
result = subprocess.run(
    [sys.executable, str(augment_script)],
    cwd=str(BASE_DIR),
    capture_output=False
)
if result.returncode != 0:
    print("[ERROR] Augmented training failed.")
    sys.exit(1)

print("\n[OK] Augmented model trained and saved.")
print("     Checkpoint: backend/models/cnn_bilstm_attention_augmented/Config_A_Augmented_best.pt")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Update predictor to use the newly trained augmented model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Activating augmented model in predictor")
print("=" * 60)

# Read and update the predictor
predictor_path = BACKEND_DIR / "ml" / "cnn_bilstm_predictor.py"
content = predictor_path.read_text(encoding="utf-8")
if "AUGMENTED = False" in content:
    content = content.replace("AUGMENTED = False", "AUGMENTED = True")
    predictor_path.write_text(content, encoding="utf-8")
    print("[OK] Predictor switched to augmented model (AUGMENTED = True).")
else:
    print("[INFO] AUGMENTED flag already set.")

print("\n[DONE] Steps 4+5 complete!")
print("  - SVD dataset downloaded from Zenodo")
print("  - CNN-BiLSTM retrained with 3x augmented training data")
print("  - Predictor now uses the augmented model")
print("\nRestart the backend server for changes to take effect.")

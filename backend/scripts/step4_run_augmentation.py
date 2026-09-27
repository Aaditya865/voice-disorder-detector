"""
STEP 4: Run Augmented CNN-BiLSTM Training (after dataset is present)
======================================================================
Run this AFTER the dataset is downloaded:
  python backend/scripts/step4_run_augmentation.py

What augmentation does:
  Original 648 training recordings
  + Variation 1: Circular shift (+4-8%) + gain (+0.8..+1.8 dB) + Gaussian noise
  + Variation 2: Circular shift (-4-8%) + attenuation (-1.8..-0.8 dB) + noise
  = 1944 total training examples (3x expansion)

Then retrains CNN-BiLSTM-Attention with this expanded set.
"""
import sys
import subprocess
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Check dataset is present ──────────────────────────────────────────────────
meta_path = BASE_DIR / "dataset" / "metadata.csv"
if not meta_path.exists():
    print("[ERROR] metadata.csv not found. Run prepare_svd_option_b_fast.py first.")
    sys.exit(1)

with open(meta_path) as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
present = sum(1 for r in train_rows if Path(BASE_DIR / r["processed_file"]).exists())
print(f"Training WAV files on disk: {present}/{len(train_rows)}")

if present < 10:
    print("[ERROR] Not enough training data on disk.")
    print("        Run: python backend/scripts/prepare_svd_option_b_fast.py")
    sys.exit(1)

print(f"\n[OK] Found {present} training files. Starting augmented training...\n")

# ── Run the augmented experiment ──────────────────────────────────────────────
result = subprocess.run(
    [sys.executable, str(BACKEND_DIR / "scripts" / "run_augmented_experiment.py")],
    cwd=str(BASE_DIR),
    capture_output=False
)

if result.returncode != 0:
    print("[ERROR] Augmented training failed.")
    sys.exit(1)

# ── Switch predictor to use augmented checkpoint ──────────────────────────────
predictor_path = BACKEND_DIR / "ml" / "cnn_bilstm_predictor.py"
content = predictor_path.read_text(encoding="utf-8")
if "AUGMENTED = False" in content:
    content = content.replace("AUGMENTED = False", "AUGMENTED = True")
    predictor_path.write_text(content, encoding="utf-8")
    print("\n[OK] Predictor switched to augmented model.")
else:
    print("\n[INFO] Predictor already using augmented model.")

print("\n[DONE] Step 4 complete!")
print("  Augmented CNN-BiLSTM checkpoint: backend/models/cnn_bilstm_attention_augmented/")
print("  Restart the backend server for changes to take effect.")

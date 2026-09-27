"""
test_samples.py
----------------
Tests all audio files in sample_recordings/ using the
Wav2Vec 2.0 + Random Forest inference engine and prints a clean report.
"""

import sys
from pathlib import Path
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from ml.predictor import VoicePredictor

def run_tests():
    sample_dir = Path(__file__).parent / "sample_recordings"
    audio_files = sorted(list(sample_dir.glob("*.wav")))

    if not audio_files:
        print(f"[ERROR] No .wav files found in {sample_dir}")
        return

    print("==========================================================================")
    print(" LARYNGEAL VOICE DISORDER CLASSIFICATION - AUDIO TEST SUITE")
    print(" Architecture: Wav2Vec 2.0 (facebook/wav2vec2-base) + Random Forest")
    print(f" Testing {len(audio_files)} audio samples from: {sample_dir}")
    print("==========================================================================\n")

    predictor = VoicePredictor.get_instance()
    if not predictor.is_ready:
        print("[ERROR] Model is not ready.")
        return

    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Audio File: {audio_path.name}")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        res = predictor.predict_audio(audio_bytes)
        pred_class = res.get("predicted_class", "Unknown")
        confidence = res.get("confidence", 0.0) * 100
        category = res.get("category", "")
        duration = res.get("audio_metadata", {}).get("duration_seconds", 0.0)
        probs = res.get("class_probabilities", {})

        # Top 3 probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        probs_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in sorted_probs])

        print(f"      Duration:    {duration:.2f} seconds")
        print(f"      Prediction:  {pred_class} ({category})")
        print(f"      Confidence:  {confidence:.1f}%")
        print(f"      Top 3:       {probs_str}")
        print("      " + "-" * 60)

    print("\n==========================================================================")
    print(" ALL AUDIO SAMPLE TESTS COMPLETED SUCCESSFULLY!")
    print("==========================================================================")


if __name__ == "__main__":
    run_tests()

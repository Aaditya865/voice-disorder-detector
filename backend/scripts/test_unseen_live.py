import requests
from pathlib import Path

API = "http://127.0.0.1:8000/predict"
sample_dir = Path("unseen_test_samples")

# True labels are encoded in the filename
files = sorted(sample_dir.glob("*.wav"))

print("=" * 75)
print("LIVE TEST: 6 UNSEEN PATIENTS (never seen during training or validation)")
print("=" * 75)
print()

correct = 0
for wav in files:
    # Parse true label from filename: UNSEEN_<CLASS>_spk...
    parts = wav.stem.split("_")
    # Class could be multi-word like Vocal_fold_paralysis
    # Format: UNSEEN_<CLASS>_spk<ID>_a_n
    # Find 'spk' to locate where class ends
    name = wav.stem  # e.g. UNSEEN_Vocal_fold_paralysis_spk1382_a_n
    after_unseen = name[7:]  # remove "UNSEEN_"
    spk_idx = after_unseen.index("_spk")
    true_class = after_unseen[:spk_idx]

    with open(wav, "rb") as f:
        resp = requests.post(API, files={"file": (wav.name, f, "audio/wav")})
    res = resp.json()

    pred = res["predicted_class"]
    conf = res["confidence"] * 100
    wins = res.get("num_windows_voted", 1)
    hit  = "CORRECT" if pred == true_class else "WRONG  "

    if pred == true_class:
        correct += 1

    print(f"True class : {true_class}")
    print(f"Predicted  : {pred}  ({conf:.1f}% confidence)  [{wins} window(s)]  [{hit}]")
    print("Probabilities:")
    for cls, p in sorted(res["class_probabilities"].items(), key=lambda x: -x[1]):
        marker = " <-- PREDICTED" if cls == pred else ("  <-- TRUE    " if cls == true_class else "")
        bar = "#" * int(p * 25)
        print(f"  {cls:<25} {p*100:>5.1f}%  {bar}{marker}")
    print()

print("=" * 75)
print(f"RESULT: {correct}/6 correct  ({correct/6*100:.0f}% accuracy on unseen patients)")
print(f"Architecture: {res['model_architecture']}")
print("=" * 75)

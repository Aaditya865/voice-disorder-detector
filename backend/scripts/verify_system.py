import requests
import json

def verify():
    print("=== 1. VERIFY HEALTH ENDPOINT ===")
    r_health = requests.get("http://127.0.0.1:8000/health").json()
    print("Health:", r_health)
    assert r_health["status"] == "online"
    assert r_health["model_trained"] is True

    print("\n=== 2. VERIFY DATASET AUDIT ===")
    r_ds = requests.get("http://127.0.0.1:8000/dataset-statistics").json()
    print(f"Total Speakers: {r_ds['totalSpeakers']}, Total Audio Files: {r_ds['totalSamples']}")
    print(f"Available Classes: {r_ds['availableClasses']}")
    print(f"Train Speakers ({len(r_ds['trainSpeakers'])}): {r_ds['trainSpeakers']}")
    print(f"Test Speakers ({len(r_ds['testSpeakers'])}): {r_ds['testSpeakers']}")
    overlap = set(r_ds["trainSpeakers"]).intersection(set(r_ds["testSpeakers"]))
    print(f"Speaker Overlap between Train and Test: {overlap} (Verified empty: {len(overlap) == 0})")
    assert len(overlap) == 0, "Speaker overlap detected!"
    assert r_ds["totalSpeakers"] == 18, f"Expected 18 speakers, got {r_ds['totalSpeakers']}"

    print("\n=== 3. VERIFY EVALUATION ENDPOINT ===")
    r_eval = requests.get("http://127.0.0.1:8000/evaluation").json()
    print(f"Status: {r_eval['status']}, Test Accuracy: {r_eval['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1: {r_eval['macro_metrics']['f1_score']:.4f}")
    print("Confusion Matrix Labels:", r_eval["confusion_matrix"]["labels"])
    print("Confusion Matrix:", r_eval["confusion_matrix"]["matrix"])

    print("\n=== 4. VERIFY PREDICTION ENDPOINT WITH REAL SVD RECORDING ===")
    test_audio = "C:/Users/ishan/.gemini/antigravity/scratch/ai-laryngeal-voice-disorder-system/dataset/Laryngozele/speaker_1602/1205-a_n.wav"
    with open(test_audio, "rb") as f:
        r_pred = requests.post("http://127.0.0.1:8000/predict", files={"file": ("1205-a_n.wav", f, "audio/wav")}).json()
    print(f"Predicted Class: {r_pred.get('predicted_class')}")
    print(f"Confidence: {r_pred.get('confidence')}")
    print("Class Probabilities:", r_pred.get("class_probabilities"))

    print("\n==================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    verify()

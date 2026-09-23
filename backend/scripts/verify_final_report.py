import json

with open("backend/models/evaluation_results.json") as f:
    ev = json.load(f)

print("=== MODEL SPECIFICATION ===")
print("Model Type:", ev.get("model_type"))
print("Pooling Strategy:", ev.get("pooling_strategy"))
print("Feature Dim:", ev.get("feature_dim"))

print("\n=== VALIDATION PERFORMANCE ===")
for k, v in ev["validation_performance"].items():
    print(f"  {k}: {v}")

print("\n=== TEST PERFORMANCE ===")
for k, v in ev["test_performance"].items():
    print(f"  {k}: {v}")

print("\n=== PER-CLASS METRICS (TEST) ===")
for c in ev["per_class_metrics"]:
    print(f"  {c['class']:22s} | P: {c['precision']:.4f} | R: {c['recall']:.4f} | F1: {c['f1_score']:.4f} | Support: {c['support']}")

print("\n=== CONFUSION MATRIX (TEST) ===")
print("Labels:", ev["confusion_matrix"]["labels"])
for row, label in zip(ev["confusion_matrix"]["matrix"], ev["confusion_matrix"]["labels"]):
    print(f"  {label:22s}: {row}")

print("\n=== SPEAKER-LEVEL EVALUATION (TEST) ===")
spk_eval = ev["speaker_level_evaluation"]
print("Soft Voting Accuracy:", spk_eval.get("speaker_accuracy_soft"))
print("Majority Voting Accuracy:", spk_eval.get("speaker_accuracy_majority"))
for c in spk_eval.get("per_class_speaker_results", []):
    acc = c["soft_voting_accuracy"] * 100
    print(f"  {c['class']:22s} | Correct: {c['correctly_classified']}/{c['total_speakers']} ({acc:.1f}%)")

print("\n=== SPEAKER BREAKDOWN (TEST) ===")
for s in spk_eval.get("speaker_breakdown", []):
    spk_id = s["speaker_id"]
    true_c = s["true_class"]
    soft_p = s["predicted_class_soft"]
    soft_ok = "CORRECT" if s["correct_soft"] else "INCORRECT"
    maj_p = s["predicted_class_majority"]
    maj_ok = "CORRECT" if s["correct_majority"] else "INCORRECT"
    conf = s["posterior_probabilities"][soft_p]
    print(f"  Speaker {spk_id:<6} | True: {true_c:<22} | Soft: {soft_p:<22} ({conf*100:.1f}%) [{soft_ok}] | Maj: {maj_p:<22} [{maj_ok}]")

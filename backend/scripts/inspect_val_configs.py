import csv
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    cache_path = backend_dir / "data" / "features_mean_std_cache.joblib"

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    features_cache = joblib.load(cache_path)

    # Strictly Train and Val rows
    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]

    # Verify zero leakage
    train_spks = set(r["speaker_id"] for r in train_rows)
    val_spks = set(r["speaker_id"] for r in val_rows)
    assert len(train_spks.intersection(val_spks)) == 0

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    X_train = np.array([features_cache[r["processed_file"]] for r in train_rows], dtype=np.float32)
    y_train = le.transform([r["class"] for r in train_rows])

    X_val = np.array([features_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
    y_val = le.transform([r["class"] for r in val_rows])

    # Distinct validation speakers and their recordings
    val_speaker_map = defaultdict(list)
    for i, r in enumerate(val_rows):
        val_speaker_map[r["speaker_id"]].append({
            "idx": i,
            "true_cls": r["class"],
            "vowel": r["vowel"],
            "pitch": r["pitch"]
        })

    configs = {
        "Config #7": {
            "name": "RF (200 trees, depth=None)",
            "params": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "class_weight": None, "random_state": 42, "n_jobs": -1}
        },
        "Config #8": {
            "name": "RF (200 trees, depth=20)",
            "params": {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt", "class_weight": None, "random_state": 42, "n_jobs": -1}
        },
        "Config #9": {
            "name": "RF (300 trees, depth=None)",
            "params": {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "class_weight": None, "random_state": 42, "n_jobs": -1}
        }
    }

    results = {}

    for cfg_key, cfg_info in configs.items():
        rf = RandomForestClassifier(**cfg_info["params"])
        rf.fit(X_train, y_train)

        val_preds = rf.predict(X_val)
        val_probs = rf.predict_proba(X_val)

        rec_acc = accuracy_score(y_val, val_preds)
        rec_prec, rec_rec, rec_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
        rec_prec_w, rec_rec_w, rec_f1_w, _ = precision_recall_fscore_support(y_val, val_preds, average="weighted", zero_division=0)

        # Speaker evaluation
        speaker_details = {}
        spk_soft_correct = 0
        spk_hard_correct = 0

        for spk_id, recs in val_speaker_map.items():
            true_c = recs[0]["true_cls"]
            # Soft voting
            avg_probs = np.mean([val_probs[r["idx"]] for r in recs], axis=0)
            soft_pred_c = classes[int(np.argmax(avg_probs))]
            soft_conf = float(np.max(avg_probs))

            # Majority voting
            rec_preds = [classes[val_preds[r["idx"]]] for r in recs]
            vote_counts = Counter(rec_preds)
            hard_pred_c, max_votes = vote_counts.most_common(1)[0]

            is_soft_corr = bool(soft_pred_c == true_c)
            is_hard_corr = bool(hard_pred_c == true_c)

            if is_soft_corr:
                spk_soft_correct += 1
            if is_hard_corr:
                spk_hard_correct += 1

            prob_dict = {classes[k]: round(float(avg_probs[k]), 4) for k in range(len(classes))}
            speaker_details[spk_id] = {
                "true_class": true_c,
                "soft_pred": soft_pred_c,
                "soft_correct": is_soft_corr,
                "soft_confidence": round(soft_conf, 4),
                "hard_pred": hard_pred_c,
                "hard_correct": is_hard_corr,
                "hard_votes": f"{max_votes}/9",
                "vote_breakdown": dict(vote_counts),
                "probabilities": prob_dict
            }

        # Class-level speaker accuracy
        per_class_spk = {}
        for c in classes:
            c_spks = [s for s, d in speaker_details.items() if d["true_class"] == c]
            if c_spks:
                c_soft_corr = sum(1 for s in c_spks if speaker_details[s]["soft_correct"])
                c_hard_corr = sum(1 for s in c_spks if speaker_details[s]["hard_correct"])
                per_class_spk[c] = {
                    "total": len(c_spks),
                    "soft_correct": c_soft_corr,
                    "soft_accuracy": round(c_soft_corr / len(c_spks), 4),
                    "hard_correct": c_hard_corr,
                    "hard_accuracy": round(c_hard_corr / len(c_spks), 4)
                }

        results[cfg_key] = {
            "name": cfg_info["name"],
            "params": cfg_info["params"],
            "recording_level": {
                "accuracy": round(float(rec_acc), 4),
                "macro_precision": round(float(rec_prec), 4),
                "macro_recall": round(float(rec_rec), 4),
                "macro_f1": round(float(rec_f1), 4),
                "weighted_f1": round(float(rec_f1_w), 4)
            },
            "speaker_level": {
                "soft_voting_accuracy": round(spk_soft_correct / len(val_speaker_map), 4),
                "soft_correct_count": spk_soft_correct,
                "majority_voting_accuracy": round(spk_hard_correct / len(val_speaker_map), 4),
                "hard_correct_count": spk_hard_correct,
                "total_speakers": len(val_speaker_map),
                "per_class": per_class_spk,
                "speaker_predictions": speaker_details
            }
        }

    # Print comparison
    print("\n=======================================================")
    print("CONFIG #7 vs #8 vs #9 VALIDATION SUMMARY")
    print("=======================================================")
    print(f"{'Metric':<32} {'Config #7 (200, None)':<24} {'Config #8 (200, d=20)':<24} {'Config #9 (300, None)':<24}")
    print("-" * 104)
    print(f"{'Recording Accuracy':<32} {results['Config #7']['recording_level']['accuracy']*100:<23.2f}% {results['Config #8']['recording_level']['accuracy']*100:<23.2f}% {results['Config #9']['recording_level']['accuracy']*100:<23.2f}%")
    print(f"{'Recording Macro F1':<32} {results['Config #7']['recording_level']['macro_f1']:<24.4f} {results['Config #8']['recording_level']['macro_f1']:<24.4f} {results['Config #9']['recording_level']['macro_f1']:<24.4f}")
    print(f"{'Recording Macro Precision':<32} {results['Config #7']['recording_level']['macro_precision']:<24.4f} {results['Config #8']['recording_level']['macro_precision']:<24.4f} {results['Config #9']['recording_level']['macro_precision']:<24.4f}")
    print(f"{'Recording Macro Recall':<32} {results['Config #7']['recording_level']['macro_recall']:<24.4f} {results['Config #8']['recording_level']['macro_recall']:<24.4f} {results['Config #9']['recording_level']['macro_recall']:<24.4f}")
    print(f"{'Speaker Soft-Voting Accuracy':<32} {results['Config #7']['speaker_level']['soft_voting_accuracy']*100:.1f}% ({results['Config #7']['speaker_level']['soft_correct_count']}/15){'':<13} {results['Config #8']['speaker_level']['soft_voting_accuracy']*100:.1f}% ({results['Config #8']['speaker_level']['soft_correct_count']}/15){'':<13} {results['Config #9']['speaker_level']['soft_voting_accuracy']*100:.1f}% ({results['Config #9']['speaker_level']['soft_correct_count']}/15)")
    print(f"{'Speaker Majority Accuracy':<32} {results['Config #7']['speaker_level']['majority_voting_accuracy']*100:.1f}% ({results['Config #7']['speaker_level']['hard_correct_count']}/15){'':<13} {results['Config #8']['speaker_level']['majority_voting_accuracy']*100:.1f}% ({results['Config #8']['speaker_level']['hard_correct_count']}/15){'':<13} {results['Config #9']['speaker_level']['majority_voting_accuracy']*100:.1f}% ({results['Config #9']['speaker_level']['hard_correct_count']}/15)")

    # Save detailed JSON
    out_file = backend_dir / "models" / "val_config_7_8_9_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed analysis to: {out_file}")


if __name__ == "__main__":
    main()

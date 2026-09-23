import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("final_evaluation")


def run_final_evaluation():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mean_std_cache_path = backend_dir / "data" / "features_mean_std_cache.joblib"

    print("==================================================", flush=True)
    print("TRAINING FINAL MODEL (CONFIG #91) & TEST EVALUATION", flush=True)
    print("==================================================", flush=True)

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} metadata rows.")

    # Load 1536-D features cache
    if not mean_std_cache_path.exists():
        raise FileNotFoundError(f"Missing mean+std feature cache: {mean_std_cache_path}")
    mean_std_cache = joblib.load(mean_std_cache_path)
    logger.info(f"Loaded mean+std feature cache with {len(mean_std_cache)} entries.")

    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    logger.info(f"Split counts: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    # Verify zero speaker overlap
    train_speakers = set(r["speaker_id"] for r in train_rows)
    val_speakers = set(r["speaker_id"] for r in val_rows)
    test_speakers = set(r["speaker_id"] for r in test_rows)

    assert len(train_speakers.intersection(test_speakers)) == 0, "Speaker leakage between Train and Test!"
    assert len(val_speakers.intersection(test_speakers)) == 0, "Speaker leakage between Val and Test!"

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    X_train = np.array([mean_std_cache[r["processed_file"]] for r in train_rows], dtype=np.float32)
    y_train = le.transform([r["class"] for r in train_rows])

    X_val = np.array([mean_std_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
    y_val = le.transform([r["class"] for r in val_rows])

    X_test = np.array([mean_std_cache[r["processed_file"]] for r in test_rows], dtype=np.float32)
    y_test = le.transform([r["class"] for r in test_rows])

    logger.info(f"Feature matrix shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Configuration #91:
    # Mean + Std pooling = 1536 features
    # n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight=None, max_features='sqrt', random_state=42
    rf_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": None,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    }

    logger.info(f"Fitting final Random Forest with Config #91 on training set ({len(X_train)} samples)...")
    t0 = time.time()
    final_rf = RandomForestClassifier(**rf_params)
    final_rf.fit(X_train, y_train)
    t_fit = time.time() - t0
    logger.info(f"Model fitting complete in {t_fit:.2f}s.")

    # Validation check
    val_preds = final_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec_m, val_rec_m, val_f1_m, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
    logger.info(f"Validation verification: Acc={val_acc*100:.2f}%, Macro F1={val_f1_m:.4f}")

    # =========================================================================
    # Held-out Test Set Evaluation (Recording-Level)
    # =========================================================================
    logger.info(f"Evaluating final model on 144 test recordings (16 unseen speakers)...")
    test_preds = final_rf.predict(X_test)
    test_probs = final_rf.predict_proba(X_test)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec_m, test_rec_m, test_f1_m, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    test_prec_w, test_rec_w, test_f1_w, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)

    p_per_cls, r_per_cls, f_per_cls, s_per_cls = precision_recall_fscore_support(
        y_test, test_preds, labels=range(len(classes)), zero_division=0
    )

    per_class_metrics = []
    for cls_name, p, r, f, s in zip(classes, p_per_cls, r_per_cls, f_per_cls, s_per_cls):
        per_class_metrics.append({
            "class": cls_name,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "support": int(s)
        })

    cm_test = confusion_matrix(y_test, test_preds, labels=range(len(classes)))
    cm_test_norm = cm_test.astype("float") / (cm_test.sum(axis=1)[:, np.newaxis] + 1e-9)

    # =========================================================================
    # Held-out Test Set Evaluation (Speaker-Level Aggregation)
    # =========================================================================
    speaker_recording_map = defaultdict(list)
    for i, r in enumerate(test_rows):
        spk = r["speaker_id"]
        speaker_recording_map[spk].append({
            "idx": i,
            "true_cls": r["class"],
            "pred_cls": classes[test_preds[i]],
            "probs": test_probs[i],
            "vowel": r["vowel"],
            "pitch": r["pitch"],
            "file": r["processed_file"]
        })

    speaker_eval_results = []
    spk_true_labels = []
    spk_pred_soft_labels = []
    spk_pred_hard_labels = []

    for spk_id, recs in speaker_recording_map.items():
        true_cls = recs[0]["true_cls"]
        avg_probs = np.mean([r["probs"] for r in recs], axis=0)
        soft_pred_idx = int(np.argmax(avg_probs))
        soft_pred_cls = classes[soft_pred_idx]

        votes = Counter([r["pred_cls"] for r in recs])
        hard_pred_cls = votes.most_common(1)[0][0]

        spk_true_labels.append(true_cls)
        spk_pred_soft_labels.append(soft_pred_cls)
        spk_pred_hard_labels.append(hard_pred_cls)

        prob_dict = {classes[k]: round(float(avg_probs[k]), 4) for k in range(len(classes))}

        speaker_eval_results.append({
            "speaker_id": spk_id,
            "true_class": true_cls,
            "predicted_class_soft": soft_pred_cls,
            "predicted_class_majority": hard_pred_cls,
            "correct_soft": bool(true_cls == soft_pred_cls),
            "correct_majority": bool(true_cls == hard_pred_cls),
            "posterior_probabilities": prob_dict,
            "recording_count": len(recs)
        })

    spk_soft_acc = float(np.mean([r["correct_soft"] for r in speaker_eval_results]))
    spk_hard_acc = float(np.mean([r["correct_majority"] for r in speaker_eval_results]))

    # Per-class speaker-level breakdown
    per_class_speaker_results = []
    for cls_name in classes:
        cls_speakers = [r for r in speaker_eval_results if r["true_class"] == cls_name]
        correct_cnt = sum(1 for r in cls_speakers if r["correct_soft"])
        total_cnt = len(cls_speakers)
        cls_acc = (correct_cnt / total_cnt) if total_cnt > 0 else 0.0
        per_class_speaker_results.append({
            "class": cls_name,
            "total_speakers": total_cnt,
            "correctly_classified": correct_cnt,
            "soft_voting_accuracy": round(float(cls_acc), 4)
        })

    # Speaker confusion matrix
    spk_cm = confusion_matrix(
        [le.transform([r["true_class"]])[0] for r in speaker_eval_results],
        [le.transform([r["predicted_class_soft"]])[0] for r in speaker_eval_results],
        labels=range(len(classes))
    )

    # Detailed per-recording predictions
    recording_predictions = []
    for i, r in enumerate(test_rows):
        recording_predictions.append({
            "speaker_id": r["speaker_id"],
            "vowel": r["vowel"],
            "pitch": r["pitch"],
            "true_class": r["class"],
            "predicted_class": classes[test_preds[i]],
            "confidence": round(float(np.max(test_probs[i])), 4),
            "correct": bool(test_preds[i] == y_test[i])
        })

    # =========================================================================
    # Save Final Model and Metadata Artifacts
    # =========================================================================
    rf_model_path = models_dir / "rf_classifier.joblib"
    le_path = models_dir / "label_encoder.joblib"
    meta_path = models_dir / "model_metadata.json"
    results_path = models_dir / "evaluation_results.json"

    joblib.dump(final_rf, rf_model_path)
    joblib.dump(le, le_path)

    metadata_payload = {
        "model_name": "Wav2Vec 2.0 + Random Forest (Config #91)",
        "feature_extractor": "facebook/wav2vec2-base",
        "pooling_strategy": "mean_std_pooling",
        "n_features": 1536,
        "n_samples": len(X_train),
        "classes": classes,
        "n_classes": len(classes),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": None,
            "max_features": "sqrt",
            "random_state": 42
        },
        "trained_at": datetime.now().isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    evaluation_payload = {
        "status": "trained",
        "trained_at": datetime.now().isoformat(),
        "model_type": "Wav2Vec 2.0 (facebook/wav2vec2-base) + Random Forest (Config #91)",
        "pooling_strategy": "mean_std_pooling",
        "feature_dim": 1536,
        "classes": classes,
        "n_classes": len(classes),
        "split_summary": {
            "train_recordings": len(train_rows),
            "val_recordings": len(val_rows),
            "test_recordings": len(test_rows),
            "train_speaker_count": len(train_speakers),
            "val_speaker_count": len(val_speakers),
            "test_speaker_count": len(test_speakers),
            "train_speakers": sorted(list(train_speakers)),
            "val_speakers": sorted(list(val_speakers)),
            "test_speakers": sorted(list(test_speakers)),
            "split_strategy": "Speaker-Independent (Zero Speaker Leakage)"
        },
        "validation_performance": {
            "accuracy": round(float(val_acc), 4),
            "macro_precision": round(float(val_prec_m), 4),
            "macro_recall": round(float(val_rec_m), 4),
            "macro_f1": round(float(val_f1_m), 4)
        },
        "test_performance": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_prec_m), 4),
            "macro_recall": round(float(test_rec_m), 4),
            "macro_f1": round(float(test_f1_m), 4),
            "weighted_f1": round(float(test_f1_w), 4),
            "speaker_level_soft_voting_accuracy": round(float(spk_soft_acc), 4),
            "speaker_level_majority_accuracy": round(float(spk_hard_acc), 4)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm_test.tolist(),
            "normalized_matrix": [[round(float(v), 4) for v in row] for row in cm_test_norm.tolist()]
        },
        "speaker_level_evaluation": {
            "speaker_accuracy_soft": round(float(spk_soft_acc), 4),
            "speaker_accuracy_majority": round(float(spk_hard_acc), 4),
            "per_class_speaker_results": per_class_speaker_results,
            "confusion_matrix": spk_cm.tolist(),
            "speaker_breakdown": speaker_eval_results
        },
        "test_predictions_recording_breakdown": recording_predictions
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    logger.info("Saved final model and evaluation results successfully.")

    # Output summary to stdout
    print("\n==================================================", flush=True)
    print("FINAL TEST EVALUATION RESULTS (CONFIG #91)", flush=True)
    print("==================================================", flush=True)
    print(f"Test Recording Accuracy: {test_acc*100:.2f}% ({int(test_acc*len(test_rows))}/{len(test_rows)})")
    print(f"Test Macro Precision:    {test_prec_m:.4f}")
    print(f"Test Macro Recall:       {test_rec_m:.4f}")
    print(f"Test Macro F1-Score:     {test_f1_m:.4f}")
    print(f"Test Weighted F1-Score:  {test_f1_w:.4f}")
    print(f"Speaker Soft-Voting Acc: {spk_soft_acc*100:.2f}% ({int(spk_soft_acc*len(speaker_eval_results))}/{len(speaker_eval_results)})")
    print(f"Speaker Majority Acc:    {spk_hard_acc*100:.2f}% ({int(spk_hard_acc*len(speaker_eval_results))}/{len(speaker_eval_results)})")


if __name__ == "__main__":
    run_final_evaluation()

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import soundfile as sf
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2Model, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expanded_training")


def extract_all_features(metadata_rows, base_dir, cache_path):
    """
    Extracts 768-dimensional Wav2Vec 2.0 representations for all 927 recordings with temporal mean pooling.
    Uses persistent caching in joblib format.
    """
    cache = {}
    if cache_path.exists():
        try:
            cache = joblib.load(cache_path)
            logger.info(f"Loaded existing feature cache with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load feature cache ({e}), recomputing.")
            cache = {}

    missing_paths = []
    for r in metadata_rows:
        rel_p = r["processed_file"]
        if rel_p not in cache:
            missing_paths.append(rel_p)

    if missing_paths:
        logger.info(f"Extracting Wav2Vec 2.0 features for {len(missing_paths)} recordings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        model.eval()

        t0 = time.time()
        for idx, rel_p in enumerate(missing_paths, 1):
            full_p = base_dir / rel_p
            audio_arr, sr = sf.read(str(full_p))
            if audio_arr.ndim > 1:
                audio_arr = audio_arr[:, 0]

            inputs = proc(audio_arr, sampling_rate=16000, return_tensors="pt", padding=False)
            input_vals = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(input_vals)
                # Temporal mean pooling over sequence length dimension
                feat = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy().astype(np.float32)

            cache[rel_p] = feat

            if idx % 100 == 0 or idx == len(missing_paths):
                elapsed = time.time() - t0
                rate = idx / elapsed
                logger.info(f"  [{idx}/{len(missing_paths)}] features extracted ({rate:.1f} audio/sec)")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path)
        logger.info(f"Feature cache saved with {len(cache)} total entries.")
    else:
        logger.info("All 927 recordings already cached in features_cache.joblib.")

    return cache


def tune_hyperparameters(X_train, y_train, X_val, y_val, class_names):
    """
    Evaluates hyperparameter configurations on the VALIDATION set only.
    Selects the configuration maximizing Validation Macro F1.
    """
    logger.info(f"Tuning Random Forest on Validation Set ({len(X_val)} samples across 15 speakers)...")

    param_grid = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2, "class_weight": "balanced", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 4, "class_weight": "balanced", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced_subsample", "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": None, "max_features": "sqrt", "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced", "max_features": "log2", "random_state": 42},
    ]

    best_score = -1.0
    best_params = None
    tuning_records = []

    for p in param_grid:
        rf = RandomForestClassifier(**p, n_jobs=-1)
        rf.fit(X_train, y_train)

        val_preds = rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)[2]

        record = {
            "params": p,
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4)
        }
        tuning_records.append(record)
        logger.info(f"  Params: n_est={p['n_estimators']}, depth={p['max_depth']}, cw={p['class_weight']}, mf={p['max_features']} -> Val Acc: {val_acc*100:.2f}%, Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_params = p

    logger.info(f"Selected Best Hyperparameters (Val Macro F1 = {best_score:.4f}): {best_params}")
    return best_params, tuning_records


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = backend_dir / "data" / "features_cache.joblib"

    print("==================================================", flush=True)
    print("AI-BASED LARYNGEAL VOICE DISORDER CLASSIFICATION", flush=True)
    print("EXPANDED SVD MODEL TRAINING & BENCHMARK EVALUATION", flush=True)
    print("==================================================", flush=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} metadata rows.")

    # 1. Feature Extraction
    features_cache = extract_all_features(metadata_rows, base_dir, cache_path)

    # 2. Partition Data
    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    logger.info(f"Partitions: Train={len(train_rows)} recordings, Val={len(val_rows)} recordings, Test={len(test_rows)} recordings.")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    X_train = np.array([features_cache[r["processed_file"]] for r in train_rows], dtype=np.float32)
    y_train = le.transform([r["class"] for r in train_rows])
    spk_train = [r["speaker_id"] for r in train_rows]

    X_val = np.array([features_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
    y_val = le.transform([r["class"] for r in val_rows])
    spk_val = [r["speaker_id"] for r in val_rows]

    X_test = np.array([features_cache[r["processed_file"]] for r in test_rows], dtype=np.float32)
    y_test = le.transform([r["class"] for r in test_rows])
    spk_test = [r["speaker_id"] for r in test_rows]

    # Verify zero speaker leakage
    train_speakers_set = set(spk_train)
    val_speakers_set = set(spk_val)
    test_speakers_set = set(spk_test)

    assert len(train_speakers_set.intersection(val_speakers_set)) == 0, "Speaker leakage between Train and Val!"
    assert len(train_speakers_set.intersection(test_speakers_set)) == 0, "Speaker leakage between Train and Test!"
    assert len(val_speakers_set.intersection(test_speakers_set)) == 0, "Speaker leakage between Val and Test!"

    logger.info(f"Verified strict speaker independence across all splits:")
    logger.info(f"  Train: {len(train_speakers_set)} speakers")
    logger.info(f"  Val:   {len(val_speakers_set)} speakers")
    logger.info(f"  Test:  {len(test_speakers_set)} speakers")

    # 3. Model Selection on Validation Set Only
    best_params, tuning_records = tune_hyperparameters(X_train, y_train, X_val, y_val, classes)

    # 4. Final Model Training
    logger.info("Fitting final Random Forest with selected hyperparameters on full training partition...")
    final_rf = RandomForestClassifier(**best_params, n_jobs=-1)
    final_rf.fit(X_train, y_train)

    # Validation Set Evaluation
    val_preds = final_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec_m, val_rec_m, val_f1_m, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
    val_prec_w, val_rec_w, val_f1_w, _ = precision_recall_fscore_support(y_val, val_preds, average="weighted", zero_division=0)

    # 5. Held-Out Test Set Evaluation (Recording-Level)
    logger.info("Evaluating on held-out test partition (144 recordings across 16 unseen speakers)...")
    test_preds = final_rf.predict(X_test)
    test_probs = final_rf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    test_prec_m, test_rec_m, test_f1_m, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    test_prec_w, test_rec_w, test_f1_w, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)

    p_per_cls, r_per_cls, f_per_cls, s_per_cls = precision_recall_fscore_support(y_test, test_preds, labels=range(len(classes)), zero_division=0)

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

    # 6. Held-Out Test Set Evaluation (Speaker-Level Aggregation)
    # Group test recordings by speaker
    speaker_recording_map = defaultdict(list)
    for i, r in enumerate(test_rows):
        spk = r["speaker_id"]
        speaker_recording_map[spk].append({
            "idx": i,
            "true_cls": r["class"],
            "pred_cls": classes[test_preds[i]],
            "probs": test_probs[i],
            "vowel": r["vowel"],
            "pitch": r["pitch"]
        })

    speaker_eval_results = []
    spk_true_labels = []
    spk_pred_soft_labels = []
    spk_pred_hard_labels = []

    for spk_id, recs in speaker_recording_map.items():
        true_cls = recs[0]["true_cls"]
        # Soft voting: average posterior probability vector
        avg_probs = np.mean([r["probs"] for r in recs], axis=0)
        soft_pred_idx = int(np.argmax(avg_probs))
        soft_pred_cls = classes[soft_pred_idx]

        # Hard voting: majority vote among 9 predictions
        from collections import Counter
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

    spk_acc_soft = float(np.mean([t == p for t, p in zip(spk_true_labels, spk_pred_soft_labels)]))
    spk_acc_hard = float(np.mean([t == p for t, p in zip(spk_true_labels, spk_pred_hard_labels)]))

    cm_spk = confusion_matrix(spk_true_labels, spk_pred_soft_labels, labels=classes)

    logger.info("==================================================")
    logger.info("FINAL EVALUATION RESULTS SUMMARY")
    logger.info("==================================================")
    logger.info(f"Recording-Level Test Accuracy:  {test_acc*100:.2f}% ({int(test_acc*len(y_test))}/{len(y_test)} correct)")
    logger.info(f"Recording-Level Macro F1-Score: {test_f1_m:.4f}")
    logger.info(f"Speaker-Level Soft Voting Acc:  {spk_acc_soft*100:.2f}% ({int(spk_acc_soft*len(spk_true_labels))}/{len(spk_true_labels)} correct)")
    logger.info(f"Speaker-Level Majority Acc:     {spk_acc_hard*100:.2f}% ({int(spk_acc_hard*len(spk_true_labels))}/{len(spk_true_labels)} correct)")

    # 7. Serialize Artifacts
    model_payload = {
        "status": "trained",
        "trained_at": datetime.now().isoformat(),
        "model_type": "Wav2Vec 2.0 (facebook/wav2vec2-base) + Random Forest",
        "dataset_source": "Saarbrücken Voice Database (SVD)",
        "dataset_type": "Expanded 103-Speaker Cohort (9 Sustained Vowels per Speaker: a/i/u at normal/low/high pitch)",
        "classes": classes,
        "n_classes": len(classes),
        "split_summary": {
            "train_recordings": len(train_rows),
            "val_recordings": len(val_rows),
            "test_recordings": len(test_rows),
            "train_speakers": sorted(list(train_speakers_set), key=lambda x: int(x) if x.isdigit() else x),
            "val_speakers": sorted(list(val_speakers_set), key=lambda x: int(x) if x.isdigit() else x),
            "test_speakers": sorted(list(test_speakers_set), key=lambda x: int(x) if x.isdigit() else x),
            "train_speaker_count": len(train_speakers_set),
            "val_speaker_count": len(val_speakers_set),
            "test_speaker_count": len(test_speakers_set),
            "total_speakers": 103,
            "total_recordings": 927,
            "split_strategy": "Speaker-Independent (Zero Speaker Leakage across Splits)"
        },
        "validation_performance": {
            "accuracy": round(float(val_acc), 4),
            "macro_precision": round(float(val_prec_m), 4),
            "macro_recall": round(float(val_rec_m), 4),
            "macro_f1": round(float(val_f1_m), 4),
            "weighted_f1": round(float(val_f1_w), 4)
        },
        "test_performance": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_prec_m), 4),
            "macro_recall": round(float(test_rec_m), 4),
            "macro_f1": round(float(test_f1_m), 4),
            "weighted_f1": round(float(test_f1_w), 4),
            "speaker_level_soft_voting_accuracy": round(spk_acc_soft, 4),
            "speaker_level_majority_accuracy": round(spk_acc_hard, 4)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm_test.tolist(),
            "normalized_matrix": [[round(float(v), 4) for v in row] for row in cm_test_norm]
        },
        "speaker_level_evaluation": {
            "confusion_matrix": cm_spk.tolist(),
            "speaker_breakdown": speaker_eval_results
        },
        "test_predictions_recording_breakdown": [
            {
                "speaker_id": test_rows[i]["speaker_id"],
                "vowel": test_rows[i]["vowel"],
                "pitch": test_rows[i]["pitch"],
                "true_class": test_rows[i]["class"],
                "predicted_class": classes[test_preds[i]],
                "confidence": round(float(np.max(test_probs[i])), 4),
                "correct": bool(test_rows[i]["class"] == classes[test_preds[i]])
            }
            for i in range(len(test_rows))
        ],
        "hyperparameter_tuning_history": tuning_records,
        "selected_hyperparameters": best_params
    }

    # Save rf model, label encoder, metadata, and evaluation results
    joblib.dump(final_rf, models_dir / "rf_classifier.joblib")
    joblib.dump(le, models_dir / "label_encoder.joblib")

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": len(X_train),
            "n_features": 768,
            "classes": classes,
            "n_classes": len(classes),
            "hyperparameters": best_params,
            "trained_at": datetime.now().isoformat()
        }, f, indent=2)

    with open(models_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    logger.info(f"Serialized all model artifacts and evaluation metrics to {models_dir}")
    print("\n==================================================", flush=True)
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()

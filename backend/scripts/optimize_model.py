import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

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
logger = logging.getLogger("model_optimization")


def extract_mean_std_features(metadata_rows, base_dir, cache_path):
    """
    Extracts 1536-dimensional Wav2Vec 2.0 representations (mean + std pooling)
    for all recordings, cached in joblib format.
    """
    cache = {}
    if cache_path.exists():
        try:
            cache = joblib.load(cache_path)
            logger.info(f"Loaded existing mean+std feature cache with {len(cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load mean+std feature cache ({e}), recomputing.")
            cache = {}

    missing_paths = [r["processed_file"] for r in metadata_rows if r["processed_file"] not in cache]

    if missing_paths:
        logger.info(f"Extracting Wav2Vec 2.0 mean+std features for {len(missing_paths)} recordings...")
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
                hidden_states = outputs.last_hidden_state  # (1, T, 768)
                mean = torch.mean(hidden_states, dim=1).squeeze(0)  # (768,)
                std = torch.std(hidden_states, dim=1).squeeze(0)   # (768,)
                feat = torch.cat([mean, std], dim=-1).cpu().numpy().astype(np.float32)

            cache[rel_p] = feat

            if idx % 100 == 0 or idx == len(missing_paths):
                elapsed = time.time() - t0
                rate = idx / elapsed
                logger.info(f"  [{idx}/{len(missing_paths)}] features extracted ({rate:.1f} audio/sec)")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path)
        logger.info(f"Mean+std feature cache saved with {len(cache)} total entries.")
    else:
        logger.info("All recordings already cached in features_mean_std_cache.joblib.")

    return cache


def run_optimization():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mean_cache_path = backend_dir / "data" / "features_cache.joblib"
    mean_std_cache_path = backend_dir / "data" / "features_mean_std_cache.joblib"

    print("==================================================", flush=True)
    print("HYPERPARAMETER & FEATURE OPTIMIZATION ON SVD", flush=True)
    print("STRICTLY USING TRAINING & VALIDATION SETS ONLY", flush=True)
    print("==================================================", flush=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} metadata rows.")

    # Load / Extract both feature representations
    mean_cache = joblib.load(mean_cache_path)
    mean_std_cache = extract_mean_std_features(metadata_rows, base_dir, mean_std_cache_path)

    # Strictly partition ONLY Train and Validation
    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    # Strict isolation check
    assert len(train_rows) == 648, f"Expected 648 train rows, got {len(train_rows)}"
    assert len(val_rows) == 135, f"Expected 135 val rows, got {len(val_rows)}"
    train_speakers = set(r["speaker_id"] for r in train_rows)
    val_speakers = set(r["speaker_id"] for r in val_rows)
    test_speakers = set(r["speaker_id"] for r in test_rows)

    assert len(train_speakers.intersection(val_speakers)) == 0, "Leakage between train and val!"
    assert len(train_speakers.intersection(test_speakers)) == 0, "Leakage between train and test!"
    assert len(val_speakers.intersection(test_speakers)) == 0, "Leakage between val and test!"

    logger.info(f"Partitions: Train = {len(train_rows)} recordings ({len(train_speakers)} speakers), Val = {len(val_rows)} recordings ({len(val_speakers)} speakers).")
    logger.info("Test set is strictly excluded and untouched.")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    y_train = le.transform([r["class"] for r in train_rows])
    y_val = le.transform([r["class"] for r in val_rows])

    # Feature matrices for both pooling strategies
    feature_sets = {
        "mean_pooling": {
            "dim": 768,
            "X_train": np.array([mean_cache[r["processed_file"]] for r in train_rows], dtype=np.float32),
            "X_val": np.array([mean_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
        },
        "mean_std_pooling": {
            "dim": 1536,
            "X_train": np.array([mean_std_cache[r["processed_file"]] for r in train_rows], dtype=np.float32),
            "X_val": np.array([mean_std_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
        }
    }

    # Group validation recordings by speaker for speaker-level evaluation
    val_speaker_map = defaultdict(list)
    for i, r in enumerate(val_rows):
        val_speaker_map[r["speaker_id"]].append({
            "idx": i,
            "true_cls": r["class"]
        })

    # Define hyperparameter grid
    pooling_options = ["mean_pooling", "mean_std_pooling"]
    n_estimators_options = [100, 200, 300]
    max_depth_options = [None, 15, 25]
    min_samples_leaf_options = [1, 2, 4]
    class_weight_options = [None, "balanced"]

    total_configs = (
        len(pooling_options) *
        len(n_estimators_options) *
        len(max_depth_options) *
        len(min_samples_leaf_options) *
        len(class_weight_options)
    )

    logger.info(f"Starting systematic grid evaluation across {total_configs} configurations...")
    results = []
    config_id = 0
    t_start = time.time()

    for pooling_type in pooling_options:
        X_tr = feature_sets[pooling_type]["X_train"]
        X_v = feature_sets[pooling_type]["X_val"]
        feat_dim = feature_sets[pooling_type]["dim"]

        for n_est in n_estimators_options:
            for depth in max_depth_options:
                for min_leaf in min_samples_leaf_options:
                    for cw in class_weight_options:
                        config_id += 1
                        rf = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_leaf=min_leaf,
                            class_weight=cw,
                            max_features="sqrt",
                            random_state=42,
                            n_jobs=-1
                        )
                        rf.fit(X_tr, y_train)

                        val_preds = rf.predict(X_v)
                        val_probs = rf.predict_proba(X_v)

                        # Recording-level metrics
                        rec_acc = float(accuracy_score(y_val, val_preds))
                        rec_prec_m, rec_rec_m, rec_f1_m, _ = precision_recall_fscore_support(
                            y_val, val_preds, average="macro", zero_division=0
                        )
                        rec_prec_w, rec_rec_w, rec_f1_w, _ = precision_recall_fscore_support(
                            y_val, val_preds, average="weighted", zero_division=0
                        )

                        # Speaker-level metrics (Soft Voting)
                        spk_correct_soft = 0
                        spk_correct_hard = 0
                        spk_true_list = []
                        spk_pred_soft_list = []

                        for spk_id, recs in val_speaker_map.items():
                            true_c = recs[0]["true_cls"]
                            spk_probs = np.mean([val_probs[r["idx"]] for r in recs], axis=0)
                            soft_pred_c = classes[int(np.argmax(spk_probs))]

                            # Majority vote
                            pred_classes = [classes[val_preds[r["idx"]]] for r in recs]
                            hard_pred_c = Counter(pred_classes).most_common(1)[0][0]

                            if soft_pred_c == true_c:
                                spk_correct_soft += 1
                            if hard_pred_c == true_c:
                                spk_correct_hard += 1

                            spk_true_list.append(true_c)
                            spk_pred_soft_list.append(soft_pred_c)

                        spk_soft_acc = float(spk_correct_soft / len(val_speaker_map))
                        spk_hard_acc = float(spk_correct_hard / len(val_speaker_map))
                        _, _, spk_macro_f1, _ = precision_recall_fscore_support(
                            spk_true_list, spk_pred_soft_list, average="macro", zero_division=0
                        )

                        entry = {
                            "config_id": config_id,
                            "pooling": pooling_type,
                            "feature_dim": feat_dim,
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "min_samples_leaf": min_leaf,
                            "class_weight": cw if cw is not None else "None (normal)",
                            "val_recording_acc": round(rec_acc, 4),
                            "val_recording_macro_f1": round(float(rec_f1_m), 4),
                            "val_recording_macro_prec": round(float(rec_prec_m), 4),
                            "val_recording_macro_rec": round(float(rec_rec_m), 4),
                            "val_recording_weighted_f1": round(float(rec_f1_w), 4),
                            "val_speaker_soft_acc": round(spk_soft_acc, 4),
                            "val_speaker_hard_acc": round(spk_hard_acc, 4),
                            "val_speaker_macro_f1": round(float(spk_macro_f1), 4),
                        }
                        results.append(entry)

                        if config_id % 18 == 0 or config_id == total_configs:
                            logger.info(
                                f"  [{config_id}/{total_configs}] {pooling_type} | n_est={n_est}, depth={depth}, leaf={min_leaf}, cw={cw} "
                                f"-> Val Acc: {rec_acc*100:.2f}%, Val Macro F1: {rec_f1_m:.4f}, Val Spk Soft Acc: {spk_soft_acc*100:.1f}%"
                            )

    elapsed = time.time() - t_start
    logger.info(f"Finished evaluating {total_configs} configurations in {elapsed:.1f}s.")

    # Sort results by validation macro F1 descending, then val recording acc, then val speaker soft acc
    results_sorted = sorted(
        results,
        key=lambda x: (x["val_recording_macro_f1"], x["val_recording_acc"], x["val_speaker_soft_acc"]),
        reverse=True
    )

    best_config = results_sorted[0]
    logger.info("==================================================")
    logger.info(f"BEST CONFIGURATION SELECTED (Rank #1):")
    logger.info(f"  Config ID:         {best_config['config_id']}")
    logger.info(f"  Pooling:           {best_config['pooling']} ({best_config['feature_dim']} dims)")
    logger.info(f"  n_estimators:      {best_config['n_estimators']}")
    logger.info(f"  max_depth:         {best_config['max_depth']}")
    logger.info(f"  min_samples_leaf:  {best_config['min_samples_leaf']}")
    logger.info(f"  class_weight:      {best_config['class_weight']}")
    logger.info(f"  Val Recording Acc: {best_config['val_recording_acc']*100:.2f}%")
    logger.info(f"  Val Macro F1:      {best_config['val_recording_macro_f1']:.4f}")
    logger.info(f"  Val Weighted F1:   {best_config['val_recording_weighted_f1']:.4f}")
    logger.info(f"  Val Spk Soft Acc:  {best_config['val_speaker_soft_acc']*100:.2f}%")
    logger.info("==================================================")

    # Save complete optimization report
    output_report_path = models_dir / "validation_optimization_results.json"
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "cohort_summary": {
                "train_speakers": len(train_speakers),
                "train_recordings": len(train_rows),
                "val_speakers": len(val_speakers),
                "val_recordings": len(val_rows),
                "test_speakers_held_out": len(test_speakers),
                "test_recordings_held_out": len(test_rows),
            },
            "best_configuration": best_config,
            "all_configurations_evaluated": results,
            "ranked_configurations": results_sorted
        }, f, indent=2)

    logger.info(f"Saved full optimization results to: {output_report_path}")
    return best_config, results_sorted


if __name__ == "__main__":
    run_optimization()

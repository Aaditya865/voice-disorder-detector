import csv
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("focused_comparison")


def evaluate_on_val(model, X_val, y_val, val_rows, classes):
    # Recording-level predictions & probabilities
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    rec_acc = float(accuracy_score(y_val, val_preds))
    rec_prec_m, rec_rec_m, rec_f1_m, _ = precision_recall_fscore_support(
        y_val, val_preds, average="macro", zero_division=0
    )

    # Speaker-level soft-voting across 9 recordings per speaker
    spk_map = defaultdict(list)
    for i, r in enumerate(val_rows):
        spk_map[r["speaker_id"]].append({
            "idx": i,
            "true_cls": r["class"]
        })

    spk_correct_soft = 0
    for spk_id, recs in spk_map.items():
        true_c = recs[0]["true_cls"]
        avg_probs = np.mean([val_probs[r["idx"]] for r in recs], axis=0)
        soft_pred_c = classes[int(np.argmax(avg_probs))]
        if soft_pred_c == true_c:
            spk_correct_soft += 1

    spk_soft_acc = float(spk_correct_soft / len(spk_map))

    return {
        "val_recording_acc": round(rec_acc, 4),
        "val_recording_macro_f1": round(float(rec_f1_m), 4),
        "val_recording_macro_prec": round(float(rec_prec_m), 4),
        "val_recording_macro_rec": round(float(rec_rec_m), 4),
        "val_speaker_soft_acc": round(spk_soft_acc, 4),
        "spk_correct_soft_count": spk_correct_soft,
        "total_val_speakers": len(spk_map)
    }


def main():
    t_start_total = time.time()
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = backend_dir / "data" / "features_mean_std_cache.joblib"

    print("==================================================", flush=True)
    print("FOCUSED CLASSIFIER COMPARISON (TRAIN & VAL ONLY)", flush=True)
    print("==================================================", flush=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_csv}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached features: {cache_path}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    features_cache = joblib.load(cache_path)
    logger.info(f"Loaded {len(features_cache)} cached 1536-D features.")

    # Strictly separate partitions
    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    train_speakers = set(r["speaker_id"] for r in train_rows)
    val_speakers = set(r["speaker_id"] for r in val_rows)
    test_speakers = set(r["speaker_id"] for r in test_rows)

    # Verification: zero leakage
    assert len(train_speakers.intersection(val_speakers)) == 0, "Leakage train-val!"
    assert len(train_speakers.intersection(test_speakers)) == 0, "Leakage train-test!"
    assert len(val_speakers.intersection(test_speakers)) == 0, "Leakage val-test!"
    assert len(train_rows) == 648, f"Expected 648 train rows, got {len(train_rows)}"
    assert len(val_rows) == 135, f"Expected 135 val rows, got {len(val_rows)}"
    assert len(test_rows) == 144, f"Expected 144 test rows, got {len(test_rows)}"

    logger.info("Cohort verified: 72 training speakers (648 recs), 15 validation speakers (135 recs).")
    logger.info("The 16 held-out test speakers (144 recs) are STRICTLY UNTOUCHED.")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    y_train = le.transform([r["class"] for r in train_rows])
    y_val = le.transform([r["class"] for r in val_rows])

    X_train_raw = np.array([features_cache[r["processed_file"]] for r in train_rows], dtype=np.float32)
    X_val_raw = np.array([features_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)

    # Scaled features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # Exact 10 configurations
    configs = [
        # 1. SVM with RBF kernel (3 configs)
        {
            "id": 1,
            "classifier": "SVM (RBF)",
            "params_desc": "C=1, gamma='scale', StandardScaler=True",
            "model": SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=42),
            "scaled": True
        },
        {
            "id": 2,
            "classifier": "SVM (RBF)",
            "params_desc": "C=10, gamma='scale', StandardScaler=True",
            "model": SVC(C=10.0, gamma="scale", kernel="rbf", probability=True, random_state=42),
            "scaled": True
        },
        {
            "id": 3,
            "classifier": "SVM (RBF)",
            "params_desc": "C=100, gamma='scale', StandardScaler=True",
            "model": SVC(C=100.0, gamma="scale", kernel="rbf", probability=True, random_state=42),
            "scaled": True
        },

        # 2. Logistic Regression (3 configs)
        {
            "id": 4,
            "classifier": "Logistic Regression",
            "params_desc": "C=0.1, solver='lbfgs', max_iter=2000, StandardScaler=True",
            "model": LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, random_state=42),
            "scaled": True
        },
        {
            "id": 5,
            "classifier": "Logistic Regression",
            "params_desc": "C=1.0, solver='lbfgs', max_iter=2000, StandardScaler=True",
            "model": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42),
            "scaled": True
        },
        {
            "id": 6,
            "classifier": "Logistic Regression",
            "params_desc": "C=10.0, solver='lbfgs', max_iter=2000, StandardScaler=True",
            "model": LogisticRegression(C=10.0, solver="lbfgs", max_iter=2000, random_state=42),
            "scaled": True
        },

        # 3. Random Forest (4 configs)
        {
            "id": 7,
            "classifier": "Random Forest",
            "params_desc": "n_estimators=200, max_depth=None, max_features='sqrt', class_weight=None",
            "model": RandomForestClassifier(n_estimators=200, max_depth=None, max_features="sqrt", class_weight=None, random_state=42, n_jobs=-1),
            "scaled": False
        },
        {
            "id": 8,
            "classifier": "Random Forest",
            "params_desc": "n_estimators=200, max_depth=20, max_features='sqrt', class_weight=None",
            "model": RandomForestClassifier(n_estimators=200, max_depth=20, max_features="sqrt", class_weight=None, random_state=42, n_jobs=-1),
            "scaled": False
        },
        {
            "id": 9,
            "classifier": "Random Forest",
            "params_desc": "n_estimators=300, max_depth=None, max_features='sqrt', class_weight=None",
            "model": RandomForestClassifier(n_estimators=300, max_depth=None, max_features="sqrt", class_weight=None, random_state=42, n_jobs=-1),
            "scaled": False
        },
        {
            "id": 10,
            "classifier": "Random Forest",
            "params_desc": "n_estimators=300, max_depth=20, max_features='sqrt', class_weight=None",
            "model": RandomForestClassifier(n_estimators=300, max_depth=20, max_features="sqrt", class_weight=None, random_state=42, n_jobs=-1),
            "scaled": False
        }
    ]

    results = []
    print(f"\nEvaluating {len(configs)} focused configurations...\n", flush=True)

    for cfg in configs:
        t0 = time.time()
        X_tr = X_train_scaled if cfg["scaled"] else X_train_raw
        X_va = X_val_scaled if cfg["scaled"] else X_val_raw

        model = cfg["model"]
        model.fit(X_tr, y_train)
        fit_time = round(time.time() - t0, 3)

        eval_res = evaluate_on_val(model, X_va, y_val, val_rows, classes)

        record = {
            "config_id": cfg["id"],
            "classifier": cfg["classifier"],
            "parameters": cfg["params_desc"],
            "fit_time_seconds": fit_time,
            **eval_res
        }
        results.append(record)

        print(
            f"Config #{cfg['id']:2d} | {cfg['classifier']:19s} | {cfg['params_desc']:58s} | "
            f"Val Acc: {eval_res['val_recording_acc']*100:.2f}% | "
            f"Val Macro F1: {eval_res['val_recording_macro_f1']:.4f} | "
            f"Spk Soft Acc: {eval_res['val_speaker_soft_acc']*100:.1f}% ({eval_res['spk_correct_soft_count']}/15) | "
            f"Time: {fit_time:.2f}s",
            flush=True
        )

    total_runtime = round(time.time() - t_start_total, 2)
    print(f"\nTotal runtime: {total_runtime:.2f}s", flush=True)

    # Rank by Validation Macro F1, then Recording Accuracy, then Speaker Soft Accuracy
    ranked = sorted(
        results,
        key=lambda x: (x["val_recording_macro_f1"], x["val_recording_acc"], x["val_speaker_soft_acc"]),
        reverse=True
    )

    best = ranked[0]
    print("\n==================================================", flush=True)
    print("BEST CONFIGURATION SELECTED (BY VALIDATION MACRO F1)", flush=True)
    print("==================================================", flush=True)
    print(f"Config ID:                 #{best['config_id']}")
    print(f"Classifier:                {best['classifier']}")
    print(f"Parameters:                {best['parameters']}")
    print(f"Validation Recording Acc:  {best['val_recording_acc']*100:.2f}%")
    print(f"Validation Macro F1:       {best['val_recording_macro_f1']:.4f}")
    print(f"Validation Macro Prec:     {best['val_recording_macro_prec']:.4f}")
    print(f"Validation Macro Recall:   {best['val_recording_macro_rec']:.4f}")
    print(f"Validation Spk Soft Acc:   {best['val_speaker_soft_acc']*100:.2f}% ({best['spk_correct_soft_count']}/15 speakers)")
    print(f"Total Runtime:             {total_runtime:.2f} seconds")

    # Save to focused_comparison_results.json
    output_path = models_dir / "focused_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_configurations_tested": len(configs),
            "total_runtime_seconds": total_runtime,
            "best_configuration": best,
            "all_results": results,
            "ranked_results": ranked
        }, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

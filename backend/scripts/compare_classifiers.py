import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classifier_comparison")


def evaluate_model_on_val(model, X_val, y_val, val_rows, classes):
    """
    Evaluates a trained classifier on the validation set.
    Computes both recording-level metrics and speaker-level soft/hard voting metrics.
    """
    val_preds = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)
    elif hasattr(model, "decision_function"):
        # For classifiers without predict_proba, apply softmax on decision_function
        df = model.decision_function(X_val)
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        val_probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
    else:
        # Fallback to one-hot predictions
        val_probs = np.zeros((len(val_preds), len(classes)))
        for i, p in enumerate(val_preds):
            val_probs[i, p] = 1.0

    # Recording-level metrics
    rec_acc = float(accuracy_score(y_val, val_preds))
    rec_prec_m, rec_rec_m, rec_f1_m, _ = precision_recall_fscore_support(
        y_val, val_preds, average="macro", zero_division=0
    )
    rec_prec_w, rec_rec_w, rec_f1_w, _ = precision_recall_fscore_support(
        y_val, val_preds, average="weighted", zero_division=0
    )

    # Group recordings by speaker for speaker-level evaluation
    spk_map = defaultdict(list)
    for i, r in enumerate(val_rows):
        spk_map[r["speaker_id"]].append({
            "idx": i,
            "true_cls": r["class"]
        })

    spk_correct_soft = 0
    spk_correct_hard = 0
    spk_true_list = []
    spk_pred_soft_list = []

    for spk_id, recs in spk_map.items():
        true_c = recs[0]["true_cls"]
        # Average posterior probabilities
        avg_probs = np.mean([val_probs[r["idx"]] for r in recs], axis=0)
        soft_pred_c = classes[int(np.argmax(avg_probs))]

        # Majority vote
        pred_cls_list = [classes[val_preds[r["idx"]]] for r in recs]
        hard_pred_c = Counter(pred_cls_list).most_common(1)[0][0]

        if soft_pred_c == true_c:
            spk_correct_soft += 1
        if hard_pred_c == true_c:
            spk_correct_hard += 1

        spk_true_list.append(true_c)
        spk_pred_soft_list.append(soft_pred_c)

    spk_soft_acc = float(spk_correct_soft / len(spk_map))
    spk_hard_acc = float(spk_correct_hard / len(spk_map))
    _, _, spk_macro_f1, _ = precision_recall_fscore_support(
        spk_true_list, spk_pred_soft_list, average="macro", zero_division=0
    )

    return {
        "val_recording_acc": round(rec_acc, 4),
        "val_recording_macro_prec": round(float(rec_prec_m), 4),
        "val_recording_macro_rec": round(float(rec_rec_m), 4),
        "val_recording_macro_f1": round(float(rec_f1_m), 4),
        "val_recording_weighted_f1": round(float(rec_f1_w), 4),
        "val_speaker_soft_acc": round(spk_soft_acc, 4),
        "val_speaker_hard_acc": round(spk_hard_acc, 4),
        "val_speaker_macro_f1": round(float(spk_macro_f1), 4),
        "spk_correct_soft_count": spk_correct_soft,
        "total_val_speakers": len(spk_map)
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    backend_dir = base_dir / "backend"
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mean_std_cache_path = backend_dir / "data" / "features_mean_std_cache.joblib"

    print("==================================================", flush=True)
    print("CLASSIFIER COMPARISON & OPTIMIZATION", flush=True)
    print("STRICTLY USING TRAIN (72 SPK) & VAL (15 SPK) SETS", flush=True)
    print("==================================================", flush=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    if not mean_std_cache_path.exists():
        raise FileNotFoundError(f"Features cache not found: {mean_std_cache_path}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} metadata rows.")
    features_cache = joblib.load(mean_std_cache_path)
    logger.info(f"Loaded features cache with {len(features_cache)} entries.")

    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    # Verify strict speaker isolation
    train_speakers = set(r["speaker_id"] for r in train_rows)
    val_speakers = set(r["speaker_id"] for r in val_rows)
    test_speakers = set(r["speaker_id"] for r in test_rows)

    assert len(train_speakers.intersection(val_speakers)) == 0, "Speaker leakage train-val!"
    assert len(train_speakers.intersection(test_speakers)) == 0, "Speaker leakage train-test!"
    assert len(val_speakers.intersection(test_speakers)) == 0, "Speaker leakage val-test!"

    logger.info(f"Cohorts: Train={len(train_rows)} recs ({len(train_speakers)} spk), Val={len(val_rows)} recs ({len(val_speakers)} spk). Test is untouched.")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    le = LabelEncoder()
    le.fit(classes)

    X_train_raw = np.array([features_cache[r["processed_file"]] for r in train_rows], dtype=np.float32)
    y_train = le.transform([r["class"] for r in train_rows])

    X_val_raw = np.array([features_cache[r["processed_file"]] for r in val_rows], dtype=np.float32)
    y_val = le.transform([r["class"] for r in val_rows])

    logger.info(f"Raw feature shapes: X_train={X_train_raw.shape}, X_val={X_val_raw.shape}")

    # Standard Scaler (crucial for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    configs_to_run = []

    # -------------------------------------------------------------
    # 1. SVM with RBF Kernel
    # -------------------------------------------------------------
    for c_val in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for gamma in ["scale", "auto", 0.001, 0.0005, 0.01]:
            for cw in [None, "balanced"]:
                for scaled in [True, False]:
                    configs_to_run.append({
                        "family": "SVM (RBF)",
                        "model_fn": lambda c=c_val, g=gamma, w=cw: SVC(
                            C=c, gamma=g, kernel="rbf", class_weight=w, probability=True, random_state=42
                        ),
                        "use_scaled": scaled,
                        "params": {"C": c_val, "gamma": gamma, "class_weight": str(cw), "scaled": scaled}
                    })

    # -------------------------------------------------------------
    # 2. SVM with Linear Kernel
    # -------------------------------------------------------------
    for c_val in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        for cw in [None, "balanced"]:
            for scaled in [True, False]:
                configs_to_run.append({
                    "family": "SVM (Linear)",
                    "model_fn": lambda c=c_val, w=cw: SVC(
                        C=c, kernel="linear", class_weight=w, probability=True, random_state=42
                    ),
                    "use_scaled": scaled,
                    "params": {"C": c_val, "class_weight": str(cw), "scaled": scaled}
                })

    # -------------------------------------------------------------
    # 3. Logistic Regression
    # -------------------------------------------------------------
    for c_val in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for cw in [None, "balanced"]:
            for solver in ["lbfgs", "saga"]:
                configs_to_run.append({
                    "family": "Logistic Regression",
                    "model_fn": lambda c=c_val, w=cw, s=solver: LogisticRegression(
                        C=c, class_weight=w, solver=s, max_iter=2000, random_state=42
                    ),
                    "use_scaled": True,
                    "params": {"C": c_val, "class_weight": str(cw), "solver": solver, "scaled": True}
                })

    # -------------------------------------------------------------
    # 4. Random Forest
    # -------------------------------------------------------------
    for n_est in [200, 300, 500]:
        for depth in [None, 20, 30]:
            for min_leaf in [1, 2]:
                for max_feat in ["sqrt", "log2", 0.2]:
                    configs_to_run.append({
                        "family": "Random Forest",
                        "model_fn": lambda n=n_est, d=depth, l=min_leaf, mf=max_feat: RandomForestClassifier(
                            n_estimators=n, max_depth=d, min_samples_leaf=l, max_features=mf,
                            class_weight=None, random_state=42, n_jobs=-1
                        ),
                        "use_scaled": False,
                        "params": {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": min_leaf, "max_features": max_feat}
                    })

    # -------------------------------------------------------------
    # 5. XGBoost
    # -------------------------------------------------------------
    for n_est in [100, 200, 300]:
        for lr in [0.03, 0.05, 0.1]:
            for max_d in [3, 4, 6]:
                for colsample in [0.2, 0.4, 0.7]:
                    configs_to_run.append({
                        "family": "XGBoost",
                        "model_fn": lambda n=n_est, r=lr, d=max_d, cs=colsample: XGBClassifier(
                            n_estimators=n, learning_rate=r, max_depth=d, colsample_bytree=cs,
                            subsample=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1
                        ),
                        "use_scaled": False,
                        "params": {"n_estimators": n_est, "learning_rate": lr, "max_depth": max_d, "colsample_bytree": colsample}
                    })

    # -------------------------------------------------------------
    # 6. HistGradientBoostingClassifier
    # -------------------------------------------------------------
    for max_iter in [100, 200]:
        for lr in [0.03, 0.05, 0.1]:
            for max_leaf in [15, 31]:
                configs_to_run.append({
                    "family": "Gradient Boosting (Hist)",
                    "model_fn": lambda mi=max_iter, r=lr, ml=max_leaf: HistGradientBoostingClassifier(
                        max_iter=mi, learning_rate=r, max_leaf_nodes=ml, random_state=42
                    ),
                    "use_scaled": False,
                    "params": {"max_iter": max_iter, "learning_rate": lr, "max_leaf_nodes": max_leaf}
                })

    logger.info(f"Total configurations queued across 6 model families: {len(configs_to_run)}")

    results = []
    t_start = time.time()

    for idx, cfg in enumerate(configs_to_run, 1):
        family = cfg["family"]
        use_scaled = cfg["use_scaled"]
        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_va = X_val_scaled if use_scaled else X_val_raw

        try:
            model = cfg["model_fn"]()
            t0 = time.time()
            model.fit(X_tr, y_train)
            fit_time = round(time.time() - t0, 3)

            eval_res = evaluate_model_on_val(model, X_va, y_val, val_rows, classes)

            record = {
                "config_id": idx,
                "family": family,
                "params": cfg["params"],
                "fit_time_seconds": fit_time,
                **eval_res
            }
            results.append(record)

            if idx % 25 == 0 or idx == len(configs_to_run):
                logger.info(
                    f"[{idx}/{len(configs_to_run)}] {family:20s} | "
                    f"Val Acc: {eval_res['val_recording_acc']*100:.2f}% | "
                    f"Val Macro F1: {eval_res['val_recording_macro_f1']:.4f} | "
                    f"Spk Soft Acc: {eval_res['val_speaker_soft_acc']*100:.1f}% ({eval_res['spk_correct_soft_count']}/15)"
                )
        except Exception as e:
            logger.warning(f"Config {idx} ({family}) failed: {e}")

    total_time = round(time.time() - t_start, 1)
    logger.info(f"Completed evaluation of {len(results)} configurations in {total_time}s.")

    # Sort results by validation macro F1, then recording accuracy, then speaker soft accuracy
    sorted_by_f1 = sorted(
        results,
        key=lambda x: (x["val_recording_macro_f1"], x["val_recording_acc"], x["val_speaker_soft_acc"]),
        reverse=True
    )

    # Sort results by speaker soft voting accuracy
    sorted_by_spk = sorted(
        results,
        key=lambda x: (x["val_speaker_soft_acc"], x["val_recording_macro_f1"], x["val_recording_acc"]),
        reverse=True
    )

    # Best per family
    families = sorted(list(set(r["family"] for r in results)))
    best_per_family = {}
    for fam in families:
        fam_results = [r for r in results if r["family"] == fam]
        best_fam = sorted(fam_results, key=lambda x: (x["val_recording_macro_f1"], x["val_speaker_soft_acc"]), reverse=True)[0]
        best_per_family[fam] = best_fam

    # Save to JSON
    output_path = models_dir / "classifier_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
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
            "best_overall_config": sorted_by_f1[0],
            "best_speaker_acc_config": sorted_by_spk[0],
            "best_per_family": best_per_family,
            "top_15_by_macro_f1": sorted_by_f1[:15],
            "top_15_by_speaker_acc": sorted_by_spk[:15],
            "all_results": results
        }, f, indent=2)

    logger.info(f"Saved full comparison results to: {output_path}")

    print("\n==================================================", flush=True)
    print("BEST CONFIGURATION PER MODEL FAMILY (VALIDATION SET)", flush=True)
    print("==================================================", flush=True)
    for fam, b in best_per_family.items():
        print(f"{fam:24s} -> Val Acc: {b['val_recording_acc']*100:.2f}% | Val Macro F1: {b['val_recording_macro_f1']:.4f} | Spk Soft Acc: {b['val_speaker_soft_acc']*100:.1f}% ({b['spk_correct_soft_count']}/15)")
        print(f"   Params: {b['params']}")


if __name__ == "__main__":
    main()

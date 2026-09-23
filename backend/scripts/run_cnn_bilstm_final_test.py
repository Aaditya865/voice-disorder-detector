import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

# Setup project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.cnn_bilstm_attention.dataset import LaryngealSpectrogramDataset
from ml.cnn_bilstm_attention.model import AudioCNNBilstmAttention
from ml.cnn_bilstm_attention.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cnn_bilstm_final_test")


def compute_speaker_level_per_class(
    speaker_breakdown: List[Dict[str, Any]],
    classes: List[str]
) -> Dict[str, Any]:
    """
    Computes per-class speaker-level metrics for soft-voting and majority-voting.
    """
    y_true = [s["true_class"] for s in speaker_breakdown]
    y_pred_soft = [s["predicted_class_soft"] for s in speaker_breakdown]
    y_pred_hard = [s["predicted_class_majority"] for s in speaker_breakdown]

    p_soft, r_soft, f_soft, s_soft = precision_recall_fscore_support(
        y_true, y_pred_soft, labels=classes, zero_division=0
    )
    p_hard, r_hard, f_hard, s_hard = precision_recall_fscore_support(
        y_true, y_pred_hard, labels=classes, zero_division=0
    )

    per_class_speaker = []
    for idx, cls_name in enumerate(classes):
        per_class_speaker.append({
            "class": cls_name,
            "speaker_support": int(s_soft[idx]),
            "soft_voting": {
                "precision": round(float(p_soft[idx]), 4),
                "recall": round(float(r_soft[idx]), 4),
                "f1_score": round(float(f_soft[idx]), 4),
                "correct_speakers": sum(1 for s in speaker_breakdown if s["true_class"] == cls_name and s["correct_soft"])
            },
            "majority_voting": {
                "precision": round(float(p_hard[idx]), 4),
                "recall": round(float(r_hard[idx]), 4),
                "f1_score": round(float(f_hard[idx]), 4),
                "correct_speakers": sum(1 for s in speaker_breakdown if s["true_class"] == cls_name and s["correct_majority"])
            }
        })

    return {
        "per_class_speaker_metrics": per_class_speaker,
        "macro_soft": {
            "precision": round(float(np.mean(p_soft)), 4),
            "recall": round(float(np.mean(r_soft)), 4),
            "f1": round(float(np.mean(f_soft)), 4)
        },
        "macro_hard": {
            "precision": round(float(np.mean(p_hard)), 4),
            "recall": round(float(np.mean(r_hard)), 4),
            "f1": round(float(np.mean(f_hard)), 4)
        }
    }


def main():
    print("=================================================================", flush=True)
    print("FINAL HELD-OUT TEST EVALUATION: 2D CNN + BiLSTM + ATTENTION", flush=True)
    print("STRICTLY EVALUATING ON THE 16 HELD-OUT TEST SPEAKERS (144 CLIPS)", flush=True)
    print("=================================================================", flush=True)

    metadata_csv = BASE_DIR / "dataset" / "metadata.csv"
    cache_path = BACKEND_DIR / "data" / "spectrograms_cache.joblib"
    checkpoint_path = BACKEND_DIR / "models" / "cnn_bilstm_attention" / "Config_A_Baseline_Light_best.pt"
    output_results_path = BACKEND_DIR / "models" / "cnn_bilstm_attention" / "final_test_results.json"
    rf_eval_path = BACKEND_DIR / "models" / "evaluation_results.json"
    cnn_val_path = BACKEND_DIR / "models" / "cnn_bilstm_attention" / "validation_results.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Dataset Partition Isolation Audit
    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    train_spks = set(r["speaker_id"] for r in train_rows)
    val_spks = set(r["speaker_id"] for r in val_rows)
    test_spks = set(r["speaker_id"] for r in test_rows)

    assert len(train_spks.intersection(test_spks)) == 0, "Test set contaminated with training speakers!"
    assert len(val_spks.intersection(test_spks)) == 0, "Test set contaminated with validation speakers!"
    assert len(test_rows) == 144, f"Expected 144 test recordings, found {len(test_rows)}"
    assert len(test_spks) == 16, f"Expected 16 test speakers, found {len(test_spks)}"

    logger.info("Test Set Integrity & Isolation Verified:")
    logger.info(f"  Test Speakers: {len(test_spks)} distinct speakers (completely held-out)")
    logger.info(f"  Test Recordings: {len(test_rows)} verified WAV recordings")

    # 2. Load Checkpoint & Validate Metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    logger.info(f"Loaded checkpoint '{ckpt['config_name']}' from epoch {ckpt['best_epoch']} (Val loss: {ckpt['best_val_loss']:.4f})")
    logger.info(f"Classes ({len(classes)}): {classes}")

    # 3. Load Spectrogram Cache
    spectrogram_cache = joblib.load(cache_path)
    test_dataset = LaryngealSpectrogramDataset(test_rows, spectrogram_cache, class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

    # 4. Instantiate Model Architecture (Config A)
    model = AudioCNNBilstmAttention(
        n_classes=len(classes),
        cnn_channels=[32, 64, 128],
        lstm_hidden=64,
        lstm_layers=1,
        dense_dropout=0.3,
        cnn_dropout=0.1
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    # 5. Execute Evaluation
    logger.info("Running inference on 144 held-out test recordings across 16 unseen speakers...")
    eval_results = evaluate_model(model, test_loader, classes, device)

    rec_metrics = eval_results["recording_level"]
    spk_metrics = eval_results["speaker_level"]
    cm = eval_results["confusion_matrix"]
    per_class_rec = eval_results["per_class_metrics"]
    speaker_breakdown = spk_metrics["speaker_breakdown"]

    # 6. Compute Per-Class Speaker-Level Metrics
    spk_class_analysis = compute_speaker_level_per_class(speaker_breakdown, classes)

    # 7. Load Wav2Vec2 + Random Forest Baseline & CNN Validation for Comparison
    rf_eval = {}
    if rf_eval_path.exists():
        with open(rf_eval_path, "r", encoding="utf-8") as f:
            rf_eval = json.load(f)

    cnn_val = {}
    if cnn_val_path.exists():
        with open(cnn_val_path, "r", encoding="utf-8") as f:
            cnn_data = json.load(f)
            for c in cnn_data.get("all_configurations", []):
                if c["config_name"] == "Config_A_Baseline_Light":
                    cnn_val = c
                    break

    # 8. Assemble Complete Final Test JSON Payload
    final_output = {
        "evaluation_type": "FINAL HELD-OUT TEST EVALUATION",
        "evaluated_at": datetime.now().isoformat(),
        "model_architecture": "2D CNN + BiLSTM + Attention (Config A Baseline Light)",
        "model_checkpoint": str(checkpoint_path.relative_to(BASE_DIR)),
        "checkpoint_epoch": ckpt["best_epoch"],
        "checkpoint_val_loss": round(float(ckpt["best_val_loss"]), 4),
        "classes": classes,
        "cohort_summary": {
            "test_speakers": len(test_spks),
            "test_recordings": len(test_rows),
            "recordings_per_speaker": 9,
            "speaker_isolation": "Strictly held-out; never used during training, tuning, or selection"
        },
        "test_performance": {
            "recording_level": rec_metrics,
            "speaker_level": {
                "soft_voting_accuracy": spk_metrics["soft_voting_accuracy"],
                "majority_voting_accuracy": spk_metrics["majority_voting_accuracy"],
                "correct_soft_speakers": spk_metrics["correct_soft_speakers"],
                "correct_majority_speakers": spk_metrics["correct_hard_speakers"],
                "total_speakers": spk_metrics["total_speakers"]
            }
        },
        "per_class_recording_metrics": per_class_rec,
        "per_class_speaker_metrics": spk_class_analysis["per_class_speaker_metrics"],
        "speaker_macro_metrics": {
            "soft_voting": spk_class_analysis["macro_soft"],
            "majority_voting": spk_class_analysis["macro_hard"]
        },
        "confusion_matrix": cm,
        "per_speaker_predictions": speaker_breakdown,
        "comparisons": {
            "vs_wav2vec2_rf": {
                "rf_model": "Wav2Vec2 + Random Forest (Config #9)",
                "rf_test_recording_accuracy": rf_eval.get("test_performance", {}).get("accuracy"),
                "rf_test_macro_f1": rf_eval.get("test_performance", {}).get("macro_f1"),
                "rf_test_macro_precision": rf_eval.get("test_performance", {}).get("macro_precision"),
                "rf_test_macro_recall": rf_eval.get("test_performance", {}).get("macro_recall"),
                "rf_test_speaker_soft_accuracy": rf_eval.get("test_performance", {}).get("speaker_level_soft_voting_accuracy"),
                "rf_test_speaker_majority_accuracy": rf_eval.get("test_performance", {}).get("speaker_level_majority_accuracy"),
                "cnn_test_recording_accuracy": rec_metrics["accuracy"],
                "cnn_test_macro_f1": rec_metrics["macro_f1"],
                "cnn_test_speaker_soft_accuracy": spk_metrics["soft_voting_accuracy"],
                "cnn_test_speaker_majority_accuracy": spk_metrics["majority_voting_accuracy"],
                "delta_recording_accuracy": round(rec_metrics["accuracy"] - (rf_eval.get("test_performance", {}).get("accuracy") or 0.0), 4),
                "delta_macro_f1": round(rec_metrics["macro_f1"] - (rf_eval.get("test_performance", {}).get("macro_f1") or 0.0), 4),
                "delta_speaker_soft_accuracy": round(spk_metrics["soft_voting_accuracy"] - (rf_eval.get("test_performance", {}).get("speaker_level_soft_voting_accuracy") or 0.0), 4),
                "delta_speaker_majority_accuracy": round(spk_metrics["majority_voting_accuracy"] - (rf_eval.get("test_performance", {}).get("speaker_level_majority_accuracy") or 0.0), 4)
            },
            "vs_cnn_validation": {
                "cnn_val_recording_accuracy": cnn_val.get("validation_recording_accuracy"),
                "cnn_val_macro_f1": cnn_val.get("validation_macro_f1"),
                "cnn_val_speaker_soft_accuracy": cnn_val.get("validation_speaker_soft_accuracy"),
                "cnn_val_speaker_majority_accuracy": cnn_val.get("validation_speaker_majority_accuracy"),
                "cnn_test_recording_accuracy": rec_metrics["accuracy"],
                "cnn_test_macro_f1": rec_metrics["macro_f1"],
                "cnn_test_speaker_soft_accuracy": spk_metrics["soft_voting_accuracy"],
                "cnn_test_speaker_majority_accuracy": spk_metrics["majority_voting_accuracy"],
                "delta_recording_accuracy": round(rec_metrics["accuracy"] - (cnn_val.get("validation_recording_accuracy") or 0.0), 4),
                "delta_macro_f1": round(rec_metrics["macro_f1"] - (cnn_val.get("validation_macro_f1") or 0.0), 4),
                "delta_speaker_soft_accuracy": round(spk_metrics["soft_voting_accuracy"] - (cnn_val.get("validation_speaker_soft_accuracy") or 0.0), 4),
                "delta_speaker_majority_accuracy": round(spk_metrics["majority_voting_accuracy"] - (cnn_val.get("validation_speaker_majority_accuracy") or 0.0), 4)
            }
        }
    }

    # 9. Save to backend/models/cnn_bilstm_attention/final_test_results.json
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Saved complete final test results to: {output_results_path}")

    # 10. Print Structured Final Report
    print("\n" + "=" * 90, flush=True)
    print("FINAL HELD-OUT TEST EVALUATION REPORT (2D CNN + BiLSTM + ATTENTION)", flush=True)
    print("=" * 90, flush=True)
    print(f"Model Checkpoint:       backend/models/cnn_bilstm_attention/Config_A_Baseline_Light_best.pt")
    print(f"Evaluated Test Cohort:  16 unseen speakers | 144 recordings (9 per speaker)")
    print("-" * 90, flush=True)
    print(f"1.  Test Recording-Level Accuracy:       {rec_metrics['accuracy']*100:.2f}% ({rec_metrics['correct_recordings']}/144)")
    print(f"2.  Macro Precision:                     {rec_metrics['macro_precision']*100:.2f}% ({rec_metrics['macro_precision']:.4f})")
    print(f"3.  Macro Recall:                        {rec_metrics['macro_recall']*100:.2f}% ({rec_metrics['macro_recall']:.4f})")
    print(f"4.  Macro F1-Score:                      {rec_metrics['macro_f1']:.4f}")
    print(f"5.  Weighted F1-Score:                   {rec_metrics['weighted_f1']:.4f}")
    print(f"8.  Speaker-Level Soft-Voting Accuracy:  {spk_metrics['soft_voting_accuracy']*100:.2f}% ({spk_metrics['correct_soft_speakers']}/16)")
    print(f"9.  Speaker-Level Majority-Voting Acc:   {spk_metrics['majority_voting_accuracy']*100:.2f}% ({spk_metrics['correct_hard_speakers']}/16)")
    print(f"10. Correct / Total Test Speakers:       {spk_metrics['correct_soft_speakers']}/16 (Soft) | {spk_metrics['correct_hard_speakers']}/16 (Majority)")
    print("-" * 90, flush=True)

    print("\n6. PER-CLASS RECORDING-LEVEL METRICS:")
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 72)
    for p in per_class_rec:
        print(f"{p['class']:<25} | {p['precision']:<10.4f} | {p['recall']:<10.4f} | {p['f1_score']:<10.4f} | {p['support']:<8}")

    print("\n7. CONFUSION MATRIX (Recording-Level):")
    header = "True \\ Pred".ljust(22) + " ".join([f"[{c[:8]:>8}]" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<20} | " + " ".join([f"{val:>8d} " for val in row])
        print(row_str)

    print("\n11. PER-CLASS SPEAKER-LEVEL RESULTS:")
    print(f"{'Class':<25} | {'Speakers':<9} | {'Soft Correct':<13} | {'Soft F1':<8} | {'Maj Correct':<12} | {'Maj F1':<8}")
    print("-" * 85)
    for spk_c in spk_class_analysis["per_class_speaker_metrics"]:
        print(f"{spk_c['class']:<25} | {spk_c['speaker_support']:<9} | "
              f"{spk_c['soft_voting']['correct_speakers']:<13} | {spk_c['soft_voting']['f1_score']:<8.4f} | "
              f"{spk_c['majority_voting']['correct_speakers']:<12} | {spk_c['majority_voting']['f1_score']:<8.4f}")

    print("\nPER-SPEAKER PREDICTIONS BREAKDOWN (16 HELD-OUT TEST SPEAKERS):")
    print(f"{'Speaker ID':<12} | {'True Class':<22} | {'Pred Soft':<22} | {'Pred Majority':<22} | {'Soft Correct':<12} | {'Maj Correct':<12}")
    print("-" * 110)
    for s in speaker_breakdown:
        print(f"{s['speaker_id']:<12} | {s['true_class']:<22} | {s['predicted_class_soft']:<22} | "
              f"{s['predicted_class_majority']:<22} | {str(s['correct_soft']):<12} | {str(s['correct_majority']):<12}")

    print("\nCOMPARISON WITH WAV2VEC2 + RANDOM FOREST & CNN VALIDATION:")
    print(f"{'Model':<42} | {'Rec Acc':<9} | {'Macro F1':<9} | {'Spk Soft Acc':<13} | {'Spk Maj Acc':<13}")
    print("-" * 92)
    rf_t = rf_eval.get("test_performance", {})
    print(f"{'Wav2Vec2 + Random Forest (Test)':<42} | "
          f"{rf_t.get('accuracy', 0)*100:6.2f}% | "
          f"{rf_t.get('macro_f1', 0):8.4f} | "
          f"{rf_t.get('speaker_level_soft_voting_accuracy', 0)*100:6.1f}% (8/16)  | "
          f"{rf_t.get('speaker_level_majority_accuracy', 0)*100:6.1f}% (5/16)")

    print(f"{'2D CNN + BiLSTM + Attention (Val)':<42} | "
          f"{cnn_val.get('validation_recording_accuracy', 0)*100:6.2f}% | "
          f"{cnn_val.get('validation_macro_f1', 0):8.4f} | "
          f"{cnn_val.get('validation_speaker_soft_accuracy', 0)*100:6.1f}% (6/15)  | "
          f"{cnn_val.get('validation_speaker_majority_accuracy', 0)*100:6.1f}% (8/15)")

    print(f"{'2D CNN + BiLSTM + Attention (Test)':<42} | "
          f"{rec_metrics['accuracy']*100:6.2f}% | "
          f"{rec_metrics['macro_f1']:8.4f} | "
          f"{spk_metrics['soft_voting_accuracy']*100:6.1f}% ({spk_metrics['correct_soft_speakers']}/16) | "
          f"{spk_metrics['majority_voting_accuracy']*100:6.1f}% ({spk_metrics['correct_hard_speakers']}/16)")
    print("=" * 92 + "\n", flush=True)


if __name__ == "__main__":
    main()

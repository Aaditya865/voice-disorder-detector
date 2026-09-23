import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.cnn_bilstm_attention.dataset import (
    precompute_and_cache_spectrograms,
    LaryngealSpectrogramDataset,
)
from ml.cnn_bilstm_attention.model import AudioCNNBilstmAttention
from ml.cnn_bilstm_attention.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cnn_bilstm_experiment")


def main():
    print("==================================================", flush=True)
    print("2D CNN + BiLSTM + ATTENTION MODEL EXPERIMENT", flush=True)
    print("STRICTLY USING TRAIN (72 SPK) & VAL (15 SPK) SETS", flush=True)
    print("==================================================", flush=True)

    dataset_dir = BASE_DIR / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    data_dir = BACKEND_DIR / "data"
    models_dir = BACKEND_DIR / "models" / "cnn_bilstm_attention"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "spectrograms_cache.joblib"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} total metadata rows.")

    # 1. Dataset Audit and Partition Isolation
    train_rows = [r for r in metadata_rows if r["split"] == "train"]
    val_rows = [r for r in metadata_rows if r["split"] == "val"]
    test_rows = [r for r in metadata_rows if r["split"] == "test"]

    train_spks = set(r["speaker_id"] for r in train_rows)
    val_spks = set(r["speaker_id"] for r in val_rows)
    test_spks = set(r["speaker_id"] for r in test_rows)

    assert len(train_spks.intersection(val_spks)) == 0, "Leakage between Train and Val!"
    assert len(train_spks.intersection(test_spks)) == 0, "Leakage between Train and Test!"
    assert len(val_spks.intersection(test_spks)) == 0, "Leakage between Val and Test!"

    assert len(train_rows) == 648, f"Expected 648 train rows, got {len(train_rows)}"
    assert len(val_rows) == 135, f"Expected 135 val rows, got {len(val_rows)}"
    assert len(test_rows) == 144, f"Expected 144 test rows, got {len(test_rows)}"

    logger.info(f"Strict Speaker Split Verified:")
    logger.info(f"  Train:      {len(train_spks)} distinct speakers ({len(train_rows)} recordings)")
    logger.info(f"  Validation: {len(val_spks)} distinct speakers ({len(val_rows)} recordings)")
    logger.info(f"  Test:       {len(test_spks)} distinct speakers ({len(test_rows)} recordings) [HELD-OUT & UNTOUCHED]")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    logger.info(f"Target classes ({len(classes)}): {classes}")

    # 2. Precompute Log-Mel Spectrograms
    spectrogram_cache = precompute_and_cache_spectrograms(metadata_rows, BASE_DIR, cache_path)

    # 3. Create Datasets and DataLoaders
    train_dataset = LaryngealSpectrogramDataset(train_rows, spectrogram_cache, class_to_idx)
    val_dataset = LaryngealSpectrogramDataset(val_rows, spectrogram_cache, class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 4. Define the 3 Experimental Configurations
    configs = [
        {
            "name": "Config_A_Baseline_Light",
            "description": "3-stage CNN (32->64->128) + 1-layer BiLSTM (h=64) + Attention + Standard CE Loss",
            "model_args": {
                "n_classes": len(classes),
                "cnn_channels": [32, 64, 128],
                "lstm_hidden": 64,
                "lstm_layers": 1,
                "dense_dropout": 0.3,
                "cnn_dropout": 0.1
            },
            "train_args": {
                "epochs": 35,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "use_class_weights": False,
                "label_smoothing": 0.0,
                "patience": 8
            }
        },
        {
            "name": "Config_B_Deeper_Temporal",
            "description": "4-stage CNN (32->64->128->128) + 2-layer BiLSTM (h=128) + Attention + Higher Dropout",
            "model_args": {
                "n_classes": len(classes),
                "cnn_channels": [32, 64, 128, 128],
                "lstm_hidden": 128,
                "lstm_layers": 2,
                "dense_dropout": 0.4,
                "cnn_dropout": 0.15
            },
            "train_args": {
                "epochs": 35,
                "lr": 7e-4,
                "weight_decay": 1e-4,
                "use_class_weights": False,
                "label_smoothing": 0.0,
                "patience": 8
            }
        },
        {
            "name": "Config_C_ClassWeighted_Regularized",
            "description": "3-stage CNN + 1-layer BiLSTM + Attention + Class-Balanced Loss + Label Smoothing",
            "model_args": {
                "n_classes": len(classes),
                "cnn_channels": [32, 64, 128],
                "lstm_hidden": 64,
                "lstm_layers": 1,
                "dense_dropout": 0.3,
                "cnn_dropout": 0.1
            },
            "train_args": {
                "epochs": 35,
                "lr": 1e-3,
                "weight_decay": 2e-4,
                "use_class_weights": True,
                "label_smoothing": 0.05,
                "patience": 8
            }
        }
    ]

    results = []

    for cfg in configs:
        c_name = cfg["name"]
        print(f"\n>>> Starting Training: {c_name}...", flush=True)
        print(f"    Description: {cfg['description']}", flush=True)

        model = AudioCNNBilstmAttention(**cfg["model_args"])
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable parameters: {param_count:,}", flush=True)

        trained_model, summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=classes,
            device=device,
            checkpoint_dir=models_dir,
            config_name=c_name,
            **cfg["train_args"]
        )

        val_rec = summary["validation_metrics"]["recording_level"]
        val_spk = summary["validation_metrics"]["speaker_level"]

        res_entry = {
            "config_name": c_name,
            "description": cfg["description"],
            "parameters": param_count,
            "architecture_details": {
                "model_args": cfg["model_args"],
                "train_args": cfg["train_args"]
            },
            "best_epoch": summary["best_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "total_epochs_trained": summary["total_epochs"],
            "training_time_seconds": summary["training_time_seconds"],
            "validation_recording_accuracy": val_rec["accuracy"],
            "validation_macro_f1": val_rec["macro_f1"],
            "validation_macro_precision": val_rec["macro_precision"],
            "validation_macro_recall": val_rec["macro_recall"],
            "validation_weighted_f1": val_rec["weighted_f1"],
            "validation_speaker_soft_accuracy": val_spk["soft_voting_accuracy"],
            "validation_speaker_majority_accuracy": val_spk["majority_voting_accuracy"],
            "speaker_soft_correct_count": val_spk["correct_soft_speakers"],
            "speaker_hard_correct_count": val_spk["correct_hard_speakers"],
            "per_class_metrics": summary["validation_metrics"]["per_class_metrics"],
            "speaker_breakdown": val_spk["speaker_breakdown"]
        }
        results.append(res_entry)

        print(f"    Finished {c_name} in {summary['training_time_seconds']:.1f}s (Best Epoch: {summary['best_epoch']}):")
        print(f"      Val Recording Acc: {val_rec['accuracy']*100:.2f}% | Val Macro F1: {val_rec['macro_f1']:.4f}")
        print(f"      Val Spk Soft Acc:  {val_spk['soft_voting_accuracy']*100:.1f}% ({val_spk['correct_soft_speakers']}/15)")
        print(f"      Val Spk Hard Acc:  {val_spk['majority_voting_accuracy']*100:.1f}% ({val_spk['correct_hard_speakers']}/15)")

    # 5. Select Best Configuration Based on Validation Macro F1 and Accuracy
    ranked = sorted(
        results,
        key=lambda x: (x["validation_macro_f1"], x["validation_recording_accuracy"], x["validation_speaker_soft_accuracy"]),
        reverse=True
    )
    best_config = ranked[0]

    # 6. Save Complete Validation Results to JSON
    results_path = models_dir / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "model_family": "2D CNN + BiLSTM + Attention",
            "classes": classes,
            "cohort_summary": {
                "train_speakers": len(train_spks),
                "train_recordings": len(train_rows),
                "val_speakers": len(val_spks),
                "val_recordings": len(val_rows),
                "test_speakers_held_out": len(test_spks),
                "test_recordings_held_out": len(test_rows),
            },
            "best_configuration": best_config["config_name"],
            "all_configurations": results,
            "ranked_configurations": ranked
        }, f, indent=2)

    logger.info(f"Saved complete experiment results to: {results_path}")

    # Output Final Summary
    print("\n==================================================", flush=True)
    print("ALL 3 CONFIGURATIONS COMPLETED", flush=True)
    print("==================================================", flush=True)
    for r in results:
        print(f"{r['config_name']:34s} | Time: {r['training_time_seconds']:5.1f}s | "
              f"Val Acc: {r['validation_recording_accuracy']*100:5.2f}% | "
              f"Val Macro F1: {r['validation_macro_f1']:.4f} | "
              f"Spk Soft: {r['validation_speaker_soft_accuracy']*100:4.1f}% | "
              f"Spk Hard: {r['validation_speaker_majority_accuracy']*100:4.1f}%")

    print(f"\nBEST CONFIGURATION SELECTED: {best_config['config_name']}")
    print(f"  Description:             {best_config['description']}")
    print(f"  Validation Recording Acc:{best_config['validation_recording_accuracy']*100:.2f}%")
    print(f"  Validation Macro F1:     {best_config['validation_macro_f1']:.4f}")
    print(f"  Val Speaker Soft Acc:    {best_config['validation_speaker_soft_accuracy']*100:.1f}% ({best_config['speaker_soft_correct_count']}/15)")
    print(f"  Val Speaker Hard Acc:    {best_config['validation_speaker_majority_accuracy']*100:.1f}% ({best_config['speaker_hard_correct_count']}/15)")
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.cnn_bilstm_attention.dataset import (
    compute_log_mel_spectrogram,
    precompute_and_cache_spectrograms,
    LaryngealSpectrogramDataset,
)
from ml.cnn_bilstm_attention.model import AudioCNNBilstmAttention
from ml.cnn_bilstm_attention.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("augmented_experiment")


class AugmentedSpectrogramDataset(Dataset):
    """
    Dataset wrapping precomputed in-memory or cached spectrogram arrays:
    (spectrogram, label_idx, speaker_id, identifier)
    """
    def __init__(self, samples: List[Tuple[np.ndarray, int, str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        spec_np, label_idx, spk_id, identifier = self.samples[idx]
        spec_tensor = torch.from_numpy(spec_np).float()
        return spec_tensor, label_idx, spk_id, identifier


def generate_augmented_training_data(
    train_rows: List[Dict[str, str]],
    original_cache: Dict[str, np.ndarray],
    class_to_idx: Dict[str, int],
    aug_cache_path: Path
) -> List[Tuple[np.ndarray, int, str, str]]:
    """
    Generates 2 conservative, safe acoustic variations per training recording:
    Variation 1:
      - Circular time shift: +4% to +8%
      - Volume gain: +0.8 dB to +1.8 dB
      - Subtle additive Gaussian noise (sigma = 0.0020 * std(audio))
    Variation 2:
      - Circular time shift: -4% to -8%
      - Volume attenuation: -1.8 dB to -0.8 dB
      - Subtle additive Gaussian noise (sigma = 0.0025 * std(audio))
    
    Total training size: 648 original + 648 aug1 + 648 aug2 = 1,944 samples.
    """
    aug_cache = {}
    if aug_cache_path.exists():
        try:
            aug_cache = joblib.load(aug_cache_path)
            logger.info(f"Loaded existing augmented training cache with {len(aug_cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load augmented cache ({e}), recomputing.")
            aug_cache = {}

    training_samples: List[Tuple[np.ndarray, int, str, str]] = []
    missing_augs = False

    for idx, row in enumerate(train_rows):
        rel_p = row["processed_file"]
        spk_id = row["speaker_id"]
        label_idx = class_to_idx[row["class"]]

        # 1. Original sample from primary cache
        orig_spec = original_cache[rel_p]
        training_samples.append((orig_spec, label_idx, spk_id, rel_p))

        # Check if augmentations are already cached
        key_aug1 = f"{rel_p}_aug1"
        key_aug2 = f"{rel_p}_aug2"

        if key_aug1 in aug_cache and key_aug2 in aug_cache:
            training_samples.append((aug_cache[key_aug1], label_idx, spk_id, key_aug1))
            training_samples.append((aug_cache[key_aug2], label_idx, spk_id, key_aug2))
        else:
            missing_augs = True
            full_p = BASE_DIR / rel_p
            audio, sr = sf.read(str(full_p))
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio_std = float(np.std(audio)) + 1e-7

            rng = np.random.RandomState(42 + idx)

            # --- Augmentation 1: Forward shift, positive gain, subtle noise ---
            shift_pct_1 = rng.uniform(0.04, 0.08)
            shift_samples_1 = int(len(audio) * shift_pct_1)
            gain_db_1 = rng.uniform(0.8, 1.8)
            gain_1 = 10.0 ** (gain_db_1 / 20.0)
            noise_1 = rng.normal(0, 0.0020 * audio_std, len(audio)).astype(np.float32)

            aug_audio_1 = (np.roll(audio, shift_samples_1) * gain_1 + noise_1).astype(np.float32)
            spec_aug1 = compute_log_mel_spectrogram(aug_audio_1, sr=sr)
            aug_cache[key_aug1] = spec_aug1
            training_samples.append((spec_aug1, label_idx, spk_id, key_aug1))

            # --- Augmentation 2: Backward shift, attenuation, subtle noise ---
            shift_pct_2 = rng.uniform(-0.08, -0.04)
            shift_samples_2 = int(len(audio) * shift_pct_2)
            gain_db_2 = rng.uniform(-1.8, -0.8)
            gain_2 = 10.0 ** (gain_db_2 / 20.0)
            noise_2 = rng.normal(0, 0.0025 * audio_std, len(audio)).astype(np.float32)

            aug_audio_2 = (np.roll(audio, shift_samples_2) * gain_2 + noise_2).astype(np.float32)
            spec_aug2 = compute_log_mel_spectrogram(aug_audio_2, sr=sr)
            aug_cache[key_aug2] = spec_aug2
            training_samples.append((spec_aug2, label_idx, spk_id, key_aug2))

    if missing_augs:
        aug_cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(aug_cache, aug_cache_path)
        logger.info(f"Saved {len(aug_cache)} augmented spectrograms to {aug_cache_path}")

    return training_samples


def main():
    print("==================================================", flush=True)
    print("2D CNN + BiLSTM + ATTENTION: TRAINING-ONLY AUGMENTATION EXPERIMENT", flush=True)
    print("STRICTLY USING TRAIN (72 SPK) & VAL (15 SPK) SETS", flush=True)
    print("==================================================", flush=True)

    dataset_dir = BASE_DIR / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    data_dir = BACKEND_DIR / "data"
    augmented_models_dir = BACKEND_DIR / "models" / "cnn_bilstm_attention_augmented"
    augmented_models_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results_path = BACKEND_DIR / "models" / "cnn_bilstm_attention" / "validation_results.json"
    primary_cache_path = data_dir / "spectrograms_cache.joblib"
    aug_cache_path = data_dir / "augmented_train_spectrograms_cache.joblib"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(metadata_rows)} total metadata rows.")

    # 1. Dataset Partition Audit and Strict Leakage Prevention
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

    logger.info("Strict Speaker Partitioning Verified:")
    logger.info(f"  Training:   {len(train_spks)} distinct speakers ({len(train_rows)} original recordings)")
    logger.info(f"  Validation: {len(val_spks)} distinct speakers ({len(val_rows)} recordings) [UNTOUCHED & UNMODIFIED]")
    logger.info(f"  Test:       {len(test_spks)} distinct speakers ({len(test_rows)} recordings) [HELD-OUT & COMPLETELY UNTOUCHED]")

    classes = sorted(list(set(r["class"] for r in metadata_rows)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    logger.info(f"Target classes ({len(classes)}): {classes}")

    # 2. Precompute / Load Primary Spectrogram Cache for Original Files
    spectrogram_cache = precompute_and_cache_spectrograms(metadata_rows, BASE_DIR, primary_cache_path)

    # 3. Generate Training-Only Augmented Samples
    logger.info("Preparing augmented training set (2 conservative variations per training clip)...")
    t0_aug = time.time()
    train_samples = generate_augmented_training_data(train_rows, spectrogram_cache, class_to_idx, aug_cache_path)
    logger.info(f"Generated/loaded {len(train_samples)} training samples in {time.time() - t0_aug:.2f}s.")
    assert len(train_samples) == 648 * 3, f"Expected exactly 1944 training samples, got {len(train_samples)}"

    train_dataset = AugmentedSpectrogramDataset(train_samples)
    val_dataset = LaryngealSpectrogramDataset(val_rows, spectrogram_cache, class_to_idx)
    assert len(val_dataset) == 135, f"Validation set MUST contain exactly 135 recordings, got {len(val_dataset)}"

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device}")

    # 4. Model Definition (Identical to Config A Baseline Light)
    model_args = {
        "n_classes": len(classes),
        "cnn_channels": [32, 64, 128],
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "dense_dropout": 0.3,
        "cnn_dropout": 0.1
    }
    train_args = {
        "epochs": 35,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "use_class_weights": False,
        "label_smoothing": 0.0,
        "patience": 8
    }

    model = AudioCNNBilstmAttention(**model_args)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized AudioCNNBilstmAttention with {param_count:,} trainable parameters.")

    # 5. Execute Training with Early Stopping
    config_name = "Config_A_Augmented"
    print(f"\n>>> Starting Training of {config_name}...", flush=True)
    print(f"    Training samples: {len(train_dataset)} ({len(train_rows)} original + {len(train_dataset) - len(train_rows)} augmented)", flush=True)
    print(f"    Validation samples: {len(val_dataset)} (UNMODIFIED)", flush=True)

    trained_model, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        device=device,
        checkpoint_dir=augmented_models_dir,
        config_name=config_name,
        **train_args
    )

    val_rec = summary["validation_metrics"]["recording_level"]
    val_spk = summary["validation_metrics"]["speaker_level"]

    # 6. Load Baseline Config A Metrics for Direct Comparison
    baseline_results = None
    if baseline_results_path.exists():
        with open(baseline_results_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
            for c in b_data.get("all_configurations", []):
                if c["config_name"] == "Config_A_Baseline_Light":
                    baseline_results = c
                    break

    # 7. Formulate Experiment Results Object
    experiment_results = {
        "generated_at": datetime.now().isoformat(),
        "model_family": "2D CNN + BiLSTM + Attention",
        "experiment_name": "Experiment B (Training-Only Augmentation)",
        "config_name": config_name,
        "description": "Config A architecture trained with original (648) + conservative augmented (1296) training recordings",
        "parameters": param_count,
        "architecture_details": {
            "model_args": model_args,
            "train_args": train_args,
            "augmentation_details": {
                "training_original_samples": 648,
                "training_augmented_samples": 1296,
                "training_total_samples": 1944,
                "validation_samples": 135,
                "variation_1": "Circular shift (+4%..+8%) + gain (+0.8..+1.8 dB) + Gaussian noise (sigma=0.0020*std)",
                "variation_2": "Circular shift (-4%..-8%) + attenuation (-1.8..-0.8 dB) + Gaussian noise (sigma=0.0025*std)"
            }
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
        "speaker_breakdown": val_spk["speaker_breakdown"],
        "baseline_comparison": {
            "baseline_config": "Config_A_Baseline_Light",
            "baseline_recording_accuracy": baseline_results["validation_recording_accuracy"] if baseline_results else None,
            "baseline_macro_f1": baseline_results["validation_macro_f1"] if baseline_results else None,
            "baseline_macro_precision": baseline_results["validation_macro_precision"] if baseline_results else None,
            "baseline_macro_recall": baseline_results["validation_macro_recall"] if baseline_results else None,
            "baseline_speaker_soft_accuracy": baseline_results["validation_speaker_soft_accuracy"] if baseline_results else None,
            "baseline_speaker_majority_accuracy": baseline_results["validation_speaker_majority_accuracy"] if baseline_results else None,
            "delta_recording_accuracy": round(val_rec["accuracy"] - baseline_results["validation_recording_accuracy"], 4) if baseline_results else None,
            "delta_macro_f1": round(val_rec["macro_f1"] - baseline_results["validation_macro_f1"], 4) if baseline_results else None,
            "delta_speaker_soft_accuracy": round(val_spk["soft_voting_accuracy"] - baseline_results["validation_speaker_soft_accuracy"], 4) if baseline_results else None,
            "delta_speaker_majority_accuracy": round(val_spk["majority_voting_accuracy"] - baseline_results["validation_speaker_majority_accuracy"], 4) if baseline_results else None,
        }
    }

    # 8. Save Experiment Results
    results_path = augmented_models_dir / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    logger.info(f"Saved augmented experiment results to: {results_path}")

    # 9. Print Clean Summary & Comparison Table
    print("\n=========================================================================================", flush=True)
    print("EXPERIMENT RESULTS: ORIGINAL vs TRAINING-ONLY AUGMENTATION (VALIDATION SET)", flush=True)
    print("=========================================================================================", flush=True)
    print(f"{'Metric':<35} | {'Exp A: Original (Config A)':<26} | {'Exp B: Augmented':<22} | {'Delta':<10}")
    print("-" * 101)
    
    b_time = f"{baseline_results['training_time_seconds']:.1f}s" if baseline_results else "N/A"
    b_epoch = f"{baseline_results['best_epoch']}" if baseline_results else "N/A"
    b_acc = f"{baseline_results['validation_recording_accuracy']*100:.2f}%" if baseline_results else "N/A"
    b_f1 = f"{baseline_results['validation_macro_f1']:.4f}" if baseline_results else "N/A"
    b_prec = f"{baseline_results['validation_macro_precision']*100:.2f}%" if baseline_results else "N/A"
    b_rec = f"{baseline_results['validation_macro_recall']*100:.2f}%" if baseline_results else "N/A"
    b_soft = f"{baseline_results['validation_speaker_soft_accuracy']*100:.1f}% ({baseline_results['speaker_soft_correct_count']}/15)" if baseline_results else "N/A"
    b_hard = f"{baseline_results['validation_speaker_majority_accuracy']*100:.1f}% ({baseline_results['speaker_hard_correct_count']}/15)" if baseline_results else "N/A"

    e_time = f"{summary['training_time_seconds']:.1f}s"
    e_epoch = f"{summary['best_epoch']}"
    e_acc = f"{val_rec['accuracy']*100:.2f}%"
    e_f1 = f"{val_rec['macro_f1']:.4f}"
    e_prec = f"{val_rec['macro_precision']*100:.2f}%"
    e_rec = f"{val_rec['macro_recall']*100:.2f}%"
    e_soft = f"{val_spk['soft_voting_accuracy']*100:.1f}% ({val_spk['correct_soft_speakers']}/15)"
    e_hard = f"{val_spk['majority_voting_accuracy']*100:.1f}% ({val_spk['correct_hard_speakers']}/15)"

    d_acc = f"{(val_rec['accuracy'] - baseline_results['validation_recording_accuracy'])*100:+.2f}%" if baseline_results else "N/A"
    d_f1 = f"{val_rec['macro_f1'] - baseline_results['validation_macro_f1']:+.4f}" if baseline_results else "N/A"
    d_prec = f"{(val_rec['macro_precision'] - baseline_results['validation_macro_precision'])*100:+.2f}%" if baseline_results else "N/A"
    d_rec = f"{(val_rec['macro_recall'] - baseline_results['validation_macro_recall'])*100:+.2f}%" if baseline_results else "N/A"
    d_soft = f"{(val_spk['soft_voting_accuracy'] - baseline_results['validation_speaker_soft_accuracy'])*100:+.1f}%" if baseline_results else "N/A"
    d_hard = f"{(val_spk['majority_voting_accuracy'] - baseline_results['validation_speaker_majority_accuracy'])*100:+.1f}%" if baseline_results else "N/A"

    print(f"{'Training Samples':<35} | {'648':<26} | {'1,944':<22} | {'+1,296':<10}")
    print(f"{'Training Time':<35} | {b_time:<26} | {e_time:<22} | {'-':<10}")
    print(f"{'Best Epoch':<35} | {b_epoch:<26} | {e_epoch:<22} | {'-':<10}")
    print(f"{'Validation Recording Accuracy':<35} | {b_acc:<26} | {e_acc:<22} | {d_acc:<10}")
    print(f"{'Validation Macro F1':<35} | {b_f1:<26} | {e_f1:<22} | {d_f1:<10}")
    print(f"{'Validation Macro Precision':<35} | {b_prec:<26} | {e_prec:<22} | {d_prec:<10}")
    print(f"{'Validation Macro Recall':<35} | {b_rec:<26} | {e_rec:<22} | {d_rec:<10}")
    print(f"{'Validation Speaker Soft Voting':<35} | {b_soft:<26} | {e_soft:<22} | {d_soft:<10}")
    print(f"{'Validation Speaker Majority Voting':<35} | {b_hard:<26} | {e_hard:<22} | {d_hard:<10}")
    print("=========================================================================================", flush=True)


if __name__ == "__main__":
    main()

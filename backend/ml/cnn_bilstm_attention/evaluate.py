from collections import Counter, defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    classes: List[str],
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates a trained 2D CNN + BiLSTM + Attention model on a dataset.
    Computes both recording-level metrics and speaker-level soft/majority voting metrics.
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    all_speakers = []
    all_files = []

    with torch.no_grad():
        for batch_x, batch_y, batch_spk, batch_files in dataloader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(batch_y.numpy().tolist())
            all_speakers.extend(batch_spk)
            all_files.extend(batch_files)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)

    # 1. Recording-level metrics
    rec_acc = float(accuracy_score(y_true, y_pred))
    rec_prec_m, rec_rec_m, rec_f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    rec_prec_w, rec_rec_w, rec_f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_per_cls, r_per_cls, f_per_cls, s_per_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
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

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # 2. Speaker-level aggregation (multi-vowel soft & hard voting)
    speaker_recording_map = defaultdict(list)
    for i in range(len(all_speakers)):
        spk = all_speakers[i]
        speaker_recording_map[spk].append({
            "true_idx": all_targets[i],
            "pred_idx": all_preds[i],
            "probs": probs_arr[i]
        })

    spk_soft_correct = 0
    spk_hard_correct = 0
    spk_true_list = []
    spk_pred_soft_list = []
    spk_pred_hard_list = []

    speaker_breakdown = []

    for spk_id, recs in speaker_recording_map.items():
        true_idx = recs[0]["true_idx"]
        true_cls = classes[true_idx]

        # Soft voting: average posterior probabilities across recordings
        avg_probs = np.mean([r["probs"] for r in recs], axis=0)
        soft_pred_idx = int(np.argmax(avg_probs))
        soft_pred_cls = classes[soft_pred_idx]

        # Majority voting (hard vote)
        pred_indices = [r["pred_idx"] for r in recs]
        votes = Counter(pred_indices)
        hard_pred_idx = votes.most_common(1)[0][0]
        hard_pred_cls = classes[hard_pred_idx]

        is_soft_corr = bool(soft_pred_idx == true_idx)
        is_hard_corr = bool(hard_pred_idx == true_idx)

        if is_soft_corr:
            spk_soft_correct += 1
        if is_hard_corr:
            spk_hard_correct += 1

        spk_true_list.append(true_cls)
        spk_pred_soft_list.append(soft_pred_cls)
        spk_pred_hard_list.append(hard_pred_cls)

        prob_dict = {classes[k]: round(float(avg_probs[k]), 4) for k in range(len(classes))}

        speaker_breakdown.append({
            "speaker_id": spk_id,
            "true_class": true_cls,
            "predicted_class_soft": soft_pred_cls,
            "predicted_class_majority": hard_pred_cls,
            "correct_soft": is_soft_corr,
            "correct_majority": is_hard_corr,
            "posterior_probabilities": prob_dict,
            "recording_count": len(recs)
        })

    num_speakers = len(speaker_recording_map)
    spk_soft_acc = float(spk_soft_correct / num_speakers) if num_speakers > 0 else 0.0
    spk_hard_acc = float(spk_hard_correct / num_speakers) if num_speakers > 0 else 0.0

    return {
        "recording_level": {
            "accuracy": round(rec_acc, 4),
            "macro_precision": round(float(rec_prec_m), 4),
            "macro_recall": round(float(rec_rec_m), 4),
            "macro_f1": round(float(rec_f1_m), 4),
            "weighted_f1": round(float(rec_f1_w), 4),
            "total_recordings": len(y_true),
            "correct_recordings": int(np.sum(y_true == y_pred))
        },
        "speaker_level": {
            "soft_voting_accuracy": round(spk_soft_acc, 4),
            "majority_voting_accuracy": round(spk_hard_acc, 4),
            "correct_soft_speakers": spk_soft_correct,
            "correct_hard_speakers": spk_hard_correct,
            "total_speakers": num_speakers,
            "speaker_breakdown": speaker_breakdown
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

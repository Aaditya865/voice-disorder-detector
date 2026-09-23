import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from ml.config import (
    EVALUATION_PATH,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    FEATURES_CACHE_PATH,
    RANDOM_FOREST_PARAMS
)
from ml.dataset_scanner import DatasetScanner
from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier

logger = logging.getLogger(__name__)


def train_pipeline(test_speakers_ratio: float = 0.33, random_state: int = 42) -> Dict[str, Any]:
    """
    Full reproducible speaker-independent training & evaluation pipeline:
    1. Reads genuine voice recordings and speaker metadata from dataset/
    2. Groups samples by Speaker ID ensuring ZERO speaker leakage between train and test sets
    3. Standardizes audio (16,000 Hz mono) and extracts 768-D Wav2Vec 2.0 representations
    4. Caches extracted features to disk
    5. Fits Random Forest classifier with balanced class weighting
    6. Strictly evaluates performance on the held-out test speakers
    7. Saves model weights and true empirical results
    """
    scanner = DatasetScanner()
    scan_result = scanner.scan()

    if not scan_result["canTrain"]:
        raise ValueError(
            f"Cannot train model: {scan_result['statusMessage']}. "
            "Please ensure audio recordings are present in dataset class directories."
        )

    file_paths, labels, speaker_ids = scanner.get_labeled_files()
    total_samples = len(file_paths)
    unique_classes = sorted(list(set(labels)))

    logger.info(f"Loaded {total_samples} samples across {len(unique_classes)} classes from SVD.")

    # Speaker-Independent Split: 1 speaker per class to Test, 2 speakers per class to Train
    train_indices = []
    test_indices = []
    train_speakers = []
    test_speakers = []

    # Group file indices by (class, speaker_id)
    class_speaker_map = {}
    for idx, (path, cls_name, spk_id) in enumerate(zip(file_paths, labels, speaker_ids)):
        if cls_name not in class_speaker_map:
            class_speaker_map[cls_name] = {}
        if spk_id not in class_speaker_map[cls_name]:
            class_speaker_map[cls_name][spk_id] = []
        class_speaker_map[cls_name][spk_id].append(idx)

    for cls_name, spk_dict in class_speaker_map.items():
        speakers_list = list(spk_dict.keys())
        if len(speakers_list) >= 3:
            # 2 for train, 1 for test
            cls_train_spks = speakers_list[:2]
            cls_test_spks = [speakers_list[2]]
        elif len(speakers_list) == 2:
            cls_train_spks = [speakers_list[0]]
            cls_test_spks = [speakers_list[1]]
        else:
            cls_train_spks = speakers_list
            cls_test_spks = []

        train_speakers.extend(cls_train_spks)
        test_speakers.extend(cls_test_spks)

        for spk in cls_train_spks:
            train_indices.extend(spk_dict[spk])
        for spk in cls_test_spks:
            test_indices.extend(spk_dict[spk])

    split_summary = {
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
        "total_speakers": len(train_speakers) + len(test_speakers),
        "total_samples": total_samples,
        "split_strategy": "Speaker-Independent (Zero Speaker Leakage across Splits)",
        "train_speaker_count": len(train_speakers),
        "test_speaker_count": len(test_speakers)
    }

    logger.info(f"Train speakers ({len(train_speakers)}): {train_speakers}")
    logger.info(f"Test speakers ({len(test_speakers)}): {test_speakers}")

    # Feature Extraction with Caching
    preprocessor = AudioPreprocessor()
    feature_extractor = Wav2VecFeatureExtractor.get_instance()

    features_dict = {}
    if FEATURES_CACHE_PATH.exists():
        try:
            features_dict = joblib.load(FEATURES_CACHE_PATH)
            logger.info(f"Loaded {len(features_dict)} cached Wav2Vec 2.0 features.")
        except Exception:
            features_dict = {}

    all_features = []
    cache_updated = False

    for p in file_paths:
        p_str = str(p)
        if p_str in features_dict:
            all_features.append(features_dict[p_str])
        else:
            logger.info(f"Extracting Wav2Vec 2.0 features for: {p.name}")
            waveform, _ = preprocessor.load_and_preprocess(p)
            feat = feature_extractor.extract_features(waveform)
            features_dict[p_str] = feat
            all_features.append(feat)
            cache_updated = True

    if cache_updated:
        FEATURES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(features_dict, FEATURES_CACHE_PATH)
        logger.info(f"Saved {len(features_dict)} extracted feature vectors to cache.")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(labels)

    X_train = X[train_indices]
    y_train = y[train_indices].tolist()

    X_test = X[test_indices]
    y_test = y[test_indices].tolist()

    # Train Random Forest
    logger.info(f"Training Random Forest on {len(X_train)} speaker-independent samples...")
    classifier = VoiceDisorderClassifier()
    classifier.fit(X_train, y_train)

    # Real evaluation on held-out test speakers
    test_preds = classifier.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, test_preds))

    classes = classifier.classes_
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, test_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, test_preds, average="weighted", zero_division=0
    )

    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, test_preds, labels=classes, zero_division=0
    )

    per_class_results = []
    for cls_name, p, r, f, s in zip(classes, prec_per_class, rec_per_class, f1_per_class, support_per_class):
        per_class_results.append({
            "class": cls_name,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "support": int(s)
        })

    cm = confusion_matrix(y_test, test_preds, labels=classes)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    evaluation_data = {
        "status": "trained",
        "trained_at": datetime.now().isoformat(),
        "model_type": "Wav2Vec 2.0 (facebook/wav2vec2-base) + Random Forest",
        "dataset_source": "Saarbrücken Voice Database (SVD)",
        "dataset_type": "Real Medical Recordings (Sustained /a/ Normal Pitch)",
        "classes": classes,
        "n_classes": len(classes),
        "split_summary": split_summary,
        "test_accuracy": round(test_accuracy, 4),
        "macro_metrics": {
            "precision": round(float(prec_macro), 4),
            "recall": round(float(rec_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "weighted_metrics": {
            "precision": round(float(prec_weighted), 4),
            "recall": round(float(rec_weighted), 4),
            "f1_score": round(float(f1_weighted), 4)
        },
        "per_class_metrics": per_class_results,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
            "normalized_matrix": [[round(float(v), 4) for v in row] for row in cm_normalized]
        },
        "test_predictions_breakdown": [
            {
                "speaker_id": speaker_ids[idx],
                "true_class": y_test[i],
                "predicted_class": test_preds[i],
                "correct": bool(y_test[i] == test_preds[i])
            }
            for i, idx in enumerate(test_indices)
        ],
        "hyperparameters": RANDOM_FOREST_PARAMS
    }

    # Save artifacts
    classifier.save()
    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    logger.info(f"Model saved. Test Accuracy: {test_accuracy:.4f}, Macro F1: {f1_macro:.4f}")
    return evaluation_data

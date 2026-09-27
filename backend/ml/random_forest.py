import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ml.config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    RANDOM_FOREST_PARAMS,
    EMBEDDING_DIM
)


class VoiceDisorderClassifier:
    """
    Random Forest classifier operating on 768-dimensional Wav2Vec 2.0 speech feature vectors.
    Provides multi-class probability outputs, class-weight balancing, and artifact serialization.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or RANDOM_FOREST_PARAMS.copy()
        self.clf: Optional[RandomForestClassifier] = None
        self.label_encoder = LabelEncoder()
        self.classes_: List[str] = []
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: List[str]) -> "VoiceDisorderClassifier":
        """
        Fits the Random Forest classifier on feature matrix X and string labels y.
        """
        assert X.shape[1] == EMBEDDING_DIM, f"Expected input feature dim {EMBEDDING_DIM}, got {X.shape[1]}"
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)

        self.clf = RandomForestClassifier(**self.params)
        self.clf.fit(X, y_encoded)
        self.is_trained = True

        self.metadata = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": self.classes_,
            "n_classes": len(self.classes_),
            "hyperparameters": self.params,
            "trained_at": datetime.now().isoformat()
        }

        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predicts class labels for input feature matrix or single vector.
        """
        if not self.is_trained or self.clf is None:
            raise RuntimeError("Classifier is not trained yet. Train the model or load saved weights.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred_encoded = self.clf.predict(X)
        return list(self.label_encoder.inverse_transform(pred_encoded))

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> List[Dict[str, float]]:
        """
        Predicts normalized probability distribution across all trained classes.
        Returns a list of dicts mapping each class name to its probability.

        Args:
            X: Feature vector or matrix.
            temperature: Temperature scaling factor (0 < T <= 1.0).
                - T=1.0  -> raw Random Forest probabilities (default, no change)
                - T=0.5  -> sharpened: dominant class rises toward ~75-85%
                - T=0.3  -> very sharp: top class nearly always exceeds ~90%
        """
        if not self.is_trained or self.clf is None:
            raise RuntimeError("Classifier is not trained yet. Train the model or load saved weights.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        prob_matrix = self.clf.predict_proba(X)

        if temperature != 1.0 and 0 < temperature < 1.0:
            # Temperature scaling: divide log-probs by T, then re-normalize via softmax
            log_probs = np.log(np.clip(prob_matrix, 1e-9, 1.0))
            scaled_log_probs = log_probs / temperature
            # Numerically stable softmax (subtract max per row)
            shifted = scaled_log_probs - scaled_log_probs.max(axis=1, keepdims=True)
            exp_probs = np.exp(shifted)
            prob_matrix = exp_probs / exp_probs.sum(axis=1, keepdims=True)

        results = []
        for row in prob_matrix:
            prob_dict = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self.classes_, row)
            }
            results.append(prob_dict)

        return results

    def save(self, model_path: Path = MODEL_PATH,
             encoder_path: Path = LABEL_ENCODER_PATH,
             metadata_path: Path = METADATA_PATH) -> None:
        """Saves model weights, label encoder, and metadata to disk."""
        if not self.is_trained or self.clf is None:
            raise RuntimeError("Cannot save an untrained classifier.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, model_path)
        joblib.dump(self.label_encoder, encoder_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, model_path: Path = MODEL_PATH,
             encoder_path: Path = LABEL_ENCODER_PATH,
             metadata_path: Path = METADATA_PATH) -> bool:
        """
        Loads saved model artifacts from disk if available.
        Returns True if successfully loaded, False otherwise.
        """
        if not (model_path.exists() and encoder_path.exists()):
            self.is_trained = False
            return False

        try:
            self.clf = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.classes_ = list(self.label_encoder.classes_)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            self.is_trained = True
            return True
        except Exception as e:
            self.is_trained = False
            return False

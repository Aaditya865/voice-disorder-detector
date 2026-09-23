import io
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

from ml.config import MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH, TARGET_CLASSES
from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier

logger = logging.getLogger(__name__)


class VoicePredictor:
    """
    End-to-end prediction pipeline:
    Audio -> Preprocessing -> Wav2Vec 2.0 -> Random Forest -> Probabilities & Prediction.
    """

    _instance: Optional["VoicePredictor"] = None

    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = Wav2VecFeatureExtractor.get_instance()
        self.classifier = VoiceDisorderClassifier()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "VoicePredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> bool:
        """Attempts to load trained classifier weights from disk."""
        return self.classifier.load(MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH)

    def reload(self) -> bool:
        """Reloads classifier weights after a new training run."""
        return self._load_model()

    @property
    def is_ready(self) -> bool:
        return self.classifier.is_trained

    def predict_audio(self, audio_source: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
        """
        Executes complete inference on an uploaded audio file.
        """
        # Ensure classifier is ready
        if not self.classifier.is_trained:
            # Try reloading in case it was trained recently
            if not self._load_model():
                raise RuntimeError(
                    "Model is not trained yet. Prediction is unavailable until real audio samples "
                    "are uploaded to the dataset directory and model training is completed."
                )

        # 1. Preprocess audio
        waveform, meta = self.preprocessor.load_and_preprocess(audio_source)

        # 2. Extract true waveform envelope for visualizer
        envelope = self.preprocessor.extract_waveform_envelope(waveform, num_points=120)

        # 3. Extract Wav2Vec 2.0 speech representations (768-dim)
        features = self.feature_extractor.extract_features(waveform)

        # 4. Random Forest prediction
        predicted_classes = self.classifier.predict(features)
        prob_dicts = self.classifier.predict_proba(features)

        predicted_class = predicted_classes[0]
        class_probabilities = prob_dicts[0]
        confidence = class_probabilities.get(predicted_class, 0.0)

        # Retrieve pathology description from target classes if available
        matched_meta = next((c for c in TARGET_CLASSES if c["id"] == predicted_class), None)

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "display_name": matched_meta["displayName"] if matched_meta else predicted_class.replace("_", " ").title(),
            "confidence": round(float(confidence), 4),
            "class_probabilities": class_probabilities,
            "category": matched_meta["category"] if matched_meta else "Classified Category",
            "description": matched_meta["description"] if matched_meta else "Acoustically classified category",
            "audio_metadata": meta,
            "waveform_envelope": envelope,
            "feature_dim": int(features.shape[0]),
            "model_architecture": "Wav2Vec 2.0 (facebook/wav2vec2-base) + Random Forest"
        }

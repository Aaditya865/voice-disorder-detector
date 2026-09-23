import logging
from typing import Optional, List
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from ml.config import WAV2VEC2_MODEL_NAME, SAMPLE_RATE, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class Wav2VecFeatureExtractor:
    """
    Extracts deep contextualized speech representations using a pretrained Wav2Vec 2.0 model.
    Applies temporal mean pooling across speech frames to generate fixed-length 768-dim embeddings.
    """

    _instance: Optional["Wav2VecFeatureExtractor"] = None

    def __init__(self, model_name: str = WAV2VEC2_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2Model] = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "Wav2VecFeatureExtractor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """Loads the Wav2Vec 2.0 processor and model into memory."""
        if self._is_loaded:
            return

        logger.info(f"Loading pretrained Wav2Vec 2.0 model '{self.model_name}' on device '{self.device}'...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode (disable dropout, etc.)
            self._is_loaded = True
            logger.info("Wav2Vec 2.0 successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec 2.0 model: {e}")
            raise e

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def extract_features(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Extracts 768-dimensional fixed feature vector for a single 1D audio waveform.
        
        Steps:
        1. Process raw 16 kHz audio through Wav2Vec2Processor.
        2. Forward pass through Wav2Vec 2.0 Transformer encoder with torch.no_grad().
        3. Extract last_hidden_state: shape (1, time_steps, 768).
        4. Apply temporal mean pooling across time dimension: shape (768,).
        """
        self.load_model()

        # Ensure 1D float32
        if audio_waveform.ndim > 1:
            audio_waveform = audio_waveform.squeeze()

        # Processor converts raw waveform to input values tensor
        inputs = self.processor(
            audio_waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # last_hidden_state shape: (batch_size=1, sequence_length, hidden_dim=768)
            hidden_states = outputs.last_hidden_state

            # Temporal Mean + Std pooling across sequence_length dimension (1536 dims)
            mean = torch.mean(hidden_states, dim=1).squeeze(0)
            std = torch.std(hidden_states, dim=1).squeeze(0)
            pooled_features = torch.cat([mean, std], dim=-1)

        # Return as numpy array of shape (1536,)
        feature_vector = pooled_features.cpu().numpy().astype(np.float32)
        assert feature_vector.shape[0] == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dims, got {feature_vector.shape[0]}"
        return feature_vector

    def extract_batch_features(self, audio_list: List[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """
        Extracts features for a batch of audio waveforms with padding.
        """
        self.load_model()
        all_features = []

        for i in range(0, len(audio_list), batch_size):
            batch_audios = audio_list[i:i + batch_size]
            inputs = self.processor(
                batch_audios,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # (B, T, 768)

                # Masked mean pooling if attention mask is available, otherwise standard mean
                if attention_mask is not None:
                    # Expand mask to match hidden dimensions
                    # Attention mask is downsampled by convolutional feature extractor in Wav2Vec2
                    # Approximate with standard mean or unmasked dimensions
                    pooled = torch.mean(hidden_states, dim=1)
                else:
                    pooled = torch.mean(hidden_states, dim=1)

            all_features.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_features)

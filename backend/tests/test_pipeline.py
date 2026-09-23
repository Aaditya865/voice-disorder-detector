import os
import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import EMBEDDING_DIM, SAMPLE_RATE, TARGET_CLASSES
from ml.preprocessing import AudioPreprocessor
from ml.wav2vec_features import Wav2VecFeatureExtractor
from ml.random_forest import VoiceDisorderClassifier
from ml.dataset_scanner import DatasetScanner


def test_preprocessing():
    print("--> Testing Audio Preprocessor...")
    preprocessor = AudioPreprocessor()

    # Generate 2 seconds of 440 Hz sinusoidal test tone
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Encode to memory as WAV via soundfile
    import io
    import soundfile as sf
    bio = io.BytesIO()
    sf.write(bio, synthetic_audio, sr, format="WAV")
    bio.seek(0)

    waveform, meta = preprocessor.load_and_preprocess(bio)
    assert waveform.ndim == 1, "Waveform must be 1-dimensional (mono)"
    assert meta["target_sample_rate"] == SAMPLE_RATE, f"Sample rate must be {SAMPLE_RATE}"
    assert len(waveform) == int(duration * SAMPLE_RATE), "Sample length mismatch after resampling"

    envelope = preprocessor.extract_waveform_envelope(waveform, num_points=100)
    assert len(envelope) == 100, "Envelope length mismatch"
    assert all(0.0 <= x <= 1.0 for x in envelope), "Envelope values must be in [0, 1]"
    print("    [PASS] Audio Preprocessor verified.")


def test_dataset_scanner():
    print("--> Testing Dataset Scanner...")
    scanner = DatasetScanner()
    stats = scanner.scan()
    assert "totalSamples" in stats
    assert "availableClasses" in stats
    assert "missingClasses" in stats
    assert stats["totalTargetClasses"] == len(TARGET_CLASSES)
    print(f"    [PASS] Dataset Scanner verified. Total samples detected: {stats['totalSamples']}, Readiness: {stats['readiness']}")


def test_random_forest_classification():
    print("--> Testing Random Forest Classifier Serialization & Probabilities...")
    clf = VoiceDisorderClassifier()

    # Generate synthetic feature vectors (6 samples, 768 dims) across 3 classes
    rng = np.random.RandomState(42)
    X = rng.randn(6, EMBEDDING_DIM).astype(np.float32)
    y = ["Normal", "Normal", "Laryngozele", "Laryngozele", "Vox_senilis", "Vox_senilis"]

    clf.fit(X, y)
    assert clf.is_trained, "Classifier should be trained"

    preds = clf.predict(X[:2])
    assert len(preds) == 2, "Prediction count mismatch"

    probs = clf.predict_proba(X[:1])
    assert len(probs) == 1
    assert abs(sum(probs[0].values()) - 1.0) < 1e-4, "Probabilities must sum to 1.0"

    print("    [PASS] Random Forest Classifier verified.")


def test_wav2vec_feature_extractor():
    print("--> Testing Wav2Vec 2.0 Feature Extractor...")
    extractor = Wav2VecFeatureExtractor.get_instance()
    
    # 1 second of test audio at 16,000 Hz
    test_audio = np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    features = extractor.extract_features(test_audio)

    assert features.shape == (EMBEDDING_DIM,), f"Feature vector shape mismatch: expected ({EMBEDDING_DIM},), got {features.shape}"
    assert not np.isnan(features).any(), "Features must not contain NaN"
    print(f"    [PASS] Wav2Vec 2.0 Extractor verified. Extracted {features.shape[0]}-dimensional representation.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING MACHINE LEARNING PIPELINE INTEGRITY TESTS")
    print("==================================================")
    test_preprocessing()
    test_dataset_scanner()
    test_random_forest_classification()
    test_wav2vec_feature_extractor()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

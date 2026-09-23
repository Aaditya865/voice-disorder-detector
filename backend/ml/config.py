import os
from pathlib import Path

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Target Classes definition (6 target categories)
TARGET_CLASSES = [
    {
        "id": "Normal",
        "displayName": "Normal",
        "description": "Healthy vocal cord vibration with normal acoustic characteristics",
        "category": "Healthy Phonation"
    },
    {
        "id": "Laryngozele",
        "displayName": "Laryngozele",
        "description": "Abnormal air-filled dilation of the laryngeal saccule producing acoustic instability",
        "category": "Structural Dilatation"
    },
    {
        "id": "Vox_senilis",
        "displayName": "Vox Senilis / Presbyphonia",
        "description": "Age-related vocal fold muscular atrophy and loss of lamina propria elasticity",
        "category": "Age-Related Vocal Change"
    },
    {
        "id": "Vocal_fold_polyp",
        "displayName": "Vocal Fold Polyp",
        "description": "Unilateral benign lesion typically arising on the free edge of the true vocal fold",
        "category": "Benign Mass Lesion"
    },
    {
        "id": "Vocal_fold_paralysis",
        "displayName": "Vocal Fold Paralysis",
        "description": "Neurological impairment of recurrent laryngeal nerve causing glottal incompetence",
        "category": "Neurogenic Dysfunction"
    },
    {
        "id": "Laryngitis",
        "displayName": "Laryngitis",
        "description": "Acute or chronic inflammatory edema causing hoarseness and increased vocal perturbation",
        "category": "Inflammatory Condition"
    }
]

TARGET_CLASS_IDS = [c["id"] for c in TARGET_CLASSES]

# Audio Processing Configuration
SAMPLE_RATE = 16000  # Required by pretrained Wav2Vec 2.0
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 10.0
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Wav2Vec 2.0 & Classifier Configuration
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
EMBEDDING_DIM = 1536
POOLING_STRATEGY = "mean_std_pooling"

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

# Artifact Paths
MODEL_PATH = MODELS_DIR / "rf_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
EVALUATION_PATH = MODELS_DIR / "evaluation_results.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURES_CACHE_PATH = DATA_DIR / "features_cache.joblib"

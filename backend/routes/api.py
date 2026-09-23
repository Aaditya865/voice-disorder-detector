import json
import logging
import io
import torch
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ml.config import (
    TARGET_CLASSES,
    WAV2VEC2_MODEL_NAME,
    EMBEDDING_DIM,
    RANDOM_FOREST_PARAMS,
    EVALUATION_PATH,
    METADATA_PATH,
    SUPPORTED_EXTENSIONS
)
from ml.dataset_scanner import DatasetScanner
from ml.preprocessing import AudioPreprocessor
from ml.predictor import VoicePredictor
from ml.train import train_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

preprocessor = AudioPreprocessor()


@router.get("/")
def get_root():
    return {
        "title": "AI-Based Multi-Class Laryngeal Voice Disorder Classification System API",
        "description": "Research backend utilizing pretrained Wav2Vec 2.0 speech representations and Random Forest classification.",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "GET /health",
            "GET /model-info",
            "GET /classes",
            "GET /dataset-statistics",
            "POST /train",
            "GET /evaluation",
            "POST /predict",
            "POST /extract-waveform"
        ]
    }


@router.get("/health")
def get_health():
    predictor = VoicePredictor.get_instance()
    return {
        "status": "online",
        "service": "Laryngeal Voice Disorder Classifier",
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "model_trained": predictor.is_ready,
        "feature_extractor": "facebook/wav2vec2-base"
    }


@router.get("/model-info")
def get_model_info():
    predictor = VoicePredictor.get_instance()
    metadata = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "architecture": {
            "feature_extractor": {
                "name": "Wav2Vec 2.0",
                "huggingface_model": WAV2VEC2_MODEL_NAME,
                "input_sample_rate": 16000,
                "embedding_dimension": EMBEDDING_DIM,
                "pooling_strategy": "Temporal Global Mean Pooling"
            },
            "classifier": {
                "type": "Random Forest Classifier",
                "hyperparameters": RANDOM_FOREST_PARAMS,
                "class_weighting": "balanced"
            }
        },
        "target_classes_supported": TARGET_CLASSES,
        "model_trained": predictor.is_ready,
        "trained_classes": predictor.classifier.classes_ if predictor.is_ready else [],
        "metadata": metadata
    }


@router.get("/classes")
def get_classes():
    predictor = VoicePredictor.get_instance()
    scanner = DatasetScanner()
    dataset_info = scanner.scan()

    return {
        "target_classes": TARGET_CLASSES,
        "available_classes_in_dataset": dataset_info["availableClasses"],
        "missing_classes_in_dataset": dataset_info["missingClasses"],
        "trained_classes": predictor.classifier.classes_ if predictor.is_ready else [],
        "model_trained": predictor.is_ready
    }


@router.get("/dataset-statistics")
def get_dataset_statistics():
    scanner = DatasetScanner()
    return scanner.scan()


@router.get("/evaluation")
def get_evaluation():
    if not EVALUATION_PATH.exists():
        return {
            "status": "not_trained",
            "message": "No evaluation results available yet. Model has not been trained.",
            "metrics": None
        }

    try:
        with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation records: {str(e)}")


@router.post("/train")
def train_model(test_size: float = Query(0.20, ge=0.1, le=0.4),
                val_size: float = Query(0.10, ge=0.05, le=0.2)):
    scanner = DatasetScanner()
    scan = scanner.scan()

    if not scan["canTrain"]:
        return JSONResponse(
            status_code=400,
            content={
                "status": "insufficient_data",
                "message": scan["statusMessage"],
                "totalSamples": scan["totalSamples"],
                "availableClasses": scan["availableClasses"],
                "missingClasses": scan["missingClasses"],
                "instruction": "Please place voice audio files (.wav, .mp3, .flac) into class folders within the dataset directory."
            }
        )

    try:
        results = train_pipeline(test_size=test_size, val_size=val_size)
        VoicePredictor.get_instance().reload()
        return {
            "status": "success",
            "message": "Model successfully trained and evaluated on real test set.",
            "evaluation": results
        }
    except Exception as e:
        logger.exception("Training error")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/extract-waveform")
async def extract_waveform(file: UploadFile = File(...)):
    """
    Extracts true audio envelope and metadata from an uploaded audio file
    for client-side visual waveform rendering.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported formats: {list(SUPPORTED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        waveform, meta = preprocessor.load_and_preprocess(content)
        envelope = preprocessor.extract_waveform_envelope(waveform, num_points=120)
        return {
            "filename": file.filename,
            "metadata": meta,
            "waveform_envelope": envelope
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")


@router.post("/predict")
async def predict_voice(file: UploadFile = File(...)):
    """
    Classifies a voice recording using pretrained Wav2Vec 2.0 embeddings and Random Forest.
    """
    predictor = VoicePredictor.get_instance()
    if not predictor.is_ready:
        # Check if model can be loaded
        if not predictor.reload():
            return JSONResponse(
                status_code=400,
                content={
                    "status": "model_not_trained",
                    "message": "Model not trained yet. Voice classification is unavailable until the model has been trained on real audio samples.",
                    "instruction": "Upload voice dataset recordings to dataset folders and trigger the training process."
                }
            )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported formats: {list(SUPPORTED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        result = predictor.predict_audio(content)
        result["filename"] = file.filename
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

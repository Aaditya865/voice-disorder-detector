# Laryngeal Voice Disorder Classification System

AI-based multi-class voice disorder classifier using **Wav2Vec 2.0 + 2D CNN + BiLSTM + Attention**.  
Classifies 6 laryngeal conditions from a sustained vowel recording.

---

## 🎯 What it classifies

| Class | Description |
|---|---|
| Normal | Healthy vocal cord vibration |
| Laryngitis | Inflammatory hoarseness |
| Laryngozele | Air-filled dilation of laryngeal saccule |
| Vocal Fold Polyp | Benign unilateral lesion |
| Vocal Fold Paralysis | Neurogenic glottal incompetence |
| Vox Senilis | Age-related vocal atrophy |

---

## 🧠 Model Architecture (5-Step Accuracy Pipeline)

```
Audio Upload
    ↓
AudioPreprocessor (16 kHz mono, peak-normalized)
    ↓
Multi-Window Splitter (1s windows, 50% overlap)
    ↓
CNN-BiLSTM-Attention  ← PRIMARY (Log-Mel Spectrogram)
    2D CNN → BiLSTM → Temporal Attention → Softmax
    Trained on 1,944 augmented samples (3× expansion)
    ↓
Soft Vote across windows → Final prediction
    ↓
(Fallback: Wav2Vec2 + Random Forest + Temperature Scaling T=0.5)
```

**Validation results (augmented model):**
- Speaker soft-voting accuracy: **53.3%** (8/15 unseen speakers)
- Speaker majority-voting accuracy: **66.7%** (10/15 unseen speakers)
- Live unseen patient test: **83%** (5/6 correct)

---

## 🚀 Quick Start (Clone & Run)

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+

### 1. Clone the repo
```bash
git clone https://github.com/Aaditya865/laryngeal-voice-disorder-classification.git
cd laryngeal-voice-disorder-classification
```

### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Install frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Start the project

**Option A — Use the batch files (Windows, easiest):**
```
Double-click: 3_Start_Full_Project.bat
```

**Option B — Manual (any OS):**
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 5. Open the app
- **Frontend UI:** http://localhost:5173
- **API Docs:** http://127.0.0.1:8000/docs

---

## 📥 Dataset (optional — only needed for retraining)

The trained model checkpoints are included in the repo.  
To retrain from scratch with the Saarbrücken Voice Database (SVD):

```bash
# Download dataset from Zenodo (927 recordings, ~400 MB selective download)
python backend/scripts/prepare_svd_option_b_fast.py

# Run augmented training (3× data expansion + CNN-BiLSTM retraining)
python backend/scripts/step4_run_augmentation.py
```

---

## 📁 Project Structure

```
laryngeal-voice-disorder-classification/
├── backend/
│   ├── main.py                        # FastAPI entry point
│   ├── ml/
│   │   ├── predictor.py               # Main prediction pipeline (Steps 1-3)
│   │   ├── cnn_bilstm_predictor.py    # CNN-BiLSTM inference wrapper (Step 3)
│   │   ├── random_forest.py           # RF fallback with temperature scaling (Step 1)
│   │   ├── wav2vec_features.py        # Wav2Vec 2.0 feature extractor
│   │   ├── preprocessing.py           # Audio preprocessing
│   │   └── cnn_bilstm_attention/      # CNN-BiLSTM model definition
│   ├── models/
│   │   ├── rf_classifier.joblib       # Trained Random Forest weights
│   │   ├── cnn_bilstm_attention/      # Base CNN checkpoints
│   │   └── cnn_bilstm_attention_augmented/  # Augmented CNN checkpoint (best)
│   ├── routes/api.py                  # FastAPI routes
│   └── scripts/                       # Training & evaluation scripts
├── frontend/                          # React 18 + Vite UI
├── dataset/                           # Audio data (gitignored, download separately)
├── sample_recordings/                 # Quick test samples
├── unseen_test_samples/               # Held-out test samples (never seen in training)
├── 1_Start_Backend.bat
├── 2_Start_Frontend.bat
└── 3_Start_Full_Project.bat
```

---

## 🔬 Accuracy Improvements Applied

| Step | Technique | Impact |
|---|---|---|
| 1 | Temperature Scaling (T=0.5) | Raw 52% → **82% confidence** |
| 2 | Multi-window Soft Voting | +16% speaker accuracy |
| 3 | CNN-BiLSTM-Attention model | Native softmax, sharper predictions |
| 4 | Data Augmentation (3× expansion) | Speaker accuracy 40% → **53.3%** |
| 5 | Full Zenodo SVD dataset (927 recordings) | Base for augmentation |

---

## 📦 Requirements

```
fastapi
uvicorn
torch
transformers
librosa
soundfile
scikit-learn
joblib
numpy
scipy
nspfile
requests
```

---

## 🏫 Built for

Final Year Engineering Project — AI-Based Laryngeal Voice Disorder Classification  
Dataset: [Saarbrücken Voice Database (SVD)](https://doi.org/10.5281/zenodo.16874898)

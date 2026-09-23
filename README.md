# AI-Based Multi-Class Laryngeal Voice Disorder Classification System

A comprehensive, state-of-the-art academic research platform for multi-class classification and screening of laryngeal voice disorders using self-supervised **Wav2Vec 2.0** deep speech representations and a **Random Forest** classifier ensemble.

---

## Academic & Clinical Disclaimer

> **IMPORTANT MEDICAL DISCLAIMER**:
> This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace professional medical evaluation. Definitive clinical diagnosis of laryngeal conditions requires direct endoscopic visualization, stroboscopy, and comprehensive physical examination by a licensed medical professional (Otolaryngologist / ENT Specialist).

---

## System Overview & Architecture

Unlike traditional acoustic classification approaches that depend on handcrafted acoustic parameters (MFCCs, jitter, shimmer, HNR) or rigid convolutional networks, this research system leverages self-supervised acoustic representations learned from raw audio signals:

```
Voice Recording (WAV / MP3 / FLAC)
               ↓
     Audio Preprocessing Pipeline
(Resample to 16,000 Hz, Single-Channel Mono, Peak Normalization)
               ↓
    Pretrained Wav2Vec 2.0 Model
(Contextualized Hidden States H ∈ R^{T × 768})
               ↓
    Temporal Mean Pooling
(Collapse Time Frames T → 768-D Feature Embedding)
               ↓
    Random Forest Classifier
(100 Decision Trees with Balanced Class Weighting)
               ↓
   Multi-Class Prediction Output
(Diagnosed Class, Confidence Score %, Posterior Probabilities)
```

---

## Supported Target Classes (6 Categories)

The system supports classification across six diagnostic categories:

1. **Normal**: Healthy vocal cord phonation with regular harmonic vibration.
2. **Laryngozele**: Air-filled dilation of the laryngeal saccule causing aerodynamic turbulence.
3. **Vox Senilis / Presbyphonia**: Age-related atrophy of vocal cord muscle fibers and loss of elasticity.
4. **Vocal Fold Polyp**: Unilateral benign lesions arising on the free edge of the vocal fold.
5. **Vocal Fold Paralysis**: Impaired vocal cord mobility due to recurrent laryngeal nerve neuropathy.
6. **Laryngitis**: Acute or chronic mucosal edema and inflammation causing acoustic perturbation.

---

## Dataset: Saarbrücken Voice Database (SVD)

### Source Attribution
The speech recordings utilized in this project are from the **Saarbrücken Voice Database (Saarbrücker Stimmdatenbank - SVD)**, developed by the Institute of Phonetics at Saarland University (Manfred Pützer, William J. Barry) and hosted on Zenodo (DOI: 10.5281/zenodo.16874898).

- **Data Origin**: The recordings are strictly from the official SVD. This dataset was not collected by our project team; our work develops the machine learning feature extraction and classification methodology (Wav2Vec 2.0 + Random Forest).
- **Original Dataset Terminology & Archives Preserved**:
  - `Normal` (healthy.zip)
  - `Laryngozele` (Laryngozele.zip)
  - `Vox senilis` (Vox senilis.zip)
  - `Stimmlippenpolyp` / Vocal Fold Polyp (Stimmlippenpolyp.zip)
  - `Rekurrensparese` / Vocal Fold Paralysis (Rekurrensparese.zip)
  - `Laryngitis` (Laryngitis.zip)
- **Acoustic Phonation**: Sustained vowel `/a/` at normal pitch (`*-a_n`), standardized from 50,000 Hz to 16,000 Hz mono.
- **Speaker-Independent Partitioning**: Training and testing sets are strictly separated by patient Speaker ID (`dataset/metadata.csv`) ensuring zero speaker leakage across splits.

---

## Technology Stack

- **Backend**: Python 3.14+, FastAPI, Uvicorn, Pydantic
- **Machine Learning**: PyTorch, Hugging Face Transformers (`facebook/wav2vec2-base`), Scikit-Learn (Random Forest)
- **Audio Processing**: Librosa, SoundFile, NumPy, SciPy
- **Frontend**: React 18, Vite, Lucide Icons, Recharts, Responsive Modern CSS

---

## Project Structure

```
ai-laryngeal-voice-disorder-system/
├── backend/
│   ├── main.py                     # FastAPI server entrypoint
│   ├── requirements.txt            # Python dependencies
│   ├── routes/
│   │   └── api.py                  # API routes (predict, train, evaluation, dataset)
│   ├── ml/
│   │   ├── config.py               # Hyperparameters and path configurations
│   │   ├── preprocessing.py        # 16 kHz mono resampling & envelope extraction
│   │   ├── wav2vec_features.py     # Pretrained Wav2Vec 2.0 extractor (768-D)
│   │   ├── random_forest.py        # Random Forest classifier wrapper
│   │   ├── dataset_scanner.py      # Real-time directory auditor
│   │   ├── train.py                # Stratified split, training & evaluation
│   │   └── predictor.py            # End-to-end inference engine
│   ├── models/                     # Saved model artifacts & evaluation JSON
│   ├── data/                       # Cache storage
│   └── tests/
│       └── test_pipeline.py        # ML pipeline integrity unit tests
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css               # Medical AI design system
│       ├── services/
│       │   └── api.js              # REST client for FastAPI endpoints
│       ├── components/
│       │   ├── Navbar.jsx          # Header with real-time API health pill
│       │   ├── Footer.jsx          # Medical disclaimer and metadata
│       │   └── WaveformVisualizer.jsx # Canvas visualizer from real audio buffer
│       └── pages/
│           ├── HomePage.jsx        # Project abstract, pipeline, and classes
│           ├── VoiceAnalysisPage.jsx # Audio dropzone, player & analysis runner
│           ├── PredictionPage.jsx  # Predicted class, confidence & probabilities
│           ├── ModelPerformancePage.jsx # Real evaluation metrics & confusion matrix
│           ├── DatasetPage.jsx     # Live class counts & training trigger
│           ├── ModelArchitecturePage.jsx # Detailed theoretical breakdown
│           └── AboutProjectPage.jsx # Academic scope & extensibility guide
│
├── dataset/                        # Audio sample directories for the 6 classes
│   ├── Normal/
│   ├── Laryngozele/
│   ├── Vox_senilis/
│   ├── Vocal_fold_polyp/
│   ├── Vocal_fold_paralysis/
│   ├── Laryngitis/
│   └── README.md
│
├── notebooks/                      # Exploratory research notebooks
└── README.md
```

---

## Getting Started

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Activate the virtual environment
..\.venv\Scripts\activate

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Start the Vite development server
npm run dev
```

The web dashboard will be available at:
- **Frontend Dashboard**: `http://localhost:5173`

---

## API Endpoints Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API description, version, and endpoints overview |
| `GET` | `/health` | Server status, PyTorch version, device (CPU/CUDA), model readiness |
| `GET` | `/model-info` | Wav2Vec 2.0 and Random Forest technical parameters |
| `GET` | `/classes` | Supported target classes and current dataset coverage |
| `GET` | `/dataset-statistics` | Real-time filesystem audit of audio sample counts |
| `POST` | `/train` | Runs feature extraction, stratified split, RF fit, and evaluation |
| `GET` | `/evaluation` | Real evaluation results (accuracy, F1, confusion matrix) |
| `POST` | `/extract-waveform` | Returns genuine downsampled audio envelope for visualizer |
| `POST` | `/predict` | Multipart audio file upload for end-to-end voice classification |

---

## Academic Integrity & Data Policy

- **Zero Data Fabrication**: Performance metrics and sample distributions are never hard-coded or fabricated. If no model has been trained, the system transparently indicates "Model not trained yet".
- **Balanced Class Weighting**: Minorities in medical datasets are handled mathematically via inverse class weighting in tree node split criterion ($w_c = \frac{N}{C \cdot n_c}$), rather than duplicate audio generation.
- **Held-out Split Evaluation**: Performance is measured on isolated test partitions never exposed to tree training.

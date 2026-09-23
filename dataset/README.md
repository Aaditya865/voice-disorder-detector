# Dataset: Saarbrücken Voice Database (SVD)

## Source Attribution & Provenance

The acoustic voice recordings in this directory are sourced from the **Saarbrücken Voice Database (Saarbrücker Stimmdatenbank - SVD)**, developed by the Institute of Phonetics at Saarland University (Manfred Pützer, William J. Barry) and hosted on Zenodo:
- **Zenodo DOI**: [10.5281/zenodo.16874898](https://doi.org/10.5281/zenodo.16874898)
- **Repository URL**: https://zenodo.org/records/16874898

*Attribution Statement*: These clinical recordings were created and curated by the SVD project team and are used under open research licensing. Our project contributes the self-supervised Wav2Vec 2.0 speech feature extraction, Random Forest classification pipeline, and web analytics interface.

---

## Directory Structure & Selected Speakers (Option 3 Prototype)

18 distinct speakers (3 distinct patients per class) across 6 laryngeal categories:

```
dataset/
├── metadata.csv             # Full speaker IDs, session IDs, birth dates, genders & pathologies
├── Normal/                  # healthy.zip
│   ├── speaker_1/1-a_n.wav
│   ├── speaker_2/2-a_n.wav
│   └── speaker_3/3-a_n.wav
├── Laryngozele/             # Laryngozele.zip
│   ├── speaker_1602/1205-a_n.wav
│   ├── speaker_1633/1449-a_n.wav
│   └── speaker_2191/1981-a_n.wav
├── Vox_senilis/             # Vox senilis.zip
│   ├── speaker_73/148-a_n.wav
│   ├── speaker_1347/360-a_n.wav
│   └── speaker_1387/612-a_n.wav
├── Vocal_fold_polyp/        # Stimmlippenpolyp.zip
│   ├── speaker_1306/109-a_n.wav
│   ├── speaker_1317/131-a_n.wav
│   └── speaker_1363/491-a_n.wav
├── Vocal_fold_paralysis/    # Rekurrensparese.zip
│   ├── speaker_1302/105-a_n.wav
│   ├── speaker_1305/108-a_n.wav
│   └── speaker_1312/120-a_n.wav
└── Laryngitis/              # Laryngitis.zip
    ├── speaker_1304/107-a_n.wav
    ├── speaker_1318/139-a_n.wav
    └── speaker_1320/141-a_n.wav
```

---

## Phonation & Technical Protocol

- **Recording Type**: Sustained vowel `/a/` at normal pitch (`*-a_n.nsp`).
- **Format Conversion**: Extracted from SVD KayPENTAX CSL `.nsp` format and stored as 16-bit Linear PCM `.wav` at 50,000 Hz.
- **Model Standardization**: Preprocessed dynamically to 16,000 Hz single-channel mono with peak normalization for Wav2Vec 2.0 transformer feature extraction.
- **Split Strategy**: Strictly Speaker-Independent (12 training speakers, 6 held-out test speakers, 0 speaker overlap).

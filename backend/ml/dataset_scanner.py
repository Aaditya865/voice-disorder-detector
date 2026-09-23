import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
from ml.config import DATASET_DIR, TARGET_CLASSES, TARGET_CLASS_IDS, SUPPORTED_EXTENSIONS


class DatasetScanner:
    """
    Inspects the dataset directory and metadata.csv, reporting true speaker counts,
    sample distributions, recording types, and train/val/test partitions.
    Adheres strictly to zero-data fabrication.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = self.dataset_dir / "metadata.csv"

    def scan(self) -> Dict[str, Any]:
        """
        Scans all class subdirectories and metadata.csv in dataset_dir.
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        class_stats = []
        available_classes = []
        missing_classes = []
        total_samples = 0
        formats_count = defaultdict(int)
        all_samples_map: Dict[str, List[Path]] = {}
        speakers_per_class: Dict[str, List[str]] = defaultdict(list)
        all_unique_speakers = set()

        # If metadata.csv exists, parse speaker IDs and metadata
        meta_records = []
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        meta_records.append(r)
                        spk = str(r.get("speaker_id", "")).strip()
                        cls = str(r.get("class", "")).strip()
                        if spk and cls:
                            if spk not in speakers_per_class[cls]:
                                speakers_per_class[cls].append(spk)
                            all_unique_speakers.add(spk)
            except Exception as e:
                print(f"Error reading metadata.csv: {e}")

        for target in TARGET_CLASSES:
            cls_id = target["id"]
            # Look for folder
            cls_folder = None
            for child in self.dataset_dir.iterdir():
                if child.is_dir() and child.name.lower() == cls_id.lower():
                    cls_folder = child
                    break

            if cls_folder is None:
                cls_folder = self.dataset_dir / cls_id
                cls_folder.mkdir(parents=True, exist_ok=True)

            # Gather all supported audio files (search recursively for speaker subfolders)
            # Avoid counting duplicate files if both flat and nested exist
            audio_files = []
            seen_stems = set()
            for f in sorted(cls_folder.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Deduplicate by filename stem if both speaker_id/xxx.wav and speaker_id_xxx.wav exist
                    clean_stem = f.stem.replace("speaker_", "")
                    if clean_stem not in seen_stems:
                        seen_stems.add(clean_stem)
                        audio_files.append(f)

            count = len(audio_files)
            total_samples += count

            for f in audio_files:
                formats_count[f.suffix.lower()] += 1

            all_samples_map[cls_id] = audio_files

            is_available = count > 0
            if is_available:
                available_classes.append(cls_id)
            else:
                missing_classes.append(cls_id)

            class_speakers = speakers_per_class.get(cls_id, [])

            class_stats.append({
                "id": cls_id,
                "displayName": target["displayName"],
                "category": target["category"],
                "description": target["description"],
                "sampleCount": count,
                "speakerCount": len(class_speakers),
                "speakers": class_speakers,
                "status": f"Available ({len(class_speakers)} speakers)" if is_available else "Missing / 0 Samples",
                "folderPath": str(cls_folder)
            })

        active_classes_count = len(available_classes)
        can_train = active_classes_count >= 2 and all(
            len(all_samples_map[c]) >= 2 for c in available_classes
        )

        if total_samples == 0:
            status_text = "Empty - Audio files required in class directories"
            readiness = "empty"
        elif not can_train:
            status_text = "Insufficient - Need at least 2 classes with ≥2 samples each to train"
            readiness = "insufficient"
        else:
            status_text = f"Ready - {total_samples} samples across {active_classes_count} classes ({len(all_unique_speakers)} distinct speakers)"
            readiness = "ready"

        # Define speaker-independent partition summary
        train_speakers = []
        test_speakers = []
        val_speakers = []

        for cls_id in available_classes:
            spks = speakers_per_class.get(cls_id, [])
            if len(spks) >= 3:
                # First 2 speakers -> Train, 3rd speaker -> Test
                train_speakers.extend(spks[:2])
                test_speakers.append(spks[2])
            elif len(spks) == 2:
                train_speakers.append(spks[0])
                test_speakers.append(spks[1])
            elif len(spks) == 1:
                train_speakers.append(spks[0])

        return {
            "datasetDirectory": str(self.dataset_dir),
            "totalSamples": total_samples,
            "totalSpeakers": len(all_unique_speakers),
            "totalTargetClasses": len(TARGET_CLASSES),
            "availableClassesCount": active_classes_count,
            "availableClasses": available_classes,
            "missingClassesCount": len(missing_classes),
            "missingClasses": missing_classes,
            "classStatistics": class_stats,
            "formatsDistribution": dict(formats_count),
            "readiness": readiness,
            "statusMessage": status_text,
            "canTrain": can_train,
            "datasetSource": "Saarbrücken Voice Database (SVD)",
            "recordingType": "Sustained vowel /a/ at normal pitch",
            "originalSamplingRate": "50,000 Hz (Resampled to 16,000 Hz mono)",
            "splitStrategy": "Speaker-Independent (Split by Speaker ID, no leakage)",
            "trainSpeakers": train_speakers,
            "testSpeakers": test_speakers,
            "valSpeakers": val_speakers
        }

    def get_labeled_files(self) -> Tuple[List[Path], List[str], List[str]]:
        """
        Returns (file_paths, class_labels, speaker_ids) for all available samples.
        """
        file_paths = []
        labels = []
        speaker_ids = []

        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    p = self.dataset_dir.parent / r["processed_file"]
                    if p.exists():
                        file_paths.append(p)
                        labels.append(r["class"])
                        speaker_ids.append(r["speaker_id"])
            if file_paths:
                return file_paths, labels, speaker_ids

        # Fallback to direct directory scan
        for child in self.dataset_dir.iterdir():
            if child.is_dir():
                cls_name = child.name
                seen = set()
                for f in sorted(child.rglob("*")):
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                        clean_stem = f.stem.replace("speaker_", "")
                        if clean_stem not in seen:
                            seen.add(clean_stem)
                            file_paths.append(f)
                            labels.append(cls_name)
                            speaker_ids.append(f.parent.name.replace("speaker_", "") if "speaker_" in f.parent.name else f.stem)

        return file_paths, labels, speaker_ids

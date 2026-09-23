import io
import csv
import os
import sys
import shutil
from pathlib import Path
import zipfile
import requests
import numpy as np
import scipy.signal
import soundfile as sf
import nspfile
from collections import defaultdict

ARCHIVES = {
    'Normal': {
        'url': 'https://zenodo.org/records/16874898/files/healthy.zip?download=1',
        'folder': 'Normal',
        'target_speakers': 20
    },
    'Laryngozele': {
        'url': 'https://zenodo.org/records/16874898/files/Laryngozele.zip?download=1',
        'folder': 'Laryngozele',
        'target_speakers': 3
    },
    'Vox_senilis': {
        'url': 'https://zenodo.org/records/16874898/files/Vox%20senilis.zip?download=1',
        'folder': 'Vox_senilis',
        'target_speakers': 20
    },
    'Vocal_fold_polyp': {
        'url': 'https://zenodo.org/records/16874898/files/Stimmlippenpolyp.zip?download=1',
        'folder': 'Vocal_fold_polyp',
        'target_speakers': 20
    },
    'Vocal_fold_paralysis': {
        'url': 'https://zenodo.org/records/16874898/files/Rekurrensparese.zip?download=1',
        'folder': 'Vocal_fold_paralysis',
        'target_speakers': 20
    },
    'Laryngitis': {
        'url': 'https://zenodo.org/records/16874898/files/Laryngitis.zip?download=1',
        'folder': 'Laryngitis',
        'target_speakers': 20
    }
}

TARGET_VOWELS = [
    ('a_n', 'sustained_vowel_a_normal', 'a', 'normal'),
    ('i_n', 'sustained_vowel_i_normal', 'i', 'normal'),
    ('u_n', 'sustained_vowel_u_normal', 'u', 'normal'),
    ('a_l', 'sustained_vowel_a_low', 'a', 'low'),
    ('i_l', 'sustained_vowel_i_low', 'i', 'low'),
    ('u_l', 'sustained_vowel_u_low', 'u', 'low'),
    ('a_h', 'sustained_vowel_a_high', 'a', 'high'),
    ('i_h', 'sustained_vowel_i_high', 'i', 'high'),
    ('u_h', 'sustained_vowel_u_high', 'u', 'high')
]

TARGET_VOWEL_KEYS = [v[0] for v in TARGET_VOWELS]

class RemoteZipFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=25)
        self.size = int(r.headers.get('content-length', 0))
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.size + offset
        return self.pos

    def tell(self): return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.size: size = self.size - self.pos
        if size <= 0: return b''
        headers = {'Range': f'bytes={self.pos}-{self.pos + size - 1}'}
        r = self.session.get(self.url, headers=headers, timeout=35)
        self.pos += len(r.content)
        return r.content

    def readable(self): return True
    def seekable(self): return True


def resample_audio(audio_array, orig_sr, target_sr=16000):
    # Ensure float32 1D
    y = np.asarray(audio_array, dtype=np.float32)
    if y.ndim > 1:
        y = y[:, 0]
    
    # Peak normalization before resampling if needed
    max_val = np.max(np.abs(y))
    if max_val > 1.0: # was integer PCM
        y = y / 32768.0

    if orig_sr != target_sr:
        # GCD reduction for resample_poly
        gcd = np.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr // gcd)
        down = int(orig_sr // gcd)
        y = scipy.signal.resample_poly(y, up, down)

    # Normalize to -0.98 to 0.98 peak
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    return y


def run_pipeline():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    scratch_dir = base_dir / "backend" / "data" / "tmp_nsp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv_path = dataset_dir / "metadata.csv"

    print("==================================================", flush=True)
    print("SVD DATASET EXPANSION (OPTION B: 103 SPEAKERS, 9 VOWELS EACH)", flush=True)
    print("==================================================", flush=True)

    all_metadata_rows = []
    class_speaker_allocations = {}

    for cls_name, info in ARCHIVES.items():
        print(f"\n--> Selecting & Downloading for Class: {cls_name}...", flush=True)
        url = info['url']
        target_count = info['target_speakers']
        cls_dir = dataset_dir / info['folder']
        cls_dir.mkdir(parents=True, exist_ok=True)

        rzf = RemoteZipFile(url)
        zf = zipfile.ZipFile(rzf)
        namelist = set(zf.namelist())
        overview_text = zf.read("overview.csv").decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(overview_text))

        # Identify all eligible speakers with all 9 vowels
        eligible_speakers = []
        seen = set()

        for row in reader:
            sess = row.get("AufnahmeID", "").strip()
            spk = row.get("SprecherID", "").strip()
            if not spk or not sess or spk in seen:
                continue

            # Verify all 9 vowel recordings exist in SVD
            has_all_9 = True
            for v_key, _, _, _ in TARGET_VOWELS:
                nsp_rel = f"{sess}/vowels/{sess}-{v_key}.nsp"
                if nsp_rel not in namelist:
                    has_all_9 = False
                    break

            if has_all_9:
                seen.add(spk)
                eligible_speakers.append({
                    "speaker_id": spk,
                    "session_id": sess,
                    "gender": row.get("Geschlecht", "").strip(),
                    "birthdate": row.get("Geburtsdatum", "").strip(),
                    "age": row.get("Alter", "").strip(),
                    "pathology": row.get("Pathologien", "").strip() or cls_name
                })
                if len(eligible_speakers) == target_count:
                    break

        print(f"    Selected {len(eligible_speakers)} distinct speakers for {cls_name}.", flush=True)
        assert len(eligible_speakers) == target_count, f"Expected {target_count} speakers for {cls_name}, found {len(eligible_speakers)}"

        # Assign speaker-independent splits
        # For 20 speakers: 14 train (70%), 3 val (15%), 3 test (15%)
        # For 3 speakers: 2 train (66.7%), 0 val (0%), 1 test (33.3%)
        if target_count == 20:
            train_spks = eligible_speakers[:14]
            val_spks = eligible_speakers[14:17]
            test_spks = eligible_speakers[17:20]
        else:
            train_spks = eligible_speakers[:2]
            val_spks = []
            test_spks = eligible_speakers[2:3]

        class_speaker_allocations[cls_name] = {
            "train": [s["speaker_id"] for s in train_spks],
            "val": [s["speaker_id"] for s in val_spks],
            "test": [s["speaker_id"] for s in test_spks]
        }

        # Process speakers and download recordings
        for split_label, spk_list in [('train', train_spks), ('val', val_spks), ('test', test_spks)]:
            for item in spk_list:
                spk_id = item["speaker_id"]
                sess_id = item["session_id"]
                spk_folder = cls_dir / f"speaker_{spk_id}"
                spk_folder.mkdir(parents=True, exist_ok=True)

                for v_key, rec_type, vowel_char, pitch_val in TARGET_VOWELS:
                    nsp_path_in_zip = f"{sess_id}/vowels/{sess_id}-{v_key}.nsp"
                    wav_filename = f"{sess_id}-{v_key}.wav"
                    final_wav_path = spk_folder / wav_filename

                    # Check if already converted and valid
                    download_needed = True
                    if final_wav_path.exists():
                        try:
                            test_info = sf.info(str(final_wav_path))
                            if test_info.samplerate == 16000 and test_info.frames > 0:
                                download_needed = False
                        except Exception:
                            download_needed = True

                    if download_needed:
                        # Download NSP bytes
                        nsp_bytes = zf.read(nsp_path_in_zip)
                        temp_nsp = scratch_dir / f"temp_{spk_id}_{sess_id}_{v_key}.nsp"
                        with open(temp_nsp, "wb") as f:
                            f.write(nsp_bytes)

                        # Convert from NSP to WAV
                        orig_sr, audio_arr = nspfile.read(str(temp_nsp))
                        if temp_nsp.exists():
                            temp_nsp.unlink()

                        # Resample to 16 kHz Mono
                        y_16k = resample_audio(audio_arr, orig_sr, target_sr=16000)

                        # Save standard 16-bit PCM WAV
                        sf.write(str(final_wav_path), y_16k, 16000, subtype='PCM_16')

                    # Compute duration
                    wav_info = sf.info(str(final_wav_path))
                    duration_sec = round(wav_info.duration, 4)

                    rel_path = str(final_wav_path.relative_to(base_dir)).replace("\\", "/")

                    all_metadata_rows.append({
                        "speaker_id": spk_id,
                        "class": cls_name,
                        "session_id": sess_id,
                        "recording_type": rec_type,
                        "vowel": vowel_char,
                        "pitch": pitch_val,
                        "original_file": nsp_path_in_zip,
                        "processed_file": rel_path,
                        "split": split_label,
                        "sampling_rate": 16000,
                        "duration_seconds": duration_sec,
                        "gender": item["gender"],
                        "birthdate": item["birthdate"],
                        "age": item["age"],
                        "pathology": item["pathology"]
                    })

        print(f"    Completed {cls_name}: {len(eligible_speakers)} speakers, {len(eligible_speakers)*9} WAV files.", flush=True)

    # Save metadata.csv
    fieldnames = [
        "speaker_id", "class", "session_id", "recording_type", "vowel", "pitch",
        "original_file", "processed_file", "split", "sampling_rate", "duration_seconds",
        "gender", "birthdate", "age", "pathology"
    ]
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata_rows)

    print("\n==================================================", flush=True)
    print("DATASET PREPARATION COMPLETE!", flush=True)
    print(f"Total Rows in metadata.csv: {len(all_metadata_rows)}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    run_pipeline()

import csv
import io
import os
import sys
from pathlib import Path
import zipfile
import requests
import soundfile as sf
import nspfile

# RemoteZipFile utilizing HTTP Range Requests to download only specific files
class RemoteZipFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        r = requests.head(url, allow_redirects=True, timeout=20)
        self.size = int(r.headers.get('content-length', 0))
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b''
        headers = {'Range': f'bytes={self.pos}-{self.pos + size - 1}'}
        r = requests.get(self.url, headers=headers, timeout=30)
        self.pos += len(r.content)
        return r.content

    def readable(self):
        return True

    def seekable(self):
        return True


ARCHIVES = {
    "Normal": {
        "url": "https://zenodo.org/records/16874898/files/healthy.zip?download=1",
        "folder": "Normal"
    },
    "Laryngozele": {
        "url": "https://zenodo.org/records/16874898/files/Laryngozele.zip?download=1",
        "folder": "Laryngozele"
    },
    "Vox_senilis": {
        "url": "https://zenodo.org/records/16874898/files/Vox%20senilis.zip?download=1",
        "folder": "Vox_senilis"
    },
    "Vocal_fold_polyp": {
        "url": "https://zenodo.org/records/16874898/files/Stimmlippenpolyp.zip?download=1",
        "folder": "Vocal_fold_polyp"
    },
    "Vocal_fold_paralysis": {
        "url": "https://zenodo.org/records/16874898/files/Rekurrensparese.zip?download=1",
        "folder": "Vocal_fold_paralysis"
    },
    "Laryngitis": {
        "url": "https://zenodo.org/records/16874898/files/Laryngitis.zip?download=1",
        "folder": "Laryngitis"
    }
}


def prepare_svd_dataset(target_speakers_per_class: int = 3):
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv_path = dataset_dir / "metadata.csv"
    scratch_dir = base_dir / "backend" / "data" / "tmp_nsp"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    total_downloaded_bytes = 0

    print("==================================================")
    print(f"FETCHING REAL SVD RECORDINGS ({target_speakers_per_class} DISTINCT SPEAKERS PER CLASS)")
    print("==================================================")

    for class_id, info in ARCHIVES.items():
        print(f"\n--> Processing Class: {class_id} ...")
        url = info["url"]
        class_folder = dataset_dir / info["folder"]
        class_folder.mkdir(parents=True, exist_ok=True)

        rzf = RemoteZipFile(url)
        zf = zipfile.ZipFile(rzf)
        namelist = set(zf.namelist())

        overview_text = zf.read("overview.csv").decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(overview_text))

        selected_for_class = []
        seen_speakers = set()

        for row in reader:
            session_id = row.get("AufnahmeID", "").strip()
            speaker_id = row.get("SprecherID", "").strip()
            if not session_id or not speaker_id:
                continue

            a_n_file = f"{session_id}/vowels/{session_id}-a_n.nsp"
            if a_n_file in namelist and speaker_id not in seen_speakers:
                seen_speakers.add(speaker_id)
                selected_for_class.append({
                    "session_id": session_id,
                    "speaker_id": speaker_id,
                    "nsp_path": a_n_file,
                    "gender": row.get("Geschlecht", "").strip(),
                    "birthdate": row.get("Geburtsdatum", "").strip(),
                    "pathology": row.get("Pathologien", "").strip() or class_id
                })
                if len(selected_for_class) == target_speakers_per_class:
                    break

        print(f"    Found {len(selected_for_class)} eligible distinct speakers in SVD.")

        for item in selected_for_class:
            spk_id = item["speaker_id"]
            sess_id = item["session_id"]
            nsp_path = item["nsp_path"]

            speaker_folder = class_folder / f"speaker_{spk_id}"
            speaker_folder.mkdir(parents=True, exist_ok=True)
            wav_filename = f"{sess_id}-a_n.wav"
            wav_path = speaker_folder / wav_filename

            # Download NSP bytes
            nsp_data = zf.read(nsp_path)
            total_downloaded_bytes += len(nsp_data)

            # Temp save for nspfile reader
            tmp_file = scratch_dir / f"temp_{spk_id}_{sess_id}.nsp"
            with open(tmp_file, "wb") as f:
                f.write(nsp_data)

            # Convert to standard WAV using nspfile
            orig_sr, audio_array = nspfile.read(str(tmp_file))
            if tmp_file.exists():
                tmp_file.unlink()

            # Save clean WAV
            sf.write(str(wav_path), audio_array, orig_sr, subtype="PCM_16")

            # Also create symlink or duplicate in class_folder root if needed so both direct scan and nested speaker scans work
            direct_wav_path = class_folder / f"speaker_{spk_id}_{wav_filename}"
            sf.write(str(direct_wav_path), audio_array, orig_sr, subtype="PCM_16")

            rel_wav_path = str(wav_path.relative_to(base_dir)).replace("\\", "/")
            metadata_rows.append({
                "speaker_id": spk_id,
                "class": class_id,
                "original_file": nsp_path,
                "processed_file": rel_wav_path,
                "recording_type": "sustained_vowel_a_normal",
                "sampling_rate": orig_sr,
                "gender": item["gender"],
                "birthdate": item["birthdate"],
                "pathology": item["pathology"]
            })
            print(f"    [SAVED] Speaker {spk_id} (Session {sess_id}): {orig_sr} Hz, {len(audio_array)} samples -> {wav_filename}")

    # Write metadata.csv
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["speaker_id", "class", "original_file", "processed_file", "recording_type", "sampling_rate", "gender", "birthdate", "pathology"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n==================================================")
    print("DATASET PREPARATION COMPLETE!")
    print(f"Total Speakers Processed: {len(metadata_rows)}")
    print(f"Total Bandwidth Downloaded: {total_downloaded_bytes / (1024*1024):.2f} MB")
    print(f"Metadata saved to: {metadata_csv_path}")
    print("==================================================")


if __name__ == "__main__":
    prepare_svd_dataset(3)

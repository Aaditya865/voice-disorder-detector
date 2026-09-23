import io
import csv
import os
import sys
import time
import zlib
import struct
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    y = np.asarray(audio_array, dtype=np.float32)
    if y.ndim > 1:
        y = y[:, 0]
    
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / 32768.0

    if orig_sr != target_sr:
        gcd = np.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr // gcd)
        down = int(orig_sr // gcd)
        y = scipy.signal.resample_poly(y, up, down)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    return y


def download_and_convert(task_info, session, scratch_dir):
    url = task_info['url']
    header_offset = task_info['header_offset']
    compress_size = task_info['compress_size']
    compress_type = task_info['compress_type']
    filename_len = task_info['filename_len']
    total_archive_size = task_info['total_archive_size']
    final_wav_path = task_info['final_wav_path']

    # If already exists and valid, skip download
    if final_wav_path.exists():
        try:
            info = sf.info(str(final_wav_path))
            if info.samplerate == 16000 and info.frames > 0:
                duration_sec = round(info.duration, 4)
                return {
                    "success": True,
                    "duration": duration_sec,
                    "task": task_info
                }
        except Exception:
            pass

    # Read local header (30 bytes + filename_len + extra buffer) in 1 shot
    buf_size = 30 + filename_len + 128 + compress_size
    req_end = min(header_offset + buf_size - 1, total_archive_size - 1)

    headers = {'Range': f'bytes={header_offset}-{req_end}'}
    
    max_retries = 10
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=45)
            if resp.status_code == 429:
                sleep_sec = (1.5 ** attempt) + np.random.uniform(1.0, 3.0)
                time.sleep(sleep_sec)
                continue
            elif resp.status_code in (200, 206):
                break
            else:
                time.sleep(1.0)
        except Exception:
            time.sleep(2.0 + attempt)
            
    if resp is None or resp.status_code not in (200, 206):
        status = resp.status_code if resp else 'None'
        raise RuntimeError(f"HTTP {status} fetching {task_info['nsp_path_in_zip']}")

    content = resp.content
    fn_len, extra_len = struct.unpack('<HH', content[26:30])
    data_start = 30 + fn_len + extra_len
    raw_compressed = content[data_start : data_start + compress_size]

    if len(raw_compressed) < compress_size:
        # Buffer was slightly short due to huge extra header; fetch tail
        remain = compress_size - len(raw_compressed)
        tail_start = req_end + 1
        tail_r = session.get(url, headers={'Range': f'bytes={tail_start}-{tail_start + remain - 1}'}, timeout=40)
        raw_compressed += tail_r.content

    if compress_type == 8: # Deflated
        decompressed_data = zlib.decompress(raw_compressed, -15)
    else:
        decompressed_data = raw_compressed

    thread_id = threading.get_ident()
    tmp_nsp = scratch_dir / f"tmp_{thread_id}_{task_info['speaker_id']}_{task_info['v_key']}.nsp"
    with open(tmp_nsp, "wb") as f:
        f.write(decompressed_data)

    try:
        orig_sr, audio_arr = nspfile.read(str(tmp_nsp))
    finally:
        if tmp_nsp.exists():
            tmp_nsp.unlink()

    y_16k = resample_audio(audio_arr, orig_sr, target_sr=16000)
    final_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(final_wav_path), y_16k, 16000, subtype='PCM_16')

    info = sf.info(str(final_wav_path))
    duration_sec = round(info.duration, 4)

    return {
        "success": True,
        "duration": duration_sec,
        "task": task_info
    }


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
    total_download_tasks = []
    globally_seen = set()

    for cls_name, info in ARCHIVES.items():
        print(f"\n--> Preparing Cohort for Class: {cls_name}...", flush=True)
        url = info['url']
        target_count = info['target_speakers']
        cls_dir = dataset_dir / info['folder']
        cls_dir.mkdir(parents=True, exist_ok=True)

        rzf = RemoteZipFile(url)
        zf = zipfile.ZipFile(rzf)
        namelist = set(zf.namelist())
        overview_text = zf.read("overview.csv").decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(overview_text))

        eligible_speakers = []

        for row in reader:
            sess = row.get("AufnahmeID", "").strip()
            spk = row.get("SprecherID", "").strip()
            if not spk or not sess or spk in globally_seen:
                continue

            has_all_9 = True
            for v_key, _, _, _ in TARGET_VOWELS:
                nsp_rel = f"{sess}/vowels/{sess}-{v_key}.nsp"
                if nsp_rel not in namelist:
                    has_all_9 = False
                    break

            if has_all_9:
                globally_seen.add(spk)
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

        if target_count == 20:
            train_spks = eligible_speakers[:14]
            val_spks = eligible_speakers[14:17]
            test_spks = eligible_speakers[17:20]
        else:
            train_spks = eligible_speakers[:2]
            val_spks = []
            test_spks = eligible_speakers[2:3]

        for split_label, spk_list in [('train', train_spks), ('val', val_spks), ('test', test_spks)]:
            for item in spk_list:
                spk_id = item["speaker_id"]
                sess_id = item["session_id"]
                spk_folder = cls_dir / f"speaker_{spk_id}"

                for v_key, rec_type, vowel_char, pitch_val in TARGET_VOWELS:
                    nsp_path_in_zip = f"{sess_id}/vowels/{sess_id}-{v_key}.nsp"
                    wav_filename = f"{sess_id}-{v_key}.wav"
                    final_wav_path = spk_folder / wav_filename

                    zinfo = zf.getinfo(nsp_path_in_zip)

                    total_download_tasks.append({
                        "url": url,
                        "class": cls_name,
                        "speaker_id": spk_id,
                        "session_id": sess_id,
                        "v_key": v_key,
                        "recording_type": rec_type,
                        "vowel": vowel_char,
                        "pitch": pitch_val,
                        "split": split_label,
                        "nsp_path_in_zip": nsp_path_in_zip,
                        "final_wav_path": final_wav_path,
                        "header_offset": zinfo.header_offset,
                        "compress_size": zinfo.compress_size,
                        "compress_type": zinfo.compress_type,
                        "filename_len": len(zinfo.filename),
                        "total_archive_size": rzf.size,
                        "gender": item["gender"],
                        "birthdate": item["birthdate"],
                        "age": item["age"],
                        "pathology": item["pathology"],
                        "base_dir": base_dir
                    })

    print(f"\n==================================================", flush=True)
    print(f"COMMENCING FAST PARALLEL CONVERSION ({len(total_download_tasks)} RECORDINGS)", flush=True)
    print(f"==================================================", flush=True)

    session = requests.Session()
    completed = 0
    total = len(total_download_tasks)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_and_convert, task, session, scratch_dir): task for task in total_download_tasks}
        for future in as_completed(futures):
            res = future.result()
            t = res["task"]
            duration_sec = res["duration"]
            rel_path = str(t["final_wav_path"].relative_to(base_dir)).replace("\\", "/")

            all_metadata_rows.append({
                "speaker_id": t["speaker_id"],
                "class": t["class"],
                "session_id": t["session_id"],
                "recording_type": t["recording_type"],
                "vowel": t["vowel"],
                "pitch": t["pitch"],
                "original_file": t["nsp_path_in_zip"],
                "processed_file": rel_path,
                "split": t["split"],
                "sampling_rate": 16000,
                "duration_seconds": duration_sec,
                "gender": t["gender"],
                "birthdate": t["birthdate"],
                "age": t["age"],
                "pathology": t["pathology"]
            })

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"    [{completed}/{total}] files converted to 16 kHz Mono WAV...", flush=True)

    # Sort metadata rows deterministically by class, split, speaker_id, vowel, pitch
    all_metadata_rows.sort(key=lambda r: (r["class"], r["split"], int(r["speaker_id"]) if r["speaker_id"].isdigit() else r["speaker_id"], r["vowel"], r["pitch"]))

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

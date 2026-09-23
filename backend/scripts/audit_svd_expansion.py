import io
import csv
import json
import zipfile
import requests
from collections import defaultdict
from pathlib import Path

ARCHIVES = {
    'Normal': 'https://zenodo.org/records/16874898/files/healthy.zip?download=1',
    'Laryngozele': 'https://zenodo.org/records/16874898/files/Laryngozele.zip?download=1',
    'Vox_senilis': 'https://zenodo.org/records/16874898/files/Vox%20senilis.zip?download=1',
    'Vocal_fold_polyp': 'https://zenodo.org/records/16874898/files/Stimmlippenpolyp.zip?download=1',
    'Vocal_fold_paralysis': 'https://zenodo.org/records/16874898/files/Rekurrensparese.zip?download=1',
    'Laryngitis': 'https://zenodo.org/records/16874898/files/Laryngitis.zip?download=1',
}

class RemoteZipFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        r = requests.head(url, allow_redirects=True, timeout=20)
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
        r = requests.get(self.url, headers=headers, timeout=30)
        self.pos += len(r.content)
        return r.content

    def readable(self): return True
    def seekable(self): return True

out_file = Path("backend/data/svd_audit_detailed.json")
out_file.parent.mkdir(parents=True, exist_ok=True)

audit_results = {}

for cls_name, url in ARCHIVES.items():
    print(f"\n[AUDIT] Starting archive for class: {cls_name}...", flush=True)
    try:
        rzf = RemoteZipFile(url)
        zf = zipfile.ZipFile(rzf)
        namelist = set(zf.namelist())
        overview_data = zf.read("overview.csv").decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(overview_data))
        
        speakers = defaultdict(list)
        recording_types_count = defaultdict(int)
        
        for row in reader:
            sess = row.get("AufnahmeID", "").strip()
            spk = row.get("SprecherID", "").strip()
            gender = row.get("Geschlecht", "").strip()
            age = row.get("Alter", "").strip()
            diag = row.get("Pathologien", "").strip()
            if spk and sess:
                # Find all files belonging to this session
                # Filter out .egg and directory entries
                files = [f for f in namelist if f.startswith(f"{sess}/") and not f.endswith("/") and not f.endswith(".egg")]
                for f in files:
                    # extract recording type e.g. vowels/a_n, sentences/phrase
                    parts = f.split("/")
                    if len(parts) >= 2:
                        rec_name = parts[-1].replace(f"{sess}-", "").replace(".nsp", "").replace(".wav", "")
                        rec_category = parts[-2]
                        recording_types_count[f"{rec_category}/{rec_name}"] += 1
                
                speakers[spk].append({
                    "session_id": sess,
                    "gender": gender,
                    "age": age,
                    "pathology": diag,
                    "audio_files": files
                })
        
        total_audio_recordings = sum(len(sess_info["audio_files"]) for s in speakers.values() for sess_info in s)
        
        audit_results[cls_name] = {
            "distinct_speakers_count": len(speakers),
            "total_sessions": sum(len(v) for v in speakers.values()),
            "total_valid_audio_recordings": total_audio_recordings,
            "speaker_ids": sorted(list(speakers.keys()), key=lambda x: int(x) if x.isdigit() else x),
            "recording_types_distribution": dict(recording_types_count),
            "speakers_detail": {spk: sess_list for spk, sess_list in list(speakers.items())[:10]} # preview
        }
        
        print(f"[AUDIT] {cls_name}: {len(speakers)} distinct speakers, {total_audio_recordings} valid non-EGG audio recordings.", flush=True)
        print(f"        Recording types found: {list(recording_types_count.keys())[:8]}", flush=True)
        
        # Save intermediate results
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
            
    except Exception as e:
        print(f"[ERROR] Failed auditing {cls_name}: {e}", flush=True)

print("\n[AUDIT] COMPLETE! Detailed results saved to backend/data/svd_audit_detailed.json", flush=True)

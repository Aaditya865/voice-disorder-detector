import requests
import json
from pathlib import Path

url = "http://127.0.0.1:8000/predict"
wav_path = Path(r"dataset/Vocal_fold_polyp/speaker_1363/491-a_n.wav").resolve()

print(f"Testing audio file: {wav_path}")
assert wav_path.exists(), f"File does not exist: {wav_path}"

with open(wav_path, "rb") as f:
    files = {"file": ("491-a_n.wav", f, "audio/wav")}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))

import requests, json, time
time.sleep(2)

# Health check
r = requests.get("http://127.0.0.1:8000/health")
h = r.json()
print("=== HEALTH CHECK ===")
print("  Status       :", h["status"])
print("  Model ready  :", h["model_trained"])
print("  Device       :", h["device"])

# Predict on Vox Senilis
print()
print("=== PREDICT: Vox Senilis sample ===")
with open("sample_recordings/3_Vox_Senilis_Sample.wav", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/predict",
                         files={"file": ("test.wav", f, "audio/wav")})
res = resp.json()
print("  Prediction   :", res["display_name"])
print("  Confidence   :", f"{res['confidence']*100:.1f}%")
print("  Architecture :", res["model_architecture"])
print("  Windows used :", res.get("num_windows_voted", 1))
print()
print("  All class probabilities:")
for cls, prob in sorted(res["class_probabilities"].items(), key=lambda x: -x[1]):
    bar = "#" * int(prob * 30)
    print(f"    {cls:<25} {prob*100:>5.1f}%  {bar}")

# Predict on Laryngozele
print()
print("=== PREDICT: Laryngozele sample ===")
with open("sample_recordings/2_Laryngocele_Sample.wav", "rb") as f:
    resp2 = requests.post("http://127.0.0.1:8000/predict",
                          files={"file": ("test.wav", f, "audio/wav")})
res2 = resp2.json()
print("  Prediction   :", res2["display_name"])
print("  Confidence   :", f"{res2['confidence']*100:.1f}%")
print()
print("  All class probabilities:")
for cls, prob in sorted(res2["class_probabilities"].items(), key=lambda x: -x[1]):
    bar = "#" * int(prob * 30)
    print(f"    {cls:<25} {prob*100:>5.1f}%  {bar}")

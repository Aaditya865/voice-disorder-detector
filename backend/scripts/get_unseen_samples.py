import csv, shutil
from pathlib import Path
from collections import defaultdict

BASE = Path(".")
with open("dataset/metadata.csv") as f:
    rows = list(csv.DictReader(f))

# Only test split files that exist on disk
test_rows = [r for r in rows if r["split"] == "test" and Path(r["processed_file"]).exists()]
print(f"Test files on disk: {len(test_rows)}")

# Group by class
by_class = defaultdict(list)
for r in test_rows:
    by_class[r["class"]].append(r)

# Output folder
out_dir = BASE / "unseen_test_samples"
out_dir.mkdir(exist_ok=True)

print()
print("Copying one unseen test sample per class...")
print()

copied = []
for cls in sorted(by_class):
    # Prefer a_n (sustained A at normal pitch) — most standard vowel
    candidates = [r for r in by_class[cls] if "a_n" in r["processed_file"]]
    if not candidates:
        candidates = by_class[cls]
    pick = candidates[0]

    src = Path(pick["processed_file"])
    dest_name = f"UNSEEN_{cls}_spk{pick['speaker_id']}_a_n.wav"
    dest = out_dir / dest_name
    shutil.copy2(str(src), str(dest))
    copied.append((dest_name, pick))

    dur = float(pick["duration_seconds"])
    gen = pick["gender"].upper()
    print(f"  [{cls}]")
    print(f"    Speaker : {pick['speaker_id']} | Gender: {gen} | Duration: {dur:.2f}s")
    print(f"    Saved   : unseen_test_samples/{dest_name}")
    print()

print(f"Done. {len(copied)} files saved to: unseen_test_samples/")

import csv
import json
from pathlib import Path
from collections import defaultdict
import soundfile as sf

def verify():
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"

    print("==================================================", flush=True)
    print("DATASET INTEGRITY AUDIT", flush=True)
    print("==================================================", flush=True)

    if not metadata_csv.exists():
        print("[ERROR] metadata.csv does not exist!", flush=True)
        return

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Total entries in metadata.csv: {len(reader)}")

    # 1. Final speaker count per class & recordings per class
    speakers_by_class = defaultdict(set)
    recordings_by_class = defaultdict(list)
    vowels_by_speaker = defaultdict(set)
    speaker_class_map = {}
    speaker_split_map = {}

    split_speakers = defaultdict(set)
    split_recordings = defaultdict(list)

    all_files_exist = True
    all_sr_16k = True
    any_egg_or_nsp = False

    for row in reader:
        spk = row["speaker_id"]
        cls = row["class"]
        split = row["split"]
        vowel_key = f"{row['vowel']}_{row['pitch']}"
        proc_file = base_dir / row["processed_file"]

        speakers_by_class[cls].add(spk)
        recordings_by_class[cls].append(row)
        vowels_by_speaker[spk].add(vowel_key)
        speaker_class_map[spk] = cls

        if spk in speaker_split_map and speaker_split_map[spk] != split:
            print(f"[ERROR] Speaker {spk} assigned to multiple splits: {speaker_split_map[spk]} and {split}!")
        speaker_split_map[spk] = split

        split_speakers[split].add(spk)
        split_recordings[split].append(row)

        if not proc_file.exists():
            print(f"[MISSING FILE] {proc_file}")
            all_files_exist = False
        else:
            try:
                info = sf.info(str(proc_file))
                if info.samplerate != 16000:
                    print(f"[SR MISMATCH] {proc_file} is {info.samplerate} Hz, expected 16000 Hz")
                    all_sr_16k = False
                if info.frames <= 0:
                    print(f"[EMPTY AUDIO] {proc_file} has 0 frames")
            except Exception as e:
                print(f"[FILE READ ERROR] {proc_file}: {e}")

    # Check for rogue NSP / EGG files in dataset/
    for p in dataset_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in [".nsp", ".egg"]:
                print(f"[FORBIDDEN FILE FOUND] {p}")
                any_egg_or_nsp = True

    # 2. Check speaker overlap between splits
    train_spks = split_speakers["train"]
    val_spks = split_speakers["val"]
    test_spks = split_speakers["test"]

    overlap_train_val = train_spks.intersection(val_spks)
    overlap_train_test = train_spks.intersection(test_spks)
    overlap_val_test = val_spks.intersection(test_spks)

    zero_speaker_leakage = (len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0)

    # 3. Check for missing recordings
    missing_recordings = []
    expected_vowels_count = 9
    for spk, v_set in vowels_by_speaker.items():
        if len(v_set) != expected_vowels_count:
            missing_recordings.append((spk, expected_vowels_count - len(v_set)))

    print("\n--- 1. Speaker & Recording Count per Class ---")
    for cls in sorted(speakers_by_class.keys()):
        spk_cnt = len(speakers_by_class[cls])
        rec_cnt = len(recordings_by_class[cls])
        print(f"  Class: {cls:<25} | Speakers: {spk_cnt:>3} | Recordings: {rec_cnt:>4} (expected {spk_cnt*9})")

    print("\n--- 2. Split Breakdown (Speaker-Independent) ---")
    for s_name in ['train', 'val', 'test']:
        spk_cnt = len(split_speakers[s_name])
        rec_cnt = len(split_recordings[s_name])
        pct_spk = spk_cnt / len(speaker_split_map) * 100
        print(f"  Split: {s_name:<6} | Speakers: {spk_cnt:>3} ({pct_spk:>5.1f}%) | Recordings: {rec_cnt:>4}")

    print("\n--- 3. Speaker Independence & Overlap Audit ---")
    print(f"  Train & Val overlap:  {len(overlap_train_val)} speakers (overlap: {overlap_train_val})")
    print(f"  Train & Test overlap: {len(overlap_train_test)} speakers (overlap: {overlap_train_test})")
    print(f"  Val & Test overlap:   {len(overlap_val_test)} speakers (overlap: {overlap_val_test})")
    print(f"  Zero Speaker Overlap Confirmed: {zero_speaker_leakage}")

    print("\n--- 4. Audio Quality & Integrity Checks ---")
    print(f"  All {len(reader)} audio files exist on disk: {all_files_exist}")
    print(f"  All audio files standardized to 16,000 Hz: {all_sr_16k}")
    print(f"  No forbidden .egg or .nsp files in dataset: {not any_egg_or_nsp}")
    print(f"  Speakers with missing vowels: {len(missing_recordings)}")

    print("==================================================", flush=True)

if __name__ == "__main__":
    verify()

import json

with open("backend/models/val_config_7_8_9_comparison.json") as f:
    data = json.load(f)

c7 = data["Config #7"]["speaker_level"]["speaker_predictions"]
c8 = data["Config #8"]["speaker_level"]["speaker_predictions"]
c9 = data["Config #9"]["speaker_level"]["speaker_predictions"]

speakers = list(c7.keys())

with open("backend/models/val_speakers_table.md", "w", encoding="utf-8") as out:
    out.write("| Speaker ID | True Diagnosis | Config #7 Soft Voting | Config #7 Majority | Config #8 Soft Voting | Config #8 Majority | Config #9 Soft Voting | Config #9 Majority |\n")
    out.write("| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for s in speakers:
        t = c7[s]["true_class"]
        s7_ok = "CORRECT" if c7[s]["soft_correct"] else "INCORRECT"
        h7_ok = "CORRECT" if c7[s]["hard_correct"] else "INCORRECT"
        s8_ok = "CORRECT" if c8[s]["soft_correct"] else "INCORRECT"
        h8_ok = "CORRECT" if c8[s]["hard_correct"] else "INCORRECT"
        s9_ok = "CORRECT" if c9[s]["soft_correct"] else "INCORRECT"
        h9_ok = "CORRECT" if c9[s]["hard_correct"] else "INCORRECT"

        s7 = f"{c7[s]['soft_pred']} ({c7[s]['soft_confidence']*100:.1f}%) [{s7_ok}]"
        h7 = f"{c7[s]['hard_pred']} ({c7[s]['hard_votes']}) [{h7_ok}]"

        s8 = f"{c8[s]['soft_pred']} ({c8[s]['soft_confidence']*100:.1f}%) [{s8_ok}]"
        h8 = f"{c8[s]['hard_pred']} ({c8[s]['hard_votes']}) [{h8_ok}]"

        s9 = f"{c9[s]['soft_pred']} ({c9[s]['soft_confidence']*100:.1f}%) [{s9_ok}]"
        h9 = f"{c9[s]['hard_pred']} ({c9[s]['hard_votes']}) [{h9_ok}]"

        out.write(f"| **{s}** | {t} | {s7} | {h7} | {s8} | {h8} | {s9} | {h9} |\n")

print("Generated val_speakers_table.md")

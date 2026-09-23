import json

with open("backend/models/val_config_7_8_9_comparison.json") as f:
    data = json.load(f)

c7 = data["Config #7"]["speaker_level"]["speaker_predictions"]
c8 = data["Config #8"]["speaker_level"]["speaker_predictions"]
c9 = data["Config #9"]["speaker_level"]["speaker_predictions"]

speakers = list(c7.keys())

print(f"{'Spk ID':<8} {'True Class':<22} | {'Config #7 (Soft)':<26} {'Hard':<22} | {'Config #8 (Soft)':<26} {'Hard':<22} | {'Config #9 (Soft)':<26} {'Hard':<22}")
print("-" * 165)
for s in speakers:
    t = c7[s]["true_class"]

    s7 = f"{c7[s]['soft_pred']} ({c7[s]['soft_confidence']*100:.1f}%) {'[OK]' if c7[s]['soft_correct'] else '[X]'}"
    h7 = f"{c7[s]['hard_pred']} ({c7[s]['hard_votes']}) {'[OK]' if c7[s]['hard_correct'] else '[X]'}"

    s8 = f"{c8[s]['soft_pred']} ({c8[s]['soft_confidence']*100:.1f}%) {'[OK]' if c8[s]['soft_correct'] else '[X]'}"
    h8 = f"{c8[s]['hard_pred']} ({c8[s]['hard_votes']}) {'[OK]' if c8[s]['hard_correct'] else '[X]'}"

    s9 = f"{c9[s]['soft_pred']} ({c9[s]['soft_confidence']*100:.1f}%) {'[OK]' if c9[s]['soft_correct'] else '[X]'}"
    h9 = f"{c9[s]['hard_pred']} ({c9[s]['hard_votes']}) {'[OK]' if c9[s]['hard_correct'] else '[X]'}"

    print(f"{s:<8} {t:<22} | {s7:<26} {h7:<22} | {s8:<26} {h8:<22} | {s9:<26} {h9:<22}")

diff_spks = [s for s in speakers if (c7[s]['soft_correct'] != c9[s]['soft_correct']) or (c8[s]['soft_correct'] != c9[s]['soft_correct']) or (c7[s]['hard_correct'] != c9[s]['hard_correct']) or (c8[s]['hard_correct'] != c9[s]['hard_correct'])]
print("\n=== SPEAKERS WHERE PREDICTIONS DIFFER ===")
for s in diff_spks:
    print(f"\nSpeaker {s} (True Diagnosis: {c7[s]['true_class']}):")
    print(f"  Config #7: Soft={c7[s]['soft_pred']} (conf={c7[s]['soft_confidence']:.4f}, true_prob={c7[s]['probabilities'][c7[s]['true_class']]:.4f}), Hard={c7[s]['hard_pred']} ({c7[s]['hard_votes']})")
    print(f"  Config #8: Soft={c8[s]['soft_pred']} (conf={c8[s]['soft_confidence']:.4f}, true_prob={c8[s]['probabilities'][c8[s]['true_class']]:.4f}), Hard={c8[s]['hard_pred']} ({c8[s]['hard_votes']})")
    print(f"  Config #9: Soft={c9[s]['soft_pred']} (conf={c9[s]['soft_confidence']:.4f}, true_prob={c9[s]['probabilities'][c9[s]['true_class']]:.4f}), Hard={c9[s]['hard_pred']} ({c9[s]['hard_votes']})")
    print(f"  Vote breakdown #7: {c7[s]['vote_breakdown']}")
    print(f"  Vote breakdown #8: {c8[s]['vote_breakdown']}")
    print(f"  Vote breakdown #9: {c9[s]['vote_breakdown']}")
    print(f"  Probabilities #7: {c7[s]['probabilities']}")
    print(f"  Probabilities #8: {c8[s]['probabilities']}")
    print(f"  Probabilities #9: {c9[s]['probabilities']}")

import json

with open("backend/models/validation_optimization_results.json") as f:
    data = json.load(f)

ranked = data["ranked_configurations"]
print("=== TOP 10 CONFIGURATIONS ON VALIDATION SET ===")
for i, c in enumerate(ranked[:10], 1):
    print(f"#{i:2d} | ID:{c['config_id']:3d} | {c['pooling']:17s} | n_est:{c['n_estimators']:3d} | depth:{str(c['max_depth']):4s} | leaf:{c['min_samples_leaf']} | cw:{str(c['class_weight']):13s} | Val Acc:{c['val_recording_acc']*100:.2f}% | Val Macro F1:{c['val_recording_macro_f1']:.4f} | Spk Soft:{c['val_speaker_soft_acc']*100:.1f}%")

print("\n=== BEST CONFIG BY POOLING TYPE ===")
for p in ["mean_pooling", "mean_std_pooling"]:
    best_p = [c for c in ranked if c["pooling"] == p][0]
    print(f"{p:17s} -> Val Acc: {best_p['val_recording_acc']*100:.2f}%, Val Macro F1: {best_p['val_recording_macro_f1']:.4f}, Config ID: {best_p['config_id']}, n_est: {best_p['n_estimators']}, depth: {best_p['max_depth']}, leaf: {best_p['min_samples_leaf']}, cw: {best_p['class_weight']}")

all_c = data["all_configurations_evaluated"]

def avg_stat(field):
    vals = set(str(c[field]) for c in all_c)
    print(f"\n--- Average Metrics by {field} ---")
    for v in sorted(vals):
        subset = [c["val_recording_macro_f1"] for c in all_c if str(c[field]) == v]
        acc_sub = [c["val_recording_acc"] for c in all_c if str(c[field]) == v]
        spk_sub = [c["val_speaker_soft_acc"] for c in all_c if str(c[field]) == v]
        print(f"  {v:20s}: Mean Macro F1={sum(subset)/len(subset):.4f} | Mean Rec Acc={sum(acc_sub)/len(acc_sub)*100:.2f}% | Mean Spk Acc={sum(spk_sub)/len(spk_sub)*100:.1f}%")

avg_stat("pooling")
avg_stat("class_weight")
avg_stat("max_depth")
avg_stat("min_samples_leaf")
avg_stat("n_estimators")

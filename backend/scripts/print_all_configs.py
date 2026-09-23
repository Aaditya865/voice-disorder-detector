import json

with open("backend/models/validation_optimization_results.json") as f:
    data = json.load(f)

for p in ["mean_pooling", "mean_std_pooling"]:
    confs = [c for c in data["all_configurations_evaluated"] if c["pooling"] == p]
    print(f"\n=======================================================")
    print(f"=== {p.upper()} ({len(confs)} CONFIGURATIONS) ===")
    print(f"=======================================================")
    for c in confs:
        cw_str = "None" if "None" in c["class_weight"] else "balanced"
        d_str = str(c["max_depth"])
        print(f"ID:{c['config_id']:3d} | n_est:{c['n_estimators']:3d} | depth:{d_str:4s} | leaf:{c['min_samples_leaf']} | cw:{cw_str:8s} | Val Acc:{c['val_recording_acc']*100:.2f}% | Val Macro F1:{c['val_recording_macro_f1']:.4f} | Spk Soft:{c['val_speaker_soft_acc']*100:.1f}%")

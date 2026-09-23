import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train import train_pipeline

if __name__ == "__main__":
    print("--> Starting training pipeline on real SVD dataset...")
    results = train_pipeline()
    print("\n=== REAL EMPIRICAL EVALUATION RESULTS ===")
    print(f"Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score: {results['macro_metrics']['f1_score']:.4f}")
    print(f"Macro Precision: {results['macro_metrics']['precision']:.4f}")
    print(f"Macro Recall: {results['macro_metrics']['recall']:.4f}")
    print("\nPer-Class Breakdown:")
    for row in results["per_class_metrics"]:
        print(f"  {row['class']}: Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, F1={row['f1_score']:.4f}, Support={row['support']}")
    print("\nConfusion Matrix Labels:", results["confusion_matrix"]["labels"])
    print("Confusion Matrix Rows:")
    for r in results["confusion_matrix"]["matrix"]:
        print(" ", r)
    print("\nTest Predictions Breakdown:")
    for p in results["test_predictions_breakdown"]:
        print(f"  Speaker {p['speaker_id']}: True={p['true_class']}, Pred={p['predicted_class']}, Correct={p['correct']}")

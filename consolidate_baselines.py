"""
FedShield: Consolidate Baseline Metrics
Combines Centralized and Local baseline metrics into structured CSV and JSON artifacts.
"""

import os
import json
import pandas as pd


def consolidate():
    results_dir = "results"
    cent_metrics_path = os.path.join(results_dir, "centralized_metrics.json")
    local_metrics_path = os.path.join(results_dir, "local_metrics.json")
    cent_hist_path = os.path.join(results_dir, "centralized_history.json")
    local_hist_path = os.path.join(results_dir, "local_histories.json")
    
    if not os.path.exists(cent_metrics_path) or not os.path.exists(local_metrics_path):
        raise FileNotFoundError("Baseline metrics missing. Run train_centralized.py and train_local.py first.")
        
    with open(cent_metrics_path, "r") as f:
        cent_metrics = json.load(f)
    with open(local_metrics_path, "r") as f:
        local_metrics_list = json.load(f)
        
    all_models = [cent_metrics] + local_metrics_list
    
    # 1. Save baseline_metrics.json
    with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(all_models, f, indent=2)
        
    # 2. Save confusion_matrices.json
    cm_dict = {}
    for m in all_models:
        key = f"{m['condition']}_{m['client_id']}"
        cm_dict[key] = {
            "model_name": m["model_name"],
            "confusion_matrix": m["confusion_matrix"],
            "total_test_samples": m["total_test_samples"],
            "accuracy": m["accuracy"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1_macro": m["f1_macro"],
            "false_positive_rate": m["false_positive_rate"]
        }
    with open(os.path.join(results_dir, "confusion_matrices.json"), "w") as f:
        json.dump(cm_dict, f, indent=2)
        
    # 3. Save training_history.json
    with open(cent_hist_path, "r") as f:
        cent_hist = json.load(f)
    with open(local_hist_path, "r") as f:
        local_hist = json.load(f)
        
    full_history = {"centralized": cent_hist}
    full_history.update(local_hist)
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(full_history, f, indent=2)
        
    # 4. Save baseline_metrics.csv
    table_rows = []
    for m in all_models:
        cm = m["confusion_matrix"]
        table_rows.append({
            "Model": m["model_name"],
            "Condition": m["condition"],
            "Client": m["client_id"],
            "Seed": m["seed"],
            "Train Samples": m["num_train_samples"],
            "Test Samples": m["total_test_samples"],
            "Epochs": m["epochs_completed"],
            "Training Time (s)": m["total_training_time_sec"],
            "Test Loss": round(m["test_loss"], 4),
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision (%)": round(m["precision"] * 100, 2),
            "Recall (%)": round(m["recall"] * 100, 2),
            "Macro-F1 (%)": round(m["f1_macro"] * 100, 2),
            "FPR (%)": round(m["false_positive_rate"] * 100, 2),
            "TN": cm["TN"],
            "FP": cm["FP"],
            "FN": cm["FN"],
            "TP": cm["TP"]
        })
        
    df = pd.DataFrame(table_rows)
    csv_path = os.path.join(results_dir, "baseline_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 90)
    print(" FedShield Phase 3: Consolidated Baseline Performance on Held-Out Test Set (N=22,544)")
    print("=" * 90)
    print(df[["Model", "Train Samples", "Training Time (s)", "Accuracy (%)", "Precision (%)", "Recall (%)", "Macro-F1 (%)", "FPR (%)"]].to_string(index=False))
    print("=" * 90)
    print(f"\nSaved consolidated metrics to:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(results_dir, 'baseline_metrics.json')}")
    print(f"  {os.path.join(results_dir, 'confusion_matrices.json')}")
    print(f"  {os.path.join(results_dir, 'training_history.json')}")


if __name__ == "__main__":
    consolidate()

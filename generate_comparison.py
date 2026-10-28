"""
FedShield: Full Comparative Evaluation
Generates the comprehensive benchmark table comparing:
- Centralized Baseline
- IID Local Models & IID Local Average
- Non-IID Local Models & Non-IID Local Average
- IID FedAvg Global Model
- Non-IID FedAvg Global Model
All evaluated on the same 22,544-sample official held-out test set.
"""

import os
import json
import pandas as pd
import numpy as np


def generate_comparison_table():
    results_dir = "results"
    baseline_csv = os.path.join(results_dir, "baseline_metrics.csv")
    fedavg_csv = os.path.join(results_dir, "fedavg_metrics.csv")
    
    df_base = pd.read_csv(baseline_csv)
    df_fed = pd.read_csv(fedavg_csv)
    
    # Extract entries
    centralized_row = df_base[df_base["Condition"] == "centralized"].iloc[0].to_dict()
    
    iid_local_rows = df_base[df_base["Condition"] == "iid"]
    non_iid_local_rows = df_base[df_base["Condition"] == "non_iid"]
    
    iid_fed_row = df_fed[df_fed["Condition"] == "iid"].iloc[0].to_dict()
    non_iid_fed_row = df_fed[df_fed["Condition"] == "non_iid"].iloc[0].to_dict()
    
    # Compute IID Local Average
    iid_avg_row = {
        "Model": "IID Local Models (Average of 3 Clients)",
        "Condition": "iid_local_average",
        "Client": "average",
        "Train Samples": int(iid_local_rows["Train Samples"].mean()),
        "Training Time (s)": round(iid_local_rows["Training Time (s)"].mean(), 2),
        "Accuracy (%)": round(iid_local_rows["Accuracy (%)"].mean(), 2),
        "Precision (%)": round(iid_local_rows["Precision (%)"].mean(), 2),
        "Recall (%)": round(iid_local_rows["Recall (%)"].mean(), 2),
        "Macro-F1 (%)": round(iid_local_rows["Macro-F1 (%)"].mean(), 2),
        "FPR (%)": round(iid_local_rows["FPR (%)"].mean(), 2),
        "Comm Payload (MB)": 0.0
    }
    
    # Compute Non-IID Local Average
    non_iid_avg_row = {
        "Model": "Non-IID Local Models (Average of 3 Clients)",
        "Condition": "non_iid_local_average",
        "Client": "average",
        "Train Samples": int(non_iid_local_rows["Train Samples"].mean()),
        "Training Time (s)": round(non_iid_local_rows["Training Time (s)"].mean(), 2),
        "Accuracy (%)": round(non_iid_local_rows["Accuracy (%)"].mean(), 2),
        "Precision (%)": round(non_iid_local_rows["Precision (%)"].mean(), 2),
        "Recall (%)": round(non_iid_local_rows["Recall (%)"].mean(), 2),
        "Macro-F1 (%)": round(non_iid_local_rows["Macro-F1 (%)"].mean(), 2),
        "FPR (%)": round(non_iid_local_rows["FPR (%)"].mean(), 2),
        "Comm Payload (MB)": 0.0
    }
    
    # Format comparison list
    comparison_rows = [
        # Centralized
        {
            "Category": "Centralized Upper Bound",
            "Architecture / Model": centralized_row["Model"],
            "Train Data / Scope": f"{centralized_row['Train Samples']:,} (All Data)",
            "Train Time (s)": centralized_row["Training Time (s)"],
            "Comm (MB)": 0.0,
            "Accuracy (%)": centralized_row["Accuracy (%)"],
            "Precision (%)": centralized_row["Precision (%)"],
            "Recall (%)": centralized_row["Recall (%)"],
            "Macro-F1 (%)": centralized_row["Macro-F1 (%)"],
            "FPR (%)": centralized_row["FPR (%)"],
            "TN": centralized_row["TN"], "FP": centralized_row["FP"],
            "FN": centralized_row["FN"], "TP": centralized_row["TP"]
        },
        # IID Local Avg
        {
            "Category": "IID Local Baselines",
            "Architecture / Model": iid_avg_row["Model"],
            "Train Data / Scope": f"~{iid_avg_row['Train Samples']:,} / client",
            "Train Time (s)": iid_avg_row["Training Time (s)"],
            "Comm (MB)": 0.0,
            "Accuracy (%)": iid_avg_row["Accuracy (%)"],
            "Precision (%)": iid_avg_row["Precision (%)"],
            "Recall (%)": iid_avg_row["Recall (%)"],
            "Macro-F1 (%)": iid_avg_row["Macro-F1 (%)"],
            "FPR (%)": iid_avg_row["FPR (%)"],
            "TN": "-", "FP": "-", "FN": "-", "TP": "-"
        },
        # IID FedAvg
        {
            "Category": "IID Federated",
            "Architecture / Model": iid_fed_row["Model"],
            "Train Data / Scope": "125,973 (Federated 3x)",
            "Train Time (s)": iid_fed_row["Training Time (s)"],
            "Comm (MB)": iid_fed_row["Comm Payload (MB)"],
            "Accuracy (%)": iid_fed_row["Accuracy (%)"],
            "Precision (%)": iid_fed_row["Precision (%)"],
            "Recall (%)": iid_fed_row["Recall (%)"],
            "Macro-F1 (%)": iid_fed_row["Macro-F1 (%)"],
            "FPR (%)": iid_fed_row["FPR (%)"],
            "TN": iid_fed_row["TN"], "FP": iid_fed_row["FP"],
            "FN": iid_fed_row["FN"], "TP": iid_fed_row["TP"]
        },
        # Non-IID Local Avg
        {
            "Category": "Non-IID Local Baselines",
            "Architecture / Model": non_iid_avg_row["Model"],
            "Train Data / Scope": f"~{non_iid_avg_row['Train Samples']:,} / client",
            "Train Time (s)": non_iid_avg_row["Training Time (s)"],
            "Comm (MB)": 0.0,
            "Accuracy (%)": non_iid_avg_row["Accuracy (%)"],
            "Precision (%)": non_iid_avg_row["Precision (%)"],
            "Recall (%)": non_iid_avg_row["Recall (%)"],
            "Macro-F1 (%)": non_iid_avg_row["Macro-F1 (%)"],
            "FPR (%)": non_iid_avg_row["FPR (%)"],
            "TN": "-", "FP": "-", "FN": "-", "TP": "-"
        },
        # Non-IID FedAvg
        {
            "Category": "Non-IID Federated",
            "Architecture / Model": non_iid_fed_row["Model"],
            "Train Data / Scope": "125,973 (Federated 3x)",
            "Train Time (s)": non_iid_fed_row["Training Time (s)"],
            "Comm (MB)": non_iid_fed_row["Comm Payload (MB)"],
            "Accuracy (%)": non_iid_fed_row["Accuracy (%)"],
            "Precision (%)": non_iid_fed_row["Precision (%)"],
            "Recall (%)": non_iid_fed_row["Recall (%)"],
            "Macro-F1 (%)": non_iid_fed_row["Macro-F1 (%)"],
            "FPR (%)": non_iid_fed_row["FPR (%)"],
            "TN": non_iid_fed_row["TN"], "FP": non_iid_fed_row["FP"],
            "FN": non_iid_fed_row["FN"], "TP": non_iid_fed_row["TP"]
        }
    ]
    
    df_comp = pd.DataFrame(comparison_rows)
    out_csv = os.path.join(results_dir, "full_comparison_table.csv")
    out_json = os.path.join(results_dir, "full_comparison_table.json")
    
    df_comp.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(comparison_rows, f, indent=2)
        
    print("\n" + "=" * 105)
    print(" FedShield: Consolidated Benchmark Comparison Table (Held-Out Test Set N = 22,544)")
    print("=" * 105)
    print(df_comp[["Category", "Architecture / Model", "Comm (MB)", "Accuracy (%)", "Precision (%)", "Recall (%)", "Macro-F1 (%)", "FPR (%)"]].to_string(index=False))
    print("=" * 105)
    print(f"\nSaved full comparison table to:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    generate_comparison_table()

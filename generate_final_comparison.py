"""
FedShield: Final Audited Benchmark Comparison Table Generator
Produces results/final_comparison.csv and results/final_comparison.json
Includes:
- Centralized baseline
- All individual local models (IID and Non-IID)
- Clearly labelled local averages
- FedAvg global models (IID and Non-IID)
- Absolute percentage-point differences (Delta vs Local Avg, Delta vs Centralized) calculated from full-precision saved metrics before rounding
- Corrected decimal MB and binary MiB communication units
"""

import os
import json
import pandas as pd
import numpy as np


def generate_final_comparison():
    results_dir = "results"
    baseline_json = os.path.join(results_dir, "baseline_metrics.json")
    fedavg_json = os.path.join(results_dir, "fedavg_metrics.json")
    
    with open(baseline_json, "r") as f:
        base_list = json.load(f)
    with open(fedavg_json, "r") as f:
        fed_list = json.load(f)
    
    cent_item = [x for x in base_list if x["condition"] == "centralized"][0]
    iid_locals = [x for x in base_list if x["condition"] == "iid"]
    non_iid_locals = [x for x in base_list if x["condition"] == "non_iid"]
    
    fed_iid = [x for x in fed_list if x["condition"] == "iid"][0]
    fed_non = [x for x in fed_list if x["condition"] == "non_iid"][0]
    
    cent_acc = cent_item["accuracy"]
    cent_rec = cent_item["recall"]
    
    iid_mean_acc = sum(x["accuracy"] for x in iid_locals) / len(iid_locals)
    iid_mean_rec = sum(x["recall"] for x in iid_locals) / len(iid_locals)
    iid_mean_prec = sum(x["precision"] for x in iid_locals) / len(iid_locals)
    iid_mean_f1 = sum(x["f1_macro"] for x in iid_locals) / len(iid_locals)
    iid_mean_fpr = sum(x["false_positive_rate"] for x in iid_locals) / len(iid_locals)
    iid_mean_time = sum(x["total_training_time_sec"] for x in iid_locals) / len(iid_locals)
    
    non_mean_acc = sum(x["accuracy"] for x in non_iid_locals) / len(non_iid_locals)
    non_mean_rec = sum(x["recall"] for x in non_iid_locals) / len(non_iid_locals)
    non_mean_prec = sum(x["precision"] for x in non_iid_locals) / len(non_iid_locals)
    non_mean_f1 = sum(x["f1_macro"] for x in non_iid_locals) / len(non_iid_locals)
    non_mean_fpr = sum(x["false_positive_rate"] for x in non_iid_locals) / len(non_iid_locals)
    non_mean_time = sum(x["total_training_time_sec"] for x in non_iid_locals) / len(non_iid_locals)
    
    # Serialized bytes = 2,566,380 B -> Decimal MB = 2.566, Binary MiB = 2.447
    comm_mb_dec = round(2566380 / 1000000.0, 3)
    comm_mib_bin = round(2566380 / 1048576.0, 3)
    
    final_rows = []
    
    # 1. Centralized Baseline
    final_rows.append({
        "Model": "Centralized Baseline",
        "Condition": "Centralized",
        "Role / Scope": "Global Upper Bound (All Data)",
        "Train Samples": int(cent_item["num_train_samples"]),
        "Training Time (s)": round(cent_item["total_training_time_sec"], 3),
        "Comm Payload (MB dec)": 0.0,
        "Comm Payload (MiB bin)": 0.0,
        "Accuracy (%)": round(cent_acc * 100, 2),
        "Precision (%)": round(cent_item["precision"] * 100, 2),
        "Attack Recall (%)": round(cent_rec * 100, 2),
        "Macro-F1 (%)": round(cent_item["f1_macro"] * 100, 2),
        "FPR (%)": round(cent_item["false_positive_rate"] * 100, 2),
        "Delta Acc vs Local Avg (pp)": None,
        "Delta Acc vs Centralized (pp)": 0.00,
        "Delta Rec vs Local Avg (pp)": None,
        "Delta Rec vs Centralized (pp)": 0.00,
        "TN": int(cent_item["confusion_matrix"]["TN"]),
        "FP": int(cent_item["confusion_matrix"]["FP"]),
        "FN": int(cent_item["confusion_matrix"]["FN"]),
        "TP": int(cent_item["confusion_matrix"]["TP"])
    })
    
    # 2. IID Local Models
    for item in iid_locals:
        final_rows.append({
            "Model": f"Local Model ({item['client_id'].replace('_', ' ').title()})",
            "Condition": "IID",
            "Role / Scope": f"Isolated Client Partition ({item['client_id']})",
            "Train Samples": int(item["num_train_samples"]),
            "Training Time (s)": round(item["total_training_time_sec"], 3),
            "Comm Payload (MB dec)": 0.0,
            "Comm Payload (MiB bin)": 0.0,
            "Accuracy (%)": round(item["accuracy"] * 100, 2),
            "Precision (%)": round(item["precision"] * 100, 2),
            "Attack Recall (%)": round(item["recall"] * 100, 2),
            "Macro-F1 (%)": round(item["f1_macro"] * 100, 2),
            "FPR (%)": round(item["false_positive_rate"] * 100, 2),
            "Delta Acc vs Local Avg (pp)": round((item["accuracy"] - iid_mean_acc) * 100, 2),
            "Delta Acc vs Centralized (pp)": round((item["accuracy"] - cent_acc) * 100, 2),
            "Delta Rec vs Local Avg (pp)": round((item["recall"] - iid_mean_rec) * 100, 2),
            "Delta Rec vs Centralized (pp)": round((item["recall"] - cent_rec) * 100, 2),
            "TN": int(item["confusion_matrix"]["TN"]),
            "FP": int(item["confusion_matrix"]["FP"]),
            "FN": int(item["confusion_matrix"]["FN"]),
            "TP": int(item["confusion_matrix"]["TP"])
        })
        
    # 3. IID Local Average
    final_rows.append({
        "Model": "IID Local-Only Average",
        "Condition": "IID",
        "Role / Scope": "Mean of 3 Local IID Clients",
        "Train Samples": int(round(sum(x['num_train_samples'] for x in iid_locals) / len(iid_locals))),
        "Training Time (s)": round(iid_mean_time, 2),
        "Comm Payload (MB dec)": 0.0,
        "Comm Payload (MiB bin)": 0.0,
        "Accuracy (%)": round(iid_mean_acc * 100, 2),
        "Precision (%)": round(iid_mean_prec * 100, 2),
        "Attack Recall (%)": round(iid_mean_rec * 100, 2),
        "Macro-F1 (%)": round(iid_mean_f1 * 100, 2),
        "FPR (%)": round(iid_mean_fpr * 100, 2),
        "Delta Acc vs Local Avg (pp)": 0.00,
        "Delta Acc vs Centralized (pp)": round((iid_mean_acc - cent_acc) * 100, 2),
        "Delta Rec vs Local Avg (pp)": 0.00,
        "Delta Rec vs Centralized (pp)": round((iid_mean_rec - cent_rec) * 100, 2),
        "TN": None, "FP": None, "FN": None, "TP": None
    })
    
    # 4. IID FedAvg Global Model
    final_rows.append({
        "Model": "FedAvg Global Model (IID)",
        "Condition": "IID",
        "Role / Scope": "Federated Aggregation (10 Rounds, 3 Clients)",
        "Train Samples": int(fed_iid["total_train_samples"]),
        "Training Time (s)": round(fed_iid["total_training_time_sec"], 3),
        "Comm Payload (MB dec)": comm_mb_dec,
        "Comm Payload (MiB bin)": comm_mib_bin,
        "Accuracy (%)": round(fed_iid["accuracy"] * 100, 2),
        "Precision (%)": round(fed_iid["precision"] * 100, 2),
        "Attack Recall (%)": round(fed_iid["recall"] * 100, 2),
        "Macro-F1 (%)": round(fed_iid["f1_macro"] * 100, 2),
        "FPR (%)": round(fed_iid["false_positive_rate"] * 100, 2),
        "Delta Acc vs Local Avg (pp)": round((fed_iid["accuracy"] - iid_mean_acc) * 100, 2),
        "Delta Acc vs Centralized (pp)": round((fed_iid["accuracy"] - cent_acc) * 100, 2),
        "Delta Rec vs Local Avg (pp)": round((fed_iid["recall"] - iid_mean_rec) * 100, 2),
        "Delta Rec vs Centralized (pp)": round((fed_iid["recall"] - cent_rec) * 100, 2),
        "TN": int(fed_iid["confusion_matrix"]["TN"]),
        "FP": int(fed_iid["confusion_matrix"]["FP"]),
        "FN": int(fed_iid["confusion_matrix"]["FN"]),
        "TP": int(fed_iid["confusion_matrix"]["TP"])
    })
    
    # 5. Non-IID Local Models
    for item in non_iid_locals:
        final_rows.append({
            "Model": f"Local Model ({item['client_id'].replace('_', ' ').title()})",
            "Condition": "Non-IID",
            "Role / Scope": f"Skewed Client Partition ({item['client_id']})",
            "Train Samples": int(item["num_train_samples"]),
            "Training Time (s)": round(item["total_training_time_sec"], 3),
            "Comm Payload (MB dec)": 0.0,
            "Comm Payload (MiB bin)": 0.0,
            "Accuracy (%)": round(item["accuracy"] * 100, 2),
            "Precision (%)": round(item["precision"] * 100, 2),
            "Attack Recall (%)": round(item["recall"] * 100, 2),
            "Macro-F1 (%)": round(item["f1_macro"] * 100, 2),
            "FPR (%)": round(item["false_positive_rate"] * 100, 2),
            "Delta Acc vs Local Avg (pp)": round((item["accuracy"] - non_mean_acc) * 100, 2),
            "Delta Acc vs Centralized (pp)": round((item["accuracy"] - cent_acc) * 100, 2),
            "Delta Rec vs Local Avg (pp)": round((item["recall"] - non_mean_rec) * 100, 2),
            "Delta Rec vs Centralized (pp)": round((item["recall"] - cent_rec) * 100, 2),
            "TN": int(item["confusion_matrix"]["TN"]),
            "FP": int(item["confusion_matrix"]["FP"]),
            "FN": int(item["confusion_matrix"]["FN"]),
            "TP": int(item["confusion_matrix"]["TP"])
        })
        
    # 6. Non-IID Local Average
    final_rows.append({
        "Model": "Non-IID Local-Only Average",
        "Condition": "Non-IID",
        "Role / Scope": "Mean of 3 Local Non-IID Clients",
        "Train Samples": int(round(sum(x['num_train_samples'] for x in non_iid_locals) / len(non_iid_locals))),
        "Training Time (s)": round(non_mean_time, 2),
        "Comm Payload (MB dec)": 0.0,
        "Comm Payload (MiB bin)": 0.0,
        "Accuracy (%)": round(non_mean_acc * 100, 2),
        "Precision (%)": round(non_mean_prec * 100, 2),
        "Attack Recall (%)": round(non_mean_rec * 100, 2),
        "Macro-F1 (%)": round(non_mean_f1 * 100, 2),
        "FPR (%)": round(non_mean_fpr * 100, 2),
        "Delta Acc vs Local Avg (pp)": 0.00,
        "Delta Acc vs Centralized (pp)": round((non_mean_acc - cent_acc) * 100, 2),
        "Delta Rec vs Local Avg (pp)": 0.00,
        "Delta Rec vs Centralized (pp)": round((non_mean_rec - cent_rec) * 100, 2),
        "TN": None, "FP": None, "FN": None, "TP": None
    })
    
    # 7. Non-IID FedAvg Global Model
    final_rows.append({
        "Model": "FedAvg Global Model (Non-IID)",
        "Condition": "Non-IID",
        "Role / Scope": "Federated Aggregation (10 Rounds, 3 Clients)",
        "Train Samples": int(fed_non["total_train_samples"]),
        "Training Time (s)": round(fed_non["total_training_time_sec"], 3),
        "Comm Payload (MB dec)": comm_mb_dec,
        "Comm Payload (MiB bin)": comm_mib_bin,
        "Accuracy (%)": round(fed_non["accuracy"] * 100, 2),
        "Precision (%)": round(fed_non["precision"] * 100, 2),
        "Attack Recall (%)": round(fed_non["recall"] * 100, 2),
        "Macro-F1 (%)": round(fed_non["f1_macro"] * 100, 2),
        "FPR (%)": round(fed_non["false_positive_rate"] * 100, 2),
        "Delta Acc vs Local Avg (pp)": round((fed_non["accuracy"] - non_mean_acc) * 100, 2),
        "Delta Acc vs Centralized (pp)": round((fed_non["accuracy"] - cent_acc) * 100, 2),
        "Delta Rec vs Local Avg (pp)": round((fed_non["recall"] - non_mean_rec) * 100, 2),
        "Delta Rec vs Centralized (pp)": round((fed_non["recall"] - cent_rec) * 100, 2),
        "TN": int(fed_non["confusion_matrix"]["TN"]),
        "FP": int(fed_non["confusion_matrix"]["FP"]),
        "FN": int(fed_non["confusion_matrix"]["FN"]),
        "TP": int(fed_non["confusion_matrix"]["TP"])
    })
    
    df_final = pd.DataFrame(final_rows)
    out_csv = os.path.join(results_dir, "final_comparison.csv")
    out_json = os.path.join(results_dir, "final_comparison.json")
    
    df_final.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(final_rows, f, indent=2)
        
    print("\n" + "=" * 115)
    print(" FedShield: Final Audited Performance Comparison Table (Test Set N = 22,544)")
    print("=" * 115)
    cols_display = ["Model", "Condition", "Accuracy (%)", "Attack Recall (%)", "Macro-F1 (%)", "FPR (%)",
                    "Delta Acc vs Local Avg (pp)", "Delta Rec vs Local Avg (pp)", "Delta Acc vs Centralized (pp)", "Delta Rec vs Centralized (pp)"]
    print(df_final[cols_display].to_string(index=False))
    print("=" * 115)
    print(f"\nSaved final comparison table to:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    generate_final_comparison()

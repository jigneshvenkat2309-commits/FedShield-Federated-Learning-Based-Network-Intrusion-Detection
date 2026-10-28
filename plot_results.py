"""
FedShield: Visualization Suite
Generates 5 publication-quality figures saved under results/figures/:
1. Baseline Comparison (Accuracy, Attack Recall, Macro-F1, FPR)
2. Client Heterogeneity (Class distribution skew and local metric disparity)
3. FedAvg Training Trajectory (Loss and test metrics across 10 rounds)
4. Confusion Matrix Heatmaps (Centralized, IID FedAvg, Non-IID FedAvg)
5. Communication vs Performance Trade-off (Dual-unit payload vs accuracy/recall)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory configuration
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication styling settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

COLORS = {
    "centralized": "#1f77b4",     # Deep Blue
    "iid_local": "#aec7e8",       # Light Blue
    "iid_fed": "#2ca02c",         # Forest Green
    "non_iid_local": "#ffbb78",   # Light Coral
    "non_iid_fed": "#d62728",     # Crimson
    "normal": "#2ca02c",          # Green for Benign
    "attack": "#d62728"           # Red for Attack
}


def plot_1_baseline_comparison(df_final):
    """Figure 1: Comparative bar chart across primary evaluation metrics."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    # Select key models
    target_models = [
        "Centralized Baseline",
        "IID Local-Only Average",
        "FedAvg Global Model (IID)",
        "Non-IID Local-Only Average",
        "FedAvg Global Model (Non-IID)"
    ]
    df_sub = df_final[df_final["Model"].isin(target_models)].copy()
    # Preserve order
    df_sub["sort_order"] = df_sub["Model"].map({m: i for i, m in enumerate(target_models)})
    df_sub = df_sub.sort_values("sort_order")
    
    metrics = ["Accuracy (%)", "Attack Recall (%)", "Macro-F1 (%)", "FPR (%)"]
    metric_labels = ["Accuracy", "Attack Recall", "Macro-F1", "False Positive Rate (FPR)"]
    
    x = np.arange(len(metrics))
    num_models = len(df_sub)
    width = 0.15
    
    palette = [COLORS["centralized"], COLORS["iid_local"], COLORS["iid_fed"],
               COLORS["non_iid_local"], COLORS["non_iid_fed"]]
    
    for i, (_, row) in enumerate(df_sub.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - num_models / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=row["Model"], color=palette[i],
                       edgecolor="black", linewidth=0.7, alpha=0.9)
        # Annotate values
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, rotation=0)
            
    ax.set_ylabel("Metric Percentage (%)", fontweight="bold")
    ax.set_title("FedShield: Centralized vs Local Baselines vs FedAvg (Held-Out Test Set N = 22,544)",
                 fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "1_baseline_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_2_client_heterogeneity():
    """Figure 2: Client class skew and standalone local performance variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    clients = ["IID C1", "IID C2", "IID C3", "Non-IID C1", "Non-IID C2", "Non-IID C3"]
    normal_pct = [53.46, 53.46, 53.46, 92.28, 9.48, 69.83]
    attack_pct = [46.54, 46.54, 46.54, 7.72, 90.52, 30.17]
    
    # Subplot A: Class Distribution Stacked Bar
    x = np.arange(len(clients))
    width = 0.55
    ax1.bar(x, normal_pct, width, label="Normal (Class 0)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax1.bar(x, attack_pct, width, bottom=normal_pct, label="Attack (Class 1)", color="#d62728", edgecolor="black", alpha=0.85)
    
    for i in range(len(clients)):
        ax1.text(i, normal_pct[i] / 2, f"{normal_pct[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        ax1.text(i, normal_pct[i] + attack_pct[i] / 2, f"{attack_pct[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        
    ax1.set_ylabel("Local Class Proportion (%)", fontweight="bold")
    ax1.set_title("(A) Local Training Data Composition", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clients, rotation=20, ha="right", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", framealpha=0.9)
    
    # Subplot B: Local Evaluation Performance
    acc = [74.51, 75.38, 74.48, 71.33, 81.93, 73.19]
    rec = [60.87, 62.42, 60.83, 54.69, 74.96, 58.30]
    fpr = [7.48, 7.51, 7.49, 6.68, 8.85, 7.12]
    
    w = 0.25
    r1 = ax2.bar(x - w, acc, w, label="Accuracy", color="#1f77b4", edgecolor="black", alpha=0.85)
    r2 = ax2.bar(x, rec, w, label="Attack Recall", color="#ff7f0e", edgecolor="black", alpha=0.85)
    r3 = ax2.bar(x + w, fpr, w, label="FPR (False Alarms)", color="#9467bd", edgecolor="black", alpha=0.85)
    
    # Highlight Client 1 recall drop and Client 2 high FPR
    ax2.annotate("Recall Blindness\n(54.69%)", xy=(3, 54.69), xytext=(3, 35),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 ha="center", fontsize=8.5, fontweight="bold")
    ax2.annotate("Highest False Alarms\n(8.85% FPR)", xy=(4 + w, 8.85), xytext=(4 + w, 22),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 ha="center", fontsize=8.5, fontweight="bold")
                 
    ax2.set_ylabel("Metric Percentage (%)", fontweight="bold")
    ax2.set_title("(B) Local Model Performance Disparity (Held-Out Test Set)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(clients, rotation=20, ha="right", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "2_client_heterogeneity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_3_fedavg_training_trajectory():
    """Figure 3: Round-by-round convergence of Loss, Accuracy, and Attack Recall."""
    with open(os.path.join(RESULTS_DIR, "fedavg_round_history.json"), "r") as f:
        hist = json.load(f)
        
    rounds = [r["round"] for r in hist["iid"]]
    iid_loss = [np.mean(list(r["client_losses"].values())) for r in hist["iid"]]
    non_iid_loss = [np.mean(list(r["client_losses"].values())) for r in hist["non_iid"]]
    
    iid_acc = [r["test_accuracy"] for r in hist["iid"]]
    non_iid_acc = [r["test_accuracy"] for r in hist["non_iid"]]
    
    iid_rec = [r["test_recall"] for r in hist["iid"]]
    non_iid_rec = [r["test_recall"] for r in hist["non_iid"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot A: Training Loss Trajectory
    ax1.plot(rounds, iid_loss, marker="o", linewidth=2.2, color="#2ca02c", label="IID (Avg Client Loss)")
    ax1.plot(rounds, non_iid_loss, marker="s", linewidth=2.2, color="#d62728", linestyle="--", label="Non-IID (Avg Client Loss)")
    ax1.set_xlabel("Communication Round", fontweight="bold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("(A) Federated Training Loss Trajectory", fontweight="bold")
    ax1.set_xticks(rounds)
    ax1.legend(loc="upper right")
    
    # Subplot B: Test Metrics Trajectory
    ax2.plot(rounds, iid_acc, marker="o", linewidth=2.0, color="#1f77b4", label="IID Test Accuracy")
    ax2.plot(rounds, iid_rec, marker="^", linewidth=2.0, color="#2ca02c", label="IID Attack Recall")
    ax2.plot(rounds, non_iid_acc, marker="s", linewidth=2.0, color="#ff7f0e", linestyle="--", label="Non-IID Test Accuracy")
    ax2.plot(rounds, non_iid_rec, marker="v", linewidth=2.0, color="#d62728", linestyle="--", label="Non-IID Attack Recall")
    
    ax2.axhline(78.71, color="gray", linestyle=":", linewidth=1.5, label="Centralized Acc (78.7%)")
    ax2.axhline(64.68, color="black", linestyle=":", linewidth=1.5, label="Centralized Recall (64.7%)")
    
    ax2.set_xlabel("Communication Round", fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontweight="bold")
    ax2.set_title("(B) Held-Out Test Set Convergence Trajectory", fontweight="bold")
    ax2.set_xticks(rounds)
    ax2.set_ylim(50, 85)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "3_fedavg_training_trajectory.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_4_confusion_matrices():
    """Figure 4: Confusion Matrix Heatmaps for Centralized, IID FedAvg, Non-IID FedAvg."""
    with open(os.path.join(RESULTS_DIR, "confusion_matrices.json"), "r") as f:
        cms = json.load(f)
        
    models_to_plot = [
        ("Centralized Baseline", cms["centralized_all"]["confusion_matrix"]),
        ("FedAvg Global (IID)", {
            "TN": 8995, "FP": 716, "FN": 4622, "TP": 8211, "total": 22544
        }),
        ("FedAvg Global (Non-IID)", {
            "TN": 8981, "FP": 730, "FN": 5090, "TP": 7743, "total": 22544
        })
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    
    for idx, (title, cm) in enumerate(models_to_plot):
        ax = axes[idx]
        mat = np.array([[cm["TN"], cm["FP"]],
                        [cm["FN"], cm["TP"]]])
        total = cm["total"]
        
        im = ax.imshow(mat, cmap="Blues", interpolation="nearest", vmin=0, vmax=11000)
        
        # Quadrant labels
        labels = [
            [f"TN\n{cm['TN']:,}\n({cm['TN']/total*100:.1f}%)", f"FP\n{cm['FP']:,}\n({cm['FP']/total*100:.1f}%)"],
            [f"FN\n{cm['FN']:,}\n({cm['FN']/total*100:.1f}%)", f"TP\n{cm['TP']:,}\n({cm['TP']/total*100:.1f}%)"]
        ]
        
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > 5500 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", color=color,
                        fontweight="bold", fontsize=10.5)
                
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal (0)", "Attack (1)"], fontweight="bold")
        ax.set_yticklabels(["Normal (0)", "Attack (1)"], fontweight="bold")
        ax.set_xlabel("Predicted Class", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Class", fontweight="bold")
        ax.set_title(f"{title}\nTotal: {total:,}", fontweight="bold", pad=10)
        
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "4_confusion_matrices.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_5_communication_tradeoff():
    """Figure 5: Communication Cost (Decimal MB & Binary MiB) vs Accuracy/Recall."""
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    
    rounds = np.arange(1, 11)
    # Per-round serialized transfer: 256,638 bytes per round
    # Decimal MB = bytes / 1,000,000; Binary MiB = bytes / 1,048,576
    cum_bytes = rounds * 256638
    cum_mb = cum_bytes / 1000000.0   # Decimal MB
    cum_mib = cum_bytes / 1048576.0  # Binary MiB
    
    with open(os.path.join(RESULTS_DIR, "fedavg_round_history.json"), "r") as f:
        hist = json.load(f)
    iid_acc = [r["test_accuracy"] for r in hist["iid"]]
    non_iid_acc = [r["test_accuracy"] for r in hist["non_iid"]]
    
    ax1.plot(cum_mb, iid_acc, marker="o", linewidth=2.4, color="#2ca02c", label="IID FedAvg Accuracy")
    ax1.plot(cum_mb, non_iid_acc, marker="s", linewidth=2.4, color="#d62728", linestyle="--", label="Non-IID FedAvg Accuracy")
    
    # Baselines
    ax1.axhline(78.71, color="#1f77b4", linestyle="-.", linewidth=1.5, label="Centralized (78.71%, 0 MB Comm)")
    ax1.axhline(74.79, color="#aec7e8", linestyle=":", linewidth=1.8, label="IID Local Average (74.79%, 0 MB Comm)")
    ax1.axhline(75.48, color="#ffbb78", linestyle=":", linewidth=1.8, label="Non-IID Local Average (75.48%, 0 MB Comm)")
    
    # Annotate final cumulative transfers
    final_mb = cum_mb[-1]
    final_mib = cum_mib[-1]
    ax1.annotate(f"Final Transfer (Round 10):\n{final_mb:.3f} MB (Decimal)\n{final_mib:.3f} MiB (Binary)",
                 xy=(final_mb, iid_acc[-1]), xytext=(final_mb - 0.7, 72.5),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", lw=1),
                 fontsize=9, fontweight="bold")
                 
    ax1.set_xlabel("Cumulative Simulated Model Payload Transfer (Decimal MB = bytes / 10^6)", fontweight="bold")
    ax1.set_ylabel("Held-Out Test Accuracy (%)", fontweight="bold")
    ax1.set_title("Communication Overhead vs Accuracy Convergence Trade-off", fontweight="bold", pad=12)
    ax1.set_ylim(70, 81)
    ax1.legend(loc="lower left", framealpha=0.95, edgecolor="gray")
    
    # Secondary top X axis showing binary MiB
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # Convert tick marks to binary MiB
    dec_ticks = ax1.get_xticks()
    mib_labels = [f"{t * 1000000 / 1048576:.2f}" for t in dec_ticks]
    ax2.set_xticks(dec_ticks)
    ax2.set_xticklabels(mib_labels)
    ax2.set_xlabel("Cumulative Simulated Model Payload Transfer (Binary MiB = bytes / 2^20)", fontweight="bold")
    ax2.grid(False)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "5_communication_tradeoff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 70)
    print(" FedShield: Generating Publication-Quality Figures")
    print("=" * 70)
    df_final = pd.read_csv(os.path.join(RESULTS_DIR, "final_comparison.csv"))
    
    plot_1_baseline_comparison(df_final)
    plot_2_client_heterogeneity()
    plot_3_fedavg_training_trajectory()
    plot_4_confusion_matrices()
    plot_5_communication_tradeoff()
    
    print("=" * 70)
    print(" All 5 figures successfully generated in results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
FedShield: Client Partitioning (IID vs Non-IID)
Partitions the preprocessed training dataset across 3 simulated federated clients.
Supports:
1. IID: Seeded stratified uniform distribution across all clients.
2. Non-IID: Dirichlet label-skew with configurable alpha parameter.
"""

import os
import sys
import json
import argparse
import numpy as np


def partition_iid(X, y, num_clients=3, seed=42):
    """
    Generate an IID partition using stratified random sampling across classes.
    Ensures equal sample counts and identical class proportions across all clients.
    """
    rng = np.random.RandomState(seed)
    client_indices = {k: [] for k in range(num_clients)}
    
    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        splits = np.array_split(idx_c, num_clients)
        for k in range(num_clients):
            client_indices[k].extend(splits[k])
            
    # Convert to sorted/shuffled numpy arrays
    for k in range(num_clients):
        c_idx = np.array(client_indices[k], dtype=np.int64)
        rng.shuffle(c_idx)
        client_indices[k] = c_idx
        
    return client_indices


def partition_non_iid(X, y, num_clients=3, alpha=0.3, min_class_ratio=0.05, seed=42):
    """
    Generate a Non-IID partition using Dirichlet label-skew.
    - alpha: Dirichlet concentration parameter (lower alpha -> stronger skew).
    - min_class_ratio: minimum proportion of class samples per client to ensure
      local models have sufficient samples of all classes for training.
    """
    rng = np.random.RandomState(seed)
    client_indices = {k: [] for k in range(num_clients)}
    
    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        
        # Draw Dirichlet proportions ensuring min_class_ratio
        attempts = 0
        while attempts < 1000:
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            if np.all(proportions >= min_class_ratio):
                break
            attempts += 1
            
        if attempts >= 1000:
            # Fallback clipping and renormalization if extreme alpha
            proportions = np.maximum(proportions, min_class_ratio)
            proportions /= proportions.sum()
            
        counts = (proportions * len(idx_c)).astype(int)
        # Remainder adjustment to preserve exact count
        counts[-1] = len(idx_c) - counts[:-1].sum()
        
        splits = np.split(idx_c, np.cumsum(counts)[:-1])
        for k in range(num_clients):
            client_indices[k].extend(splits[k])
            
    for k in range(num_clients):
        c_idx = np.array(client_indices[k], dtype=np.int64)
        rng.shuffle(c_idx)
        client_indices[k] = c_idx
        
    return client_indices


def save_partitions(X, y, client_indices, condition_name, output_dir, config):
    """Save client partitions as compressed .npz archives and save metadata."""
    cond_dir = os.path.join(output_dir, condition_name)
    os.makedirs(cond_dir, exist_ok=True)
    
    client_stats = {}
    all_assigned_indices = []
    
    for k, idx in client_indices.items():
        client_id = f"client_{k + 1}"
        X_k = X[idx]
        y_k = y[idx]
        
        # Save client archive
        npz_path = os.path.join(cond_dir, f"{client_id}.npz")
        np.savez_compressed(npz_path, X=X_k, y=y_k, indices=idx)
        
        # Compute statistics
        unique_labels, counts = np.unique(y_k, return_counts=True)
        class_dist = {int(label): int(cnt) for label, cnt in zip(unique_labels, counts)}
        class_props = {int(label): round(cnt / len(y_k) * 100, 2) for label, cnt in zip(unique_labels, counts)}
        
        client_stats[client_id] = {
            "num_samples": int(len(idx)),
            "class_counts": class_dist,
            "class_proportions_pct": class_props,
            "feature_shape": list(X_k.shape),
            "file_size_bytes": os.path.getsize(npz_path)
        }
        all_assigned_indices.extend(idx)
        
    # Verify index integrity
    all_assigned_indices = np.array(all_assigned_indices)
    assert len(all_assigned_indices) == len(X), "Sample count mismatch!"
    assert len(np.unique(all_assigned_indices)) == len(X), "Duplicate indices across clients!"
    
    metadata = {
        "condition": condition_name,
        "config": config,
        "num_clients": len(client_indices),
        "total_samples": int(len(X)),
        "clients": client_stats
    }
    
    with open(os.path.join(cond_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata


def print_summary_table(condition_name, metadata):
    print(f"\n========================================================")
    print(f" Condition: {condition_name.upper()} Partition Summary")
    print(f"========================================================")
    print(f"{'Client':<10} | {'Total':<10} | {'Normal (0)':<22} | {'Attack (1)':<22}")
    print("-" * 72)
    for client_id, stats in metadata["clients"].items():
        total = stats["num_samples"]
        c0 = stats["class_counts"].get(0, 0)
        p0 = stats["class_proportions_pct"].get(0, 0.0)
        c1 = stats["class_counts"].get(1, 0)
        p1 = stats["class_proportions_pct"].get(1, 0.0)
        print(f"{client_id:<10} | {total:<10,d} | {c0:>7,d} ({p0:6.2f}%)       | {c1:>7,d} ({p1:6.2f}%)")
    print("-" * 72)
    tot = metadata["total_samples"]
    print(f"{'Total':<10} | {tot:<10,d} | Complete partition of {tot:,} training records.")


def run_partitioning(data_dir="data/processed", output_dir="data/partitions",
                     condition="both", alpha=0.3, seed=42, num_clients=3):
    """Execute partitioning and persist all client files."""
    train_feat_path = os.path.join(data_dir, "train_features.npy")
    train_label_path = os.path.join(data_dir, "train_labels.npy")
    
    if not os.path.exists(train_feat_path) or not os.path.exists(train_label_path):
        raise FileNotFoundError(f"Processed training files missing in {data_dir}. Run preprocess.py first.")
        
    print(f"Loading preprocessed training data from {data_dir}...")
    X_train = np.load(train_feat_path)
    y_train = np.load(train_label_path)
    
    print(f"Dataset shape: X={X_train.shape}, y={y_train.shape}")
    
    summary = {}
    
    if condition in ["iid", "both"]:
        print(f"\n[Generating IID Partition (Stratified Balanced, Seed={seed})...]")
        iid_indices = partition_iid(X_train, y_train, num_clients=num_clients, seed=seed)
        iid_config = {"seed": seed, "num_clients": num_clients, "strategy": "stratified_random"}
        summary["iid"] = save_partitions(X_train, y_train, iid_indices, "iid", output_dir, iid_config)
        print_summary_table("iid", summary["iid"])
        
    if condition in ["non_iid", "both"]:
        print(f"\n[Generating Non-IID Partition (Dirichlet Label Skew, Alpha={alpha}, Seed={seed})...]")
        non_iid_indices = partition_non_iid(X_train, y_train, num_clients=num_clients,
                                            alpha=alpha, seed=seed)
        non_iid_config = {
            "seed": seed,
            "num_clients": num_clients,
            "strategy": "dirichlet_label_skew",
            "alpha": alpha,
            "min_class_ratio": 0.05
        }
        summary["non_iid"] = save_partitions(X_train, y_train, non_iid_indices, "non_iid", output_dir, non_iid_config)
        print_summary_table("non_iid", summary["non_iid"])
        
    # Top-level summary
    summary_path = os.path.join(output_dir, "partition_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nAll partition archives saved to {os.path.abspath(output_dir)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="FedShield Client Partitioning (IID vs Non-IID)")
    parser.add_argument("--condition", choices=["iid", "non_iid", "both"], default="both",
                        help="Partition condition to generate: iid, non_iid, or both")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet concentration parameter for non-IID skew (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of simulated federated clients (default: 3)")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Directory containing preprocessed data arrays")
    parser.add_argument("--output-dir", default="data/partitions",
                        help="Directory to save generated client partitions")
                        
    args = parser.parse_args()
    run_partitioning(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        condition=args.condition,
        alpha=args.alpha,
        seed=args.seed,
        num_clients=args.num_clients
    )


if __name__ == "__main__":
    main()

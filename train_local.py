"""
FedShield: Independent Local-Only Baseline Training
Trains 6 independent models:
- 3 models on IID client partitions (Client 1, Client 2, Client 3)
- 3 models on Non-IID client partitions (Client 1, Client 2, Client 3)
All models evaluate on the shared official held-out test set (22,544 records).
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import create_model
from evaluation import evaluate_model, format_metrics_table


def train_single_client(client_npz_path, condition, client_id,
                        test_x, test_y, models_dir="models",
                        batch_size=1024, epochs=10, lr=0.001,
                        seed=42, device="cpu"):
    """Trains a single client model on its private partition and evaluates on test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load private client data
    client_data = np.load(client_npz_path)
    X_train_client = client_data["X"]
    y_train_client = client_data["y"]
    num_samples = len(X_train_client)
    
    # 2. Build DataLoader
    dataset = TensorDataset(
        torch.from_numpy(X_train_client).float(),
        torch.from_numpy(y_train_client).float().unsqueeze(1)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, pin_memory=False
    )
    
    # 3. Model, Loss, Optimizer
    model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2, seed=seed)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    history = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            
        avg_loss = epoch_loss / num_samples
        duration = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "epoch_duration_sec": round(duration, 3)
        })
        
    total_time = time.time() - start_time
    
    # 5. Evaluate on shared held-out test set
    metrics = evaluate_model(model, test_x, test_y, device=device, threshold=0.5)
    
    # Record metadata
    metrics["model_name"] = f"Local Model ({condition.upper()} - {client_id})"
    metrics["condition"] = condition
    metrics["client_id"] = client_id
    metrics["seed"] = seed
    metrics["num_train_samples"] = num_samples
    metrics["batch_size"] = batch_size
    metrics["epochs_completed"] = epochs
    metrics["learning_rate"] = lr
    metrics["total_training_time_sec"] = round(total_time, 3)
    
    # 6. Save model checkpoint
    cond_model_dir = os.path.join(models_dir, condition)
    os.makedirs(cond_model_dir, exist_ok=True)
    model_path = os.path.join(cond_model_dir, f"{client_id}.pt")
    
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": 122,
            "hidden_dims": (64, 32),
            "dropout_rate": 0.2,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "client_id": client_id,
            "condition": condition
        },
        "metrics": metrics,
        "history": history
    }, model_path)
    
    return metrics, history, model_path


def run_all_local_baselines(partitions_dir="data/partitions",
                            processed_dir="data/processed",
                            models_dir="models",
                            results_dir="results",
                            batch_size=1024, epochs=10, lr=0.001,
                            base_seed=100, device="cpu"):
    """Trains all 6 local-only models (3 IID, 3 Non-IID)."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load shared test set
    test_x = np.load(os.path.join(processed_dir, "test_features.npy"))
    test_y = np.load(os.path.join(processed_dir, "test_labels.npy"))
    
    all_local_metrics = []
    all_histories = {}
    
    conditions = ["iid", "non_iid"]
    
    print("=" * 70)
    print(" FedShield: Independent Local-Only Baseline Training (6 Models)")
    print("=" * 70)
    
    for cond_idx, condition in enumerate(conditions):
        print(f"\n>>> Running condition: {condition.upper()} <<<")
        for k in [1, 2, 3]:
            client_id = f"client_{k}"
            client_npz = os.path.join(partitions_dir, condition, f"{client_id}.npz")
            
            # Distinct reproducible seed per client
            client_seed = base_seed + (cond_idx * 10) + k
            
            print(f"Training {condition} - {client_id} (Seed: {client_seed})...")
            t_start = time.time()
            metrics, history, saved_path = train_single_client(
                client_npz_path=client_npz,
                condition=condition,
                client_id=client_id,
                test_x=test_x,
                test_y=test_y,
                models_dir=models_dir,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                seed=client_seed,
                device=device
            )
            elapsed = time.time() - t_start
            print(f"  [DONE] {condition}_{client_id} in {elapsed:.2f}s | Acc: {metrics['accuracy']*100:.2f}% | Rec: {metrics['recall']*100:.2f}% | F1: {metrics['f1_macro']*100:.2f}% | FPR: {metrics['false_positive_rate']*100:.2f}%")
            
            all_local_metrics.append(metrics)
            all_histories[f"{condition}_{client_id}"] = history
            
    # Save local metrics summary
    with open(os.path.join(results_dir, "local_metrics.json"), "w") as f:
        json.dump(all_local_metrics, f, indent=2)
    with open(os.path.join(results_dir, "local_histories.json"), "w") as f:
        json.dump(all_histories, f, indent=2)
        
    return all_local_metrics, all_histories


def main():
    parser = argparse.ArgumentParser(description="FedShield Local-Only Baselines Training")
    parser.add_argument("--partitions-dir", default="data/partitions")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    run_all_local_baselines(
        partitions_dir=args.partitions_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        base_seed=args.base_seed,
        device=args.device
    )


if __name__ == "__main__":
    main()

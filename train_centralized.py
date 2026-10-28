"""
FedShield: Centralized Training Baseline
Trains the IntrusionMLP on the entire official training set (125,973 records)
and evaluates on the official held-out test set (22,544 records).
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


def train_centralized(data_dir="data/processed", models_dir="models",
                      results_dir="results", batch_size=1024, epochs=10,
                      lr=0.001, seed=42, device="cpu"):
    """Execute centralized baseline training and evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load data
    train_x_path = os.path.join(data_dir, "train_features.npy")
    train_y_path = os.path.join(data_dir, "train_labels.npy")
    test_x_path = os.path.join(data_dir, "test_features.npy")
    test_y_path = os.path.join(data_dir, "test_labels.npy")
    
    print(f"Loading training data from {train_x_path}...")
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    
    num_train_samples = len(X_train)
    num_test_samples = len(X_test)
    print(f"Loaded {num_train_samples:,} train samples, {num_test_samples:,} test samples.")
    
    # 2. Build DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().unsqueeze(1)
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=generator, pin_memory=False
    )
    
    # 3. Initialize Model, Loss, Optimizer
    model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2, seed=seed)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    param_count, model_bytes = model.count_parameters()
    print(f"Model initialized: {param_count:,} parameters ({model_bytes/1024:.2f} KB).")
    print(f"Beginning Centralized Training (Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}, Seed: {seed})...\n")
    
    # 4. Training Loop
    history = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            
        avg_loss = epoch_loss / num_train_samples
        duration = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "epoch_duration_sec": round(duration, 3)
        })
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.5f} | Time: {duration:.2f}s")
        
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/epochs:.2f}s/epoch).")
    
    # 5. Evaluate on Official Held-out Test Set
    print("\nEvaluating on official held-out test set (N=22,544)...")
    metrics = evaluate_model(model, X_test, y_test, device=device, threshold=0.5)
    
    # Add metadata
    metrics["model_name"] = "Centralized Baseline"
    metrics["condition"] = "centralized"
    metrics["client_id"] = "all"
    metrics["seed"] = seed
    metrics["num_train_samples"] = num_train_samples
    metrics["batch_size"] = batch_size
    metrics["epochs_completed"] = epochs
    metrics["learning_rate"] = lr
    metrics["total_training_time_sec"] = round(total_training_time, 3)
    
    print(format_metrics_table(metrics, title="Centralized Baseline Test Results"))
    
    # 6. Save Model and Outputs
    model_save_path = os.path.join(models_dir, "centralized.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": 122,
            "hidden_dims": (64, 32),
            "dropout_rate": 0.2,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        },
        "metrics": metrics,
        "history": history
    }, model_save_path)
    print(f"Model checkpoint saved to: {model_save_path}")
    
    # Save standalone JSON metrics and history
    with open(os.path.join(results_dir, "centralized_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "centralized_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    return metrics, history


def main():
    parser = argparse.ArgumentParser(description="FedShield Centralized Baseline Training")
    parser.add_argument("--data-dir", default="data/processed", help="Path to preprocessed data directory")
    parser.add_argument("--models-dir", default="models", help="Directory to save model checkpoints")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cpu", help="Device (cpu)")
    
    args = parser.parse_args()
    train_centralized(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()

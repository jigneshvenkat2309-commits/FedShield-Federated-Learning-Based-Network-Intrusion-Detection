"""
FedShield: Genuine Federated Averaging (FedAvg) Simulation
Implements server-client federated training on client partitions weighted by local sample size:
w_{t+1} = sum_{k=1}^K (n_k / N) * w_{t+1}^{(k)}
"""

import os
import io
import time
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import create_model
from evaluation import evaluate_model, format_metrics_table


def fedavg_aggregate(client_states, client_sample_counts):
    """
    Computes the weighted parameter average of client model states:
      w_{t+1} = sum_{k=1}^K (n_k / N) * w_{t+1}^{(k)}
      
    Parameters:
      client_states: list of dicts (state_dict from each client)
      client_sample_counts: list of ints (sample counts n_k for each client)
      
    Returns:
      aggregated_state: dict of aggregated PyTorch parameter tensors
      weights: list of normalized client weights (sum to 1.0)
    """
    K = len(client_states)
    if K == 0:
        raise ValueError("Cannot aggregate an empty list of client states.")
    if len(client_sample_counts) != K:
        raise ValueError(f"Mismatch: {K} client states but {len(client_sample_counts)} sample counts.")
    if any(n <= 0 for n in client_sample_counts):
        raise ValueError(f"All sample counts must be strictly positive, got: {client_sample_counts}")
        
    total_samples = sum(client_sample_counts)
    weights = [n / total_samples for n in client_sample_counts]
    
    # Assert weights sum to 1.0 within numerical precision
    assert abs(sum(weights) - 1.0) < 1e-6, f"Aggregation weights do not sum to 1: {sum(weights)}"
    
    # Verify matching keys and shapes across all client models
    reference_keys = list(client_states[0].keys())
    for k in range(1, K):
        if list(client_states[k].keys()) != reference_keys:
            raise ValueError(f"Parameter keys mismatch between client 0 and client {k}.")
        for key in reference_keys:
            if client_states[k][key].shape != client_states[0][key].shape:
                raise ValueError(f"Tensor shape mismatch for '{key}' between client 0 and client {k}.")
                
    # Weighted aggregation
    aggregated_state = {}
    for key in reference_keys:
        tensors = [client_states[k][key] for k in range(K)]
        
        # Check if tensor is floating point
        if tensors[0].is_floating_point():
            weighted_tensors = [w * t.to(dtype=tensors[0].dtype) for w, t in zip(weights, tensors)]
            aggregated_state[key] = sum(weighted_tensors)
        else:
            # For non-floating parameters (e.g. integer counters), take from client 0
            aggregated_state[key] = copy.deepcopy(tensors[0])
            
    return aggregated_state, weights


def unit_test_fedavg_aggregate():
    """Unit test for fedavg_aggregate using synthetic model states."""
    # Synthetic 1-parameter model states
    state_A = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    state_B = {"weight": torch.tensor([2.0, 4.0], dtype=torch.float32)}
    state_C = {"weight": torch.tensor([5.0, 10.0], dtype=torch.float32)}
    
    # Sample counts: A=100, B=200, C=700 (Total = 1000)
    # Expected weights: 0.1, 0.2, 0.7
    counts = [100, 200, 700]
    agg_state, weights = fedavg_aggregate([state_A, state_B, state_C], counts)
    
    # Expected: 0.1*[1, 2] + 0.2*[2, 4] + 0.7*[5, 10] = [0.1+0.4+3.5, 0.2+0.8+7.0] = [4.0, 8.0]
    expected = torch.tensor([4.0, 8.0], dtype=torch.float32)
    unweighted = torch.tensor([(1.0 + 2.0 + 5.0) / 3.0, (2.0 + 4.0 + 10.0) / 3.0], dtype=torch.float32)
    
    assert torch.allclose(agg_state["weight"], expected), f"Unit test failed! Got {agg_state['weight']}, expected {expected}"
    assert not torch.allclose(agg_state["weight"], unweighted), "Unit test failed: output should differ from unweighted mean!"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights do not sum to 1.0!"
    print("  [PASS] fedavg_aggregate unit test passed: Weighted sum matches expected value (differs from unweighted mean).")


def measure_model_payload(model):
    """Measures raw parameter bytes and serialized state_dict bytes."""
    sd = model.state_dict()
    raw_bytes = sum(p.numel() * p.element_size() for p in sd.values())
    
    buf = io.BytesIO()
    torch.save(sd, buf)
    serialized_bytes = buf.tell()
    
    return {
        "raw_parameter_bytes": raw_bytes,
        "raw_parameter_kb": round(raw_bytes / 1024, 2),
        "serialized_bytes": serialized_bytes,
        "serialized_kb": round(serialized_bytes / 1024, 2)
    }


def train_client_local_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    """Executes local gradient descent for a client over its local DataLoader."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_x)
        total_samples += len(batch_x)
        
    return total_loss / total_samples if total_samples > 0 else 0.0


def run_fedavg_simulation(condition="iid", num_rounds=10, local_epochs=1,
                          batch_size=1024, lr=0.001, seed=42,
                          data_dir="data/partitions", processed_dir="data/processed",
                          models_dir="models/fedavg", results_dir="results",
                          device="cpu"):
    """
    Executes genuine FedAvg simulation across 3 simulated clients.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(os.path.join(models_dir, condition), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load shared held-out test set
    test_x = np.load(os.path.join(processed_dir, "test_features.npy"))
    test_y = np.load(os.path.join(processed_dir, "test_labels.npy"))
    num_test_samples = len(test_x)
    
    # 2. Load private client datasets
    client_ids = ["client_1", "client_2", "client_3"]
    client_loaders = {}
    client_counts = {}
    
    cond_dir = os.path.join(data_dir, condition)
    for c_id in client_ids:
        npz_path = os.path.join(cond_dir, f"{c_id}.npz")
        data = np.load(npz_path)
        X_c = torch.from_numpy(data["X"]).float()
        y_c = torch.from_numpy(data["y"]).float().unsqueeze(1)
        client_counts[c_id] = len(X_c)
        
        dataset = TensorDataset(X_c, y_c)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        client_loaders[c_id] = loader
        
    total_train_samples = sum(client_counts.values())
    sample_counts_list = [client_counts[c] for c in client_ids]
    
    # 3. Initialize Global Model
    global_model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2, seed=seed)
    global_model.to(device)
    
    # Measure payload size
    payload_info = measure_model_payload(global_model)
    model_bytes = payload_info["serialized_bytes"]
    raw_param_bytes = payload_info["raw_parameter_bytes"]
    
    print("\n" + "=" * 76)
    print(f" FedShield: FedAvg Simulation (Condition: {condition.upper()}, Seed: {seed})")
    print("=" * 76)
    print(f"Clients: {len(client_ids)} | Communication Rounds: {num_rounds} | Local Epochs: {local_epochs}")
    print(f"Batch Size: {batch_size} | Learning Rate: {lr} | Optimizer: Adam | Loss: BCEWithLogitsLoss")
    print(f"Client Sample Counts: {client_counts} (Total Training Records: {total_train_samples:,})")
    print(f"Model Size: {payload_info['serialized_kb']} KB serialized ({payload_info['raw_parameter_kb']} KB raw parameters)")
    
    round_history = []
    total_start_time = time.time()
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Federated Communication Rounds
    for r in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = global_model.state_dict()
        
        client_updated_states = []
        client_round_losses = {}
        
        # Local client execution
        for c_id in client_ids:
            # Step 1 & 2: Client receives broadcast global weights
            client_model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2)
            client_model.load_state_dict(copy.deepcopy(global_state))
            client_model.to(device)
            
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            
            # Step 3: Train local model for E local epochs on client partition
            local_loss = 0.0
            for ep in range(local_epochs):
                ep_loss = train_client_local_epoch(
                    client_model, client_loaders[c_id], criterion, optimizer, device=device
                )
                local_loss += ep_loss
            local_loss /= local_epochs
            
            client_round_losses[c_id] = float(local_loss)
            
            # Step 4: Client returns updated local state
            client_updated_states.append(client_model.state_dict())
            
        # Step 5: Server aggregates client weights proportional to local sample counts
        aggregated_state, norm_weights = fedavg_aggregate(client_updated_states, sample_counts_list)
        global_model.load_state_dict(aggregated_state)
        
        round_duration = time.time() - round_start
        
        # Diagnostic evaluation on test set (for trajectory monitoring only)
        diag_metrics = evaluate_model(global_model, test_x, test_y, device=device, threshold=0.5)
        
        round_info = {
            "round": r,
            "client_losses": client_round_losses,
            "aggregation_weights": {c_id: round(w, 5) for c_id, w in zip(client_ids, norm_weights)},
            "round_duration_sec": round(round_duration, 3),
            "test_accuracy": round(diag_metrics["accuracy"] * 100, 2),
            "test_recall": round(diag_metrics["recall"] * 100, 2),
            "test_precision": round(diag_metrics["precision"] * 100, 2),
            "test_macro_f1": round(diag_metrics["f1_macro"] * 100, 2),
            "test_fpr": round(diag_metrics["false_positive_rate"] * 100, 2),
            "test_loss": round(diag_metrics["test_loss"], 4)
        }
        round_history.append(round_info)
        
        avg_client_loss = np.mean(list(client_round_losses.values()))
        print(f"  Round {r:2d}/{num_rounds:2d} ({round_duration:4.2f}s) | Avg Client Loss: {avg_client_loss:.5f} | Test Acc: {round_info['test_accuracy']}% | Rec: {round_info['test_recall']}% | F1: {round_info['test_macro_f1']}% | FPR: {round_info['test_fpr']}%")
        
    total_training_time = time.time() - total_start_time
    print(f"\nFedAvg completed {num_rounds} rounds in {total_training_time:.2f} seconds.")
    
    # 5. Final Evaluation on Official Held-Out Test Set
    final_metrics = evaluate_model(global_model, test_x, test_y, device=device, threshold=0.5)
    
    # 6. Communication Accounting
    # Downlink: 3 clients receive global model per round
    # Uplink: 3 clients send updated model per round
    downlink_bytes_per_round = len(client_ids) * model_bytes
    uplink_bytes_per_round = len(client_ids) * model_bytes
    total_round_bytes = downlink_bytes_per_round + uplink_bytes_per_round
    cumulative_bytes = total_round_bytes * num_rounds
    
    raw_downlink_round = len(client_ids) * raw_param_bytes
    raw_uplink_round = len(client_ids) * raw_param_bytes
    raw_cumulative_bytes = (raw_downlink_round + raw_uplink_round) * num_rounds
    
    comm_stats = {
        "num_clients": len(client_ids),
        "num_rounds": num_rounds,
        "serialized_model_bytes": model_bytes,
        "raw_parameter_bytes": raw_param_bytes,
        "downlink_bytes_per_round": downlink_bytes_per_round,
        "uplink_bytes_per_round": uplink_bytes_per_round,
        "total_bytes_per_round": total_round_bytes,
        "cumulative_bytes": cumulative_bytes,
        "cumulative_mb": round(cumulative_bytes / (1024 * 1024), 3),
        "raw_cumulative_bytes": raw_cumulative_bytes,
        "raw_cumulative_mb": round(raw_cumulative_bytes / (1024 * 1024), 3),
        "description": "Simulated parameter payload transfer; does not include TCP/IP headers, serialization protocol overhead, or TLS."
    }
    
    # Compile metadata
    final_metrics["model_name"] = f"FedAvg Global ({condition.upper()})"
    final_metrics["condition"] = condition
    final_metrics["client_id"] = "global"
    final_metrics["seed"] = seed
    final_metrics["num_rounds"] = num_rounds
    final_metrics["local_epochs_per_round"] = local_epochs
    final_metrics["total_local_epochs_per_client"] = num_rounds * local_epochs
    final_metrics["batch_size"] = batch_size
    final_metrics["learning_rate"] = lr
    final_metrics["total_train_samples"] = total_train_samples
    final_metrics["client_sample_counts"] = client_counts
    final_metrics["aggregation_weights"] = {c_id: round(w, 5) for c_id, w in zip(client_ids, norm_weights)}
    final_metrics["total_training_time_sec"] = round(total_training_time, 3)
    final_metrics["total_sample_passes"] = total_train_samples * local_epochs * num_rounds
    final_metrics["communication"] = comm_stats
    
    print(format_metrics_table(final_metrics, title=f"FedAvg {condition.upper()} Final Global Model Evaluation"))
    
    # 7. Save Checkpoint
    checkpoint_path = os.path.join(models_dir, condition, "global_final.pt")
    torch.save({
        "state_dict": global_model.state_dict(),
        "config": {
            "condition": condition,
            "seed": seed,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "client_counts": client_counts
        },
        "metrics": final_metrics,
        "round_history": round_history,
        "communication": comm_stats
    }, checkpoint_path)
    print(f"Global model saved to: {checkpoint_path}")
    
    return final_metrics, round_history, comm_stats


def run_both_experiments(data_dir="data/partitions", processed_dir="data/processed",
                         models_dir="models/fedavg", results_dir="results",
                         num_rounds=10, local_epochs=1, batch_size=1024, lr=0.001):
    """Executes both IID and Non-IID FedAvg experiments and saves consolidated artifacts."""
    print("=" * 76)
    print(" Running FedAvg Unit Test")
    print("=" * 76)
    unit_test_fedavg_aggregate()
    
    all_fed_metrics = []
    all_round_histories = {}
    all_comm_stats = {}
    
    # 1. Run IID FedAvg (seed 42)
    iid_metrics, iid_history, iid_comm = run_fedavg_simulation(
        condition="iid", num_rounds=num_rounds, local_epochs=local_epochs,
        batch_size=batch_size, lr=lr, seed=42,
        data_dir=data_dir, processed_dir=processed_dir,
        models_dir=models_dir, results_dir=results_dir
    )
    all_fed_metrics.append(iid_metrics)
    all_round_histories["iid"] = iid_history
    all_comm_stats["iid"] = iid_comm
    
    # 2. Run Non-IID FedAvg (seed 84 - distinct documented seed)
    non_iid_metrics, non_iid_history, non_iid_comm = run_fedavg_simulation(
        condition="non_iid", num_rounds=num_rounds, local_epochs=local_epochs,
        batch_size=batch_size, lr=lr, seed=84,
        data_dir=data_dir, processed_dir=processed_dir,
        models_dir=models_dir, results_dir=results_dir
    )
    all_fed_metrics.append(non_iid_metrics)
    all_round_histories["non_iid"] = non_iid_history
    all_comm_stats["non_iid"] = non_iid_comm
    
    # 3. Save JSON artifacts
    with open(os.path.join(results_dir, "fedavg_metrics.json"), "w") as f:
        json.dump(all_fed_metrics, f, indent=2)
    with open(os.path.join(results_dir, "fedavg_round_history.json"), "w") as f:
        json.dump(all_round_histories, f, indent=2)
    with open(os.path.join(results_dir, "fedavg_communication.json"), "w") as f:
        json.dump(all_comm_stats, f, indent=2)
        
    # 4. Save CSV artifact
    csv_rows = []
    for m in all_fed_metrics:
        cm = m["confusion_matrix"]
        comm = m["communication"]
        csv_rows.append({
            "Model": m["model_name"],
            "Condition": m["condition"],
            "Rounds": m["num_rounds"],
            "Local Epochs": m["local_epochs_per_round"],
            "Total Samples": m["total_train_samples"],
            "Training Time (s)": m["total_training_time_sec"],
            "Comm Payload (MB)": comm["cumulative_mb"],
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
    import pandas as pd
    df_fed = pd.DataFrame(csv_rows)
    df_fed.to_csv(os.path.join(results_dir, "fedavg_metrics.csv"), index=False)
    
    print("\nSaved FedAvg artifacts to results/fedavg_metrics.csv, results/fedavg_metrics.json, results/fedavg_round_history.json, results/fedavg_communication.json")
    return all_fed_metrics


def main():
    parser = argparse.ArgumentParser(description="FedShield Federated Averaging (FedAvg)")
    parser.add_argument("--condition", choices=["iid", "non_iid", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    if args.condition == "both":
        run_both_experiments(num_rounds=args.rounds, local_epochs=args.local_epochs,
                             batch_size=args.batch_size, lr=args.lr)
    else:
        unit_test_fedavg_aggregate()
        run_fedavg_simulation(condition=args.condition, num_rounds=args.rounds,
                              local_epochs=args.local_epochs, batch_size=args.batch_size,
                              lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()

"""
FedShield Interactive Dashboard Backend
Provides REST & Server-Sent Events (SSE) APIs for visual simulation,
real-time metrics streaming, and live PyTorch hardware execution.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, send_from_directory, jsonify, Response, request

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import create_model
from evaluation import evaluate_model
from federated import fedavg_aggregate, train_client_local_epoch

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def get_status():
    has_results = os.path.exists(os.path.join(RESULTS_DIR, "final_comparison.json"))
    has_partitions = os.path.exists(os.path.join(DATA_DIR, "partitions", "partition_summary.json"))
    return jsonify({
        "status": "ready",
        "dataset": "NSL-KDD",
        "clients": 3,
        "features": 122,
        "has_results": has_results,
        "has_partitions": has_partitions,
        "supported_conditions": ["iid", "non_iid"],
        "supported_modes": ["visual", "live"]
    })


@app.route("/api/results/final")
def get_final_results():
    data = load_json(os.path.join(RESULTS_DIR, "final_comparison.json"))
    return jsonify(data)


@app.route("/api/results/confusion_matrices")
def get_confusion_matrices():
    data = load_json(os.path.join(RESULTS_DIR, "confusion_matrices.json"))
    return jsonify(data)


@app.route("/api/results/round_history")
def get_round_history():
    data = load_json(os.path.join(RESULTS_DIR, "fedavg_round_history.json"))
    return jsonify(data)


@app.route("/api/results/communication")
def get_communication():
    data = load_json(os.path.join(RESULTS_DIR, "fedavg_communication.json"))
    return jsonify(data)


@app.route("/api/data/partitions")
def get_partitions():
    data = load_json(os.path.join(DATA_DIR, "partitions", "partition_summary.json"))
    return jsonify(data)


@app.route("/api/figures/<filename>")
def get_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


def stream_visual_replay(condition, speed):
    """Streams the audited 10 rounds with 60 FPS animation timing."""
    step_delay = max(0.1, 0.45 / speed)
    round_history_all = load_json(os.path.join(RESULTS_DIR, "fedavg_round_history.json"))
    rounds_data = round_history_all.get(condition, [])

    init_payload = {
        "event": "init",
        "condition": condition,
        "mode": "visual",
        "total_rounds": len(rounds_data),
        "clients": [
            {"id": "client_1", "name": "Client 1 (Subnet)", "samples": 41992 if condition == "iid" else 47008,
             "role": "Balanced Traffic" if condition == "iid" else "Enterprise Workstations (92.3% Benign)"},
            {"id": "client_2", "name": "Client 2 (DMZ)", "samples": 41991 if condition == "iid" else 51654,
             "role": "Balanced Traffic" if condition == "iid" else "Honeypot / DMZ Server (90.5% Malicious)"},
            {"id": "client_3", "name": "Client 3 (Gateway)", "samples": 41990 if condition == "iid" else 27311,
             "role": "Balanced Traffic" if condition == "iid" else "Mixed Gateway (69.8% Benign)"}
        ]
    }
    yield f"data: {json.dumps(init_payload)}\n\n"
    time.sleep(step_delay)

    cumulative_bytes = 0
    total_model_bytes_round = 256638

    for r_info in rounds_data:
        round_num = r_info["round"]

        # Phase 1: Downlink Broadcast
        yield f"data: {json.dumps({'event': 'phase_downlink', 'round': round_num, 'transfer_kb': 125.31, 'mode': 'visual'})}\n\n"
        time.sleep(step_delay)

        # Phase 2: Local Client Training
        yield f"data: {json.dumps({'event': 'phase_training', 'round': round_num, 'client_losses': r_info['client_losses'], 'mode': 'visual'})}\n\n"
        time.sleep(step_delay * 1.3)

        # Phase 3: Uplink Upload
        yield f"data: {json.dumps({'event': 'phase_uplink', 'round': round_num, 'transfer_kb': 125.31, 'mode': 'visual'})}\n\n"
        time.sleep(step_delay)

        # Phase 4: Server FedAvg Aggregation
        weights = r_info["aggregation_weights"]
        cumulative_bytes += total_model_bytes_round
        yield f"data: {json.dumps({'event': 'phase_aggregation', 'round': round_num, 'aggregation_weights': weights, 'mode': 'visual'})}\n\n"
        time.sleep(step_delay)

        # Phase 5: Evaluation
        eval_payload = {
            "event": "round_metrics",
            "round": round_num,
            "test_accuracy": r_info.get("test_accuracy"),
            "test_recall": r_info.get("test_recall"),
            "test_precision": r_info.get("test_precision"),
            "test_macro_f1": r_info.get("test_macro_f1"),
            "test_fpr": r_info.get("test_fpr"),
            "test_loss": r_info.get("test_loss"),
            "client_losses": r_info["client_losses"],
            "round_duration_sec": r_info.get("round_duration_sec", 1.8),
            "cumulative_mb": round(cumulative_bytes / 1000000.0, 3),
            "mode": "visual"
        }
        yield f"data: {json.dumps(eval_payload)}\n\n"
        time.sleep(step_delay)

    final_payload = {
        "event": "simulation_complete",
        "condition": condition,
        "mode": "visual",
        "final_metrics": rounds_data[-1] if rounds_data else {},
        "cumulative_mb": round(cumulative_bytes / 1000000.0, 3)
    }
    yield f"data: {json.dumps(final_payload)}\n\n"


def stream_live_pytorch_training(condition, num_rounds=10, batch_size=1024, lr=0.001):
    """
    Executes REAL PyTorch training on CPU live in the background,
    streaming real-time gradient descent losses and held-out test evaluations.
    """
    seed = 42 if condition == "iid" else 84
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load test data
    test_x = np.load(os.path.join(DATA_DIR, "processed", "test_features.npy"))
    test_y = np.load(os.path.join(DATA_DIR, "processed", "test_labels.npy"))

    # 2. Load client partitions
    client_ids = ["client_1", "client_2", "client_3"]
    client_loaders = {}
    client_counts = {}
    cond_dir = os.path.join(DATA_DIR, "partitions", condition)

    for c_id in client_ids:
        data = np.load(os.path.join(cond_dir, f"{c_id}.npz"))
        X_c = torch.from_numpy(data["X"]).float()
        y_c = torch.from_numpy(data["y"]).float().unsqueeze(1)
        client_counts[c_id] = len(X_c)

        ds = TensorDataset(X_c, y_c)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        client_loaders[c_id] = loader

    sample_counts_list = [client_counts[c] for c in client_ids]

    # 3. Create fresh PyTorch model
    global_model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2, seed=seed)
    criterion = nn.BCEWithLogitsLoss()

    init_payload = {
        "event": "init",
        "condition": condition,
        "mode": "live",
        "total_rounds": num_rounds,
        "clients": [
            {"id": "client_1", "name": "Client 1 (Subnet)", "samples": client_counts["client_1"], "role": "Private Partition"},
            {"id": "client_2", "name": "Client 2 (DMZ)", "samples": client_counts["client_2"], "role": "Private Partition"},
            {"id": "client_3", "name": "Client 3 (Gateway)", "samples": client_counts["client_3"], "role": "Private Partition"}
        ]
    }
    yield f"data: {json.dumps(init_payload)}\n\n"

    cumulative_bytes = 0
    total_model_bytes_round = 256638

    for r in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = global_model.state_dict()

        # Phase 1: Broadcast
        yield f"data: {json.dumps({'event': 'phase_downlink', 'round': r, 'transfer_kb': 125.31, 'mode': 'live'})}\n\n"

        # Phase 2: Live Local SGD Execution on CPU
        client_updated_states = []
        client_losses = {}
        for c_id in client_ids:
            c_model = create_model(input_dim=122, hidden_dims=(64, 32), dropout_rate=0.2)
            c_model.load_state_dict(copy.deepcopy(global_state))
            opt = torch.optim.Adam(c_model.parameters(), lr=lr)

            ep_loss = train_client_local_epoch(c_model, client_loaders[c_id], criterion, opt, device="cpu")
            client_losses[c_id] = float(ep_loss)
            client_updated_states.append(c_model.state_dict())

        yield f"data: {json.dumps({'event': 'phase_training', 'round': r, 'client_losses': client_losses, 'mode': 'live'})}\n\n"

        # Phase 3: Uplink
        yield f"data: {json.dumps({'event': 'phase_uplink', 'round': r, 'transfer_kb': 125.31, 'mode': 'live'})}\n\n"

        # Phase 4: FedAvg Aggregation
        agg_state, norm_weights = fedavg_aggregate(client_updated_states, sample_counts_list)
        global_model.load_state_dict(agg_state)
        cumulative_bytes += total_model_bytes_round

        weights_dict = {c_id: round(w, 5) for c_id, w in zip(client_ids, norm_weights)}
        yield f"data: {json.dumps({'event': 'phase_aggregation', 'round': r, 'aggregation_weights': weights_dict, 'mode': 'live'})}\n\n"

        # Phase 5: Live Test Evaluation on 22,544 held-out flows
        eval_metrics = evaluate_model(global_model, test_x, test_y, device="cpu", threshold=0.5)
        duration = time.time() - round_start

        payload = {
            "event": "round_metrics",
            "round": r,
            "test_accuracy": round(eval_metrics["accuracy"] * 100, 2),
            "test_recall": round(eval_metrics["recall"] * 100, 2),
            "test_precision": round(eval_metrics["precision"] * 100, 2),
            "test_macro_f1": round(eval_metrics["f1_macro"] * 100, 2),
            "test_fpr": round(eval_metrics["false_positive_rate"] * 100, 2),
            "test_loss": round(eval_metrics["test_loss"], 4),
            "client_losses": client_losses,
            "round_duration_sec": round(duration, 2),
            "cumulative_mb": round(cumulative_bytes / 1000000.0, 3),
            "mode": "live"
        }
        yield f"data: {json.dumps(payload)}\n\n"

    final_payload = {
        "event": "simulation_complete",
        "condition": condition,
        "mode": "live",
        "cumulative_mb": round(cumulative_bytes / 1000000.0, 3)
    }
    yield f"data: {json.dumps(final_payload)}\n\n"


@app.route("/api/simulation/stream")
def stream_simulation():
    """
    Server-Sent Events (SSE) endpoint to stream round-by-round simulation steps.
    Supports mode='visual' (fast animated replay) and mode='live' (real PyTorch execution on CPU).
    """
    condition = request.args.get("condition", "iid").lower()
    if condition not in ["iid", "non_iid"]:
        condition = "iid"

    mode = request.args.get("mode", "visual").lower()
    speed = float(request.args.get("speed", 1.0))

    if mode == "live":
        return Response(stream_live_pytorch_training(condition), mimetype="text/event-stream")
    else:
        return Response(stream_visual_replay(condition, speed), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" FedShield Interactive Visual Dashboard Server")
    print(f" Web UI: http://localhost:{port}")
    print(f" API:    http://localhost:{port}/api/status")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

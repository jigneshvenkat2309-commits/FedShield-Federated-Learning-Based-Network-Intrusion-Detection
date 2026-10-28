"""
FedShield Interactive Dashboard Backend
Provides REST & Server-Sent Events (SSE) APIs for visual simulation,
real-time metrics streaming, and experimental benchmark visualization.
"""

import os
import sys
import time
import json
from flask import Flask, send_from_directory, jsonify, Response, request

# Add project root to path so we can import project modules if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        "supported_conditions": ["iid", "non_iid"]
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


@app.route("/api/simulation/stream")
def stream_simulation():
    """
    Server-Sent Events (SSE) endpoint to stream round-by-round simulation steps.
    Supports 'iid' and 'non_iid' conditions.
    """
    condition = request.args.get("condition", "iid").lower()
    if condition not in ["iid", "non_iid"]:
        condition = "iid"

    # Delay multiplier: 1.0 = normal (approx 1s/round), 0.5 = fast, 2.0 = slow
    speed = float(request.args.get("speed", 1.0))
    step_delay = max(0.1, 0.45 / speed)

    round_history_all = load_json(os.path.join(RESULTS_DIR, "fedavg_round_history.json"))
    rounds_data = round_history_all.get(condition, [])

    final_comparison = load_json(os.path.join(RESULTS_DIR, "final_comparison.json"))
    cm_data = load_json(os.path.join(RESULTS_DIR, "confusion_matrices.json"))
    comm_data = load_json(os.path.join(RESULTS_DIR, "fedavg_communication.json"))

    def event_stream():
        # Initialization event
        init_payload = {
            "event": "init",
            "condition": condition,
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
        total_model_bytes_round = 256638  # 125.31 KB down + 125.31 KB up

        for r_info in rounds_data:
            round_num = r_info["round"]

            # Phase 1: Downlink Broadcast
            yield f"data: {json.dumps({'event': 'phase_downlink', 'round': round_num, 'description': 'Server broadcasting global model weights w_t to 3 clients...', 'transfer_kb': 125.31})}\n\n"
            time.sleep(step_delay)

            # Phase 2: Local Client Training
            yield f"data: {json.dumps({'event': 'phase_training', 'round': round_num, 'description': 'Clients executing local SGD (1 epoch, batch size 1024)...', 'client_losses': r_info['client_losses']})}\n\n"
            time.sleep(step_delay * 1.3)

            # Phase 3: Uplink Weight Upload
            yield f"data: {json.dumps({'event': 'phase_uplink', 'round': round_num, 'description': 'Clients uploading updated model state dictionaries w_{t+1}^{(k)} to server...', 'transfer_kb': 125.31})}\n\n"
            time.sleep(step_delay)

            # Phase 4: Server FedAvg Aggregation
            weights = r_info["aggregation_weights"]
            cumulative_bytes += total_model_bytes_round
            yield f"data: {json.dumps({'event': 'phase_aggregation', 'round': round_num, 'description': 'Server performing sample-weighted parameter aggregation...', 'aggregation_weights': weights})}\n\n"
            time.sleep(step_delay)

            # Phase 5: Evaluation & Metrics Update
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
                "cumulative_mb": round(cumulative_bytes / 1000000.0, 3)
            }
            yield f"data: {json.dumps(eval_payload)}\n\n"
            time.sleep(step_delay)

        # Simulation Final Complete Event
        target_cm_key = "iid_client_1" if condition == "iid" else "non_iid_client_1"
        final_payload = {
            "event": "simulation_complete",
            "condition": condition,
            "final_metrics": rounds_data[-1] if rounds_data else {},
            "cumulative_mb": round(cumulative_bytes / 1000000.0, 3),
            "summary_message": f"FedAvg ({condition.upper()}) completed 10 communication rounds across 3 clients."
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" FedShield Interactive Visual Dashboard Server")
    print(f" Web UI: http://localhost:{port}")
    print(f" API:    http://localhost:{port}/api/status")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

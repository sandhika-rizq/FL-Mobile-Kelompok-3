"""
FL Server — Federated Learning untuk Mobile Activity Recognition
================================================================
Jalankan di VM Server:
    python server/app.py

Environment Variables:
    FL_NUM_CLIENTS  : jumlah client yang harus bergabung (default: 4)
    FL_NUM_ROUNDS   : jumlah round training (default: 10)
    FL_SERVER_PORT  : port Flask (default: 5000)
    FL_LOCAL_EPOCHS : hint ke client untuk epoch lokal (default: 5)
"""

import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

# ─────────────────────────── Konfigurasi ───────────────────────────
NUM_CLIENTS   = int(os.environ.get("FL_NUM_CLIENTS",  4))
NUM_ROUNDS    = int(os.environ.get("FL_NUM_ROUNDS",  10))
SERVER_PORT   = int(os.environ.get("FL_SERVER_PORT", 5000))
LOCAL_EPOCHS  = int(os.environ.get("FL_LOCAL_EPOCHS", 5))

N_FEATURES = 561   # UCI HAR: 561 fitur
N_CLASSES  = 6     # 6 kelas aktivitas

# ─────────────────────────── State Global ──────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
logger = logging.getLogger(__name__)

lock = threading.Lock()

state = {
    "round"          : 1,
    "phase"          : "waiting",   # waiting | aggregating | done
    "client_updates" : {},          # {client_id: {coef, intercept, n_samples}}
    "global_coef"    : None,        # shape (N_CLASSES, N_FEATURES)
    "global_intercept": None,       # shape (N_CLASSES,)
    "history"        : [],          # [{round, accuracy, loss, timestamp}]
    "registered_clients": set(),
    "start_time"     : None,
}

# ─────────────────────────── Inisialisasi Model ────────────────────
def init_global_model():
    """Inisialisasi parameter model global secara random (simetris kecil)."""
    rng = np.random.default_rng(seed=42)
    state["global_coef"]      = rng.normal(0, 0.01, (N_CLASSES, N_FEATURES))
    state["global_intercept"] = np.zeros(N_CLASSES)
    logger.info("Global model diinisialisasi (coef shape: %s)", state["global_coef"].shape)


# ─────────────────────────── FedAvg ────────────────────────────────
def fedavg_aggregate(client_updates: dict):
    """
    Federated Averaging (FedAvg) — McMahan et al. 2017.

    Formula:
        θ_global = Σ_k ( n_k / N ) * θ_k

    Parameters
    ----------
    client_updates : dict
        { client_id: {"coef": list, "intercept": list, "n_samples": int} }

    Returns
    -------
    avg_coef : np.ndarray  shape (N_CLASSES, N_FEATURES)
    avg_intercept : np.ndarray  shape (N_CLASSES,)
    """
    total_samples = sum(u["n_samples"] for u in client_updates.values())
    avg_coef      = np.zeros((N_CLASSES, N_FEATURES))
    avg_intercept = np.zeros(N_CLASSES)

    for client_id, update in client_updates.items():
        weight        = update["n_samples"] / total_samples
        client_coef   = np.array(update["coef"])
        client_inter  = np.array(update["intercept"])

        avg_coef      += weight * client_coef
        avg_intercept += weight * client_inter

        logger.info(
            "  Client %s: n_samples=%d, weight=%.4f, local_acc=%.4f",
            client_id, update["n_samples"], weight,
            update.get("local_accuracy", -1)
        )

    return avg_coef, avg_intercept


def run_aggregation():
    """Jalankan FedAvg lalu pindah ke round berikutnya."""
    with lock:
        current_round  = state["round"]
        client_updates = state["client_updates"].copy()

    logger.info("=== Memulai Agregasi Round %d ===", current_round)

    avg_coef, avg_intercept = fedavg_aggregate(client_updates)

    # Hitung rata-rata akurasi lokal sebagai proxy
    avg_local_acc = np.mean([
        u.get("local_accuracy", 0) for u in client_updates.values()
    ])
    avg_local_loss = np.mean([
        u.get("local_loss", 0) for u in client_updates.values()
    ])

    record = {
        "round"          : current_round,
        "avg_local_acc"  : float(avg_local_acc),
        "avg_local_loss" : float(avg_local_loss),
        "n_clients"      : len(client_updates),
        "total_samples"  : sum(u["n_samples"] for u in client_updates.values()),
        "timestamp"      : datetime.now().isoformat(),
    }

    with lock:
        state["global_coef"]      = avg_coef
        state["global_intercept"] = avg_intercept
        state["history"].append(record)
        state["client_updates"]   = {}

        if current_round >= NUM_ROUNDS:
            state["phase"] = "done"
            logger.info("=== Training Selesai! %d rounds ===", NUM_ROUNDS)
        else:
            state["round"] += 1
            state["phase"]  = "waiting"
            logger.info("=== Round %d selesai → Round %d dimulai ===",
                        current_round, state["round"])

    logger.info("  Avg local acc round %d: %.4f", current_round, avg_local_acc)


# ─────────────────────────── REST API ──────────────────────────────

@app.route("/api/register", methods=["POST"])
def register_client():
    """Client mendaftarkan diri ke server."""
    data      = request.get_json(force=True)
    client_id = str(data.get("client_id", ""))

    if not client_id:
        return jsonify({"error": "client_id diperlukan"}), 400

    with lock:
        state["registered_clients"].add(client_id)
        if state["start_time"] is None:
            state["start_time"] = datetime.now().isoformat()
        total = len(state["registered_clients"])

    logger.info("Client %s terdaftar. Total: %d/%d", client_id, total, NUM_CLIENTS)
    return jsonify({
        "message"      : f"Client {client_id} berhasil terdaftar",
        "num_rounds"   : NUM_ROUNDS,
        "local_epochs" : LOCAL_EPOCHS,
        "num_clients"  : NUM_CLIENTS,
    })


@app.route("/api/model", methods=["GET"])
def get_model():
    """Client mengunduh parameter model global saat ini."""
    with lock:
        if state["global_coef"] is None:
            return jsonify({"error": "Model belum diinisialisasi"}), 503

        return jsonify({
            "round"     : state["round"],
            "phase"     : state["phase"],
            "coef"      : state["global_coef"].tolist(),
            "intercept" : state["global_intercept"].tolist(),
        })


@app.route("/api/update", methods=["POST"])
def receive_update():
    """
    Client mengirimkan pembaruan model lokal.

    Body JSON:
    {
        "client_id"      : "1",
        "round"          : 1,
        "coef"           : [[...], ...],   // shape (6, 561)
        "intercept"      : [...],          // shape (6,)
        "n_samples"      : 1234,
        "local_accuracy" : 0.92,
        "local_loss"     : 0.21
    }
    """
    data = request.get_json(force=True)

    required = ["client_id", "round", "coef", "intercept", "n_samples"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Field '{field}' diperlukan"}), 400

    client_id    = str(data["client_id"])
    client_round = int(data["round"])

    with lock:
        server_round = state["round"]
        phase        = state["phase"]

    if client_round != server_round:
        return jsonify({
            "error"        : f"Round tidak cocok. Server round: {server_round}",
            "server_round" : server_round,
        }), 409

    if phase == "done":
        return jsonify({"message": "Training sudah selesai"}), 200

    with lock:
        state["client_updates"][client_id] = {
            "coef"           : data["coef"],
            "intercept"      : data["intercept"],
            "n_samples"      : int(data["n_samples"]),
            "local_accuracy" : float(data.get("local_accuracy", 0)),
            "local_loss"     : float(data.get("local_loss", 0)),
        }
        received = len(state["client_updates"])
        state["phase"] = "aggregating" if received >= NUM_CLIENTS else "waiting"

    logger.info(
        "Round %d: menerima update dari client %s (%d/%d)",
        client_round, client_id, received, NUM_CLIENTS
    )

    # Trigger agregasi jika semua client sudah mengirim
    if received >= NUM_CLIENTS:
        agg_thread = threading.Thread(target=run_aggregation, daemon=True)
        agg_thread.start()

    return jsonify({
        "message"  : "Update diterima",
        "received" : received,
        "required" : NUM_CLIENTS,
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """Cek status training saat ini."""
    with lock:
        return jsonify({
            "round"              : state["round"],
            "total_rounds"       : NUM_ROUNDS,
            "phase"              : state["phase"],
            "registered_clients" : list(state["registered_clients"]),
            "updates_received"   : list(state["client_updates"].keys()),
            "updates_needed"     : NUM_CLIENTS,
            "start_time"         : state["start_time"],
            "rounds_completed"   : len(state["history"]),
        })


@app.route("/api/results", methods=["GET"])
def get_results():
    """Lihat riwayat training semua round."""
    with lock:
        return jsonify({
            "status"         : state["phase"],
            "rounds_done"    : len(state["history"]),
            "total_rounds"   : NUM_ROUNDS,
            "history"        : state["history"],
            "final_model_available": state["global_coef"] is not None,
        })


@app.route("/api/final_model", methods=["GET"])
def get_final_model():
    """Unduh model final setelah semua round selesai."""
    with lock:
        if state["phase"] != "done":
            return jsonify({
                "error": f"Training belum selesai. Status: {state['phase']}"
            }), 425  # Too Early

        return jsonify({
            "coef"      : state["global_coef"].tolist(),
            "intercept" : state["global_intercept"].tolist(),
            "history"   : state["history"],
        })


@app.route("/", methods=["GET"])
def index():
    with lock:
        return jsonify({
            "service"  : "Federated Learning Server — UCI HAR",
            "version"  : "1.0",
            "phase"    : state["phase"],
            "round"    : f"{state['round']}/{NUM_ROUNDS}",
            "endpoints": [
                "POST /api/register",
                "GET  /api/model",
                "POST /api/update",
                "GET  /api/status",
                "GET  /api/results",
                "GET  /api/final_model",
            ],
        })


# ─────────────────────────── Main ──────────────────────────────────
if __name__ == "__main__":
    init_global_model()
    logger.info("=" * 55)
    logger.info("  Federated Learning Server — UCI HAR")
    logger.info("  Menunggu %d client, %d round training", NUM_CLIENTS, NUM_ROUNDS)
    logger.info("  URL: http://0.0.0.0:%d", SERVER_PORT)
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)

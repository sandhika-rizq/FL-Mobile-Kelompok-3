"""
Challenge 2 — Analisis IID vs Non-IID
========================================
Eksperimen perbandingan performa Federated Learning
pada distribusi data IID vs Non-IID.

Jalankan:
    python challenges/challenge_2/solution.py

Requirements tambahan:
    pip install matplotlib
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
sys.path.insert(0, "../..")

# Import helper dari proyek utama
try:
    from data.partition_data import (
        load_uci_har, partition_iid, partition_noniid_dirichlet, ACTIVITY_LABELS
    )
except ImportError:
    from partition_data import load_uci_har, partition_iid, partition_noniid_dirichlet, ACTIVITY_LABELS

N_CLIENTS  = 4
N_ROUNDS   = 15
N_FEATURES = 561
N_CLASSES  = 6
LOCAL_ITER = 300   # iterasi Logistic Regression per round


# ═══════════════════════════════════════════════════════════════════
# BAGIAN A — Analisis Distribusi
# ═══════════════════════════════════════════════════════════════════

def analyze_distribution(partitions: list, title: str = "Distribusi Label"):
    """
    Analisis dan visualisasi distribusi label pada setiap partisi client.

    TODO: Lengkapi fungsi ini untuk:
    1. Menghitung persentase setiap label per client
    2. Membuat bar chart stacked (setiap warna = satu aktivitas)
    3. Print ringkasan distribusi ke terminal

    Parameters
    ----------
    partitions : list of (X, y) tuple untuk setiap client
    title      : judul grafik
    """
    activity_names = list(ACTIVITY_LABELS.values())
    classes        = sorted(ACTIVITY_LABELS.keys())
    n_clients      = len(partitions)



    distribution = np.zeros((n_clients, len(classes)))

    for i, (_, y) in enumerate(partitions):
        total = len(y)

        for j, cls in enumerate(classes):
            distribution[i][j] = (y == cls).sum() / total

    # --- Print distribusi ---
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    header = f"{'Aktivitas':<25}" + "".join(
        f"{'C'+str(i+1):>8}" for i in range(n_clients)
    )

    print(header)
    print("-" * 60)

    for j, name in enumerate(activity_names):
        row = f"{name:<25}" + "".join(
            f"{distribution[i][j]*100:>7.1f}%"
            for i in range(n_clients)
        )
        print(row)

    print("-" * 60)

    total_row = f"{'N Samples':<25}" + "".join(
        f"{len(p[1]):>8}" for p in partitions
    )

    print(total_row)

    # --- Plot stacked bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(n_clients)

    colors = plt.cm.tab20.colors

    for j, activity in enumerate(activity_names):

        values = distribution[:, j]

        ax.bar(
            range(n_clients),
            values,
            bottom=bottom,
            label=activity,
            color=colors[j]
        )

        bottom += values

    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f"Client {i+1}" for i in range(n_clients)])

    ax.set_ylabel("Proporsi")
    ax.set_title(title)

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"

    plt.savefig(filename)

    plt.show()


# ═══════════════════════════════════════════════════════════════════
# BAGIAN B — Simulasi FL Lokal
# ═══════════════════════════════════════════════════════════════════

def local_train_round(model: LogisticRegression,
                      X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Training lokal satu round."""
    model.fit(X_train, y_train)
    return model


def fedavg_simple(models: list, n_samples_list: list) -> tuple:
    """FedAvg sederhana — gunakan hasil Challenge 1 jika sudah selesai."""
    total = sum(n_samples_list)
    global_coef      = np.zeros((N_CLASSES, N_FEATURES))
    global_intercept = np.zeros(N_CLASSES)

    for model, n in zip(models, n_samples_list):
        w = n / total
        global_coef      += w * model.coef_
        global_intercept += w * model.intercept_

    return global_coef, global_intercept


def simulate_federated_learning(
    partitions: list,
    X_test: np.ndarray, y_test: np.ndarray,
    n_rounds: int = N_ROUNDS,
    label: str = "FL"
) -> list:

    accuracy_history = []

    # --- Inisialisasi global model ---
    global_model = LogisticRegression(
        max_iter=LOCAL_ITER,
        warm_start=True,
        solver='lbfgs'
    )

    # Dummy fit agar struktur model terbentuk
    X_dummy = np.random.randn(N_CLASSES, N_FEATURES)
    y_dummy = np.arange(1, N_CLASSES + 1)

    global_model.fit(X_dummy, y_dummy)

    # Reset parameter awal
    global_model.coef_ = np.zeros((N_CLASSES, N_FEATURES))
    global_model.intercept_ = np.zeros(N_CLASSES)

    # ===== Federated rounds =====
    for rnd in range(n_rounds):

        local_models = []
        n_samples_list = []

        # --- Training tiap client ---
        for client_id, (X_client, y_client) in enumerate(partitions):
            unique_classes = np.unique(y_client)
            if len(unique_classes) < 2:
                print(
                     f"Client {client_id} dilewati "
                     f"(hanya punya 1 kelas)"
                )
                continue

            # Client menerima global model
            local_model = copy.deepcopy(global_model)

            # Training lokal
            local_model = local_train_round(
                local_model,
                X_client,
                y_client
            )

            local_models.append(local_model)
            n_samples_list.append(len(y_client))

        # --- FedAvg ---
        global_coef, global_intercept = fedavg_simple(
            local_models,
            n_samples_list
        )

        # Update global model
        global_model.coef_ = global_coef
        global_model.intercept_ = global_intercept

        # --- Evaluasi ---
        y_pred = global_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        accuracy_history.append(acc)

        print(
            f"[{label}] Round {rnd+1:02d} "
            f"Accuracy: {acc:.4f}"
        )

    return accuracy_history


# ═══════════════════════════════════════════════════════════════════
# MAIN — Jalankan Eksperimen
# ═══════════════════════════════════════════════════════════════════

def plot_convergence(history_iid: list, history_noniid: list,
                     alpha: float = 0.1):
      
    rounds = np.arange(1, len(history_iid) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rounds,
        history_iid,
        label="IID",
        marker='o'
    )

    plt.plot(
        rounds,
        history_noniid,
        label=f"Non-IID (α={alpha})",
        marker='s'
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")

    plt.title("Perbandingan Konvergensi FL")

    plt.legend()

    plt.grid(True)

    plt.savefig("fl_convergence_comparison.png")

    plt.show()


def main():
    print("=" * 60)
    print("  Challenge 2 — IID vs Non-IID Analysis")
    print("=" * 60)

    # Muat dataset
    raw_path = Path("data/raw/UCI HAR Dataset")
    if not raw_path.exists():
        print("Dataset belum ada. Jalankan: python data/download_data.py")
        return

    X_train, y_train, X_test, y_test = load_uci_har()

    # Normalisasi
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Buat partisi ──────────────────────────────────────────────
    print("\n[1] Membuat partisi IID...")
    partitions_iid = partition_iid(X_train, y_train, N_CLIENTS)

    print("[2] Membuat partisi Non-IID (alpha=0.5)...")
    partitions_noniid = partition_noniid_dirichlet(
        X_train, y_train, N_CLIENTS, alpha=0.5
    )

    # ── Bagian A: Analisis distribusi ────────────────────────────
    print("\n[Bagian A] Analisis Distribusi Label")
    analyze_distribution(partitions_iid,   "Distribusi IID")
    analyze_distribution(partitions_noniid, "Distribusi Non-IID (alpha=0.5)")

    # ── Bagian B: Simulasi FL ────────────────────────────────────
    print("\n[Bagian B] Simulasi Federated Learning")
    print(f"  {N_ROUNDS} round, {N_CLIENTS} client\n")

    print("Menjalankan FL dengan data IID...")
    acc_iid = simulate_federated_learning(
        partitions_iid, X_test, y_test, label="IID"
    )

    print("Menjalankan FL dengan data Non-IID (alpha=0.5)...")
    acc_noniid = simulate_federated_learning(
        partitions_noniid, X_test, y_test, label="Non-IID"
    )

    # ── Plot ─────────────────────────────────────────────────────
    print("\n[Bagian B] Plot Kurva Konvergensi")
    plot_convergence(acc_iid, acc_noniid)

    # ── Ringkasan hasil ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RINGKASAN HASIL")
    print("=" * 50)
    if any(a > 0 for a in acc_iid):
        print(f"  Akurasi akhir IID     : {acc_iid[-1]:.4f} ({acc_iid[-1]*100:.2f}%)")
        print(f"  Akurasi akhir Non-IID : {acc_noniid[-1]:.4f} ({acc_noniid[-1]*100:.2f}%)")
        print(f"  Selisih               : {abs(acc_iid[-1]-acc_noniid[-1]):.4f}")
    else:
        print("  [!] Simulasi belum diimplementasikan (Bagian B)")

    print("\nTugas selanjutnya: Isi file analisis.md dengan jawaban pertanyaan")


if __name__ == "__main__":
    main()

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
LOCAL_ITER = 50   # iterasi Logistic Regression per round


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

    # ─── TODO Bagian A ───────────────────────────────────────────
    # 1. Untuk setiap client, hitung proporsi setiap kelas
    #    distribution[client_idx][class_idx] = persentase (0..1)
    #
    # 2. Buat stacked bar chart:
    #    - x-axis: Client 1, Client 2, Client 3, Client 4
    #    - y-axis: persentase (0-100%)
    #    - Setiap warna = satu aktivitas
    #    - Tambahkan legend
    #
    # 3. Print tabel distribusi ke terminal
    # ─────────────────────────────────────────────────────────────

    # Placeholder — hapus dan ganti dengan implementasimu
    print(f"[analyze_distribution] TODO: implementasikan untuk '{title}'")
    distribution = np.zeros((n_clients, len(classes)))

    for i, (_, y) in enumerate(partitions):
        total = len(y)
        for j, cls in enumerate(classes):
            distribution[i][j] = (y == cls).sum() / total

    # --- Print distribusi ---
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    header = f"{'Aktivitas':<25}" + "".join(f"{'C'+str(i+1):>8}" for i in range(n_clients))
    print(header)
    print("-" * 60)
    for j, name in enumerate(activity_names):
        row = f"{name:<25}" + "".join(f"{distribution[i][j]*100:>7.1f}%" for i in range(n_clients))
        print(row)
    print("-" * 60)
    total_row = f"{'N Samples':<25}" + "".join(
        f"{len(p[1]):>8}" for p in partitions
    )
    print(total_row)

    # --- TODO: Plot stacked bar chart ---
    # Hint: gunakan ax.bar() berulang dengan 'bottom' parameter
    # untuk membuat stacked bar chart


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
    """
    Simulasi Federated Learning lokal (tanpa server HTTP).

    TODO: Lengkapi fungsi ini:
    1. Inisialisasi model global (coef=0, intercept=0)
    2. Untuk setiap round:
       a. Setiap client menerima model global
       b. Setiap client melatih model lokal
       c. Server menjalankan FedAvg
       d. Update model global
       e. Evaluasi akurasi global
    3. Kembalikan list akurasi per round

    Parameters
    ----------
    partitions : list of (X_train_k, y_train_k)
    X_test     : data test global
    y_test     : label test global
    n_rounds   : jumlah round

    Returns
    -------
    accuracy_history : list of float, akurasi per round
    """

    # ─── TODO Bagian B ───────────────────────────────────────────
    # Implementasikan simulasi FL di sini.
    #
    # Gunakan fungsi:
    #   - local_train_round() untuk training lokal
    #   - fedavg_simple() untuk agregasi
    #
    # Catatan: Gunakan sklearn Logistic Regression dengan warm_start=True
    # ─────────────────────────────────────────────────────────────

    # Placeholder
    print(f"[simulate_fl] TODO: implementasikan simulasi untuk '{label}'")
    return [0.0] * n_rounds


# ═══════════════════════════════════════════════════════════════════
# MAIN — Jalankan Eksperimen
# ═══════════════════════════════════════════════════════════════════

def plot_convergence(history_iid: list, history_noniid: list,
                     alpha: float = 0.5):
    """
    Plot perbandingan kurva konvergensi IID vs Non-IID.

    TODO: Lengkapi fungsi ini:
    1. Plot accuracy_iid vs round
    2. Plot accuracy_noniid vs round
    3. Tambahkan legend, title, label axis
    4. Simpan sebagai PNG
    """
    # ─── TODO ────────────────────────────────────────────────────
    # Buat plot perbandingan konvergensi
    # Hint:
    #   plt.figure(figsize=(10, 6))
    #   plt.plot(rounds, acc_iid, label="IID", marker='o')
    #   plt.plot(rounds, acc_noniid, label=f"Non-IID (α={alpha})", marker='s')
    #   ...
    # ─────────────────────────────────────────────────────────────
    print("[plot_convergence] TODO: buat plot perbandingan")


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

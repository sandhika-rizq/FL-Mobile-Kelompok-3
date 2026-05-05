"""
Challenge 4 — Analisis Konvergensi & Efisiensi Komunikasi (BONUS)
===================================================================
Eksperimen pengaruh hyperparameter FL terhadap konvergensi
dan efisiensi komunikasi.

Jalankan:
    python challenges/challenge_4/solution.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
sys.path.insert(0, "../..")

try:
    from data.partition_data import load_uci_har, partition_iid
except ImportError:
    from partition_data import load_uci_har, partition_iid

N_CLIENTS  = 4
N_ROUNDS   = 20
N_FEATURES = 561
N_CLASSES  = 6


# ═══════════════════════════════════════════════════════════════════
# Helper — Simulasi FL
# ═══════════════════════════════════════════════════════════════════

def run_fl_experiment(partitions: list,
                      X_test: np.ndarray, y_test: np.ndarray,
                      local_epochs: int = 5,
                      n_rounds: int = N_ROUNDS,
                      client_fraction: float = 1.0,
                      seed: int = 42) -> dict:
    """
    Satu run eksperimen FL dengan konfigurasi tertentu.

    Parameters
    ----------
    local_epochs     : jumlah iterasi training lokal per round
    n_rounds         : jumlah round komunikasi
    client_fraction  : proporsi client yang berpartisipasi (0.0 - 1.0)

    Returns
    -------
    dict berisi:
        - "accuracy_history"   : list of float
        - "time_per_round"     : list of float (detik)
        - "comm_cost_mb"       : total komunikasi dalam MB
    """
    rng             = np.random.default_rng(seed=seed)
    accuracy_history = []
    time_per_round   = []

    # Hitung biaya komunikasi per round per client (upload + download)
    # Ukuran parameter: coef (N_CLASSES * N_FEATURES) + intercept (N_CLASSES)
    params_size  = (N_CLASSES * N_FEATURES + N_CLASSES) * 4  # float32 = 4 bytes
    comm_per_round = params_size * 2 / (1024 * 1024)  # MB (upload + download)

    # Inisialisasi
    global_coef      = np.zeros((N_CLASSES, N_FEATURES))
    global_intercept = np.zeros(N_CLASSES)

    # ─── TODO ────────────────────────────────────────────────────
    # Implementasikan loop FL di sini:
    #
    # Untuk setiap round:
    # 1. Pilih subset client sesuai client_fraction
    #    (gunakan rng.choice() untuk random sampling)
    # 2. Setiap client yang terpilih: training lokal dengan local_epochs
    # 3. FedAvg dari client yang berpartisipasi
    # 4. Evaluasi dan catat akurasi + waktu
    #
    # Hint untuk partial participation:
    #   n_selected = max(1, int(N_CLIENTS * client_fraction))
    #   selected = rng.choice(N_CLIENTS, n_selected, replace=False)
    # ─────────────────────────────────────────────────────────────

    print(f"  [run_fl_experiment] TODO: implementasikan (E={local_epochs}, "
          f"fraction={client_fraction:.2f})")

    # Placeholder
    accuracy_history = [0.0] * n_rounds
    time_per_round   = [0.0] * n_rounds

    n_participating = max(1, int(N_CLIENTS * client_fraction))
    total_comm_mb   = comm_per_round * n_participating * n_rounds

    return {
        "accuracy_history" : accuracy_history,
        "time_per_round"   : time_per_round,
        "comm_cost_mb"     : total_comm_mb,
        "local_epochs"     : local_epochs,
        "client_fraction"  : client_fraction,
    }


# ═══════════════════════════════════════════════════════════════════
# BAGIAN A — Pengaruh Local Epochs
# ═══════════════════════════════════════════════════════════════════

def analyze_convergence(partitions: list,
                        X_test: np.ndarray, y_test: np.ndarray):
    """
    Analisis pengaruh jumlah local epochs terhadap konvergensi.

    TODO:
    1. Jalankan FL dengan E ∈ {1, 5, 10, 20, 50}
    2. Bandingkan: akurasi per round, waktu total, biaya komunikasi
    3. Plot tiga grafik:
       a. Akurasi vs Round untuk berbagai E
       b. Final accuracy vs E
       c. Akurasi vs Total Waktu (time efficiency)
    """
    local_epochs_list = [1, 5, 10, 20, 50]
    results           = {}

    print("\n[Bagian A] Analisis Pengaruh Local Epochs")
    for E in local_epochs_list:
        print(f"\n  Menjalankan FL dengan E={E}...")
        results[E] = run_fl_experiment(
            partitions, X_test, y_test,
            local_epochs=E, n_rounds=N_ROUNDS
        )

    # ─── TODO Bagian A ────────────────────────────────────────────
    # Buat tiga plot:
    # 1. plt.subplot(1,3,1): accuracy vs round untuk tiap E
    # 2. plt.subplot(1,3,2): final accuracy vs E (bar chart)
    # 3. plt.subplot(1,3,3): accuracy vs cumulative time
    #
    # Simpan sebagai: challenges/challenge_4/plot_convergence.png
    # ─────────────────────────────────────────────────────────────

    print("\n[Bagian A] TODO: buat plot analisis konvergensi")

    # Print tabel ringkasan
    print("\nRingkasan:")
    print(f"{'E':>5} | {'Final Acc':>10} | {'Total Waktu':>12} | {'Comm (MB)':>10}")
    print("-" * 45)
    for E, r in results.items():
        final_acc   = r["accuracy_history"][-1] if r["accuracy_history"] else 0.0
        total_time  = sum(r["time_per_round"])
        comm_mb     = r["comm_cost_mb"]
        print(f"{E:>5} | {final_acc:>10.4f} | {total_time:>11.2f}s | {comm_mb:>9.2f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# BAGIAN B — Partial Client Participation
# ═══════════════════════════════════════════════════════════════════

def analyze_partial_participation(partitions: list,
                                   X_test: np.ndarray, y_test: np.ndarray):
    """
    Analisis pengaruh partial client participation.

    Dalam FL nyata, tidak semua client online setiap round.
    Di sini kita simulasikan dengan memilih subset client secara acak.

    TODO:
    1. Jalankan FL dengan fraction ∈ {0.25, 0.5, 0.75, 1.0}
       (fraction = proporsi client yang berpartisipasi per round)
    2. Plot akurasi vs round untuk setiap fraction
    3. Analisis: berapa fraction minimal agar FL tetap stabil?
    """
    fractions = [0.25, 0.5, 0.75, 1.0]
    results   = {}

    print("\n[Bagian B] Analisis Partial Client Participation")
    for frac in fractions:
        n_selected = max(1, int(N_CLIENTS * frac))
        print(f"\n  Fraction={frac:.2f} ({n_selected}/{N_CLIENTS} client per round)...")
        results[frac] = run_fl_experiment(
            partitions, X_test, y_test,
            local_epochs=10, n_rounds=N_ROUNDS,
            client_fraction=frac
        )

    # ─── TODO Bagian B ────────────────────────────────────────────
    # Buat plot akurasi vs round untuk semua fraksi
    # Simpan sebagai: challenges/challenge_4/plot_participation.png
    # ─────────────────────────────────────────────────────────────

    print("\n[Bagian B] TODO: buat plot partial participation")

    print("\nRingkasan Partial Participation:")
    print(f"{'Fraction':>10} | {'N Client':>8} | {'Final Acc':>10} | {'Variance':>10}")
    print("-" * 50)
    for frac, r in results.items():
        n_sel     = max(1, int(N_CLIENTS * frac))
        final_acc = r["accuracy_history"][-1] if r["accuracy_history"] else 0.0
        variance  = np.var(r["accuracy_history"][-5:]) if len(r["accuracy_history"]) >= 5 else 0.0
        print(f"{frac:>10.2f} | {n_sel:>8d} | {final_acc:>10.4f} | {variance:>10.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Challenge 4 — Konvergensi & Efisiensi Komunikasi")
    print("=" * 60)

    raw_path = Path("data/raw/UCI HAR Dataset")
    if not raw_path.exists():
        print("Dataset belum ada. Jalankan: python data/download_data.py")
        return

    # Muat dan normalisasi data
    X_train, y_train, X_test, y_test = load_uci_har()
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Buat partisi IID
    partitions = partition_iid(X_train, y_train, N_CLIENTS)

    # Jalankan eksperimen
    results_a = analyze_convergence(partitions, X_test, y_test)
    results_b = analyze_partial_participation(partitions, X_test, y_test)

    print("\n" + "=" * 60)
    print("Selesai! Isi analisis.md dengan:")
    print("1. Nilai E optimal berdasarkan eksperimen Bagian A")
    print("2. Fraction minimal agar FL stabil (Bagian B)")
    print("3. Perkiraan total komunikasi (MB) untuk konfigurasi optimal")
    print("=" * 60)


if __name__ == "__main__":
    main()

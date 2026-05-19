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

    from utils.model_utils import deserialize_model, serialize_model, fedavg_manual

    # Inisialisasi parameter global
    global_params = {
        "coef": np.zeros((N_CLASSES, N_FEATURES)).tolist(),
        "intercept": np.zeros(N_CLASSES).tolist()
    }

    n_participating = max(1, int(N_CLIENTS * client_fraction))
    total_comm_mb = comm_per_round * n_participating * n_rounds

    for round_num in range(n_rounds):
        start_time = time.time()
        
        # 1. Pilih subset client
        selected_clients = rng.choice(N_CLIENTS, n_participating, replace=False)
        
        client_params_list = []
        client_weights = []

        # 2. Training lokal untuk tiap client yang terpilih
        for client_idx in selected_clients:
            X_local, y_local = partitions[client_idx]
            
            # Load model global ke model lokal client
            local_model = deserialize_model(global_params, n_classes=N_CLASSES, n_features=N_FEATURES, max_iter=local_epochs)
            
            # Train model lokal
            local_model.fit(X_local, y_local)
            
            # Simpan parameter hasil training
            client_params_list.append(serialize_model(local_model))
            client_weights.append(len(y_local))  # Bobot berdasarkan jumlah data

        # 3. Agregasi FedAvg di Server
        global_params = fedavg_manual(client_params_list, client_weights)
        
        # 4. Evaluasi model global yang baru
        global_model = deserialize_model(global_params, n_classes=N_CLASSES, n_features=N_FEATURES, max_iter=1)
        y_pred = global_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        round_time = time.time() - start_time
        
        accuracy_history.append(acc)
        time_per_round.append(round_time)
        
        # Print progress untuk memonitor jalannya simulasi
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"      Round {round_num + 1:2d}/{n_rounds} - Acc: {acc:.4f} - Time: {round_time:.2f}s")

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

    plt.figure(figsize=(15, 5))
    
    # 1. Accuracy vs Round
    plt.subplot(1, 3, 1)
    for E, r in results.items():
        plt.plot(range(1, N_ROUNDS + 1), r["accuracy_history"], marker='o', label=f'E={E}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Communication Round')
    plt.legend()
    plt.grid(True)
    
    # 2. Final accuracy vs E
    plt.subplot(1, 3, 2)
    final_accs = [r["accuracy_history"][-1] for E in local_epochs_list]
    plt.bar([str(E) for E in local_epochs_list], final_accs, color='skyblue')
    plt.xlabel('Local Epochs (E)')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy vs Local Epochs')
    plt.ylim(0, 1.05)
    plt.grid(axis='y')
    
    # 3. Accuracy vs Cumulative Time
    plt.subplot(1, 3, 3)
    for E, r in results.items():
        cum_time = np.cumsum(r["time_per_round"])
        plt.plot(cum_time, r["accuracy_history"], marker='o', label=f'E={E}')
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Time Efficiency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('challenges/challenge_4/plot_convergence.png')
    plt.close()
    
    print("\n[Bagian A] Plot berhasil disimpan di challenges/challenge_4/plot_convergence.png")

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

    plt.figure(figsize=(8, 6))
    for frac, r in results.items():
        n_sel = max(1, int(N_CLIENTS * frac))
        plt.plot(range(1, N_ROUNDS + 1), r["accuracy_history"], marker='o', label=f'Frac={frac} ({n_sel} clients)')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Impact of Partial Client Participation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('challenges/challenge_4/plot_participation.png')
    plt.close()

    print("\n[Bagian B] Plot berhasil disimpan di challenges/challenge_4/plot_participation.png")

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

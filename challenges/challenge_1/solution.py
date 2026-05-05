"""
Challenge 1 — Implementasi FedAvg
===================================
Lengkapi fungsi fedavg_aggregate() di bawah.

Petunjuk:
    - Bobot setiap client = n_samples_k / total_samples
    - Agregasi dilakukan terpisah untuk coef dan intercept
    - Semua operasi menggunakan numpy

Jalankan test dengan:
    python test_fedavg.py
"""

import numpy as np
from typing import List, Tuple, Dict


def fedavg_aggregate(
    client_params: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementasi Federated Averaging (FedAvg).

    Parameters
    ----------
    client_params : List of dict, masing-masing berisi:
        {
            "client_id"  : str,          # ID client (opsional)
            "coef"       : np.ndarray,   # shape (n_classes, n_features)
            "intercept"  : np.ndarray,   # shape (n_classes,)
            "n_samples"  : int,          # jumlah data lokal
        }

    Returns
    -------
    global_coef      : np.ndarray, shape (n_classes, n_features)
    global_intercept : np.ndarray, shape (n_classes,)

    Contoh Penggunaan:
    ------------------
    params = [
        {"coef": coef_1, "intercept": inter_1, "n_samples": 1000},
        {"coef": coef_2, "intercept": inter_2, "n_samples": 2000},
    ]
    global_coef, global_intercept = fedavg_aggregate(params)
    """

    # ─────────────────────────────────────────────────────────────
    # TODO: Implementasikan algoritma FedAvg di sini
    #
    # Langkah-langkah:
    # 1. Hitung total_samples = jumlah seluruh n_samples dari semua client
    # 2. Untuk setiap client, hitung bobotnya: weight_k = n_samples_k / total_samples
    # 3. Inisialisasi global_coef dan global_intercept dengan nol
    # 4. Tambahkan kontribusi setiap client: global_coef += weight_k * coef_k
    # 5. Return global_coef dan global_intercept
    #
    # Catatan: Pastikan output memiliki shape yang SAMA dengan input
    # ─────────────────────────────────────────────────────────────

    # Hapus baris ini dan tulis implementasimu:
    raise NotImplementedError("Implementasikan fedavg_aggregate() terlebih dahulu!")

    # return global_coef, global_intercept   # ← uncomment setelah implementasi


# ─────────────────────────── Contoh Penggunaan ─────────────────────
if __name__ == "__main__":
    # Buat data dummy untuk mencoba implementasimu
    np.random.seed(42)
    N_CLASSES, N_FEATURES = 6, 561

    # Simulasikan 4 client dengan n_samples berbeda
    client_params = []
    sample_counts = [1200, 1800, 900, 1400]

    for i, n in enumerate(sample_counts):
        client_params.append({
            "client_id" : str(i + 1),
            "coef"      : np.random.randn(N_CLASSES, N_FEATURES) * 0.1,
            "intercept" : np.random.randn(N_CLASSES) * 0.01,
            "n_samples" : n,
        })
        print(f"Client {i+1}: n_samples={n}, "
              f"coef mean={client_params[-1]['coef'].mean():.6f}")

    print(f"\nTotal samples: {sum(sample_counts)}")
    print(f"Bobot seharusnya: {[n/sum(sample_counts) for n in sample_counts]}")
    print()

    try:
        global_coef, global_intercept = fedavg_aggregate(client_params)
        print(f"Global coef shape      : {global_coef.shape}")
        print(f"Global intercept shape : {global_intercept.shape}")
        print(f"Global coef mean       : {global_coef.mean():.6f}")
        print("\nBerhasil! Sekarang jalankan: python test_fedavg.py")
    except NotImplementedError:
        print("Belum diimplementasikan. Isi bagian TODO terlebih dahulu.")

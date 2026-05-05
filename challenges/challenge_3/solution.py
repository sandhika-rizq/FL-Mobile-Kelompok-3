"""
Challenge 3 — Differential Privacy (BONUS)
============================================
Tambahkan privasi diferensial (Gaussian Mechanism) ke FL pipeline.

Jalankan:
    python challenges/challenge_3/solution.py
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

try:
    from data.partition_data import load_uci_har, partition_iid, ACTIVITY_LABELS
except ImportError:
    from partition_data import load_uci_har, partition_iid, ACTIVITY_LABELS

N_CLIENTS  = 4
N_ROUNDS   = 10
N_FEATURES = 561
N_CLASSES  = 6
LOCAL_ITER = 50


# ═══════════════════════════════════════════════════════════════════
# BAGIAN A — Gaussian Mechanism untuk Differential Privacy
# ═══════════════════════════════════════════════════════════════════

def clip_parameters(coef: np.ndarray, intercept: np.ndarray,
                    max_norm: float = 1.0) -> tuple:
    """
    Clipping L2 norm untuk membatasi sensitivity.
    Ini adalah langkah preprocessing sebelum menambahkan noise.

    Parameter model yang terlalu besar akan diclip sehingga
    L2 norm-nya tidak melebihi max_norm.

    Parameters
    ----------
    coef       : np.ndarray, shape (N_CLASSES, N_FEATURES)
    intercept  : np.ndarray, shape (N_CLASSES,)
    max_norm   : batas maksimum L2 norm

    Returns
    -------
    clipped_coef, clipped_intercept
    """
    # Flatten semua parameter menjadi satu vektor
    flat = np.concatenate([coef.flatten(), intercept])
    l2_norm = np.linalg.norm(flat)

    # Clip jika norm melebihi batas
    if l2_norm > max_norm:
        scale     = max_norm / l2_norm
        coef      = coef * scale
        intercept = intercept * scale

    return coef, intercept


def compute_sigma(epsilon: float, delta: float = 1e-5,
                  sensitivity: float = 1.0) -> float:
    """
    Hitung standar deviasi noise Gaussian untuk (ε, δ)-DP.

    Formula:
        σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε

    Parameters
    ----------
    epsilon     : privacy budget (lebih kecil = lebih privat)
    delta       : probability of privacy violation
    sensitivity : L2 sensitivity = max_norm dari clipping

    Returns
    -------
    sigma : float, standar deviasi noise yang harus ditambahkan
    """
    # ─── TODO Bagian A ────────────────────────────────────────────
    # Implementasikan rumus compute_sigma di sini
    # ─────────────────────────────────────────────────────────────
    raise NotImplementedError("Implementasikan compute_sigma()")


def add_gaussian_noise(coef: np.ndarray, intercept: np.ndarray,
                       epsilon: float, delta: float = 1e-5,
                       max_norm: float = 1.0,
                       rng: np.random.Generator = None) -> tuple:
    """
    Tambahkan noise Gaussian ke parameter model (Gaussian Mechanism).

    Langkah-langkah:
    1. Clip parameter (batasi sensitivity)
    2. Hitung sigma berdasarkan (epsilon, delta)
    3. Tambahkan noise N(0, sigma^2) ke setiap elemen parameter

    Parameters
    ----------
    coef      : model coefficients
    intercept : model intercept
    epsilon   : privacy budget
    delta     : DP delta parameter
    max_norm  : batas clipping
    rng       : numpy random generator (untuk reproducibility)

    Returns
    -------
    noisy_coef, noisy_intercept
    """
    if rng is None:
        rng = np.random.default_rng()

    # ─── TODO Bagian A ────────────────────────────────────────────
    # 1. Clip parameter menggunakan clip_parameters()
    # 2. Hitung sigma menggunakan compute_sigma()
    # 3. Tambahkan noise Gaussian ke coef dan intercept
    #    noise = rng.normal(0, sigma, shape)
    # 4. Return noisy_coef, noisy_intercept
    # ─────────────────────────────────────────────────────────────
    raise NotImplementedError("Implementasikan add_gaussian_noise()")


# ═══════════════════════════════════════════════════════════════════
# FL Pipeline dengan DP
# ═══════════════════════════════════════════════════════════════════

def simulate_fl_with_dp(partitions: list,
                        X_test: np.ndarray, y_test: np.ndarray,
                        epsilon: float = 1.0,
                        n_rounds: int = N_ROUNDS) -> list:
    """
    Simulasi FL dengan Differential Privacy.

    Sama seperti FL biasa, tetapi setiap client menambahkan
    noise DP sebelum mengirim parameter ke server.

    Parameters
    ----------
    partitions : list of (X_train_k, y_train_k)
    epsilon    : privacy budget (float('inf') = tanpa DP)
    """
    rng     = np.random.default_rng(seed=42)
    history = []

    # Inisialisasi model global
    global_coef      = np.zeros((N_CLASSES, N_FEATURES))
    global_intercept = np.zeros(N_CLASSES)

    for round_num in range(1, n_rounds + 1):
        local_coefs      = []
        local_intercepts = []
        n_samples_list   = []

        for X_local, y_local in partitions:
            # Training lokal
            model = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=LOCAL_ITER, warm_start=True, random_state=42
            )
            model.coef_          = global_coef.copy()
            model.intercept_     = global_intercept.copy()
            model.classes_       = np.arange(1, N_CLASSES + 1)
            model.n_features_in_ = N_FEATURES
            model.fit(X_local, y_local)

            coef      = model.coef_.copy()
            intercept = model.intercept_.copy()

            # Tambahkan noise DP jika epsilon berhingga
            if not np.isinf(epsilon):
                coef, intercept = add_gaussian_noise(
                    coef, intercept, epsilon=epsilon, rng=rng
                )

            local_coefs.append(coef)
            local_intercepts.append(intercept)
            n_samples_list.append(len(y_local))

        # FedAvg
        total = sum(n_samples_list)
        global_coef      = sum(c * (n / total) for c, n in
                               zip(local_coefs, n_samples_list))
        global_intercept = sum(i * (n / total) for i, n in
                               zip(local_intercepts, n_samples_list))

        # Evaluasi
        model.coef_      = global_coef
        model.intercept_ = global_intercept
        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        history.append(acc)

        print(f"  Round {round_num:2d}/{n_rounds} | ε={epsilon:.2f} | acc={acc:.4f}")

    return history


# ═══════════════════════════════════════════════════════════════════
# BAGIAN B — Eksperimen Privacy-Utility Trade-off
# ═══════════════════════════════════════════════════════════════════

def experiment_privacy_tradeoff(partitions: list,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray):
    """
    Eksperimen: akurasi vs epsilon untuk berbagai nilai privacy budget.

    TODO: Lengkapi fungsi ini:
    1. Uji beberapa nilai epsilon: [0.1, 0.5, 1.0, 5.0, 10.0, inf]
    2. Untuk setiap epsilon, jalankan simulate_fl_with_dp()
    3. Ambil akurasi akhir (round terakhir)
    4. Plot grafik epsilon vs final accuracy
    5. Buat kesimpulan: epsilon berapa yang memberikan trade-off terbaik?
    """
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0, float("inf")]
    # float("inf") = tanpa DP (baseline)

    # ─── TODO Bagian B ────────────────────────────────────────────
    # 1. Loop untuk setiap epsilon
    # 2. Jalankan simulate_fl_with_dp() dan catat akurasi akhir
    # 3. Plot grafik
    # ─────────────────────────────────────────────────────────────

    print("[experiment_privacy_tradeoff] TODO: implementasikan eksperimen")
    results = {eps: 0.0 for eps in epsilons}
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Challenge 3 — Differential Privacy")
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

    # Bagian A: Test noise
    print("\n[Bagian A] Test Gaussian Noise")
    np.random.seed(42)
    test_coef      = np.random.randn(N_CLASSES, N_FEATURES)
    test_intercept = np.random.randn(N_CLASSES)

    try:
        sigma = compute_sigma(epsilon=1.0)
        print(f"  sigma untuk ε=1.0, δ=1e-5: {sigma:.6f}")

        noisy_coef, noisy_inter = add_gaussian_noise(
            test_coef, test_intercept, epsilon=1.0
        )
        diff_coef  = np.abs(noisy_coef - test_coef).mean()
        diff_inter = np.abs(noisy_inter - test_intercept).mean()
        print(f"  Rata-rata perubahan coef     : {diff_coef:.6f}")
        print(f"  Rata-rata perubahan intercept: {diff_inter:.6f}")
        print("  Gaussian Noise OK!")
    except NotImplementedError:
        print("  [!] Bagian A belum diimplementasikan")

    # Bagian B: Eksperimen trade-off
    print("\n[Bagian B] Privacy-Utility Trade-off")
    try:
        results = experiment_privacy_tradeoff(partitions, X_test, y_test)
        print("\nRingkasan:")
        for eps, acc in results.items():
            label = f"ε={eps:.1f}" if not np.isinf(eps) else "ε=∞ (no DP)"
            print(f"  {label:<20}: acc={acc:.4f} ({acc*100:.2f}%)")
    except NotImplementedError:
        print("  [!] Bagian B belum diimplementasikan")


if __name__ == "__main__":
    main()

"""
Test Suite untuk Challenge 1 — FedAvg
======================================
Jalankan:
    python challenges/challenge_1/test_fedavg.py
"""

import sys
import numpy as np

# Tambahkan path root ke sys.path
sys.path.insert(0, ".")
sys.path.insert(0, "../..")

try:
    from challenges.challenge_1.solution import fedavg_aggregate
except ImportError:
    from solution import fedavg_aggregate

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

N_CLASSES  = 6
N_FEATURES = 10   # kecil untuk test


def make_params(coef_val, intercept_val, n_samples):
    return {
        "coef"      : np.full((N_CLASSES, N_FEATURES), coef_val),
        "intercept" : np.full(N_CLASSES, intercept_val),
        "n_samples" : n_samples,
    }


def run_test(name, fn):
    try:
        fn()
        print(f"[{PASS}] {name}")
        return True
    except NotImplementedError:
        print(f"[SKIP] {name} — Belum diimplementasikan")
        return False
    except AssertionError as e:
        print(f"[{FAIL}] {name}: {e}")
        return False
    except Exception as e:
        print(f"[{FAIL}] {name}: Error tidak terduga: {e}")
        return False


# ─────────────────────────── Test Cases ────────────────────────────

def test_uniform_weights():
    """Bobot seimbang → hasil harus rata-rata biasa."""
    params = [
        make_params(coef_val=1.0, intercept_val=0.5, n_samples=100),
        make_params(coef_val=3.0, intercept_val=1.5, n_samples=100),
    ]
    coef, intercept = fedavg_aggregate(params)

    expected_coef      = np.full((N_CLASSES, N_FEATURES), 2.0)
    expected_intercept = np.full(N_CLASSES, 1.0)

    assert np.allclose(coef, expected_coef, atol=1e-6), \
        f"Coef salah. Expected 2.0, got {coef.mean():.4f}"
    assert np.allclose(intercept, expected_intercept, atol=1e-6), \
        f"Intercept salah. Expected 1.0, got {intercept.mean():.4f}"


def test_nonuniform_weights():
    """Bobot tidak seimbang → client dengan lebih banyak data lebih berpengaruh."""
    # Client 1: coef=0, n=1; Client 2: coef=4, n=3 → expected = (0*0.25 + 4*0.75) = 3
    params = [
        make_params(coef_val=0.0, intercept_val=0.0, n_samples=1),
        make_params(coef_val=4.0, intercept_val=4.0, n_samples=3),
    ]
    coef, intercept = fedavg_aggregate(params)

    expected = 3.0
    assert np.allclose(coef, expected, atol=1e-6), \
        f"Coef salah. Expected {expected}, got {coef.mean():.4f}"
    assert np.allclose(intercept, expected, atol=1e-6), \
        f"Intercept salah. Expected {expected}, got {intercept.mean():.4f}"


def test_four_clients():
    """Test dengan 4 client seperti skenario nyata."""
    n_samples = [1200, 1800, 900, 1400]
    total     = sum(n_samples)

    # Gunakan coef yang nilainya = indeks client
    params = [make_params(float(i + 1), float(i + 1), n) for i, n in enumerate(n_samples)]

    coef, intercept = fedavg_aggregate(params)

    # Hitung expected secara manual
    expected_val = sum(
        (i + 1) * (n / total) for i, n in enumerate(n_samples)
    )
    assert np.allclose(coef, expected_val, atol=1e-5), \
        f"Coef salah. Expected {expected_val:.4f}, got {coef.mean():.4f}"


def test_output_shape():
    """Output harus memiliki shape yang sama dengan input."""
    params = [make_params(1.0, 0.5, 100) for _ in range(4)]
    coef, intercept = fedavg_aggregate(params)

    assert coef.shape      == (N_CLASSES, N_FEATURES), \
        f"Shape coef salah: {coef.shape}"
    assert intercept.shape == (N_CLASSES,), \
        f"Shape intercept salah: {intercept.shape}"


def test_single_client():
    """Dengan 1 client, output harus sama persis dengan input."""
    params = [make_params(coef_val=0.42, intercept_val=0.13, n_samples=500)]
    coef, intercept = fedavg_aggregate(params)

    assert np.allclose(coef, 0.42, atol=1e-6), \
        f"Single client: coef salah. Expected 0.42, got {coef.mean():.4f}"
    assert np.allclose(intercept, 0.13, atol=1e-6), \
        f"Single client: intercept salah"


def test_weighted_vs_uniform():
    """Pastikan weighted average ≠ simple average saat bobot tidak seimbang."""
    params = [
        make_params(0.0, 0.0, n_samples=100),
        make_params(10.0, 10.0, n_samples=900),
    ]
    coef, _ = fedavg_aggregate(params)
    simple_avg = 5.0   # (0 + 10) / 2

    # Weighted: (0*0.1 + 10*0.9) = 9.0 ≠ 5.0
    assert not np.allclose(coef, simple_avg, atol=1e-3), \
        "Implementasi terlihat menggunakan rata-rata sederhana, bukan berbobot!"
    assert np.allclose(coef, 9.0, atol=1e-5), \
        f"Weighted avg salah. Expected 9.0, got {coef.mean():.4f}"


# ─────────────────────────── Main ──────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Challenge 1 — Test FedAvg")
    print("=" * 50)

    tests = [
        ("FedAvg dengan bobot seimbang",           test_uniform_weights),
        ("FedAvg dengan bobot tidak seimbang",     test_nonuniform_weights),
        ("FedAvg dengan 4 client",                 test_four_clients),
        ("Konsistensi dimensi output",             test_output_shape),
        ("FedAvg single client",                   test_single_client),
        ("Weighted avg vs simple avg",             test_weighted_vs_uniform),
    ]

    results = []
    for name, fn in tests:
        results.append(run_test(name, fn))

    passed = sum(1 for r in results if r is True)
    total  = len(tests)

    print("─" * 50)
    if passed == total:
        print(f"\nSemua {total} test lulus! Implementasi FedAvg benar.")
        print("Lanjut ke Bagian B: buat file analisis.md")
    else:
        print(f"\n{passed}/{total} test lulus.")
        print("Periksa kembali implementasi fedavg_aggregate()")

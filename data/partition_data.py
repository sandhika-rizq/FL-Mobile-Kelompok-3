"""
Partisi Dataset UCI HAR untuk 4 Client
========================================
Membagi data training UCI HAR menjadi partisi untuk masing-masing client
dengan dua mode: IID (i.i.d.) dan Non-IID (label skewed).

Penggunaan:
    # Partisi IID (seimbang, acak)
    python data/partition_data.py --mode iid --num_clients 4

    # Partisi Non-IID (distribusi label miring)
    python data/partition_data.py --mode noniid --num_clients 4

    # Non-IID dengan tingkat kemiringan kustom (0.1 = sangat miring)
    python data/partition_data.py --mode noniid --num_clients 4 --alpha 0.5

Output:
    data/partitions/
    ├── client_1/
    │   ├── train.npz    ← X_train, y_train lokal
    │   └── test.npz     ← X_test, y_test (sama untuk semua = global test)
    ├── client_2/
    │   ├── train.npz
    │   └── test.npz
    ├── ...
    └── partition_info.json   ← statistik distribusi label per client

Label Activities (UCI HAR):
    1: WALKING
    2: WALKING_UPSTAIRS
    3: WALKING_DOWNSTAIRS
    4: SITTING
    5: STANDING
    6: LAYING
"""

import argparse
import json
import numpy as np
from pathlib import Path

RAW_DIR      = Path("data/raw/UCI HAR Dataset")
PARTITIONS   = Path("data/partitions")

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


# ─────────────────────────── Loader ────────────────────────────────
def load_uci_har():
    """
    Muat dataset UCI HAR dari file teks asli.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    def _load_set(split: str):
        X = np.loadtxt(RAW_DIR / split / f"X_{split}.txt")
        y = np.loadtxt(RAW_DIR / split / f"y_{split}.txt", dtype=int)
        return X, y

    print("Memuat dataset UCI HAR...")
    X_train, y_train = _load_set("train")
    X_test,  y_test  = _load_set("test")

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Kelas: {np.unique(y_train).tolist()}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────── Partisi IID ───────────────────────────
def partition_iid(X: np.ndarray, y: np.ndarray, num_clients: int, seed: int = 42):
    """
    IID Partition: acak kemudian bagi rata.
    Setiap client mendapat subset acak yang representatif.

    Returns
    -------
    list of (X_i, y_i) untuk setiap client
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    splits  = np.array_split(indices, num_clients)
    return [(X[idx], y[idx]) for idx in splits]


# ─────────────────────────── Partisi Non-IID ───────────────────────
def partition_noniid_dirichlet(X: np.ndarray, y: np.ndarray,
                                num_clients: int, alpha: float = 0.5,
                                seed: int = 42):
    """
    Non-IID Partition menggunakan distribusi Dirichlet.

    Parameter alpha mengontrol tingkat heterogenitas:
    - alpha = 100.0  → hampir IID (merata)
    - alpha = 1.0    → heterogen sedang
    - alpha = 0.1    → sangat heterogen (tiap client dominan 1-2 kelas)

    Referensi: Hsieh et al. (2020), "Quagmire of Non-IID Data"

    Returns
    -------
    list of (X_i, y_i) untuk setiap client
    """
    rng        = np.random.default_rng(seed)
    classes    = np.unique(y)
    n_classes  = len(classes)

    # Untuk setiap kelas, distribusikan ke client dengan Dirichlet
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        # Sampel proporsi dari distribusi Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Konversi ke jumlah sampel (integer)
        counts = (proportions * len(cls_indices)).astype(int)
        # Distribusi sisa untuk menghindari kehilangan sampel
        remainder = len(cls_indices) - counts.sum()
        counts[:remainder] += 1

        ptr = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(cls_indices[ptr:ptr + count].tolist())
            ptr += count

    # Acak indeks tiap client
    partitions = []
    for idx_list in client_indices:
        idx = np.array(idx_list)
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))

    return partitions


# ─────────────────────────── Simpan Partisi ────────────────────────
def save_partitions(partitions, X_test, y_test, output_dir: Path):
    """Simpan partisi training dan test ke file .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_info = {}

    for i, (X_part, y_part) in enumerate(partitions):
        client_dir = output_dir / f"client_{i + 1}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Simpan train dan test
        np.savez_compressed(client_dir / "train.npz", X=X_part, y=y_part)
        np.savez_compressed(client_dir / "test.npz",  X=X_test, y=y_test)

        # Statistik label
        label_counts = {}
        for label in np.unique(y_part):
            label_counts[ACTIVITY_LABELS[label]] = int((y_part == label).sum())

        partition_info[f"client_{i + 1}"] = {
            "n_samples"    : int(len(y_part)),
            "label_dist"   : label_counts,
        }

        print(f"  client_{i + 1}: {len(y_part)} sampel | {label_counts}")

    # Simpan info partisi
    with open(output_dir / "partition_info.json", "w") as f:
        json.dump(partition_info, f, indent=2)

    print(f"\nPartisi disimpan di: {output_dir}")
    print(f"Info distribusi: {output_dir / 'partition_info.json'}")


# ─────────────────────────── Report ────────────────────────────────
def print_distribution_report(partitions):
    """Tampilkan laporan distribusi label per client."""
    print("\n" + "=" * 60)
    print("DISTRIBUSI LABEL PER CLIENT")
    print("=" * 60)
    print(f"{'Aktivitas':<25}", end="")
    for i in range(len(partitions)):
        print(f"{'Client ' + str(i + 1):>10}", end="")
    print()
    print("-" * 60)

    classes = sorted(ACTIVITY_LABELS.keys())
    for cls in classes:
        print(f"{ACTIVITY_LABELS[cls]:<25}", end="")
        for X_part, y_part in partitions:
            count = int((y_part == cls).sum())
            print(f"{count:>10}", end="")
        print()

    print("-" * 60)
    print(f"{'TOTAL':<25}", end="")
    for X_part, y_part in partitions:
        print(f"{len(y_part):>10}", end="")
    print("\n")


# ─────────────────────────── Main ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Partisi dataset UCI HAR untuk Federated Learning"
    )
    parser.add_argument("--mode",        default="iid",
                        choices=["iid", "noniid"],
                        help="Mode partisi: iid atau noniid (default: iid)")
    parser.add_argument("--num_clients", default=4, type=int,
                        help="Jumlah client (default: 4)")
    parser.add_argument("--alpha",       default=0.5, type=float,
                        help="Parameter Dirichlet untuk non-iid (default: 0.5)")
    parser.add_argument("--seed",        default=42, type=int,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Validasi
    if not RAW_DIR.exists():
        print(f"Dataset tidak ditemukan di: {RAW_DIR}")
        print("Jalankan terlebih dahulu: python data/download_data.py")
        return

    # Muat data
    X_train, y_train, X_test, y_test = load_uci_har()

    # Buat partisi
    print(f"\nMode partisi : {args.mode.upper()}")
    print(f"Jumlah client: {args.num_clients}")

    if args.mode == "iid":
        partitions = partition_iid(X_train, y_train, args.num_clients, args.seed)
        out_dir = PARTITIONS
    else:
        print(f"Alpha Dirichlet: {args.alpha}")
        partitions = partition_noniid_dirichlet(
            X_train, y_train, args.num_clients, args.alpha, args.seed
        )
        out_dir = PARTITIONS

    # Tampilkan distribusi
    print_distribution_report(partitions)

    # Simpan
    save_partitions(partitions, X_test, y_test, out_dir)

    print("Selesai! Distribusikan folder client_X ke masing-masing VM client.")
    if args.mode == "noniid":
        print(f"\nCatatan: Non-IID alpha={args.alpha}")
        print("  Semakin kecil alpha → distribusi semakin tidak merata (heterogen)")
        print("  Ini mensimulasikan situasi nyata di mana tiap pengguna")
        print("  memiliki pola aktivitas berbeda-beda.")


if __name__ == "__main__":
    main()

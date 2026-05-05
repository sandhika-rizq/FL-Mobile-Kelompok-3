"""
Download Dataset UCI HAR
========================
Unduh dan ekstrak dataset Human Activity Recognition (HAR)
dari UCI Machine Learning Repository.

Jalankan:
    python data/download_data.py

Output:
    data/raw/UCI HAR Dataset/
"""

import os
import urllib.request
import zipfile
import hashlib
from pathlib import Path

# URL dataset di UCI Repository
DATASET_URL  = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
DATASET_MD5  = "53e099137392d0a7d538d9809f3c50f6"   # MD5 checksum untuk verifikasi
OUTPUT_DIR   = Path("data/raw")
ZIP_PATH     = OUTPUT_DIR / "UCI_HAR_Dataset.zip"


def check_md5(filepath: Path, expected_md5: str) -> bool:
    """Verifikasi integritas file dengan MD5."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def download_with_progress(url: str, dest: Path):
    """Unduh file dengan tampilan progress."""
    print(f"Mengunduh: {url}")
    print(f"Tujuan   : {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            mb_done  = downloaded / (1024 * 1024)
            mb_total = total_size  / (1024 * 1024)
            print(f"\r  Progress: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline setelah progress


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = OUTPUT_DIR / "UCI HAR Dataset"

    # Cek apakah dataset sudah ada
    if extract_dir.exists():
        print(f"Dataset sudah ada di: {extract_dir}")
        print("Gunakan --force untuk mengunduh ulang")
        return

    # Unduh file zip
    if not ZIP_PATH.exists():
        download_with_progress(DATASET_URL, ZIP_PATH)
    else:
        print(f"File zip sudah ada: {ZIP_PATH}")

    # Verifikasi MD5 (opsional — skip jika checksum berubah)
    print("Memverifikasi file...")
    actual_md5 = hashlib.md5(open(ZIP_PATH, "rb").read()).hexdigest()
    if actual_md5 != DATASET_MD5:
        print(f"  Peringatan: MD5 tidak cocok ({actual_md5} != {DATASET_MD5})")
        print("  File mungkin versi berbeda, tetap coba ekstrak...")
    else:
        print("  MD5 OK")

    # Ekstrak
    print(f"Mengekstrak ke: {OUTPUT_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUTPUT_DIR)

    # Verifikasi hasil ekstraksi
    required_files = [
        extract_dir / "train" / "X_train.txt",
        extract_dir / "train" / "y_train.txt",
        extract_dir / "test"  / "X_test.txt",
        extract_dir / "test"  / "y_test.txt",
        extract_dir / "features.txt",
        extract_dir / "activity_labels.txt",
    ]

    print("\nVerifikasi file:")
    all_ok = True
    for f in required_files:
        status = "OK" if f.exists() else "TIDAK DITEMUKAN"
        print(f"  [{status}] {f.relative_to(OUTPUT_DIR)}")
        if not f.exists():
            all_ok = False

    if all_ok:
        print("\nDataset berhasil diunduh dan diekstrak!")
        print(f"Lokasi: {extract_dir}")
        print("\nLangkah berikutnya:")
        print("  python data/partition_data.py --mode iid --num_clients 4")
    else:
        print("\nAda file yang hilang. Coba hapus folder data/raw dan ulangi.")


if __name__ == "__main__":
    import sys
    if "--force" in sys.argv:
        import shutil
        shutil.rmtree(OUTPUT_DIR / "UCI HAR Dataset", ignore_errors=True)
        ZIP_PATH.unlink(missing_ok=True)
    main()

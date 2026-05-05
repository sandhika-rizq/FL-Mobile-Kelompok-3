"""
Evaluasi Model Global
======================
Unduh model final dari server dan evaluasi pada test set penuh.

Jalankan setelah semua round selesai:
    python evaluation/evaluate_global.py --server_url http://<IP_SERVER>:5000

Atau evaluasi dari file lokal:
    python evaluation/evaluate_global.py --model_file final_model.json
"""

import sys
import json
import argparse
import numpy as np
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, ".")
sys.path.insert(0, "..")

try:
    from data.partition_data import load_uci_har, ACTIVITY_LABELS
    from utils.model_utils import deserialize_model, evaluate_model
except ImportError:
    # Fallback jika dijalankan dari folder lain
    raise ImportError("Jalankan dari root folder proyek: python evaluation/evaluate_global.py")

OUTPUT_DIR = Path("evaluation/output")


def fetch_final_model(server_url: str) -> dict:
    """Unduh model final dari FL server."""
    print(f"Menghubungi server: {server_url}")
    try:
        resp = requests.get(f"{server_url}/api/final_model", timeout=15)
        if resp.status_code == 425:
            print("Training belum selesai. Tunggu semua round selesai.")
            sys.exit(1)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"Tidak bisa terhubung ke server: {server_url}")
        sys.exit(1)


def load_model_from_file(filepath: str) -> dict:
    """Muat model dari file JSON lokal."""
    with open(filepath) as f:
        return json.load(f)


def plot_training_history(history: list):
    """Plot kurva akurasi dan loss per round."""
    if not history:
        print("Tidak ada data history training.")
        return

    rounds     = [h["round"]           for h in history]
    local_accs = [h["avg_local_acc"]   for h in history]
    local_loss = [h["avg_local_loss"]  for h in history]
    n_samples  = [h.get("total_samples", 0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot akurasi
    axes[0].plot(rounds, local_accs, "b-o", markersize=5, linewidth=2)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Rata-rata Akurasi Lokal")
    axes[0].set_title("Konvergensi FL — Akurasi per Round")
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=local_accs[-1], color="r", linestyle="--",
                    alpha=0.5, label=f"Final: {local_accs[-1]:.4f}")
    axes[0].legend()

    # Plot loss
    axes[1].plot(rounds, local_loss, "r-s", markersize=5, linewidth=2)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Rata-rata Loss Lokal")
    axes[1].set_title("Konvergensi FL — Loss per Round")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot disimpan: {save_path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    cm    = confusion_matrix(y_test, y_pred)
    labels = list(ACTIVITY_LABELS.values())

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Model Global FL")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix disimpan: {save_path}")
    plt.show()


def compare_with_centralized(X_train, y_train, X_test, y_test,
                              global_acc: float):
    """
    Buat baseline dengan Logistic Regression terpusat (centralized).
    Bandingkan dengan model global FL.
    """
    from sklearn.linear_model import LogisticRegression
    print("\nMelatih model terpusat (baseline)...")
    model_central = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=500, C=1.0, random_state=42
    )
    model_central.fit(X_train, y_train)
    central_acc = model_central.score(X_test, y_test)

    print(f"\n{'=' * 50}")
    print("  PERBANDINGAN: FL vs Centralized")
    print(f"{'=' * 50}")
    print(f"  Akurasi Model FL (Global)   : {global_acc:.4f} ({global_acc*100:.2f}%)")
    print(f"  Akurasi Model Terpusat      : {central_acc:.4f} ({central_acc*100:.2f}%)")
    diff = global_acc - central_acc
    print(f"  Selisih (FL - Centralized)  : {diff:+.4f} ({diff*100:+.2f}%)")

    if abs(diff) < 0.02:
        print("\n  Hasil: FL hampir setara dengan centralized!")
    elif diff < 0:
        print(f"\n  Hasil: FL lebih rendah {abs(diff)*100:.2f}% dari centralized.")
        print("         Ini normal — privasi ada biayanya.")
    else:
        print(f"\n  Hasil: FL lebih tinggi {diff*100:.2f}% dari centralized.")

    return central_acc


# ─────────────────────────── Main ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluasi model global FL")
    parser.add_argument("--server_url",  default="",
                        help="URL FL server, misal: http://10.0.0.1:5000")
    parser.add_argument("--model_file",  default="",
                        help="Path ke file model JSON lokal")
    parser.add_argument("--no_plot",     action="store_true",
                        help="Jangan tampilkan plot")
    parser.add_argument("--compare",     action="store_true", default=True,
                        help="Bandingkan dengan centralized (default: True)")
    args = parser.parse_args()

    # Muat model
    if args.model_file:
        print(f"Memuat model dari file: {args.model_file}")
        model_data = load_model_from_file(args.model_file)
    elif args.server_url:
        model_data = fetch_final_model(args.server_url)
    else:
        print("Gunakan --server_url atau --model_file")
        parser.print_help()
        sys.exit(1)

    # Muat dataset
    raw_path = Path("data/raw/UCI HAR Dataset")
    if not raw_path.exists():
        print("Dataset belum ada. Jalankan: python data/download_data.py")
        sys.exit(1)

    print("\nMemuat dataset UCI HAR...")
    X_train, y_train, X_test, y_test = load_uci_har()

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Rekonstruksi model
    model = deserialize_model(model_data, n_classes=6, n_features=561)

    # Evaluasi
    print("\nMengevaluasi model global pada test set...")
    metrics = evaluate_model(model, X_test, y_test, verbose=True)

    # Confusion matrix
    if not args.no_plot:
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred)

        # Plot training history jika ada
        history = model_data.get("history", [])
        if history:
            plot_training_history(history)

    # Perbandingan dengan centralized
    if args.compare:
        compare_with_centralized(X_train, y_train, X_test, y_test,
                                 metrics["accuracy"])

    # Simpan hasil evaluasi
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "evaluation_results.json"
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nHasil evaluasi disimpan: {result_path}")

    return metrics


if __name__ == "__main__":
    main()

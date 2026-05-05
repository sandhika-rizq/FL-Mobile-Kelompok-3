"""
FL Client — Federated Learning untuk Mobile Activity Recognition
================================================================
Jalankan di masing-masing VM Client:

    # VM Client 1:
    CLIENT_ID=1 SERVER_URL=http://<IP_SERVER>:5000 python client/client.py

    # VM Client 2:
    CLIENT_ID=2 SERVER_URL=http://<IP_SERVER>:5000 python client/client.py

    # dst.

Environment Variables:
    CLIENT_ID        : ID client (1-4)
    SERVER_URL       : URL server FL
    DATA_PATH        : path ke folder partisi data lokal
    LOCAL_EPOCHS     : jumlah epoch lokal (default: dari server)
    MAX_ITER         : max iterasi Logistic Regression (default: 200)
"""

import os
import sys
import time
import logging
import numpy as np
import requests
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── Konfigurasi ───────────────────────────
CLIENT_ID   = str(os.environ.get("CLIENT_ID", "1"))
SERVER_URL  = os.environ.get("SERVER_URL", "http://127.0.0.1:5000").rstrip("/")
DATA_PATH   = Path(os.environ.get("DATA_PATH", f"data/partitions/client_{CLIENT_ID}"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 0))   # 0 = ikuti server
MAX_ITER     = int(os.environ.get("MAX_ITER", 200))
RETRY_DELAY  = 5   # detik antar retry
MAX_RETRIES  = 30

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [CLIENT-{CLIENT_ID}] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────── Data Loading ──────────────────────────
def load_local_data(data_path: Path):
    """
    Muat data partisi lokal client.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    train_file = data_path / "train.npz"
    test_file  = data_path / "test.npz"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Data tidak ditemukan: {train_file}\n"
            f"Pastikan sudah menjalankan: python data/partition_data.py"
        )

    train_data = np.load(train_file)
    test_data  = np.load(test_file)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test,  y_test  = test_data["X"],  test_data["y"]

    logger.info(
        "Data dimuat: train=%d, test=%d, fitur=%d, kelas=%s",
        len(X_train), len(X_test), X_train.shape[1],
        np.unique(y_train).tolist()
    )
    return X_train, y_train, X_test, y_test


# ─────────────────────────── Model Utils ───────────────────────────
def create_model(n_classes: int, max_iter: int = 200) -> LogisticRegression:
    """Buat instance Logistic Regression multi-kelas."""
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        warm_start=True,    # penting: lanjutkan dari bobot sebelumnya
        C=1.0,
        random_state=42,
    )


def set_model_params(model: LogisticRegression,
                     coef: np.ndarray,
                     intercept: np.ndarray,
                     n_classes: int,
                     n_features: int):
    """
    Terapkan parameter global ke model lokal.
    Diperlukan karena sklearn LR harus di-fit dulu sebelum bisa di-set.
    """
    model.coef_      = coef.copy()
    model.intercept_ = intercept.copy()
    model.classes_   = np.arange(1, n_classes + 1)   # label 1..6 di UCI HAR

    # Paksa sklearn tahu model sudah fitted
    model.n_features_in_ = n_features
    return model


def get_model_params(model: LogisticRegression):
    """Ambil parameter model (coef + intercept)."""
    return model.coef_.copy(), model.intercept_.copy()


# ─────────────────────────── Server Comm ───────────────────────────
def register_with_server() -> dict:
    """Daftarkan client ke server, dapatkan konfigurasi training."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{SERVER_URL}/api/register",
                json={"client_id": CLIENT_ID},
                timeout=10
            )
            resp.raise_for_status()
            config = resp.json()
            logger.info("Terdaftar di server. Config: %s", config)
            return config
        except requests.exceptions.RequestException as e:
            logger.warning("Gagal terhubung ke server (percobaan %d/%d): %s",
                           attempt + 1, MAX_RETRIES, e)
            time.sleep(RETRY_DELAY)

    raise ConnectionError(f"Tidak bisa terhubung ke server setelah {MAX_RETRIES} percobaan")


def fetch_global_model(current_round: int) -> dict:
    """Unduh parameter model global dari server."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(f"{SERVER_URL}/api/model", timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Tunggu jika server sedang agregasi
            if data.get("phase") == "aggregating":
                logger.info("Server sedang agregasi, tunggu %ds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
                continue

            # Tunggu jika round belum sesuai
            server_round = data.get("round", 1)
            if server_round < current_round:
                logger.info("Server masih di round %d, tunggu...", server_round)
                time.sleep(RETRY_DELAY)
                continue

            return data

        except requests.exceptions.RequestException as e:
            logger.warning("Gagal fetch model (percobaan %d): %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)

    raise ConnectionError("Gagal mengunduh model dari server")


def submit_update(round_num: int, coef: np.ndarray, intercept: np.ndarray,
                  n_samples: int, local_acc: float, local_loss: float):
    """Kirim parameter model lokal ke server."""
    payload = {
        "client_id"      : CLIENT_ID,
        "round"          : round_num,
        "coef"           : coef.tolist(),
        "intercept"      : intercept.tolist(),
        "n_samples"      : n_samples,
        "local_accuracy" : local_acc,
        "local_loss"     : local_loss,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{SERVER_URL}/api/update",
                json=payload,
                timeout=30
            )
            if resp.status_code == 409:
                # Round mismatch — tunggu dan coba lagi
                logger.warning("Round mismatch, tunggu server...")
                time.sleep(RETRY_DELAY)
                continue

            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "Update round %d dikirim. Server: %d/%d update diterima",
                round_num, result.get("received", "?"), result.get("required", "?")
            )
            return result

        except requests.exceptions.RequestException as e:
            logger.warning("Gagal mengirim update (percobaan %d): %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)

    raise ConnectionError("Gagal mengirim update ke server")


# ─────────────────────────── Training Loop ─────────────────────────
def local_train(model: LogisticRegression,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                n_epochs: int):
    """
    Latih model lokal selama n_epochs.

    Catatan: sklearn LR menggunakan iterasi LBFGS, bukan epoch per batch.
    'epoch' di sini berarti iterasi pengulangan fit.
    """
    logger.info("  Memulai training lokal (%d iterasi)...", n_epochs)

    # Penting: sesuaikan max_iter dengan local epochs
    model.max_iter = n_epochs
    model.fit(X_train, y_train)

    # Evaluasi lokal
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)
    local_acc  = accuracy_score(y_test, y_pred)
    local_loss = log_loss(y_test, y_prob)

    logger.info("  Training lokal selesai — acc: %.4f, loss: %.4f",
                local_acc, local_loss)
    return local_acc, local_loss


def wait_for_next_round(expected_round: int):
    """Tunggu server pindah ke round berikutnya."""
    logger.info("Menunggu server memulai round %d...", expected_round)
    for _ in range(MAX_RETRIES * 2):
        try:
            resp = requests.get(f"{SERVER_URL}/api/status", timeout=10)
            data = resp.json()
            server_round = data.get("round", 0)
            phase        = data.get("phase", "")

            if phase == "done":
                return False   # training selesai
            if server_round == expected_round and phase == "waiting":
                return True

        except requests.exceptions.RequestException:
            pass
        time.sleep(RETRY_DELAY)

    return False


# ─────────────────────────── Main ──────────────────────────────────
def main():
    logger.info("=" * 50)
    logger.info("  FL Client %s dimulai", CLIENT_ID)
    logger.info("  Server : %s", SERVER_URL)
    logger.info("  Data   : %s", DATA_PATH)
    logger.info("=" * 50)

    # 1. Muat data lokal
    X_train, y_train, X_test, y_test = load_local_data(DATA_PATH)
    n_samples  = len(X_train)
    n_classes  = len(np.unique(y_train))
    n_features = X_train.shape[1]

    # Normalisasi fitur (scaler difit di data lokal saja — FL principle!)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 2. Daftar ke server
    config       = register_with_server()
    num_rounds   = config.get("num_rounds",   10)
    local_epochs = LOCAL_EPOCHS if LOCAL_EPOCHS > 0 else config.get("local_epochs", 5)

    logger.info("Training: %d rounds, %d epoch lokal per round", num_rounds, local_epochs)

    # 3. Buat model
    model = create_model(n_classes, max_iter=local_epochs)

    # 4. FL Training Loop
    for round_num in range(1, num_rounds + 1):
        logger.info("\n--- ROUND %d/%d ---", round_num, num_rounds)

        # a. Unduh model global
        model_data = fetch_global_model(round_num)
        global_coef      = np.array(model_data["coef"])
        global_intercept = np.array(model_data["intercept"])

        # b. Terapkan model global ke lokal
        model = set_model_params(model, global_coef, global_intercept,
                                 n_classes, n_features)

        # c. Training lokal
        local_acc, local_loss = local_train(
            model, X_train, y_train, X_test, y_test, local_epochs
        )

        # d. Ambil parameter hasil training
        coef, intercept = get_model_params(model)

        # e. Kirim update ke server
        submit_update(round_num, coef, intercept,
                      n_samples, local_acc, local_loss)

        # f. Tunggu server selesai agregasi dan mulai round berikutnya
        if round_num < num_rounds:
            ok = wait_for_next_round(round_num + 1)
            if not ok:
                logger.info("Training selesai (sinyal dari server)")
                break

    # 5. Evaluasi akhir dengan model global final
    logger.info("\n=== Training Selesai ===")
    try:
        resp = requests.get(f"{SERVER_URL}/api/final_model", timeout=15)
        if resp.status_code == 200:
            final_data  = resp.json()
            final_coef  = np.array(final_data["coef"])
            final_inter = np.array(final_data["intercept"])
            model = set_model_params(model, final_coef, final_inter,
                                     n_classes, n_features)

            y_pred    = model.predict(X_test)
            final_acc = accuracy_score(y_test, y_pred)
            logger.info("Akurasi akhir (model global) pada data lokal: %.4f", final_acc)
        else:
            logger.info("Model final belum tersedia (server belum selesai semua round)")
    except requests.exceptions.RequestException as e:
        logger.warning("Tidak bisa ambil model final: %s", e)


if __name__ == "__main__":
    main()

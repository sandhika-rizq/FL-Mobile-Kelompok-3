#!/usr/bin/env bash
# ============================================================
#  setup.sh — Instalasi otomatis semua dependencies
#  Jalankan: bash setup.sh
# ============================================================

set -e   # hentikan jika ada error

echo "=============================================="
echo "  Federated Learning — UCI HAR Setup"
echo "=============================================="

# Deteksi Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &>/dev/null; then
    PYTHON=python
    PIP=pip
else
    echo "[ERROR] Python tidak ditemukan. Install Python 3.8+ terlebih dahulu."
    exit 1
fi

echo ""
echo "Python: $($PYTHON --version)"
echo "Pip   : $($PIP --version)"
echo ""

# Buat virtual environment (opsional tapi disarankan)
if [ ! -d "venv" ]; then
    echo "[1/4] Membuat virtual environment..."
    $PYTHON -m venv venv
fi

# Aktifkan venv
echo "[2/4] Mengaktifkan virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install dependencies
echo "[3/4] Menginstal dependencies..."

# Common
$PIP install --upgrade pip -q
$PIP install numpy>=1.24.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 requests>=2.31.0 -q

# Server
$PIP install flask>=2.3.0 -q

echo "[4/4] Membuat direktori yang diperlukan..."
mkdir -p data/raw
mkdir -p data/partitions
mkdir -p evaluation/output
mkdir -p logs

echo ""
echo "=============================================="
echo "  Setup selesai!"
echo ""
echo "  Langkah selanjutnya:"
echo ""
echo "  [SERVER] Unduh dataset:"
echo "    python data/download_data.py"
echo ""
echo "  [SERVER] Buat partisi data:"
echo "    python data/partition_data.py --mode iid --num_clients 4"
echo ""
echo "  [SERVER] Jalankan FL server:"
echo "    FL_NUM_CLIENTS=4 FL_NUM_ROUNDS=10 python server/app.py"
echo ""
echo "  [CLIENT] Jalankan FL client:"
echo "    CLIENT_ID=1 SERVER_URL=http://<IP>:5000 python client/client.py"
echo ""
echo "  Aktifkan venv sebelum menjalankan:"
echo "    source venv/bin/activate"
echo "=============================================="

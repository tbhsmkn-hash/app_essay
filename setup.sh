#!/bin/bash
# setup.sh — jalankan sekali untuk setup environment proyek
# Cara pakai: bash setup.sh

set -e

echo "======================================================"
echo "  Setup: Automated Essay Scoring — Multimodal"
echo "======================================================"

# 1. Buat virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/5] Membuat virtual environment..."
    python3 -m venv .venv
else
    echo "[1/5] Virtual environment sudah ada, dilewati."
fi

source .venv/bin/activate

# 2. Upgrade pip
echo "[2/5] Upgrade pip..."
pip install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/5] Install library dari requirements.txt..."
pip install -r requirements.txt --quiet

# 4. Install PyTorch dengan CUDA 12.1 (sesuaikan dengan GPU kamu)
echo "[4/5] Install PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
# Jika tidak punya GPU, ganti baris di atas dengan:
# pip install torch torchvision --quiet

# 5. Buat struktur folder proyek
echo "[5/5] Membuat struktur folder..."
mkdir -p data/{images,texts,.cache}
mkdir -p models
mkdir -p app
mkdir -p notebooks
mkdir -p outputs/{checkpoints,logs,figures}

echo ""
echo "======================================================"
echo "  Setup selesai!"
echo ""
echo "  Aktivasi environment : source .venv/bin/activate"
echo "  Verifikasi           : python setup_env.py"
echo "  Download dataset     : lihat README — bagian Dataset"
echo "======================================================"

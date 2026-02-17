#!/usr/bin/env bash
set -euo pipefail

# Setup MusicGen environment for Apple Silicon (MPS).
# Requires: conda (miniconda/miniforge), Homebrew, ffmpeg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="musicgen"

echo "=== MusicGen Setup (Apple Silicon) ==="

# --- Conda init ---
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
    eval "$(/opt/homebrew/Caskroom/miniconda/base/bin/conda shell.bash hook)"
else
    echo "Error: conda not found. Install miniconda first."
    exit 1
fi

# --- Homebrew dependencies ---
echo "Checking Homebrew dependencies..."
for pkg in pkg-config ffmpeg; do
    if ! brew list "$pkg" &>/dev/null; then
        echo "Installing $pkg..."
        brew install "$pkg"
    fi
done

# --- Conda environment ---
if conda info --envs | grep -q "^${CONDA_ENV} "; then
    echo "Conda env '$CONDA_ENV' already exists, skipping creation."
else
    echo "Creating conda env '$CONDA_ENV' (Python 3.11)..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

conda activate "$CONDA_ENV"

# --- PyTorch (MPS build, no CUDA) ---
echo "Installing PyTorch (Apple Silicon)..."
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# --- audiocraft dependencies ---
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/src/3rd-party/requirements.txt"

# --- Output directory ---
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== Setup complete ==="
echo "Usage:"
echo "  ./generate.sh --prompt \"your prompt\" --duration 10"
echo "  ./batch_generate.sh --duration 10"

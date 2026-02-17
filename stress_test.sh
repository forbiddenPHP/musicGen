#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="musicgen"

# Init conda
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
    eval "$(/opt/homebrew/Caskroom/miniconda/base/bin/conda shell.bash hook)"
else
    echo "Error: conda not found." >&2; exit 1
fi

conda activate "$CONDA_ENV"

exec python "$SCRIPT_DIR/src/stress_test.py" "$@"

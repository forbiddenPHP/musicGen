#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="musicgen"
PID_FILE="$SCRIPT_DIR/.web.pid"
LOG_DIR="$SCRIPT_DIR/log"
LOG_FILE="$LOG_DIR/web.log"

# Init conda
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/conda" ]; then
    eval "$(/opt/homebrew/Caskroom/miniconda/base/bin/conda shell.bash hook)"
else
    echo "Error: conda not found." >&2; exit 1
fi

conda activate "$CONDA_ENV"

ACTION="${1:-start}"

case "$ACTION" in
    --start|start)
        # Check if already running
        if [ -f "$PID_FILE" ]; then
            OLD_PID=$(cat "$PID_FILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "Already running (PID $OLD_PID)"
                echo ">>> http://localhost:7777/musicGen <<<"
                exit 0
            fi
            rm -f "$PID_FILE"
        fi

        echo "Starting MusicGen web server..."
        mkdir -p "$LOG_DIR"
        nohup python "$SCRIPT_DIR/src/web.py" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "PID $(cat "$PID_FILE") — log: $LOG_FILE"
        echo ">>> http://localhost:7777/musicGen <<<"
        ;;

    --stop|stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "Not running (no PID file)"
            exit 0
        fi
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Stopped (PID $PID)"
        else
            echo "Process $PID already dead"
        fi
        rm -f "$PID_FILE"
        ;;

    *)
        echo "Usage: ./web.sh [--start|--stop]" >&2
        exit 1
        ;;
esac

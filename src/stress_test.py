#!/usr/bin/env python3
"""Stress test: generate 1s..30s and log memory usage per step."""

import argparse
import gc
import sys
import time
from pathlib import Path

import soundfile as sf

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "3rd-party"))

OUTPUT_DIR = _project_root.parent / "output"

import torch
from audiocraft.models import MusicGen
from generate import apply_fade

DEFAULT_PROMPT = "upbeat pop song with catchy melody and drums"
MODEL_NAME = "facebook/musicgen-large"


def get_mem_mb():
    """Get current process RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS returns bytes


def get_mps_allocated_mb():
    """Get MPS allocated memory in MB (if available)."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            return -1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=30, help="Max duration in seconds (default: 30)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"Prompt (default: '{DEFAULT_PROMPT}')")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"{'Duration':>8} | {'Time':>8} | {'RSS MB':>10} | {'MPS MB':>10} | Status")
    print("-" * 62)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model '{MODEL_NAME}'...")

    model = MusicGen.get_pretrained(MODEL_NAME, device=device)
    mem_after_load = get_mem_mb()
    mps_after_load = get_mps_allocated_mb()
    print(f"Model loaded. RSS: {mem_after_load:.0f} MB, MPS: {mps_after_load:.0f} MB\n")

    for duration in range(1, args.max + 1):
        model.set_generation_params(duration=float(duration))

        start = time.time()
        try:
            with torch.no_grad():
                wav = model.generate([args.prompt])

            # Save WAV file
            audio = wav[0].cpu().numpy()
            if audio.ndim == 2:
                audio = audio.T
            apply_fade(audio, model.sample_rate)
            filepath = OUTPUT_DIR / f"stresstest_{duration:02d}s.wav"
            sf.write(str(filepath), audio, samplerate=model.sample_rate)

            del wav, audio
            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            elapsed = time.time() - start
            rss = get_mem_mb()
            mps = get_mps_allocated_mb()

            print(f"{duration:>6}s  | {elapsed:>6.1f}s  | {rss:>8.0f}  | {mps:>8.0f}  | OK -> {filepath.name}")

        except Exception as e:
            elapsed = time.time() - start
            rss = get_mem_mb()
            mps = get_mps_allocated_mb()
            print(f"{duration:>6}s  | {elapsed:>6.1f}s  | {rss:>8.0f}  | {mps:>8.0f}  | FAIL: {e}")

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()

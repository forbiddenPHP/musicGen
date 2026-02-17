#!/usr/bin/env python3
"""Sequenced generation: generate once, save cumulative snapshots per second."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

# Use patched audiocraft from 3rd-party/
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "3rd-party"))

from generate import get_device, load_model, apply_fade


def generate_sequenced(prompt: str, duration: float, model_name: str,
                       output_dir: str, force_device: str | None = None):
    device = get_device(force_device)

    print(f"Loading model '{model_name}'...")
    model = load_model(model_name, device)
    model.set_generation_params(duration=duration)

    print(f"Generating {duration}s of audio for: '{prompt}'")
    with torch.no_grad():
        wav, tokens = model.generate([prompt], return_tokens=True)

    # Free generation memory before snapshot decoding
    del wav
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # tokens shape: [B, K, T] — B=batch, K=codebooks, T=timesteps
    tokens_per_sec = int(model.frame_rate)
    total_tokens = tokens.shape[-1]
    total_secs = int(duration)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nSaving {total_secs} snapshots ({tokens_per_sec} tokens/s, {total_tokens} total tokens):\n")

    for sec in range(1, total_secs + 1):
        t = min(sec * tokens_per_sec, total_tokens)
        partial = tokens[:, :, :t]

        with torch.no_grad():
            audio_tensor = model.generate_audio(partial)

        audio = audio_tensor[0].cpu().numpy()
        if audio.ndim == 2:
            audio = audio.T
        apply_fade(audio, model.sample_rate)

        filepath = out / f"musicgen_{timestamp}_sequenced_{sec:02d}s.wav"
        sf.write(str(filepath), audio, samplerate=model.sample_rate)
        print(f"  {sec:>2}s  ({t:>4} tokens) -> {filepath.name}")

        del audio_tensor, audio
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\nDone. {total_secs} snapshots saved to {out}/")


def main():
    parser = argparse.ArgumentParser(description="Generate music with per-second snapshots")
    parser.add_argument("--prompt", required=True, help="Text prompt for music generation")
    parser.add_argument("--duration", type=float, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--model", default="facebook/musicgen-large", help="Model name (default: facebook/musicgen-large)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--device", choices=["mps", "cpu"], default=None, help="Force device (default: auto-detect)")
    args = parser.parse_args()

    generate_sequenced(args.prompt, args.duration, args.model, args.output, args.device)


if __name__ == "__main__":
    main()

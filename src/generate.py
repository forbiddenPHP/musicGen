#!/usr/bin/env python3
"""Generate music from a text prompt using MusicGen on Apple Silicon (MPS)."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Use patched audiocraft from 3rd-party/ (Apple Silicon MPS fixes)
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "3rd-party"))

import numpy as np
import soundfile as sf
import torch
from audiocraft.models import MusicGen

FADE_DURATION = 0.5  # seconds


def apply_fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply linear fade-in and fade-out (0.5s each) to audio.

    Args:
        audio: numpy array, shape (samples,) or (samples, channels).
        sample_rate: audio sample rate in Hz.
    Returns:
        Audio with fades applied (in-place).
    """
    fade_samples = int(FADE_DURATION * sample_rate)
    if fade_samples == 0 or len(audio) < fade_samples * 2:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    if audio.ndim == 2:
        fade_in = fade_in[:, np.newaxis]
        fade_out = fade_out[:, np.newaxis]

    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def get_device(force_device: str | None = None):
    if force_device:
        print(f"Using {force_device.upper()} (forced)")
        return force_device
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return "mps"
    print("MPS not available, falling back to CPU")
    return "cpu"


def load_model(model_name: str, device: str) -> MusicGen:
    try:
        return MusicGen.get_pretrained(model_name, device=device)
    except Exception as e:
        if device == "mps":
            print(f"MPS load failed ({e}), retrying on CPU...")
            return MusicGen.get_pretrained(model_name, device="cpu")
        raise


def generate(prompt: str, duration: float, model_name: str, output_dir: str, force_device: str | None = None) -> str:
    device = get_device(force_device)

    print(f"Loading model '{model_name}'...")
    try:
        model = load_model(model_name, device)
    except Exception as e:
        print(f"Error loading '{model_name}': {e}")
        print("Falling back to 'facebook/musicgen-medium'...")
        model = load_model("facebook/musicgen-medium", device)

    model.set_generation_params(duration=duration)

    print(f"Generating {duration}s of audio for: '{prompt}'")
    try:
        with torch.no_grad():
            wav = model.generate([prompt])
    except Exception as e:
        if device == "mps":
            print(f"MPS generation failed ({e}), retrying on CPU...")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            model = load_model(model_name, "cpu")
            model.set_generation_params(duration=duration)
            with torch.no_grad():
                wav = model.generate([prompt])
        else:
            raise

    # Free MPS cache after generation
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # wav shape: (batch, channels, samples)
    audio = wav[0].cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T  # (samples, channels) for soundfile
    apply_fade(audio, model.sample_rate)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"musicgen_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)

    sf.write(filepath, audio, samplerate=model.sample_rate)
    print(f"Saved: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate music with MusicGen")
    parser.add_argument("--prompt", required=True, help="Text prompt for music generation")
    parser.add_argument("--duration", type=float, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--model", default="facebook/musicgen-large", help="Model name (default: facebook/musicgen-large)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--device", choices=["mps", "cpu"], default=None, help="Force device (default: auto-detect)")
    args = parser.parse_args()

    generate(args.prompt, args.duration, args.model, args.output, args.device)


if __name__ == "__main__":
    main()

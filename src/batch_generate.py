#!/usr/bin/env python3
"""Batch generate music from a prompts file using MusicGen."""

import argparse
import sys
from pathlib import Path

from generate import generate


def main():
    parser = argparse.ArgumentParser(description="Batch generate music from a prompts file")
    _src_dir = Path(__file__).resolve().parent
    parser.add_argument("--prompts", default=str(_src_dir / "prompts.txt"), help="Path to prompts file (default: src/prompts.txt)")
    parser.add_argument("--duration", type=float, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--model", default="facebook/musicgen-large", help="Model name (default: facebook/musicgen-large)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--device", choices=["mps", "cpu"], default=None, help="Force device (default: auto-detect)")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"Error: prompts file not found: {prompts_path}")
        sys.exit(1)

    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip() and not line.startswith("#")]

    if not prompts:
        print("No prompts found in file.")
        sys.exit(1)

    print(f"Found {len(prompts)} prompts. Starting batch generation...")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        try:
            generate(prompt, args.duration, args.model, args.output, args.device)
        except Exception as e:
            print(f"Error generating '{prompt}': {e}")
            continue

    print("\nBatch generation complete.")


if __name__ == "__main__":
    main()

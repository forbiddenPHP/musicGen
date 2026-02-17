# MusicGen Local (Apple Silicon)

Local AI music generation using Meta's MusicGen on macOS Apple Silicon (MPS backend).

## Requirements

- macOS with Apple Silicon (M-series)
- [Homebrew](https://brew.sh)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Miniforge

## Setup

```console
./setup.sh
```

This script handles everything automatically:
- Installs `pkg-config` and `ffmpeg` via Homebrew (if missing)
- Creates the `musicgen` Conda environment (Python 3.11)
- Installs PyTorch with MPS support and all other dependencies

**audiocraft is NOT installed via pip.** A patched version ships with this repo under `src/3rd-party/audiocraft/` and is imported directly from there.

## Usage

### Single generation

```console
./generate.sh --prompt "upbeat chiptune melody, 8-bit retro style" --duration 10
```

Options:
- `--prompt` (required) — text description of the desired music
- `--duration` — length in seconds (default: 10)
- `--model` — model name (default: `facebook/musicgen-large`)
- `--output` — output directory (default: `./output/`)
- `--device` — force `mps` or `cpu` (default: auto-detect)

### Batch generation

```console
./batch_generate.sh --duration 15
```

Reads prompts from `prompts.txt` (one per line, `#` for comments) and generates one WAV file per prompt.

Options:
- `--prompts` — path to prompts file (default: `src/prompts.txt`)
- `--duration`, `--model`, `--output` — same as above

### Sequenced generation

```console
./generate_sequenced.sh --prompt "90s TV jingle, bright synth melody" --duration 10
```

Generates a single piece of music and saves cumulative per-second snapshots. The same audio, growing from 1 second to the full duration:

```
output/musicgen_20260217_192045_sequenced_01s.wav   (first second)
output/musicgen_20260217_192045_sequenced_02s.wav   (first two seconds)
...
output/musicgen_20260217_192045_sequenced_10s.wav   (full piece)
```

Options: same as single generation (`--prompt`, `--duration`, `--model`, `--output`, `--device`).

### Web interface

```console
./web.sh              # start in background (default)
./web.sh --stop       # stop the server
```

Starts a local web app in the background at [http://localhost:7777/musicGen](http://localhost:7777/musicGen). The model is loaded once at startup and stays in memory — subsequent generations are much faster. Enter a prompt, pick a duration, hit Generate, and listen in the browser. Log output goes to `log/web.log`.

### Output

WAV files are saved to `./output/` with timestamped filenames:
```
output/musicgen_20260217_180735.wav
```

## Project structure

```
├── setup.sh                 # Creates Conda env + installs dependencies
├── generate.sh              # Shell wrapper (handles Conda activation)
├── generate_sequenced.sh    # Shell wrapper for sequenced generation
├── batch_generate.sh        # Shell wrapper for batch mode
├── web.sh                   # Web server (--start/--stop, runs in background)
├── stress_test.sh           # Shell wrapper for stress test
├── readme.md
├── sources.md
├── src/
│   ├── generate.py          # Generation script
│   ├── generate_sequenced.py # Sequenced generation (per-second snapshots)
│   ├── batch_generate.py    # Batch generation script
│   ├── web.py               # Flask web app (persistent model)
│   ├── stress_test.py       # Memory stress test (1s..30s)
│   ├── prompts.txt          # Example prompts
│   ├── environment.yml      # Conda environment definition
│   └── 3rd-party/
│       ├── audiocraft/      # Patched audiocraft library (MPS-compatible)
│       └── requirements.txt # Pip dependencies
└── output/                  # Generated WAV files
```

## audiocraft patches

The original audiocraft library (v1.4.0a2) requires `xformers` and `spacy`, neither of which is readily available on Apple Silicon. The local copy in `src/3rd-party/audiocraft/` contains the following modifications:

**`modules/transformer.py`**
- `from xformers import ops` → conditional import, falls back to `None`
- `_verify_xformers_memory_efficient_compat()` is skipped when xformers is absent
- `ops.unbind()` → fallback to `torch.unbind()`
- `LowerTriangularMask` → replaced with native `is_causal=True` in PyTorch SDPA
- `ops.memory_efficient_attention()` → fallback to `torch.nn.functional.scaled_dot_product_attention()`

**`modules/conditioners.py`**
- `import spacy` → conditional import, falls back to `None`

**`models/genmodel.py`**
- `generate_audio()` → EnCodec decoder runs on CPU when device is MPS (FFT ops are not supported on MPS and `PYTORCH_ENABLE_MPS_FALLBACK` causes memory leaks)

**`models/musicgen.py`**
- `get_pretrained()` → `max_duration=10` on MPS to enable sliding window chunked generation (avoids OOM from MPS attention memory spikes on longer sequences)
- `set_generation_params()` → `extend_stride` defaults to `max_duration - 6` (6s overlap per chunk for musical continuity)

## Stress test results

Mac Studio M2 Ultra, 192 GB RAM, `musicgen-large`, prompt: "upbeat pop song with catchy melody and drums":

```
./stress_test.sh --max 30

Duration |     Time |     RSS MB |     MPS MB | Status
--------------------------------------------------------------
     1s  |    2.2s  |     6932  |     7115  | OK
     2s  |    3.3s  |     6932  |     7116  | OK
     3s  |    4.9s  |     6932  |     7116  | OK
     4s  |    6.5s  |     6932  |     7116  | OK
     5s  |    8.6s  |     6932  |     7116  | OK
     6s  |   10.8s  |     6932  |     7116  | OK
     7s  |   13.2s  |     6932  |     7116  | OK
     8s  |   15.6s  |     6932  |     7116  | OK
     9s  |   18.9s  |     6932  |     7116  | OK
    10s  |   22.1s  |     6932  |     7117  | OK
    11s  |   26.7s  |     6932  |     7117  | OK
    12s  |   30.9s  |     6981  |     7117  | OK
    13s  |   33.0s  |     7148  |     7117  | OK
    14s  |   37.8s  |     7336  |     7117  | OK
    15s  |   42.8s  |     7530  |     7117  | OK
    16s  |   45.1s  |     7728  |     7117  | OK
    17s  |   49.6s  |     7947  |     7117  | OK
    18s  |   51.9s  |     8040  |     7117  | OK
    19s  |   57.0s  |     8158  |     7117  | OK
    20s  |   61.3s  |     8277  |     7117  | OK
    21s  |   63.6s  |     8396  |     7117  | OK
    22s  |   69.3s  |     8527  |     7117  | OK
    23s  |   73.0s  |     8659  |     7117  | OK
    24s  |   75.2s  |     8786  |     7117  | OK
    25s  |   79.5s  |     8925  |     7117  | OK
    26s  |   84.3s  |     9061  |     7117  | OK
    27s  |   88.5s  |     9203  |     7117  | OK
    28s  |   88.8s  |     9352  |     7117  | OK
    29s  |   94.3s  |     9499  |     7117  | OK
    30s  |   98.1s  |     9650  |     7117  | OK
```

MPS memory stays stable at ~7117 MB. RSS grows linearly with audio length (decoded WAV data on CPU). Sliding window kicks in at >10s with 6s overlap per chunk.

## Notes

- On the first run the model is downloaded from HuggingFace (~3.3 GB for `musicgen-large`).
- Token generation runs on MPS (fast), audio decoding runs on CPU (avoids FFT/MPS issues). This hybrid approach prevents memory leaks without any noticeable slowdown.
- For durations >10s, a sliding window generates audio in 10s chunks with 6s overlap for seamless transitions.
- If MPS fails entirely, the script falls back to CPU gracefully.

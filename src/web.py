#!/usr/bin/env python3
"""MusicGen web interface — keeps model in memory for fast generation."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Use patched audiocraft from 3rd-party/ (Apple Silicon MPS fixes)
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "3rd-party"))

import torch
import soundfile as sf
from flask import Flask, request, send_file, jsonify

from generate import get_device, load_model

app = Flask(__name__)

OUTPUT_DIR = _project_root.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Model (loaded once at startup) ---
MODEL = None
MODEL_NAME = "facebook/musicgen-large"


def get_model():
    global MODEL
    if MODEL is None:
        device = get_device()
        print(f"Loading model '{MODEL_NAME}' (this only happens once)...")
        MODEL = load_model(MODEL_NAME, device)
    return MODEL


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MusicGen</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { width: 100%; max-width: 520px; padding: 2rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1.5rem; color: #fff; }
  label { display: block; font-size: 0.85rem; color: #999; margin-bottom: 0.3rem; }
  input[type=text], input[type=range] { width: 100%; }
  input[type=text] { background: #1a1a1a; border: 1px solid #333; color: #fff; padding: 0.6rem 0.8rem; border-radius: 6px; font-size: 0.95rem; margin-bottom: 1rem; }
  input[type=text]:focus { outline: none; border-color: #666; }
  .range-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem; }
  .range-row input[type=range] { flex: 1; }
  .range-val { font-size: 0.95rem; color: #fff; min-width: 2.5rem; text-align: right; }
  button { width: 100%; padding: 0.7rem; background: #2563eb; color: #fff; border: none; border-radius: 6px; font-size: 0.95rem; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; cursor: wait; }
  #status { margin-top: 1rem; font-size: 0.85rem; color: #999; min-height: 1.2rem; }
  #player { margin-top: 1rem; display: none; }
  audio { width: 100%; }
  #history { margin-top: 2rem; }
  #history h2 { font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; }
  .hist-item { font-size: 0.8rem; color: #555; padding: 0.3rem 0; border-bottom: 1px solid #1a1a1a; cursor: pointer; }
  .hist-item:hover { color: #999; }
</style>
</head>
<body>
<div class="container">
  <h1>MusicGen</h1>
  <form id="form">
    <label for="prompt">Prompt</label>
    <input type="text" id="prompt" name="prompt" placeholder="upbeat chiptune melody, 8-bit retro style" required>
    <label for="duration">Duration</label>
    <div class="range-row">
      <input type="range" id="duration" name="duration" min="3" max="30" value="10" step="1">
      <span class="range-val" id="dur-val">10s</span>
    </div>
    <button type="submit" id="btn">Generate</button>
  </form>
  <div id="status"></div>
  <div id="player"><audio id="audio" controls></audio></div>
  <div id="history"><h2>History</h2><div id="hist-list"></div></div>
</div>
<script>
const form = document.getElementById('form');
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const player = document.getElementById('player');
const audio = document.getElementById('audio');
const durSlider = document.getElementById('duration');
const durVal = document.getElementById('dur-val');
const histList = document.getElementById('hist-list');

durSlider.addEventListener('input', () => durVal.textContent = durSlider.value + 's');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  btn.disabled = true;
  status.textContent = 'Generating...';
  player.style.display = 'none';
  const start = Date.now();
  try {
    const res = await fetch('/musicGen/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, duration: parseFloat(durSlider.value)})
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.error || 'Generation failed'); }
    const data = await res.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    status.textContent = `Done in ${elapsed}s — ${data.filename}`;
    audio.src = '/musicGen/output/' + data.filename;
    player.style.display = 'block';
    audio.play();
    addHistory(prompt, data.filename);
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
  }
});

function addHistory(prompt, filename) {
  const div = document.createElement('div');
  div.className = 'hist-item';
  div.textContent = prompt;
  div.addEventListener('click', () => {
    audio.src = '/musicGen/output/' + filename;
    player.style.display = 'block';
    audio.play();
  });
  histList.prepend(div);
}
</script>
</body>
</html>"""


@app.route("/musicGen")
def index():
    return HTML


@app.route("/musicGen/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    duration = float(data.get("duration", 10))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if duration < 1 or duration > 60:
        return jsonify({"error": "duration must be between 1 and 60"}), 400

    model = get_model()
    model.set_generation_params(duration=duration)

    try:
        with torch.no_grad():
            wav = model.generate([prompt])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    audio_data = wav[0].cpu().numpy()
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"musicgen_{timestamp}.wav"
    filepath = OUTPUT_DIR / filename

    sf.write(str(filepath), audio_data, samplerate=model.sample_rate)
    return jsonify({"filename": filename})


@app.route("/musicGen/output/<filename>")
def serve_output(filename):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or not filepath.name.endswith(".wav"):
        return "Not found", 404
    return send_file(filepath, mimetype="audio/wav")


if __name__ == "__main__":
    get_model()  # pre-load on startup
    print("\n>>> http://localhost:7777/musicGen <<<\n")
    app.run(host="0.0.0.0", port=7777, debug=False)

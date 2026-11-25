"""
gen_sounds.py
-------------
Reads events.json, generates 2 audio variants (seeds 0 and 1) for each event
using AudioLDM on CUDA in float32, saves unnormalized (clipped [-1, 1]) 16-bit PCM WAVs
to outputs/sounds/, and computes the strongest onset time using librosa.
"""

import os
import sys
import json
import torch
import numpy as np
import scipy.io.wavfile
import librosa
from diffusers import AudioLDMPipeline

# ---------------------------------------------------------------------------
# 1. Configuration & Load Events
# ---------------------------------------------------------------------------
EVENTS_FILE = "events.json"
MODEL_ID = "cvssp/audioldm-s-full-v2"
OUTPUT_DIR = os.path.join("outputs", "sounds")
SAMPLE_RATE = 16000
SEEDS = [0, 1]

# Generation hyperparameters
AUDIO_LENGTH_S = 5.12
NUM_STEPS = 50
NEGATIVE_PROMPT = "low quality"
GUIDANCE_SCALE = 3.5

if not os.path.exists(EVENTS_FILE):
    sys.exit(f"[error] {EVENTS_FILE} not found in project root.")

with open(EVENTS_FILE, "r", encoding="utf-8") as f:
    events_data = json.load(f)

events = events_data.get("events", [])
print(f"Loaded {len(events)} events from {EVENTS_FILE}.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load AudioLDM Pipeline (float32 on cuda)
# ---------------------------------------------------------------------------
print(f"\nLoading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...")
pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")
print("Pipeline loaded successfully.\n")

# ---------------------------------------------------------------------------
# 3. Generate Audio Variants
# ---------------------------------------------------------------------------
generated_files = []

for ev in events:
    ev_id = ev["id"]
    ev_type = ev["type"]
    prompt = ev["prompt"]

    print(f"--- Event {ev_id} ({ev_type}) ---")
    print(f"  Prompt: \"{prompt}\"")

    for seed in SEEDS:
        filename = f"event{ev_id}_{ev_type}_v{seed}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"  Generating seed {seed} -> {filename} ...")
        generator = torch.Generator("cuda").manual_seed(seed)

        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            audio_length_in_s=AUDIO_LENGTH_S,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )

        audio = out.audios[0] if hasattr(out, "audios") else out[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.squeeze(audio)

        # Handle NaNs if any, clip to [-1.0, 1.0], convert to 16-bit PCM
        audio_clean = np.nan_to_num(audio, nan=0.0)
        audio_clipped = np.clip(audio_clean, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_int16)
        generated_files.append((filename, filepath))

print("\nAll audio variants generated successfully.")

# ---------------------------------------------------------------------------
# 4. Onset Analysis using Librosa
# ---------------------------------------------------------------------------
print("\nAnalyzing onsets and signal stats with librosa...")
table_rows = []

for filename, filepath in generated_files:
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    min_val = float(np.min(y))
    max_val = float(np.max(y))
    std_val = float(np.std(y))

    # Detect onset envelope and candidate onset frames
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        units="frames",
    )

    if len(onset_frames) > 0:
        strengths = onset_env[onset_frames]
        strongest_frame = onset_frames[np.argmax(strengths)]
        onset_time = float(librosa.frames_to_time(strongest_frame, sr=sr))
    else:
        strongest_frame = int(np.argmax(onset_env))
        onset_time = float(librosa.frames_to_time(strongest_frame, sr=sr))

    table_rows.append({
        "filename": filename,
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "onset_time": onset_time,
    })

# ---------------------------------------------------------------------------
# 5. Print Results Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'File Name':<30} {'Min':<12} {'Max':<12} {'Std Dev':<12} {'Strongest Onset':<16}")
print("-" * 80)
for r in table_rows:
    print(
        f"{r['filename']:<30} "
        f"{r['min']:<12.4f} "
        f"{r['max']:<12.4f} "
        f"{r['std']:<12.4f} "
        f"{r['onset_time']:<16.3f}s"
    )
print("=" * 80)


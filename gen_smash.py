"""
gen_smash.py
------------
Generates glass smash sound candidates using AudioLDM (float32 on cuda).
Tests two prompts with seeds 0, 1, 2. Analyzes onset strength and frequency,
removes FLAT files (std < 0.005), and prints a summary evaluation table.
"""

import os
import sys
import torch
import numpy as np
import scipy.io.wavfile
import librosa
from diffusers import AudioLDMPipeline

# ---------------------------------------------------------------------------
# 1. Configuration & Prompts
# ---------------------------------------------------------------------------
MODEL_ID = "cvssp/audioldm-s-full-v2"
OUTPUT_DIR = os.path.join("outputs", "sounds")
SAMPLE_RATE = 16000
SEEDS = [0, 1, 2]

AUDIO_LENGTH_S = 5.12
NUM_STEPS = 50
NEGATIVE_PROMPT = "low quality"
GUIDANCE_SCALE = 3.5
STD_THRESHOLD = 0.005

PROMPTS = [
    (1, "glass bottle shattering into many pieces, loud crash, breaking glass"),
    (2, "drinking glass smashing on a hard table and breaking into shards, crashing glass"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load Pipeline
# ---------------------------------------------------------------------------
print(f"Loading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...")
pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")
print("Pipeline loaded successfully.\n")

# ---------------------------------------------------------------------------
# 3. Generation & Evaluation Loop
# ---------------------------------------------------------------------------
results = []

for p_num, prompt_text in PROMPTS:
    print(f"==================================================")
    print(f"Prompt P{p_num}: \"{prompt_text}\"")
    print(f"==================================================")

    for seed in SEEDS:
        filename = f"smash_P{p_num}_s{seed}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"\nGenerating P{p_num} Seed {seed} -> {filename} ...")
        generator = torch.Generator("cuda").manual_seed(seed)

        out = pipe(
            prompt=prompt_text,
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

        # Clean NaNs, clip to [-1, 1], save 16-bit PCM WAV
        audio_clean = np.nan_to_num(audio, nan=0.0)
        audio_clipped = np.clip(audio_clean, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_int16)

        # Analyze using librosa
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        std_val = float(np.std(y))

        # Onset envelope and detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            onset_envelope=onset_env,
            units="frames",
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Onset count in first 3 seconds
        onset_count_3s = int(np.sum(onset_times <= 3.0))

        # Time of strongest onset
        if len(onset_frames) > 0:
            strengths = onset_env[onset_frames]
            strongest_frame = onset_frames[np.argmax(strengths)]
            strongest_onset_time = float(librosa.frames_to_time(strongest_frame, sr=sr))
        else:
            strongest_frame = int(np.argmax(onset_env))
            strongest_onset_time = float(librosa.frames_to_time(strongest_frame, sr=sr))

        # Filter: check if FLAT
        if std_val < STD_THRESHOLD:
            status = "FLAT"
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"  [FLAT] std={std_val:.6f} < {STD_THRESHOLD} (removed {filename})")
        else:
            status = "OK"
            print(f"  [OK] std={std_val:.4f}, strongest_onset={strongest_onset_time:.3f}s, onsets<=3s={onset_count_3s}")

        results.append({
            "filename": filename,
            "std": std_val,
            "strongest_onset": strongest_onset_time,
            "onsets_3s": onset_count_3s,
            "status": status,
        })

# ---------------------------------------------------------------------------
# 4. Print Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'File Name':<22} {'Std Dev':<12} {'Strongest Onset':<18} {'Onsets <= 3s':<16} {'Status':<8}")
print("-" * 80)
for r in results:
    onset_str = f"{r['strongest_onset']:.3f} s"
    print(
        f"{r['filename']:<22} "
        f"{r['std']:<12.4f} "
        f"{onset_str:<18} "
        f"{r['onsets_3s']:<16} "
        f"{r['status']:<8}"
    )
print("=" * 80)


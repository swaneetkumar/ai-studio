"""
gen_door_test.py
----------------
Generates test audio for door slam and subtle sounds using AudioLDM (float32 on cuda)
with validated settings (5.12s, 50 steps, guidance_scale 3.5, negative_prompt="low quality").
Saves to outputs/door_test/ and prints statistics to evaluate signal quality and clipping.
"""

import os
import torch
import numpy as np
import scipy.io.wavfile
from diffusers import AudioLDMPipeline

MODEL_ID = "cvssp/audioldm-s-full-v2"
OUTPUT_DIR = os.path.join("outputs", "door_test")
SAMPLE_RATE = 16000
SEEDS = [0, 1, 2]

AUDIO_LENGTH_S = 5.12
NUM_STEPS = 50
NEGATIVE_PROMPT = "low quality"
GUIDANCE_SCALE = 3.5

PROMPTS = {
    "A": "door slam",
    "B": "a heavy wooden door slamming shut",
    "C": "footsteps on gravel",
    "D": "cloth rustling",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...", flush=True)
pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to("cuda")
print("Pipeline loaded successfully.\n", flush=True)

results = []

for key, prompt in PROMPTS.items():
    print(f"Generating for Prompt {key}: \"{prompt}\"...", flush=True)
    for seed in SEEDS:
        filename = f"door_{key}_s{seed}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

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

        audio_clean = np.nan_to_num(audio, nan=0.0)

        std_dev = float(np.std(audio_clean))
        min_val = float(np.min(audio_clean))
        max_val = float(np.max(audio_clean))
        clip_frac = float(np.mean(np.abs(audio_clean) > 0.99))

        # Clip to [-1, 1], no normalization
        audio_clipped = np.clip(audio_clean, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_int16)

        results.append({
            "prompt_key": key,
            "prompt": prompt,
            "filename": filename,
            "std_dev": std_dev,
            "min_val": min_val,
            "max_val": max_val,
            "clip_frac": clip_frac,
        })
        print(f"  Saved {filename} (std={std_dev:.4f}, min={min_val:.4f}, max={max_val:.4f}, clip_frac={clip_frac:.6f})", flush=True)

print("\n" + "=" * 90, flush=True)
print(f"{'File Name':<18} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Clip Frac (|x|>0.99)':<22} {'Prompt'}", flush=True)
print("-" * 90, flush=True)
for r in results:
    print(
        f"{r['filename']:<18} "
        f"{r['std_dev']:<12.4f} "
        f"{r['min_val']:<12.4f} "
        f"{r['max_val']:<12.4f} "
        f"{r['clip_frac']:<22.6f} "
        f"\"{r['prompt']}\"",
        flush=True
    )
print("=" * 90 + "\n", flush=True)


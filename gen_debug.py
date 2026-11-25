"""
gen_debug.py
------------
Runs 4 AudioLDM configurations with a fixed seed to diagnose output amplitude,
shape, and distribution differences across audio length, guidance scale, and output_type.
"""

import os
import torch
import numpy as np
import scipy.io.wavfile
from diffusers import AudioLDMPipeline

# ---------------------------------------------------------------------------
# Setup & Model Loading
# ---------------------------------------------------------------------------
MODEL_ID = "cvssp/audioldm-s-full-v2"
PROMPT = "a glass cup shattering on a table"
SAMPLE_RATE = 16000
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...")
pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")
print("Model loaded successfully.\n")

# ---------------------------------------------------------------------------
# Configurations to Test
# ---------------------------------------------------------------------------
configs = [
    {
        "name": "A",
        "description": "5.12s, 50 steps, default guidance",
        "kwargs": {
            "audio_length_in_s": 5.12,
            "num_inference_steps": 50,
        },
        "output_file": os.path.join(OUTPUT_DIR, "debug_A.wav"),
    },
    {
        "name": "B",
        "description": "2.0s, 50 steps, default guidance",
        "kwargs": {
            "audio_length_in_s": 2.0,
            "num_inference_steps": 50,
        },
        "output_file": os.path.join(OUTPUT_DIR, "debug_B.wav"),
    },
    {
        "name": "C",
        "description": "5.12s, 50 steps, neg_prompt='low quality', guidance=3.5",
        "kwargs": {
            "audio_length_in_s": 5.12,
            "num_inference_steps": 50,
            "negative_prompt": "low quality",
            "guidance_scale": 3.5,
        },
        "output_file": os.path.join(OUTPUT_DIR, "debug_C.wav"),
    },
    {
        "name": "D",
        "description": "5.12s, 50 steps, output_type='np'",
        "kwargs": {
            "audio_length_in_s": 5.12,
            "num_inference_steps": 50,
            "output_type": "np",
        },
        "output_file": os.path.join(OUTPUT_DIR, "debug_D.wav"),
    },
]

results = []

for cfg in configs:
    name = cfg["name"]
    print(f"Running Config {name}: {cfg['description']}...")

    generator = torch.Generator("cuda").manual_seed(0)
    out = pipe(
        prompt=PROMPT,
        generator=generator,
        **cfg["kwargs"],
    )

    audio = out.audios[0] if hasattr(out, "audios") else out[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.squeeze(audio)

    # Compute statistics
    has_nan = bool(np.isnan(audio).any())
    min_val = float(np.nanmin(audio)) if not np.all(np.isnan(audio)) else float("nan")
    max_val = float(np.nanmax(audio)) if not np.all(np.isnan(audio)) else float("nan")
    std_val = float(np.nanstd(audio)) if not np.all(np.isnan(audio)) else float("nan")

    results.append({
        "config": name,
        "shape": str(audio.shape),
        "min": min_val,
        "max": max_val,
        "std": std_val,
        "has_nan": has_nan,
        "desc": cfg["description"],
    })

    # Save without normalization: clip to [-1.0, 1.0], convert to 16-bit PCM
    clean_audio = np.nan_to_num(audio, nan=0.0)
    audio_clipped = np.clip(clean_audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    scipy.io.wavfile.write(cfg["output_file"], rate=SAMPLE_RATE, data=audio_int16)
    print(f"  Saved -> {cfg['output_file']}")

# ---------------------------------------------------------------------------
# Output Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'Config':<8} {'Shape':<12} {'Min':<14} {'Max':<14} {'Std Dev':<14} {'Has NaN':<10} {'Description'}")
print("-" * 90)
for r in results:
    print(f"{r['config']:<8} {r['shape']:<12} {r['min']:<14.6f} {r['max']:<14.6f} {r['std']:<14.6f} {str(r['has_nan']):<10} {r['desc']}")
print("=" * 90)


"""
gen_test.py
-----------
Tests AudioLDM pipeline from diffusers on GPU using torch.float32.
Generates sound effect from a text prompt, checks and normalizes audio,
converts to 16-bit PCM, and saves to outputs/test_smash.wav.
"""

import os
import time
import torch
import numpy as np
import scipy.io.wavfile
from diffusers import AudioLDMPipeline

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "cvssp/audioldm-s-full-v2"
PROMPT = "single drinking glass shattering on a dark wooden table, sharp impact, indoor room"
NEGATIVE_PROMPT = "low quality, noise, music, speech"
AUDIO_LENGTH_S = 2.0
NUM_INFERENCE_STEPS = 25
SAMPLE_RATE = 16000
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_smash.wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load AudioLDM Pipeline in float32 on CUDA
# ---------------------------------------------------------------------------
print(f"Loading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...")
load_start = time.perf_counter()

pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")

load_time = time.perf_counter() - load_start
print(f"Model loaded in {load_time:.2f} s")

# ---------------------------------------------------------------------------
# 3. Generate Audio
# ---------------------------------------------------------------------------
print(f"\nGenerating sound...")
print(f"  Prompt          : {PROMPT}")
print(f"  Negative prompt : {NEGATIVE_PROMPT}")
print(f"  Length          : {AUDIO_LENGTH_S} s")
print(f"  Steps           : {NUM_INFERENCE_STEPS}")

torch.cuda.reset_peak_memory_stats()
gen_start = time.perf_counter()

output = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    audio_length_in_s=AUDIO_LENGTH_S,
    num_inference_steps=NUM_INFERENCE_STEPS,
)

gen_time = time.perf_counter() - gen_start
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
curr_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)

audio = output.audios[0]
if hasattr(audio, "squeeze"):
    audio = audio.squeeze()

# ---------------------------------------------------------------------------
# 4. Check Raw Audio Output
# ---------------------------------------------------------------------------
has_nan = bool(np.isnan(audio).any())
raw_min = float(np.nanmin(audio)) if not np.all(np.isnan(audio)) else float("nan")
raw_max = float(np.nanmax(audio)) if not np.all(np.isnan(audio)) else float("nan")
raw_peak_abs = float(np.nanmax(np.abs(audio))) if not np.all(np.isnan(audio)) else float("nan")

print(f"\n--- Raw Audio Inspection ---")
print(f"  Dtype           : {audio.dtype}")
print(f"  Shape           : {audio.shape}")
print(f"  Has NaN         : {has_nan}")
print(f"  Min value       : {raw_min:.6f}")
print(f"  Max value       : {raw_max:.6f}")
print(f"  Peak abs value  : {raw_peak_abs:.6f}")

# ---------------------------------------------------------------------------
# 5. Clean, Normalize, and Convert to 16-bit PCM WAV
# ---------------------------------------------------------------------------
# Replace NaN with 0 if present
if has_nan:
    print("  [warning] NaN values found; replacing with 0.0")
    audio = np.nan_to_num(audio, nan=0.0)

# Normalize so peak is 0.9
peak = float(np.max(np.abs(audio)))
if peak > 0:
    audio = audio * (0.9 / peak)
    print(f"  Normalized audio peak to 0.9 (original peak: {peak:.6f})")
else:
    print("  [warning] Peak is 0; cannot normalize.")

# Convert to 16-bit PCM
audio_int16 = np.int16(audio * 32767)
final_peak_int16 = int(np.max(np.abs(audio_int16)))
final_peak_float = float(np.max(np.abs(audio)))

# Save to WAV
scipy.io.wavfile.write(OUTPUT_PATH, rate=SAMPLE_RATE, data=audio_int16)
print(f"\nSaved audio to: {OUTPUT_PATH}")

print(f"\n--- Final Verification ---")
print(f"  Final peak (float, target 0.9) : {final_peak_float:.4f}")
print(f"  Final peak (int16 PCM, max 32767): {final_peak_int16}")

# ---------------------------------------------------------------------------
# 6. Performance Summary
# ---------------------------------------------------------------------------
print(f"\n--- Performance Summary ---")
print(f"Generation time : {gen_time:.2f} s")
print(f"Peak GPU memory : {peak_mem_mb:.2f} MB ({peak_mem_mb / 1024:.2f} GB)")
print(f"Current GPU mem : {curr_mem_mb:.2f} MB ({curr_mem_mb / 1024:.2f} GB)")

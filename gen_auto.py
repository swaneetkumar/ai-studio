"""
gen_auto.py
-----------
Generates audio candidates for each event in events_auto.json using AudioLDM (float32 on cuda).
Produces 3 seed variants (0, 1, 2) per event, filters out FLAT signals (std < 0.005),
saves OK audio files to outputs/auto/{type}_s{seed}.wav, selects the candidate with
the highest std as the default pick, writes outputs/auto/picks.json, and prints a summary table.
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
# 1. Configuration & Prompt Overrides
# ---------------------------------------------------------------------------
MODEL_ID = "cvssp/audioldm-s-full-v2"
EVENTS_FILE = "events_auto.json"
OUTPUT_DIR = os.path.join("outputs", "auto")
SAMPLE_RATE = 16000
SEEDS = [0, 1, 2]
STD_THRESHOLD = 0.005

AUDIO_LENGTH_S = 5.12
NUM_STEPS = 50
NEGATIVE_PROMPT = "low quality"
GUIDANCE_SCALE = 3.5

# Prompt overrides dictionary (event type -> custom prompt)
PROMPT_OVERRIDES = {
    "glass_smash": "glass bottle shattering into many pieces, loud crash, breaking glass",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Read events_auto.json
# ---------------------------------------------------------------------------
if not os.path.exists(EVENTS_FILE):
    sys.exit(f"[error] {EVENTS_FILE} not found in project root.")

with open(EVENTS_FILE, "r", encoding="utf-8") as f:
    events_data = json.load(f)

events = events_data.get("events", [])
print(f"Loaded {len(events)} events from {EVENTS_FILE}.\n")

# ---------------------------------------------------------------------------
# 3. Load AudioLDM Pipeline (float32 on cuda)
# ---------------------------------------------------------------------------
print(f"Loading pipeline '{MODEL_ID}' (torch_dtype=float32, device=cuda)...")
pipe = AudioLDMPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")
print("Pipeline loaded successfully.\n")

# ---------------------------------------------------------------------------
# 4. Helper Functions
# ---------------------------------------------------------------------------
def compute_strongest_onset(audio, sr=16000):
    """Finds strongest onset timestamp using librosa."""
    if len(audio) == 0:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        onset_envelope=onset_env,
        units="frames",
    )
    if len(onset_frames) > 0:
        strongest_frame = onset_frames[np.argmax(onset_env[onset_frames])]
    else:
        strongest_frame = int(np.argmax(onset_env))
    return float(librosa.frames_to_time(strongest_frame, sr=sr))

# ---------------------------------------------------------------------------
# 5. Generation Loop
# ---------------------------------------------------------------------------
all_results = []
default_picks = {}

for ev in events:
    ev_type = ev.get("type", "sound")
    prompt = PROMPT_OVERRIDES.get(ev_type, ev.get("prompt", ""))

    print(f"==================================================")
    print(f"Event: {ev_type}")
    print(f"Prompt: \"{prompt}\"")
    print(f"==================================================")

    event_candidates = []

    for seed in SEEDS:
        print(f"  Generating seed {seed} ...")
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

        # Clean NaNs
        audio_clean = np.nan_to_num(audio, nan=0.0)
        std_val = float(np.std(audio_clean))
        onset_time = compute_strongest_onset(audio_clean, sr=SAMPLE_RATE)

        filename = f"{ev_type}_s{seed}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if std_val < STD_THRESHOLD:
            status = "FLAT"
            print(f"    -> [FLAT] std={std_val:.6f} < {STD_THRESHOLD} (skipped)")
        else:
            status = "OK"
            # Clip to [-1, 1] and save 16-bit PCM WAV
            audio_clipped = np.clip(audio_clean, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_int16)
            print(f"    -> [OK] std={std_val:.4f}, onset={onset_time:.3f}s, saved {filename}")

        cand_info = {
            "type": ev_type,
            "seed": seed,
            "std": std_val,
            "onset": onset_time,
            "status": status,
            "filename": filename,
            "is_pick": False,
        }
        event_candidates.append(cand_info)
        all_results.append(cand_info)

    # Pick the candidate with highest std among OK status
    ok_candidates = [c for c in event_candidates if c["status"] == "OK"]
    if ok_candidates:
        best_cand = max(ok_candidates, key=lambda c: c["std"])
        best_cand["is_pick"] = True
        default_picks[ev_type] = best_cand["filename"]
        print(f"  ==> Pick for {ev_type}: {best_cand['filename']} (std={best_cand['std']:.4f})\n")
    else:
        print(f"  ==> [warning] No OK candidate found for {ev_type}\n")

# ---------------------------------------------------------------------------
# 6. Save Default Picks to outputs/auto/picks.json
# ---------------------------------------------------------------------------
picks_path = os.path.join(OUTPUT_DIR, "picks.json")
with open(picks_path, "w", encoding="utf-8") as f:
    json.dump(default_picks, f, indent=2)

print(f"Saved default picks to: {picks_path}")

# ---------------------------------------------------------------------------
# 7. Print Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 76)
print(f"{'Event Type':<18} {'Seed':<6} {'Std Dev':<12} {'Onset Time':<14} {'Status':<8} {'Pick':<8}")
print("-" * 76)
for r in all_results:
    pick_marker = "* PICK" if r["is_pick"] else ""
    print(
        f"{r['type']:<18} "
        f"{r['seed']:<6} "
        f"{r['std']:<12.4f} "
        f"{r['onset']:<14.3f} "
        f"{r['status']:<8} "
        f"{pick_marker:<8}"
    )
print("=" * 76)


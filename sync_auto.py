"""
sync_auto.py
-----------
Synchronizes, processes, and mixes Foley audio events from events_auto.json.
1. Reads events_auto.json.
2. Uses editable PICKS and GAINS dictionaries to configure sound selections and volume.
3. Finds strongest attack onset for each sound, trims to start 0.1s before onset
   with 0.05s fade-in, 0.3s fade-out, and cuts at most 2.5s post-onset.
4. Places each sound on a master timeline matching the video duration.
5. Scales mix if peak exceeds 0.95 and saves outputs/mix_auto.wav (16-bit PCM, 16000 Hz).
6. Muxes video with mix_auto.wav using ffmpeg -> outputs/final_auto.mp4.
7. Prints alignment verification table.
"""

import os
import sys
import json
import subprocess
import cv2
import numpy as np
import librosa
import scipy.io.wavfile

# ---------------------------------------------------------------------------
# 1. Editable Dictionaries (Picks and Gains)
# ---------------------------------------------------------------------------
PICKS = {
    "footstep": "footstep_s2.wav",
    "glass_pick_up": "glass_pick_up_s1.wav",
    "whoosh": "whoosh_s2.wav",
    "glass_smash": "glass_smash_s2.wav",
    "glass_tinkle": "glass_tinkle_s2.wav",
}

GAINS = {
    "footstep": 0.5,
    "glass_pick_up": 0.5,
    "whoosh": 0.6,
    "glass_smash": 1.0,
    "glass_tinkle": 0.5,
}

TIME_OFFSETS = {
    "glass_smash": 0.12,
}

SAMPLE_RATE = 16000
PRE_ONSET_LEAD_S = 0.1
POST_ONSET_MAX_S = 2.5
FADE_IN_S = 0.05
FADE_OUT_S = 0.3
PEAK_LIMIT = 0.95

SOUNDS_DIR = os.path.join("outputs", "auto")
OUTPUT_DIR = "outputs"
MIX_WAV_PATH = os.path.join(OUTPUT_DIR, "mix_auto.wav")
FINAL_MP4_PATH = os.path.join(OUTPUT_DIR, "final_auto.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Read events_auto.json & Video Metadata
# ---------------------------------------------------------------------------
events_file = "events_auto.json"
if not os.path.exists(events_file):
    sys.exit(f"[error] {events_file} not found in project root.")

with open(events_file, "r", encoding="utf-8") as f:
    events_data = json.load(f)

video_path = events_data.get("video", "videos/test1.mp4")
if not os.path.exists(video_path):
    alt = video_path + ".mp4" if not video_path.endswith(".mp4.mp4") else "videos/test1.mp4"
    if os.path.exists(alt):
        video_path = alt
    else:
        sys.exit(f"[error] Video file not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or float(events_data.get("fps", 24.0))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps
cap.release()

print(f"Video           : {video_path}")
print(f"FPS             : {fps:.2f}")
print(f"Total frames    : {total_frames}")
print(f"Duration        : {video_duration:.2f} s")

# Master timeline
timeline_len = int(round(video_duration * SAMPLE_RATE))
master_mix = np.zeros(timeline_len, dtype=np.float32)

# ---------------------------------------------------------------------------
# 3. Process, Trim, and Align Each Sound Event
# ---------------------------------------------------------------------------
alignment_records = []
fade_in_len = int(round(FADE_IN_S * SAMPLE_RATE))
fade_out_len = int(round(FADE_OUT_S * SAMPLE_RATE))

for ev in events_data.get("events", []):
    ev_type = ev.get("type", "unknown")
    ev_time = float(ev.get("time", 0.0))

    sound_filename = PICKS.get(ev_type)
    if not sound_filename:
        print(f"[warning] No sound file configured in PICKS for '{ev_type}'. Skipping.")
        continue

    sound_path = os.path.join(SOUNDS_DIR, sound_filename)
    if not os.path.exists(sound_path):
        sys.exit(f"[error] Sound file not found: {sound_path}")

    # Load audio at 16000 Hz
    y, sr = librosa.load(sound_path, sr=SAMPLE_RATE)

    # Strongest onset detection (onset_strength peak)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        backtrack=False,
        units="frames",
    )
    if len(onset_frames) > 0:
        strongest_frame = onset_frames[np.argmax(onset_env[onset_frames])]
    else:
        strongest_frame = int(np.argmax(onset_env))

    onset_time = float(librosa.frames_to_time(strongest_frame, sr=sr))
    onset_sample = int(round(onset_time * sr))

    # Trim sound: start 0.1s before onset, cut at most 2.5s after onset
    lead_in_samples = int(round(PRE_ONSET_LEAD_S * sr))
    max_tail_samples = int(round(POST_ONSET_MAX_S * sr))

    start_idx = max(0, onset_sample - lead_in_samples)
    end_idx = min(len(y), onset_sample + max_tail_samples)
    trimmed = y[start_idx:end_idx].copy()

    # Apply fades (0.05s in, 0.3s out)
    if len(trimmed) > (fade_in_len + fade_out_len):
        trimmed[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)
        trimmed[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len)
    elif len(trimmed) > 0:
        half = len(trimmed) // 2
        trimmed[:half] *= np.linspace(0.0, 1.0, half)
        trimmed[half:] *= np.linspace(1.0, 0.0, len(trimmed) - half)

    # Place on timeline: start sample = round((effective_time - 0.1) * 16000)
    # where effective_time = event_time + TIME_OFFSETS.get(type, 0)
    offset = float(TIME_OFFSETS.get(ev_type, 0.0))
    effective_time = ev_time + offset

    start_sample = int(round((effective_time - PRE_ONSET_LEAD_S) * sr))
    if start_sample < 0:
        cut_samples = -start_sample
        trimmed = trimmed[cut_samples:]
        start_sample = 0

    timeline_start_s = start_sample / sr

    # Multiply by gain and add to master timeline
    gain = GAINS.get(ev_type, 1.0)
    end_sample = start_sample + len(trimmed)

    if start_sample < timeline_len and end_sample > 0:
        mix_s = start_sample
        mix_e = min(timeline_len, end_sample)
        valid_len = mix_e - mix_s
        master_mix[mix_s:mix_e] += (trimmed * gain)[:valid_len]

    alignment_records.append({
        "type": ev_type,
        "event_time": ev_time,
        "offset": offset,
        "effective_time": effective_time,
        "sound_file": sound_filename,
        "onset_time": onset_time,
        "timeline_start_s": timeline_start_s,
        "start_sample": start_sample,
        "gain": gain,
    })

# ---------------------------------------------------------------------------
# 4. Master Normalization (if peak > 0.95)
# ---------------------------------------------------------------------------
max_peak = float(np.max(np.abs(master_mix))) if len(master_mix) > 0 else 0.0
print(f"\nMaster mix peak amplitude: {max_peak:.4f}")

if max_peak > PEAK_LIMIT:
    scale = PEAK_LIMIT / max_peak
    master_mix *= scale
    print(f"[info] Scaled master mix down by factor of {scale:.4f} to cap peak at {PEAK_LIMIT}.")
else:
    print(f"[info] Master mix peak within limit (<= {PEAK_LIMIT}), no scaling applied.")

# ---------------------------------------------------------------------------
# 5. Save outputs/mix_auto.wav (16-bit PCM, 16000 Hz)
# ---------------------------------------------------------------------------
clipped_mix = np.clip(master_mix, -1.0, 1.0)
mix_int16 = (clipped_mix * 32767).astype(np.int16)
scipy.io.wavfile.write(MIX_WAV_PATH, rate=SAMPLE_RATE, data=mix_int16)
print(f"Saved mix audio : {MIX_WAV_PATH}")

# ---------------------------------------------------------------------------
# 6. Multiplex with Video using FFmpeg -> outputs/final_auto.mp4
# ---------------------------------------------------------------------------
print(f"\nMuxing video with mix_auto.wav using FFmpeg...")
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-i", video_path,
    "-i", MIX_WAV_PATH,
    "-c:v", "copy",
    "-c:a", "aac",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    FINAL_MP4_PATH,
]

result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"[error] FFmpeg failed with exit code {result.returncode}:\n{result.stderr}")
    sys.exit(result.returncode)

print(f"Saved final video: {FINAL_MP4_PATH}")

# ---------------------------------------------------------------------------
# 7. Print Alignment Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'Event Type':<16} {'Event Time':<12} {'Offset':<10} {'Sound File':<24} {'Sound Onset':<14} {'Timeline Start':<20}")
print("-" * 100)
for r in alignment_records:
    start_str = f"{r['timeline_start_s']:.2f} s (sample {r['start_sample']})"
    offset_str = f"{r['offset']:+.2f} s"
    print(
        f"{r['type']:<16} "
        f"{r['event_time']:<12.2f} "
        f"{offset_str:<10} "
        f"{r['sound_file']:<24} "
        f"{r['onset_time']:<14.3f} "
        f"{start_str:<20}"
    )
print("=" * 100)


"""
sync_mix.py
-----------
Synchronizes, processes, and mixes Foley audio events to match video timestamps.
1. Reads events.json.
2. Maps events to selected generated audio candidates.
3. Detects the strongest attack onset for each sound, trims to start 0.1s before onset
   with fades (0.05s in, 0.3s out) and max length 2.5s post-onset.
4. Places each sound on a timeline matching the video duration.
5. Applies per-event gains and scales mix if peak > 0.95.
6. Saves outputs/mix.wav.
7. Multiplexes mix.wav with the original video using ffmpeg -> outputs/final_test1.mp4.
8. Prints alignment verification details.
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
# 1. Configuration & Gain Settings
# ---------------------------------------------------------------------------
EVENT_GAINS = {
    "whoosh": 0.6,
    "glass_smash": 1.0,
    "glass_tinkle": 0.5,
    # Aliases
    "smash": 1.0,
    "tinkle": 0.5,
}

EVENT_SOUND_MAP = {
    1: os.path.join("outputs", "sounds", "event1_whoosh_v0.wav"),
    2: os.path.join("outputs", "sounds", "smash_P1_s2.wav"),
    3: os.path.join("outputs", "sounds", "event3_glass_tinkle_v1.wav"),
}

SAMPLE_RATE = 16000
PRE_ONSET_LEAD_S = 0.1
POST_ONSET_MAX_S = 2.5
FADE_IN_S = 0.05
FADE_OUT_S = 0.3
PEAK_LIMIT = 0.95

OUTPUT_DIR = "outputs"
MIX_WAV_PATH = os.path.join(OUTPUT_DIR, "mix.wav")
FINAL_MP4_PATH = os.path.join(OUTPUT_DIR, "final_test1.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Read events.json & Video Metadata
# ---------------------------------------------------------------------------
events_file = "events.json"
if not os.path.exists(events_file):
    sys.exit(f"[error] {events_file} not found.")

with open(events_file, "r", encoding="utf-8") as f:
    events_data = json.load(f)

video_rel_path = events_data.get("video", "videos/test1.mp4")
if not os.path.exists(video_rel_path):
    alt = video_rel_path + ".mp4" if not video_rel_path.endswith(".mp4.mp4") else "videos/test1.mp4"
    if os.path.exists(alt):
        video_rel_path = alt
    else:
        sys.exit(f"[error] Video file not found: {video_rel_path}")

cap = cv2.VideoCapture(video_rel_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_rel_path}")

video_fps = cap.get(cv2.CAP_PROP_FPS) or float(events_data.get("fps", 24))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / video_fps
cap.release()

print(f"Video           : {video_rel_path}")
print(f"FPS             : {video_fps:.2f}")
print(f"Frames          : {total_frames}")
print(f"Duration        : {video_duration:.2f} s")

# Create master timeline matching video duration
timeline_len = int(round(video_duration * SAMPLE_RATE))
master_mix = np.zeros(timeline_len, dtype=np.float32)

# ---------------------------------------------------------------------------
# 3. Process and Align Each Sound Event
# ---------------------------------------------------------------------------
alignment_records = []

fade_in_len = int(round(FADE_IN_S * SAMPLE_RATE))
fade_out_len = int(round(FADE_OUT_S * SAMPLE_RATE))

for ev in events_data.get("events", []):
    ev_id = ev["id"]
    ev_type = ev["type"]
    ev_time = float(ev["time"])

    sound_path = EVENT_SOUND_MAP.get(ev_id)
    if not sound_path or not os.path.exists(sound_path):
        sys.exit(f"[error] Sound file not found for event {ev_id}: {sound_path}")

    # Load audio at 16000 Hz
    y, sr = librosa.load(sound_path, sr=SAMPLE_RATE)

    # Strongest onset detection: peak of onset_strength with backtrack=False
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

    # Trim: start 0.1s before onset, end at most 2.5s after onset
    lead_in_samples = int(round(PRE_ONSET_LEAD_S * sr))
    max_tail_samples = int(round(POST_ONSET_MAX_S * sr))

    start_idx = max(0, onset_sample - lead_in_samples)
    end_idx = min(len(y), onset_sample + max_tail_samples)
    trimmed = y[start_idx:end_idx].copy()

    # Apply fade-in (0.05s) and fade-out (0.3s)
    if len(trimmed) > (fade_in_len + fade_out_len):
        trimmed[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)
        trimmed[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len)
    elif len(trimmed) > 0:
        half = len(trimmed) // 2
        trimmed[:half] *= np.linspace(0.0, 1.0, half)
        trimmed[half:] *= np.linspace(1.0, 0.0, len(trimmed) - half)

    # Gain adjustment
    gain = EVENT_GAINS.get(ev_type, EVENT_GAINS.get(ev_type.split("_")[-1], 1.0))
    trimmed = trimmed * gain

    # Timeline placement: start sample = round((event_time - 0.1) * 16000)
    start_pos_s = ev_time - PRE_ONSET_LEAD_S
    timeline_start = int(round(start_pos_s * sr))
    timeline_end = timeline_start + len(trimmed)

    # Mix into master timeline
    if timeline_start < timeline_len and timeline_end > 0:
        mix_s = max(0, timeline_start)
        mix_e = min(timeline_len, timeline_end)
        trim_s = max(0, -timeline_start)
        trim_e = trim_s + (mix_e - mix_s)
        master_mix[mix_s:mix_e] += trimmed[trim_s:trim_e]

    alignment_records.append({
        "id": ev_id,
        "type": ev_type,
        "sound_file": os.path.basename(sound_path),
        "event_time": ev_time,
        "onset_time": onset_time,
        "start_pos_s": start_pos_s,
        "timeline_start": timeline_start,
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
# 5. Save outputs/mix.wav
# ---------------------------------------------------------------------------
clipped_mix = np.clip(master_mix, -1.0, 1.0)
mix_int16 = (clipped_mix * 32767).astype(np.int16)
scipy.io.wavfile.write(MIX_WAV_PATH, rate=SAMPLE_RATE, data=mix_int16)
print(f"Saved mix audio : {MIX_WAV_PATH}")

# ---------------------------------------------------------------------------
# 6. Multiplex Audio with Video using FFmpeg
# ---------------------------------------------------------------------------
print(f"\nMuxing video with mix.wav using FFmpeg...")
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-i", video_rel_path,
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
    print(f"[error] FFmpeg failed with code {result.returncode}:\n{result.stderr}")
    sys.exit(result.returncode)

print(f"Saved final video: {FINAL_MP4_PATH}")

# ---------------------------------------------------------------------------
# 7. Print Alignment Verification Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 94)
print(f"{'ID':<4} {'Type':<14} {'Sound File':<28} {'Event Time':<12} {'Sound Onset':<13} {'Timeline Start':<20}")
print("-" * 94)
for rec in alignment_records:
    start_desc = f"{rec['start_pos_s']:.2f} s (sample {rec['timeline_start']})"
    print(
        f"{rec['id']:<4} "
        f"{rec['type']:<14} "
        f"{rec['sound_file']:<28} "
        f"{rec['event_time']:<12.2f} "
        f"{rec['onset_time']:<13.3f} "
        f"{start_desc:<20}"
    )
print("=" * 94)


"""
snap_events.py
--------------
Aligns high-level sound events from Gemini (events_gemini.json) with physical motion
peaks detected via optical flow (from detect_impacts.py logic).
Snaps event timestamps to detected impacts within +/- 0.5 seconds, saves to
events_auto.json, and prints an alignment comparison table.
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# 1. Configuration Constants
# ---------------------------------------------------------------------------
PROMINENCE_FACTOR = 3.0
MIN_GAP_SEC = 0.3
TARGET_WIDTH = 320
MAX_IMPACTS = 5
SNAP_WINDOW_SEC = 0.5

EVENTS_GEMINI_FILE = "events_gemini.json"
EVENTS_AUTO_FILE = "events_auto.json"

# ---------------------------------------------------------------------------
# 2. Resolve Video Path & Load Gemini Events
# ---------------------------------------------------------------------------
if not os.path.exists(EVENTS_GEMINI_FILE):
    sys.exit(f"[error] {EVENTS_GEMINI_FILE} not found.")

with open(EVENTS_GEMINI_FILE, "r", encoding="utf-8") as f:
    gemini_data = json.load(f)

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/test1.mp4"
if not os.path.exists(video_path):
    alt = video_path + ".mp4" if not video_path.endswith(".mp4.mp4") else "videos/test1.mp4"
    if os.path.exists(alt):
        video_path = alt
    else:
        sys.exit(f"[error] Video file not found: {video_path}")

print(f"Video           : {video_path}")
print(f"Gemini events   : {len(gemini_data.get('events', []))} events loaded.")

# ---------------------------------------------------------------------------
# 3. Optical Flow Motion Detection (from detect_impacts.py)
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    small = cv2.resize(frame, (width, new_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

motion_values = []
frame_indices = []

ret, prev_frame = cap.read()
if not ret:
    sys.exit("[error] Could not read first frame.")

prev_gray = resize_to_width(prev_frame, TARGET_WIDTH)
frame_idx = 0

print("Computing optical flow motion signal...")
while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    curr_gray = resize_to_width(curr_frame, TARGET_WIDTH)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    motion_values.append(float(magnitude.mean()))
    frame_indices.append(frame_idx)
    prev_gray = curr_gray

cap.release()

motion_values = np.array(motion_values)
frame_indices = np.array(frame_indices)
time_axis = frame_indices / fps

# Smoothing and detrending
def moving_average(signal, window_size):
    if window_size <= 1:
        return signal.copy()
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")

smoothed = moving_average(motion_values, 3)
trend_win = max(1, int(round(fps)))
trend = moving_average(smoothed, trend_win)
detrended = smoothed - trend

# Peak detection
std_detrended = float(np.std(detrended))
prom_threshold = PROMINENCE_FACTOR * std_detrended
min_gap_frames = max(1, int(MIN_GAP_SEC * fps))

peaks, props = find_peaks(
    detrended,
    prominence=prom_threshold,
    distance=min_gap_frames,
)

if len(peaks) > 0 and MAX_IMPACTS > 0 and len(peaks) > MAX_IMPACTS:
    top_k = np.argsort(props["prominences"])[::-1][:MAX_IMPACTS]
    kept_peaks = np.sort(peaks[top_k])
else:
    kept_peaks = peaks

detected_impact_times = time_axis[kept_peaks]
detected_impact_frames = frame_indices[kept_peaks]

print(f"Detected {len(detected_impact_times)} motion impacts: " +
      ", ".join([f"{t:.2f}s (F{f})" for t, f in zip(detected_impact_times, detected_impact_frames)]))

# ---------------------------------------------------------------------------
# 4. Snap Gemini Events to Nearest Detected Impact (+/- 0.5s)
# ---------------------------------------------------------------------------
auto_events = []
table_rows = []

for idx, ev in enumerate(gemini_data.get("events", []), start=1):
    ev_type = ev.get("type", "unknown")
    gemini_time = float(ev.get("time", 0.0))

    # Look for impacts within SNAP_WINDOW_SEC
    best_impact_time = None
    best_impact_frame = None
    min_diff = float("inf")

    for itime, iframe in zip(detected_impact_times, detected_impact_frames):
        diff = abs(itime - gemini_time)
        if diff <= SNAP_WINDOW_SEC and diff < min_diff:
            min_diff = diff
            best_impact_time = itime
            best_impact_frame = iframe

    if best_impact_time is not None:
        final_time = round(float(best_impact_time), 2)
        final_frame = int(best_impact_frame)
        snapped = True
    else:
        final_time = round(gemini_time, 2)
        final_frame = int(round(final_time * fps))
        snapped = False

    auto_events.append({
        "id": idx,
        "type": ev_type,
        "time": final_time,
        "frame": final_frame,
        "material": ev.get("material", ""),
        "intensity": ev.get("intensity", "medium"),
        "prompt": ev.get("prompt", ""),
        "snapped": snapped,
    })

    table_rows.append({
        "type": ev_type,
        "gemini_time": gemini_time,
        "final_time": final_time,
        "frame": final_frame,
        "snapped": "Yes" if snapped else "No",
    })

# ---------------------------------------------------------------------------
# 5. Save events_auto.json
# ---------------------------------------------------------------------------
output_data = {
    "video": video_path,
    "fps": round(float(fps), 2),
    "environment": gemini_data.get("environment", ""),
    "events": auto_events,
}

with open(EVENTS_AUTO_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved synced events to: {EVENTS_AUTO_FILE}")

# ---------------------------------------------------------------------------
# 6. Print Summary Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Event Type':<18} {'Gemini Time':<14} {'Final Time':<14} {'Frame':<10} {'Snapped':<8}")
print("-" * 70)
for r in table_rows:
    print(
        f"{r['type']:<18} "
        f"{r['gemini_time']:<14.2f} "
        f"{r['final_time']:<14.2f} "
        f"{r['frame']:<10} "
        f"{r['snapped']:<8}"
    )
print("=" * 70)


"""
detect_impacts.py
-----------------
Detects impact moments in a video by analysing frame-to-frame motion
computed with Farneback optical flow, with smoothing, detrending,
and prominence-based peak filtering to reduce false positives.

Usage:
    python detect_impacts.py [video_path] [--max-impacts N]

Default video path: videos/test1.mp4
"""

import sys
import os
import argparse

import cv2
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
# Prominence multiplier: peak prominence must be at least this many times
# the standard deviation of the detrended motion signal.
PROMINENCE_FACTOR = 3.0

# Minimum time gap (seconds) between successive impacts to avoid duplicates
MIN_GAP_SEC = 0.3

# Target width to resize frames for fast optical flow calculation
TARGET_WIDTH = 320


# ---------------------------------------------------------------------------
# 1. Parse command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Detect impact moments in a video.")
parser.add_argument(
    "video",
    nargs="?",
    default="videos/test1.mp4",
    help="Path to the input video file (default: videos/test1.mp4)",
)
parser.add_argument(
    "--max-impacts",
    type=int,
    default=3,
    help="Maximum number of strongest impacts to keep (default: 3)",
)
args = parser.parse_args()
video_path = args.video
max_impacts = args.max_impacts

# If the given path doesn't exist, try appending an extra '.mp4'
# (handles common double-extension case like test1.mp4.mp4)
if not os.path.exists(video_path):
    alt = video_path + ".mp4"
    if os.path.exists(alt):
        print(f"[info] '{video_path}' not found – using '{alt}' instead.")
        video_path = alt
    else:
        sys.exit(f"[error] Video file not found: {video_path}")


# ---------------------------------------------------------------------------
# 2. Open video and read basic metadata
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps if fps > 0 else 0.0

print(f"Video      : {video_path}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print(f"Duration   : {duration_sec:.2f} s")
print("Computing optical flow …")


# ---------------------------------------------------------------------------
# 3. Compute per-frame motion magnitude using Farneback optical flow
# ---------------------------------------------------------------------------
def resize_to_width(frame, width):
    """Resize frame keeping aspect ratio, return grayscale."""
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    small = cv2.resize(frame, (width, new_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


motion_values = []          # one scalar per frame (mean flow magnitude)
frame_indices = []          # corresponding frame numbers

ret, prev_frame = cap.read()
if not ret:
    sys.exit("[error] Could not read first frame.")

prev_gray = resize_to_width(prev_frame, TARGET_WIDTH)
frame_idx = 0

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    curr_gray = resize_to_width(curr_frame, TARGET_WIDTH)

    # Farneback optical flow – returns a 2-channel (dx, dy) vector field
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,   # pyramid scale between levels
        levels=3,        # number of pyramid levels
        winsize=15,      # averaging window size
        iterations=3,    # iterations at each pyramid level
        poly_n=5,        # pixel neighbourhood size
        poly_sigma=1.2,  # Gaussian std for polynomial expansion
        flags=0,
    )

    # Convert (dx, dy) flow to motion magnitude per pixel, then take the mean
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    motion_values.append(float(magnitude.mean()))
    frame_indices.append(frame_idx)

    prev_gray = curr_gray

cap.release()

motion_values = np.array(motion_values)
frame_indices = np.array(frame_indices)

# Time axis (seconds) for each computed frame
time_axis = frame_indices / fps


# ---------------------------------------------------------------------------
# 4. Smoothing and Detrending
# ---------------------------------------------------------------------------
def moving_average(signal, window_size):
    """Compute moving average with edge padding to avoid boundary artifacts."""
    if window_size <= 1:
        return signal.copy()
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


# 1. Smooth lightly (~3 frames) to reduce single-frame sensor noise
SMOOTH_WIN = 3
smoothed_signal = moving_average(motion_values, SMOOTH_WIN)

# 2. Subtract slow trend (~1 second moving average) so only sudden spikes count
TREND_WIN = max(1, int(round(fps)))
trend_signal = moving_average(smoothed_signal, TREND_WIN)
detrended_signal = smoothed_signal - trend_signal


# ---------------------------------------------------------------------------
# 5. Detect Impacts with Prominence Threshold & Max Impacts Selection
# ---------------------------------------------------------------------------
# Standard deviation of detrended signal
std_detrended = float(np.std(detrended_signal))
prom_threshold = PROMINENCE_FACTOR * std_detrended
min_gap_frames = max(1, int(MIN_GAP_SEC * fps))

# Find peaks whose prominence is at least PROMINENCE_FACTOR * std
peaks, props = find_peaks(
    detrended_signal,
    prominence=prom_threshold,
    distance=min_gap_frames,
)

# Keep only the strongest N impacts based on prominence
if len(peaks) > 0 and max_impacts is not None and max_impacts > 0 and len(peaks) > max_impacts:
    prominences = props["prominences"]
    top_k_indices = np.argsort(prominences)[::-1][:max_impacts]
    # Keep them sorted by time
    kept_peak_indices = np.sort(peaks[top_k_indices])
else:
    kept_peak_indices = peaks

# Map peak indices back to frame numbers and timestamps
impact_frames     = frame_indices[kept_peak_indices]
impact_timestamps = time_axis[kept_peak_indices]


# ---------------------------------------------------------------------------
# 6. Print Results
# ---------------------------------------------------------------------------
print(f"\nDetrended std       : {std_detrended:.4f}")
print(f"Prominence threshold: {prom_threshold:.4f} ({PROMINENCE_FACTOR:.1f}x std)")
print(f"Impacts detected    : {len(impact_frames)} (max-impacts: {max_impacts})\n")

if len(impact_frames) == 0:
    print("No impacts detected. Try lowering PROMINENCE_FACTOR.")
else:
    print(f"{'#':>4}  {'Time (s)':>10}  {'Frame':>7}")
    print("-" * 28)
    for i, (ts, fr) in enumerate(zip(impact_timestamps, impact_frames), start=1):
        print(f"{i:>4}  {round(ts, 2):>10.2f}  {fr:>7}")


# ---------------------------------------------------------------------------
# 7. Save Plot of Detrended Signal with Impact Markers
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/motion_plot.png"

# Canvas dimensions
IMG_W, IMG_H = 1400, 400
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 40, 50

plot_w = IMG_W - PAD_L - PAD_R
plot_h = IMG_H - PAD_T - PAD_B

canvas = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)  # White background

# Coordinate mapping (x: time, y: detrended motion)
t_min, t_max = time_axis[0], time_axis[-1]
d_min, d_max = float(detrended_signal.min()), float(detrended_signal.max())
margin = (d_max - d_min) * 0.15 if (d_max > d_min) else 0.1
y_min = d_min - margin
y_max = d_max + margin

def tx(t):
    return PAD_L + int((t - t_min) / (t_max - t_min) * plot_w)

def ty(v):
    return PAD_T + plot_h - int((v - y_min) / (y_max - y_min) * plot_h)

# Grid lines and y-axis labels
n_grid = 5
for gi in range(n_grid + 1):
    gv = y_min + gi * (y_max - y_min) / n_grid
    gy = ty(gv)
    cv2.line(canvas, (PAD_L, gy), (PAD_L + plot_w, gy), (230, 230, 230), 1)
    label = f"{gv:+.2f}"
    cv2.putText(canvas, label, (6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (110, 110, 110), 1, cv2.LINE_AA)

# Zero baseline line
if y_min <= 0.0 <= y_max:
    zy = ty(0.0)
    cv2.line(canvas, (PAD_L, zy), (PAD_L + plot_w, zy), (180, 180, 180), 1, cv2.LINE_AA)

# Prominence threshold reference line
th_y = ty(prom_threshold)
if PAD_T <= th_y <= PAD_T + plot_h:
    cv2.line(canvas, (PAD_L, th_y), (PAD_L + plot_w, th_y), (0, 150, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"prominence threshold={prom_threshold:.3f}", (PAD_L + 6, th_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 150, 255), 1, cv2.LINE_AA)

# Detrended signal line (blue)
pts = [(tx(time_axis[i]), ty(detrended_signal[i])) for i in range(len(time_axis))]
for i in range(1, len(pts)):
    cv2.line(canvas, pts[i - 1], pts[i], (200, 100, 30), 1, cv2.LINE_AA)

# Impact markers (red filled circles) on kept impacts
for pi in kept_peak_indices:
    cx, cy = tx(time_axis[pi]), ty(detrended_signal[pi])
    cv2.circle(canvas, (cx, cy), 6, (0, 0, 220), -1, cv2.LINE_AA)
    ts_lbl = f"{time_axis[pi]:.2f}s"
    cv2.putText(canvas, ts_lbl, (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 180), 1, cv2.LINE_AA)

# Outer border
cv2.rectangle(canvas, (PAD_L, PAD_T), (PAD_L + plot_w, PAD_T + plot_h), (80, 80, 80), 1)

# X-axis time ticks
n_x_ticks = 10
for xi in range(n_x_ticks + 1):
    tv = t_min + xi * (t_max - t_min) / n_x_ticks
    px = tx(tv)
    cv2.line(canvas, (px, PAD_T + plot_h), (px, PAD_T + plot_h + 4), (80, 80, 80), 1)
    cv2.putText(canvas, f"{tv:.1f}", (px - 12, PAD_T + plot_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

# Title
cv2.putText(canvas, "Detrended Motion Signal with Detected Impacts",
            (PAD_L, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)

# Legend
legend_items = [
    ((200, 100, 30), "Detrended motion"),
    ((0, 150, 255),  f"Threshold ({PROMINENCE_FACTOR:.1f}x std)"),
    ((0, 0, 220),    f"Impacts ({len(impact_frames)})"),
]
lx, ly = IMG_W - 250, PAD_T + 10
for color, lbl in legend_items:
    cv2.line(canvas, (lx, ly), (lx + 20, ly), color, 2)
    cv2.putText(canvas, lbl, (lx + 25, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
    ly += 18

cv2.imwrite(output_path, canvas)
print(f"\nPlot saved to: {output_path}")

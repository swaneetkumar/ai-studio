"""
extract_check2.py
-----------------
Extracts frames at specific timestamps (0.2, 0.5, 0.8, 2.5, 2.8, 3.1 seconds)
from the video, labels each frame with its timestamp and frame index,
resizes to 400px width, arranges in a 3x2 grid, and saves to outputs/frames_check2.png.

Dependencies: opencv-python, numpy
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Resolve Video Path
# ---------------------------------------------------------------------------
video_path = "videos/test1.mp4"
if not os.path.exists(video_path):
    alt_path = "videos/test1.mp4.mp4"
    if os.path.exists(alt_path):
        print(f"[info] '{video_path}' not found – using '{alt_path}' instead.")
        video_path = alt_path
    else:
        sys.exit(f"[error] Video file not found: {video_path} or {alt_path}")

# ---------------------------------------------------------------------------
# 2. Extract Frames
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video        : {video_path}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")

timestamps = [0.2, 0.5, 0.8, 2.5, 2.8, 3.1]
TARGET_WIDTH = 400
COLS = 3

frames_list = []

for t in timestamps:
    target_frame = int(round(t * fps))
    target_frame = min(max(0, target_frame), total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"[warning] Could not read frame at {t}s (frame {target_frame}). Generating blank.")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Resize to 400px width keeping aspect ratio
    h, w = frame.shape[:2]
    new_h = int(h * TARGET_WIDTH / w)
    resized = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Dark banner at the top for label readability
    banner_height = 28
    cv2.rectangle(resized, (0, 0), (TARGET_WIDTH, banner_height), (20, 20, 20), -1)

    # Label text: time and frame number
    label = f"{t:.2f}s (Frame {target_frame})"
    cv2.putText(
        resized,
        label,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    frames_list.append(resized)
    print(f"Extracted t={t:.2f}s -> Frame {target_frame}")

cap.release()

# ---------------------------------------------------------------------------
# 3. Arrange into a 3-column Grid (2 Rows)
# ---------------------------------------------------------------------------
rows = []
for i in range(0, len(frames_list), COLS):
    row = np.hstack(frames_list[i : i + COLS])
    rows.append(row)

grid = np.vstack(rows)

# ---------------------------------------------------------------------------
# 4. Save Image
# ---------------------------------------------------------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "frames_check2.png")

cv2.imwrite(output_path, grid)
print(f"\nImage saved successfully to: {output_path} ({grid.shape[1]}x{grid.shape[0]} px)")


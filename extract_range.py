"""
extract_range.py
----------------
Extracts frames 130 through 142 (inclusive) from the video, resizes each to 400px width,
labels each frame with its frame number and timestamp, and arranges them in a grid
with 4 frames per row, saving the result to outputs/frames_range.png.

Dependencies: opencv-python, numpy
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Resolve video path
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
# 2. Open video and extract range
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Opened video : {video_path}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")

START_FRAME = 130
END_FRAME = 142
TARGET_WIDTH = 400
COLS = 4

frames_list = []

# Position directly at START_FRAME and read sequentially through END_FRAME
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

for frame_no in range(START_FRAME, END_FRAME + 1):
    ret, frame = cap.read()
    if not ret:
        print(f"[warning] Could not read frame {frame_no}. Generating blank tile.")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Resize keeping aspect ratio
    h, w = frame.shape[:2]
    new_h = int(h * TARGET_WIDTH / w)
    resized = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Calculate timestamp in seconds
    timestamp_sec = frame_no / fps

    # Draw label banner at top
    banner_height = 30
    cv2.rectangle(resized, (0, 0), (TARGET_WIDTH, banner_height), (20, 20, 20), -1)

    # Text label: frame number and time in seconds
    label = f"Frame {frame_no} | {timestamp_sec:.2f}s"
    cv2.putText(
        resized,
        label,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    frames_list.append(resized)
    print(f"Extracted Frame {frame_no} ({timestamp_sec:.2f}s)")

cap.release()

# ---------------------------------------------------------------------------
# 3. Arrange frames into a grid (COLS per row)
# ---------------------------------------------------------------------------
tile_h, tile_w = frames_list[0].shape[:2]

# Pad with blank tiles so total number of frames is a multiple of COLS
remainder = len(frames_list) % COLS
if remainder != 0:
    pad_count = COLS - remainder
    blank_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for _ in range(pad_count):
        frames_list.append(blank_tile)

# Build rows
rows = []
for i in range(0, len(frames_list), COLS):
    row = np.hstack(frames_list[i : i + COLS])
    rows.append(row)

# Stack all rows vertically
grid = np.vstack(rows)

# ---------------------------------------------------------------------------
# 4. Save output
# ---------------------------------------------------------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "frames_range.png")

cv2.imwrite(output_path, grid)
print(f"\nGrid saved successfully to: {output_path} ({grid.shape[1]}x{grid.shape[0]} px)")


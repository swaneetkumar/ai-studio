"""
extract_frames.py
-----------------
Extracts frames at specific timestamps from a video, resizes each to 320px width,
labels each frame with its timestamp, and joins them side-by-side into a contact
sheet saved as outputs/frames_check.png.

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
# 2. Open video and get metadata
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"[error] Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Opened video : {video_path}")
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")

# Timestamps to extract (in seconds)
timestamps = [4.5, 4.88, 5.2, 5.54, 5.9]
TARGET_WIDTH = 320

processed_frames = []

for t in timestamps:
    target_frame = int(round(t * fps))
    target_frame = min(max(0, target_frame), total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"[warning] Could not read frame at {t}s (frame {target_frame}). Using blank.")
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame = np.zeros((h if h > 0 else 720, w if w > 0 else 1280, 3), dtype=np.uint8)

    # Resize to TARGET_WIDTH keeping aspect ratio
    h, w = frame.shape[:2]
    new_h = int(h * TARGET_WIDTH / w)
    resized = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    # Draw a clean dark banner at the top for readable text
    banner_height = 28
    cv2.rectangle(resized, (0, 0), (TARGET_WIDTH, banner_height), (20, 20, 20), -1)

    # Label with timestamp and frame number
    label = f"{t:.2f}s (F{target_frame})"
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

    processed_frames.append(resized)
    print(f"Extracted t={t:4.2f}s -> Frame {target_frame}")

cap.release()

# ---------------------------------------------------------------------------
# 3. Join frames side by side into one contact sheet
# ---------------------------------------------------------------------------
contact_sheet = np.hstack(processed_frames)

# Ensure outputs directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "frames_check.png")

# Save contact sheet
cv2.imwrite(output_path, contact_sheet)
print(f"\nContact sheet saved successfully to: {output_path}")


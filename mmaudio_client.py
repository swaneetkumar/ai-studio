"""
mmaudio_client.py
-----------------
Command-line client to invoke an MMAudio model hosted on a remote Gradio / Google Colab instance.
Accepts a remote URL, input video path, prompt, and optional duration.
Downloads the resulting video with Foley audio, extracts 44.1kHz 16-bit PCM WAV via FFmpeg,
and reports performance statistics.

Usage:
    python mmaudio_client.py <colab_url> <video_path> "<prompt>" [duration]
"""

import os
import sys
import time
import shutil
import subprocess
from gradio_client import Client, handle_file

def main():
    if len(sys.argv) < 4:
        print("Usage: python mmaudio_client.py <colab_url> <video_path> \"<prompt>\" [duration]")
        sys.exit(1)

    colab_url = sys.argv[1].strip()
    video_path = sys.argv[2].strip()
    prompt = sys.argv[3].strip()
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

    # Resolve video path if necessary
    if not os.path.exists(video_path):
        candidates = [
            video_path + ".mp4",
            "videos/test1.mp4.mp4",
            "videos/test1.mp4",
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                video_path = c
                found = True
                break
        if not found:
            print(f"[Error] Video file not found: {video_path}")
            sys.exit(1)

    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    out_video_path = os.path.join(outputs_dir, "mmaudio_test.mp4")
    out_audio_path = os.path.join(outputs_dir, "mmaudio_test.wav")

    start_time = time.time()

    # 1. Connect to remote Client
    print(f"Connecting to Gradio instance at {colab_url} ...", flush=True)
    try:
        client = Client(colab_url, httpx_kwargs={"timeout": 300.0})
    except TypeError:
        client = Client(colab_url)
    except Exception as e:
        print(f"\n[Error] Colab Gradio instance is unreachable: {e}", flush=True)
        print("Please check that your Colab notebook is running and the Gradio share link is active.", flush=True)
        sys.exit(1)

    print("Connected successfully. Sending inference request...", flush=True)
    print(f"  Video   : {video_path}")
    print(f"  Prompt  : \"{prompt}\"")
    print(f"  Duration: {duration} s")

    # 2. Call client.predict with fallback argument formats
    result = None
    format_worked = None

    # Format 1: video={"video": handle_file(video_path)}
    try:
        print("Trying argument format: video={'video': handle_file(...)} ...", flush=True)
        result = client.predict(
            video={"video": handle_file(video_path)},
            prompt=prompt,
            negative_prompt="music",
            seed=-1,
            num_steps=25,
            cfg_strength=4.5,
            duration=duration,
            api_name="/predict",
        )
        format_worked = "video={'video': handle_file(video_path)}"
    except Exception as err1:
        print(f"Format 1 raised error: {err1}", flush=True)
        print("Trying alternative format: video=handle_file(...) ...", flush=True)
        # Format 2: video=handle_file(video_path)
        try:
            result = client.predict(
                video=handle_file(video_path),
                prompt=prompt,
                negative_prompt="music",
                seed=-1,
                num_steps=25,
                cfg_strength=4.5,
                duration=duration,
                api_name="/predict",
            )
            format_worked = "video=handle_file(video_path)"
        except Exception as err2:
            print(f"\n[Error] Prediction failed with both argument formats.", flush=True)
            print(f"  Format 1 error: {err1}", flush=True)
            print(f"  Format 2 error: {err2}", flush=True)
            sys.exit(1)

    print(f"[Success] Prediction succeeded using format: {format_worked}", flush=True)

    # 3. Extract output video file path
    res_video_file = None
    if isinstance(result, dict):
        res_video_file = result.get("video") or result.get("path")
        if isinstance(res_video_file, dict):
            res_video_file = res_video_file.get("video") or res_video_file.get("path")
        if not res_video_file:
            # Fallback to first dict value
            vals = list(result.values())
            if vals:
                res_video_file = vals[0]
                if isinstance(res_video_file, dict):
                    res_video_file = res_video_file.get("video") or res_video_file.get("path")
    elif isinstance(result, (list, tuple)):
        res_video_file = result[0]
        if isinstance(res_video_file, dict):
            res_video_file = res_video_file.get("video") or res_video_file.get("path")
    elif isinstance(result, str):
        res_video_file = result

    if not res_video_file or not os.path.exists(str(res_video_file)):
        print(f"\n[Error] Could not locate result video file from prediction return: {result}", flush=True)
        sys.exit(1)

    shutil.copyfile(str(res_video_file), out_video_path)

    # 4. Extract audio using FFmpeg (44100 Hz, 16-bit PCM)
    print("Extracting audio with FFmpeg (44100 Hz, 16-bit PCM)...", flush=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", out_video_path,
        "-vn",
        "-ar", "44100",
        "-acodec", "pcm_s16le",
        out_audio_path,
    ]

    ff_res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if ff_res.returncode != 0:
        print(f"\n[Error] FFmpeg audio extraction failed:\n{ff_res.stderr}", flush=True)
        sys.exit(ff_res.returncode)

    elapsed = time.time() - start_time

    # 5. Print summary
    print("\n" + "=" * 60, flush=True)
    print("MMAudio Generation Complete", flush=True)
    print("=" * 60, flush=True)
    print(f"Time taken : {elapsed:.2f} seconds")
    print(f"Video file : {out_video_path}")
    print(f"Audio file : {out_audio_path}")
    print("=" * 60 + "\n", flush=True)

if __name__ == "__main__":
    main()


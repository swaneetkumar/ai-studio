"""
pipeline.py
-----------
End-to-end Foley AI sound generation and video synchronization pipeline.
Supports dual sound engines:
  1. AudioLDM (local float32 on CUDA, per-event candidate generation and onset snapping)
  2. MMAudio (remote Colab GPU endpoint, equal chunk video synthesis, crossfade, and 44.1kHz audio)

Exposes:
    run_pipeline(video_path, intensity=1.0, style="auto", progress=None, engine="audioldm", colab_url=None) -> path of the final mp4
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
import cv2
import numpy as np
import scipy.io.wavfile
import scipy.signal
import librosa
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from diffusers import AudioLDMPipeline
from gradio_client import Client, handle_file

# ---------------------------------------------------------------------------
# Global Pipeline Cache (AudioLDM)
# ---------------------------------------------------------------------------
_AUDIOLDM_PIPE = None

def get_audioldm_pipeline():
    """Loads and caches AudioLDM pipeline in GPU memory."""
    global _AUDIOLDM_PIPE
    if _AUDIOLDM_PIPE is None:
        print("[pipeline] Loading AudioLDM pipeline ('cvssp/audioldm-s-full-v2', float32, cuda)...", flush=True)
        _AUDIOLDM_PIPE = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-s-full-v2",
            torch_dtype=torch.float32,
        ).to("cuda")
        print("[pipeline] AudioLDM pipeline loaded and cached in VRAM.", flush=True)
    return _AUDIOLDM_PIPE

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
MMAUDIO_SR = 44100
PROMINENCE_FACTOR = 3.0
MIN_GAP_SEC = 0.3
MAX_IMPACTS = 5
SNAP_WINDOW_SEC = 0.5
TARGET_FLOW_WIDTH = 320

AUDIO_LENGTH_S = 5.12
NUM_STEPS = 50
NEGATIVE_PROMPT = "low quality"
GUIDANCE_SCALE = 3.5
STD_THRESHOLD = 0.005
SEEDS = [0, 1, 2]

PRE_ONSET_LEAD_S = 0.1
POST_ONSET_MAX_S = 2.5
FADE_IN_S = 0.05
FADE_OUT_S = 0.3
PEAK_LIMIT = 0.95

GLASS_SMASH_OVERRIDE_PROMPT = "glass bottle shattering into many pieces, loud crash, breaking glass"
GLASS_SMASH_TIME_OFFSET = 0.12

def get_prompt_and_offset(ev_type, gemini_prompt):
    """
    Applies overrides only when the event type contains both 'glass' and 'smash'.
    Every other event uses Gemini's prompt and offset 0.
    """
    ev_type_lower = ev_type.lower()
    if "glass" in ev_type_lower and "smash" in ev_type_lower:
        return GLASS_SMASH_OVERRIDE_PROMPT, GLASS_SMASH_TIME_OFFSET
    return gemini_prompt, 0.0

def get_event_gain(ev_type):
    """
    Returns 1.0 for event types containing 'smash', 'crash', or 'impact'.
    Default 0.6 for all other events.
    """
    ev_type_lower = ev_type.lower()
    if any(w in ev_type_lower for w in ["smash", "crash", "impact"]):
        return 1.0
    return 0.6

def shorten_event_prompt(prompt_text, ev_type):
    """Shortens Gemini event prompt to the core sound action."""
    if not prompt_text:
        return ev_type.replace("_", " ")
    text = prompt_text.strip(" .,")
    lower = text.lower()
    for env_kw in [" in a ", " in an ", " inside a ", " inside an "]:
        pos = lower.rfind(env_kw)
        if pos > 10:
            text = text[:pos].strip(" .,")
            break
    return text

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def moving_average(signal, window_size):
    if window_size <= 1:
        return signal.copy()
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")

def detect_motion_impacts(video_path, fps):
    """Computes Farneback optical flow and detects motion impact timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([]), np.array([])

    def resize_gray(frame):
        h, w = frame.shape[:2]
        new_h = int(h * TARGET_FLOW_WIDTH / w)
        small = cv2.resize(frame, (TARGET_FLOW_WIDTH, new_h))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.array([]), np.array([])

    prev_gray = resize_gray(prev_frame)
    motion_values = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = resize_gray(curr_frame)

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
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_values.append(float(mag.mean()))
        frame_indices.append(frame_idx)
        prev_gray = curr_gray

    cap.release()

    if len(motion_values) == 0:
        return np.array([]), np.array([])

    motion_arr = np.array(motion_values)
    frame_arr = np.array(frame_indices)
    time_arr = frame_arr / fps

    smoothed = moving_average(motion_arr, 3)
    trend_win = max(1, int(round(fps)))
    trend = moving_average(smoothed, trend_win)
    detrended = smoothed - trend

    std_val = float(np.std(detrended))
    prom_thresh = PROMINENCE_FACTOR * std_val
    min_gap_frames = max(1, int(MIN_GAP_SEC * fps))

    peaks, props = scipy.signal.find_peaks(
        detrended,
        prominence=prom_thresh,
        distance=min_gap_frames,
    )

    if len(peaks) > 0 and MAX_IMPACTS > 0 and len(peaks) > MAX_IMPACTS:
        top_k = np.argsort(props["prominences"])[::-1][:MAX_IMPACTS]
        kept_peaks = np.sort(peaks[top_k])
    else:
        kept_peaks = peaks

    return time_arr[kept_peaks], frame_arr[kept_peaks]

def compute_strongest_onset(audio, sr=16000):
    if len(audio) == 0:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        onset_envelope=onset_env,
        backtrack=False,
        units="frames",
    )
    if len(onset_frames) > 0:
        best_frame = onset_frames[np.argmax(onset_env[onset_frames])]
    else:
        best_frame = int(np.argmax(onset_env))
    return float(librosa.frames_to_time(best_frame, sr=sr))

def apply_synthetic_reverb(dry_mix, rt60, wet_ratio, sr=16000):
    """Convolves signal with exponentially decaying synthetic impulse response."""
    if rt60 <= 0.0 or wet_ratio <= 0.0:
        return dry_mix.copy()

    ir_len = int(round(rt60 * sr))
    t = np.arange(ir_len) / sr
    decay = np.exp(-6.908 * t / rt60)
    rng = np.random.default_rng(42)
    ir = rng.standard_normal(ir_len) * decay
    ir = ir / (np.linalg.norm(ir) + 1e-8)

    wet_signal = scipy.signal.fftconvolve(dry_mix, ir, mode="full")[:len(dry_mix)]
    dry_rms = np.sqrt(np.mean(dry_mix ** 2)) + 1e-8
    wet_rms = np.sqrt(np.mean(wet_signal ** 2)) + 1e-8
    wet_scaled = wet_signal * (dry_rms / wet_rms)

    return (1.0 - wet_ratio) * dry_mix + wet_ratio * wet_scaled

def call_mmaudio_predict(client, video_chunk_path, prompt, duration, negative_prompt="music, speech, voice"):
    """
    Calls MMAudio Gradio endpoint /predict with fallback between video dict and direct handle_file.
    Returns the path of the generated video with audio.
    """
    result = None
    try:
        result = client.predict(
            video={"video": handle_file(video_chunk_path)},
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=-1,
            num_steps=25,
            cfg_strength=4.5,
            duration=float(duration),
            api_name="/predict",
        )
    except Exception as err1:
        try:
            result = client.predict(
                video=handle_file(video_chunk_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=-1,
                num_steps=25,
                cfg_strength=4.5,
                duration=float(duration),
                api_name="/predict",
            )
        except Exception as err2:
            raise RuntimeError(f"MMAudio prediction failed.\nFormat 1 error: {err1}\nFormat 2 error: {err2}")

    res_video_file = None
    if isinstance(result, dict):
        res_video_file = result.get("video") or result.get("path")
        if isinstance(res_video_file, dict):
            res_video_file = res_video_file.get("video") or res_video_file.get("path")
        if not res_video_file:
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
        raise RuntimeError(f"Could not locate result video file from prediction return: {result}")

    return str(res_video_file)

# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------
def run_pipeline(video_path, intensity=1.0, style="auto", progress=None, engine="audioldm", colab_url=None):
    """
    Executes the full Foley pipeline:
      engine == 'audioldm': AudioLDM candidates, timeline onset alignment, reverb, mix.
      engine == 'mmaudio': Equal-chunk video slicing, Colab MMAudio inference, crossfade, reverb, mix.
    """
    def log(msg):
        print(f"[pipeline] {msg}", flush=True)
        if callable(progress):
            progress(msg)

    engine = engine.lower().strip() if engine else "audioldm"
    if engine not in ["audioldm", "mmaudio"]:
        raise ValueError(f"Unknown engine: {engine}. Supported choices are 'audioldm' and 'mmaudio'.")

    if engine == "mmaudio":
        if not colab_url or not str(colab_url).strip():
            raise ValueError("A valid Colab URL is required when using the 'mmaudio' sound engine.")
        colab_url = str(colab_url).strip()

    # Resolve video path
    if not os.path.exists(video_path):
        alt = video_path + ".mp4" if not video_path.endswith(".mp4.mp4") else "videos/test1.mp4"
        if os.path.exists(alt):
            video_path = alt
        else:
            raise FileNotFoundError(f"Video not found: {video_path}")

    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # Fresh runs folder for this specific execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(outputs_dir, "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    log(f"Session run directory: {run_dir} (engine: {engine})")

    # Video Metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()

    log(f"Processing video: {video_path} ({total_frames} frames, {fps:.2f} fps, {video_duration:.2f} s)")

    # -----------------------------------------------------------------------
    # Step 1: Analyze Video with Gemini + Snap with Optical Flow
    # -----------------------------------------------------------------------
    log("Step 1/6: Analyzing video with Gemini Vision...")
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=video_path)

    while uploaded_file.state in (types.FileState.PROCESSING, "PROCESSING"):
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state not in (types.FileState.ACTIVE, "ACTIVE"):
        raise RuntimeError(f"Video file processing failed with state: {uploaded_file.state}")

    analysis_prompt = """Analyze this silent video for Foley sound design.
Watch the video and list only distinct audible sound events (impacts, footsteps, movements through air, doors, scraping, scattering pieces, etc.) in time order, with approximate timestamps.
Name each event type in short snake_case based on what you actually see, and write the text-to-audio prompt describing the sound from the actual material and environment visible in the video.

Return a JSON object with this exact shape:
{
  "environment": "<brief description of the environment, e.g. small indoor room / open field / large hall / outdoors>",
  "events": [
    {
      "type": "<short snake_case name based on visible action, e.g. footstep, impact, door_close, scrape>",
      "time": <approximate seconds from video start, number>,
      "material": "<main material involved, e.g. metal, wood, concrete>",
      "intensity": "<low|medium|high>",
      "prompt": "<a short text-to-audio prompt describing this sound, mentioning the material and environment>"
    }
  ]
}
"""
    config = types.GenerateContentConfig(response_mime_type="application/json")
    
    candidate_models = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.5-flash"]
    response = None

    for m in candidate_models:
        try:
            log(f"Querying {m} for Foley sound events...")
            response = client.models.generate_content(
                model=m,
                contents=[uploaded_file, analysis_prompt],
                config=config,
            )
            if response and response.text:
                log(f"Received response from {m}.")
                break
        except Exception as e:
            log(f"[warning] {m} failed ({e}). Trying next model...")

    if response is None or not response.text:
        raise RuntimeError("Gemini video analysis failed on all candidate models. Please check your network connection and GEMINI_API_KEY.")

    gemini_data = json.loads(response.text)

    environment_desc = gemini_data.get("environment", "")
    raw_events = gemini_data.get("events", [])
    log(f"Gemini detected environment: \"{environment_desc}\" with {len(raw_events)} events.")

    # Optical flow impact snapping
    log("Snapping event timestamps using optical flow impact detection...")
    detected_times, detected_frames = detect_motion_impacts(video_path, fps)

    snapped_events = []
    for idx, ev in enumerate(raw_events, start=1):
        ev_type = ev.get("type", "sound")
        g_time = float(ev.get("time", 0.0))

        best_time, best_frame = None, None
        min_diff = float("inf")
        for itime, iframe in zip(detected_times, detected_frames):
            diff = abs(itime - g_time)
            if diff <= SNAP_WINDOW_SEC and diff < min_diff:
                min_diff = diff
                best_time = itime
                best_frame = iframe

        if best_time is not None:
            final_time = round(float(best_time), 2)
            final_frame = int(best_frame)
            snapped = True
        else:
            final_time = round(g_time, 2)
            final_frame = int(round(final_time * fps))
            snapped = False

        prompt_used, offset = get_prompt_and_offset(ev_type, ev.get("prompt", ""))

        snapped_events.append({
            "id": idx,
            "type": ev_type,
            "gemini_time": g_time,
            "time": final_time,
            "frame": final_frame,
            "material": ev.get("material", ""),
            "intensity": ev.get("intensity", "medium"),
            "prompt_used": prompt_used,
            "offset": offset,
            "snapped": snapped,
        })
        snapped_str = "snapped" if snapped else "unsnapped"
        log(f"  Event '{ev_type}': {g_time:.2f}s -> {final_time:.2f}s (Frame {final_frame}, {snapped_str})")

    # =======================================================================
    # Branch A: MMAudio Sound Engine (Colab / Cloud)
    # =======================================================================
    if engine == "mmaudio":
        log("Step 2/6: Connecting to MMAudio remote service on Colab...")
        try:
            mmaudio_client = Client(colab_url, httpx_kwargs={"timeout": 300.0})
        except TypeError:
            mmaudio_client = Client(colab_url)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MMAudio service: {e}")

        # Split video into equal chunks of at most 8s, not shorter than 3s
        if video_duration <= 8.0:
            num_chunks = 1
        else:
            num_chunks = max(1, int(np.ceil(video_duration / 8.0)))

        chunk_dur = video_duration / num_chunks
        if chunk_dur < 3.0 and num_chunks > 1:
            num_chunks = max(1, int(np.floor(video_duration / 3.0)))
            chunk_dur = video_duration / num_chunks

        log(f"Splitting video into {num_chunks} chunk(s) of ~{chunk_dur:.2f}s each...")

        chunk_records = []
        chunk_audios = []

        for i in range(num_chunks):
            c_start = i * chunk_dur
            c_end = (i + 1) * chunk_dur if i < num_chunks - 1 else video_duration
            c_len = c_end - c_start
            chunk_video_path = os.path.join(run_dir, f"chunk_{i}.mp4")

            # Cut chunk video with ffmpeg
            cut_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{c_start:.3f}",
                "-i", video_path,
                "-t", f"{c_len:.3f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-an",
                chunk_video_path,
            ]
            res = subprocess.run(cut_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg chunk split failed for chunk {i}:\n{res.stderr}")

            # Events falling in this chunk
            chunk_evs = [
                ev for ev in snapped_events
                if (c_start <= ev["time"] < c_end) or (i == num_chunks - 1 and abs(ev["time"] - c_end) <= 0.05)
            ]

            # Build text prompt
            sound_clauses = [shorten_event_prompt(ev["prompt_used"], ev["type"]) for ev in chunk_evs]
            if environment_desc:
                sound_clauses.append(environment_desc.strip(" .,/"))
            chunk_prompt = ", ".join([c for c in sound_clauses if c]) or "ambient background sound"

            range_str = f"{c_start:.2f}s - {c_end:.2f}s"
            log(f"  Sending chunk {i+1}/{num_chunks} ({range_str}, dur: {c_len:.2f}s)...")
            log(f"    Prompt: \"{chunk_prompt}\"")

            chunk_records.append({
                "index": i + 1,
                "range_str": range_str,
                "events": chunk_evs,
                "prompt": chunk_prompt,
            })

            # Remote MMAudio inference
            ret_video_path = call_mmaudio_predict(
                client=mmaudio_client,
                video_chunk_path=chunk_video_path,
                prompt=chunk_prompt,
                duration=round(c_len, 1),
                negative_prompt="music, speech, voice",
            )

            # Extract 44100 Hz mono PCM audio
            chunk_wav_path = os.path.join(run_dir, f"chunk_{i}.wav")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(ret_video_path),
                "-vn",
                "-ar", "44100",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                chunk_wav_path,
            ]
            res = subprocess.run(extract_cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg audio extraction failed for chunk {i}:\n{res.stderr}")

            audio_chunk, _ = librosa.load(chunk_wav_path, sr=MMAUDIO_SR, mono=True)
            chunk_audios.append(audio_chunk)

        # Crossfade join (0.15s crossfade)
        log("Step 3/6: Joining chunk audios with 0.15s crossfade...")
        fade_len = int(round(0.15 * MMAUDIO_SR))

        if len(chunk_audios) == 1:
            full_audio = chunk_audios[0]
        else:
            full_audio = chunk_audios[0]
            for next_audio in chunk_audios[1:]:
                if len(full_audio) >= fade_len and len(next_audio) >= fade_len:
                    fade_out = np.linspace(1.0, 0.0, fade_len)
                    fade_in = np.linspace(0.0, 1.0, fade_len)
                    overlap = full_audio[-fade_len:] * fade_out + next_audio[:fade_len] * fade_in
                    full_audio = np.concatenate([
                        full_audio[:-fade_len],
                        overlap,
                        next_audio[fade_len:],
                    ])
                else:
                    full_audio = np.concatenate([full_audio, next_audio])

        # Trim or pad to exact video duration
        target_len = int(round(video_duration * MMAUDIO_SR))
        if len(full_audio) > target_len:
            full_audio = full_audio[:target_len]
        elif len(full_audio) < target_len:
            full_audio = np.pad(full_audio, (0, target_len - len(full_audio)))

        # Reverb styling at 44100 Hz
        resolved_style = style
        if style == "auto":
            env_lower = environment_desc.lower()
            if any(w in env_lower for w in ["hall", "large", "spacious", "arena", "cathedral"]):
                resolved_style = "large_hall"
            elif any(w in env_lower for w in ["outdoors", "field", "street", "park", "open"]):
                resolved_style = "dry"
            elif any(w in env_lower for w in ["small", "indoor", "room", "kitchen"]):
                resolved_style = "small_room"
            else:
                resolved_style = "small_room"

        log(f"Step 4/6: Applying reverb style: '{resolved_style}' (requested: '{style}')...")
        if resolved_style == "small_room":
            rt60 = 0.4
            wet_ratio = 0.20
        elif resolved_style == "large_hall":
            rt60 = 1.5
            wet_ratio = 0.35
        else:
            rt60 = 0.0
            wet_ratio = 0.0

        styled_mix = apply_synthetic_reverb(full_audio, rt60=rt60, wet_ratio=wet_ratio, sr=MMAUDIO_SR)

        # Intensity scaling & peak limiting
        intensity = max(0.2, min(2.0, float(intensity)))
        log(f"Step 5/6: Scaling intensity by {intensity:.2f}x and limiting peak to {PEAK_LIMIT}...")
        styled_mix = styled_mix * intensity

        peak = float(np.max(np.abs(styled_mix)))
        if peak > PEAK_LIMIT:
            scale = PEAK_LIMIT / peak
            styled_mix *= scale
            log(f"  Peak ({peak:.4f}) exceeded limit. Scaled by {scale:.4f} to {PEAK_LIMIT}.")
        else:
            log(f"  Peak ({peak:.4f}) within limit (<= {PEAK_LIMIT}).")

        # Save mix WAV
        mix_wav_path = os.path.join(outputs_dir, f"mix_mmaudio_{resolved_style}.wav")
        mix_int16 = (np.clip(styled_mix, -1.0, 1.0) * 32767).astype(np.int16)
        scipy.io.wavfile.write(mix_wav_path, rate=MMAUDIO_SR, data=mix_int16)
        log(f"  Mix audio saved: {mix_wav_path}")

        # Attach audio to video with ffmpeg
        final_video_name = f"result_mmaudio_{style}.mp4"
        final_video_path = os.path.join(outputs_dir, final_video_name)

        log(f"Step 6/6: Multiplexing audio and video to {final_video_path}...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mix_wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_video_path,
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {res.returncode}:\n{res.stderr}")

        log(f"Successfully generated final video: {final_video_path}")

        # Summary table for MMAudio chunks
        table_lines = []
        table_lines.append("=" * 110)
        table_lines.append(f"{'Chunk':<8} {'Time Range':<18} {'Events in Chunk':<28} {'Prompt Sent'}")
        table_lines.append("-" * 110)
        for rec in chunk_records:
            ev_list_str = ", ".join([f"{ev['type']} ({ev['time']:.2f}s)" for ev in rec["events"]]) or "None (ambient)"
            if len(ev_list_str) > 26:
                ev_list_str = ev_list_str[:23] + "..."
            table_lines.append(
                f"{rec['index']:<8} "
                f"{rec['range_str']:<18} "
                f"{ev_list_str:<28} "
                f"\"{rec['prompt']}\""
            )
        table_lines.append("=" * 110)
        full_table = "\n".join(table_lines)
        print("\n" + full_table + "\n", flush=True)
        if callable(progress):
            progress("\n" + full_table)

        return final_video_path

    # =======================================================================
    # Branch B: AudioLDM Sound Engine (Local GPU)
    # =======================================================================
    log("Step 2/6: Generating audio candidates with AudioLDM...")
    pipe = get_audioldm_pipeline()

    chosen_audio_files = {}

    for ev in snapped_events:
        ev_id = ev["id"]
        ev_type = ev["type"]
        prompt = ev["prompt_used"]

        best_cand_path = None
        highest_std = -1.0

        for seed in SEEDS:
            filename = f"{ev_type}_s{seed}.wav"
            filepath = os.path.join(run_dir, filename)

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

            audio_clean = np.nan_to_num(audio, nan=0.0)
            std_val = float(np.std(audio_clean))

            if std_val < STD_THRESHOLD:
                log(f"  Candidate {filename} FLAT (std={std_val:.5f}), skipped.")
                continue

            audio_clipped = np.clip(audio_clean, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            scipy.io.wavfile.write(filepath, rate=SAMPLE_RATE, data=audio_int16)

            if std_val > highest_std:
                highest_std = std_val
                best_cand_path = filepath

        if best_cand_path:
            chosen_audio_files[ev_id] = best_cand_path
            log(f"  Selected for '{ev_type}': {os.path.basename(best_cand_path)} (std={highest_std:.4f})")
        else:
            log(f"  [warning] No valid candidates for '{ev_type}'.")

    # Step 3: Audio Synchronization and Dry Mix
    log("Step 3/6: Synchronizing sounds onto master timeline...")
    timeline_len = int(round(video_duration * SAMPLE_RATE))
    dry_mix = np.zeros(timeline_len, dtype=np.float32)

    fade_in_len = int(round(FADE_IN_S * SAMPLE_RATE))
    fade_out_len = int(round(FADE_OUT_S * SAMPLE_RATE))

    summary_records = []

    for ev in snapped_events:
        ev_id = ev["id"]
        ev_type = ev["type"]
        ev_time = float(ev["time"])
        g_time = float(ev.get("gemini_time", ev_time))
        sound_path = chosen_audio_files.get(ev_id)
        if not sound_path or not os.path.exists(sound_path):
            summary_records.append({
                "type": ev_type,
                "gemini_time": g_time,
                "final_time": ev_time,
                "snapped": ev.get("snapped", False),
                "prompt_used": ev["prompt_used"],
                "sound_file": "None",
            })
            continue

        y, sr = librosa.load(sound_path, sr=SAMPLE_RATE)
        onset_time = compute_strongest_onset(y, sr=SAMPLE_RATE)
        onset_sample = int(round(onset_time * sr))

        # Trim: start 0.1s before onset, at most 2.5s post-onset
        lead_in_samples = int(round(PRE_ONSET_LEAD_S * sr))
        max_tail_samples = int(round(POST_ONSET_MAX_S * sr))

        start_idx = max(0, onset_sample - lead_in_samples)
        end_idx = min(len(y), onset_sample + max_tail_samples)
        trimmed = y[start_idx:end_idx].copy()

        # Apply fades
        if len(trimmed) > (fade_in_len + fade_out_len):
            trimmed[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)
            trimmed[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len)
        elif len(trimmed) > 0:
            half = len(trimmed) // 2
            trimmed[:half] *= np.linspace(0.0, 1.0, half)
            trimmed[half:] *= np.linspace(1.0, 0.0, len(trimmed) - half)

        # Place on timeline
        offset = float(ev.get("offset", 0.0))
        effective_time = ev_time + offset

        start_sample = int(round((effective_time - PRE_ONSET_LEAD_S) * sr))
        if start_sample < 0:
            cut_samples = -start_sample
            trimmed = trimmed[cut_samples:]
            start_sample = 0

        timeline_start_s = start_sample / sr
        gain = get_event_gain(ev_type)
        end_sample = start_sample + len(trimmed)

        if start_sample < timeline_len and end_sample > 0:
            mix_s = start_sample
            mix_e = min(timeline_len, end_sample)
            valid_len = mix_e - mix_s
            dry_mix[mix_s:mix_e] += (trimmed * gain)[:valid_len]

        offset_info = f", offset: {offset:+.2f}s" if offset != 0.0 else ""
        log(f"  Aligned '{ev_type}' at {timeline_start_s:.2f}s (onset: {ev_time:.2f}s{offset_info}, gain: {gain})")

        summary_records.append({
            "type": ev_type,
            "gemini_time": g_time,
            "final_time": ev_time,
            "snapped": ev.get("snapped", False),
            "prompt_used": ev["prompt_used"],
            "sound_file": os.path.basename(sound_path),
            "timeline_start_s": timeline_start_s,
            "start_sample": start_sample,
            "onset_time": onset_time,
        })

    # Step 4: Reverb Styling
    resolved_style = style
    if style == "auto":
        env_lower = environment_desc.lower()
        if any(w in env_lower for w in ["hall", "large", "spacious", "arena", "cathedral"]):
            resolved_style = "large_hall"
        elif any(w in env_lower for w in ["outdoors", "field", "street", "park", "open"]):
            resolved_style = "dry"
        elif any(w in env_lower for w in ["small", "indoor", "room", "kitchen"]):
            resolved_style = "small_room"
        else:
            resolved_style = "small_room"

    log(f"Step 4/6: Applying reverb style: '{resolved_style}' (requested: '{style}')...")

    if resolved_style == "small_room":
        rt60 = 0.4
        wet_ratio = 0.20
    elif resolved_style == "large_hall":
        rt60 = 1.5
        wet_ratio = 0.35
    else:
        rt60 = 0.0
        wet_ratio = 0.0

    styled_mix = apply_synthetic_reverb(dry_mix, rt60=rt60, wet_ratio=wet_ratio, sr=SAMPLE_RATE)

    # Step 5: Intensity Scaling & Peak Limiting
    intensity = max(0.2, min(2.0, float(intensity)))
    log(f"Step 5/6: Scaling intensity by {intensity:.2f}x and limiting peak to {PEAK_LIMIT}...")
    styled_mix = styled_mix * intensity

    peak = float(np.max(np.abs(styled_mix)))
    if peak > PEAK_LIMIT:
        scale = PEAK_LIMIT / peak
        styled_mix *= scale
        log(f"  Peak ({peak:.4f}) exceeded limit. Scaled by {scale:.4f} to {PEAK_LIMIT}.")
    else:
        log(f"  Peak ({peak:.4f}) within limit (<= {PEAK_LIMIT}).")

    # Save mix WAV
    mix_wav_path = os.path.join(outputs_dir, f"mix_{resolved_style}.wav")
    mix_int16 = (np.clip(styled_mix, -1.0, 1.0) * 32767).astype(np.int16)
    scipy.io.wavfile.write(mix_wav_path, rate=SAMPLE_RATE, data=mix_int16)
    log(f"  Mix audio saved: {mix_wav_path}")

    # Step 6: Multiplex with FFmpeg
    final_video_name = f"result_{style}.mp4"
    final_video_path = os.path.join(outputs_dir, final_video_name)

    log(f"Step 6/6: Multiplexing audio and video to {final_video_path}...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", mix_wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_video_path,
    ]
    res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {res.returncode}:\n{res.stderr}")

    log(f"Successfully generated final video: {final_video_path}")

    # Final Summary Table
    table_lines = []
    table_lines.append("=" * 135)
    table_lines.append(f"{'Type':<16} {'Gemini Time':<13} {'Final Time':<12} {'Snapped':<9} {'Sound File':<22} {'Prompt Used'}")
    table_lines.append("-" * 135)
    for r in summary_records:
        snapped_str = "Yes" if r["snapped"] else "No"
        table_lines.append(
            f"{r['type']:<16} "
            f"{r['gemini_time']:<13.2f} "
            f"{r['final_time']:<12.2f} "
            f"{snapped_str:<9} "
            f"{r['sound_file']:<22} "
            f"{r['prompt_used']}"
        )
    table_lines.append("=" * 135)
    full_table = "\n".join(table_lines)
    print("\n" + full_table + "\n", flush=True)
    if callable(progress):
        progress("\n" + full_table)

    return final_video_path

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run complete Foley AI pipeline on a video.")
    parser.add_argument("video", nargs="?", default="videos/test1.mp4", help="Path to input video")
    parser.add_argument("style", nargs="?", default="auto", help="Reverb style: dry, small_room, large_hall, auto")
    parser.add_argument("intensity", nargs="?", type=float, default=1.0, help="Volume intensity (0.2 to 2.0)")
    parser.add_argument("--engine", choices=["audioldm", "mmaudio"], default="audioldm", help="Sound engine")
    parser.add_argument("--colab-url", default=None, help="Gradio link for MMAudio on Colab")

    args = parser.parse_args()

    out_file = run_pipeline(
        video_path=args.video,
        intensity=args.intensity,
        style=args.style,
        engine=args.engine,
        colab_url=args.colab_url,
    )
    print(f"\nPipeline finished. Output saved to: {out_file}", flush=True)

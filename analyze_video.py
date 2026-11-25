"""
analyze_video.py
----------------
Analyzes a video using Gemini (via the google-genai SDK) for Foley sound design.
Uploads the video using the Files API, requests structured JSON with environment
and time-ordered sound events, prints the formatted JSON, and saves it as
events_gemini.json.
"""

import os
import sys
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-3.8-flash"

# ---------------------------------------------------------------------------
# 1. Initialize Gemini Client from .env
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("[error] GEMINI_API_KEY not found in .env file or environment.")

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# 2. Resolve Input Video Path
# ---------------------------------------------------------------------------
video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/test1.mp4"
if not os.path.exists(video_path):
    alt = video_path + ".mp4" if not video_path.endswith(".mp4.mp4") else "videos/test1.mp4"
    if os.path.exists(alt):
        video_path = alt
    else:
        sys.exit(f"[error] Video file not found: {video_path}")

print(f"Target video: {video_path}")

# ---------------------------------------------------------------------------
# 3. Upload Video with Files API & Wait for ACTIVE State
# ---------------------------------------------------------------------------
print("Uploading video via Gemini Files API...")
uploaded_file = client.files.upload(file=video_path)
print(f"File uploaded: {uploaded_file.name} (state: {uploaded_file.state})")

while uploaded_file.state in (types.FileState.PROCESSING, "PROCESSING"):
    print("Waiting for video processing...")
    time.sleep(2)
    uploaded_file = client.files.get(name=uploaded_file.name)

if uploaded_file.state not in (types.FileState.ACTIVE, "ACTIVE"):
    sys.exit(f"[error] File processing ended in non-active state: {uploaded_file.state}")

print(f"File is ready and ACTIVE.\n")

# ---------------------------------------------------------------------------
# 4. Generate Foley Sound Analysis Content
# ---------------------------------------------------------------------------
prompt = """Analyze this silent video for Foley sound design.
List only distinct audible sound events (an impact, a swing through air, scattering pieces, etc.), in time order, with timestamps as accurate as you can.

Return a JSON object with this exact shape:
{
  "environment": "<e.g. small indoor room / open field / large hall / outdoors>",
  "events": [
    {
      "type": "<short snake_case name e.g. footstep, whoosh, glass_smash, glass_tinkle, door_close>",
      "time": <approximate seconds from video start, number>,
      "material": "<main material involved>",
      "intensity": "<low|medium|high>",
      "prompt": "<a short text-to-audio prompt describing this sound, mentioning material and environment>"
    }
  ]
}
"""

config = types.GenerateContentConfig(
    response_mime_type="application/json",
)

print(f"Querying {MODEL} for Foley sound events...")
try:
    response = client.models.generate_content(
        model=MODEL,
        contents=[uploaded_file, prompt],
        config=config,
    )
except ClientError as e:
    if "no longer available" in str(e) or e.code == 404:
        print(f"[warning] {MODEL} is unavailable on this account ({e.message}). Falling back to {FALLBACK_MODEL}...")
        response = client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=[uploaded_file, prompt],
            config=config,
        )
    else:
        raise

# ---------------------------------------------------------------------------
# 5. Parse, Print, and Save JSON Response
# ---------------------------------------------------------------------------
raw_text = response.text or ""

try:
    data = json.loads(raw_text)
    formatted_json = json.dumps(data, indent=2)

    print("\n" + "=" * 60)
    print("Foley Analysis JSON Result:")
    print("=" * 60)
    print(formatted_json)
    print("=" * 60)

    output_file = "events_gemini.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_json + "\n")

    print(f"\nSaved analysis to: {output_file}")

except json.JSONDecodeError as err:
    print(f"\n[error] Response is not valid JSON: {err}")
    print("Raw model response:")
    print(raw_text)
    sys.exit(1)

# AI Foley Studio

**Synchronized Foley Art and Sound Physics Synthesis from Silent Video**

AI Foley Studio is a multimodal audio-visual pipeline that transforms raw silent video into frame-accurate, material-aware Foley sound effects. Built with computer vision, optical flow impact snapping, generative audio synthesis (**MMAudio** and **AudioLDM**), synthetic room reverb convolution, and a **Gradio** studio interface.

*(Built for Hack\<N\>Pitch — Machine Learning Track, Problem 1: Generative AI Foley)*

---

## Key Features

- **Vision-to-Audio Event Extraction**: Uses Google Gemini Vision (`google-genai`) to watch silent video and extract the scene environment along with time-ordered audible events (event type, approximate timestamp, physical material, intensity, and text-to-audio prompt).
- **Optical Flow Impact Snapping**: Computes dense Farneback optical flow with OpenCV, detrends camera/object drift, detects true collision peaks using prominence filtering, and snaps Gemini timestamps to physical impact frames within `±0.5 s`.
- **Dual Generative Sound Engines**:
  - **MMAudio (Cloud / Colab GPU — default in UI)**: Splits longer videos into equal chunks (`≤ 8 s`, `≥ 3 s`), sends scene- and event-aware prompts to a remote MMAudio (44.1 kHz) instance via `gradio_client`, and joins chunk audio with a `0.15 s` crossfade.
  - **AudioLDM (Local CUDA GPU)**: Generates 3 seed candidates (`cvssp/audioldm-s-full-v2`, `16 kHz`, `5.12 s`) per event, filters out near-silent/flat generations (`std < 0.005`), picks the strongest candidate, and aligns audio via `librosa` onset detection.
- **Acoustic Spatialization & Mastering**: Applies synthetic exponential-decay impulse response reverb (`dry`, `small_room`, `large_hall`, or `auto` inferred from the Gemini environment description), master intensity scaling (`0.2x`–`2.0x`), peak limiting (`0.95`), and `ffmpeg` muxing into the final `.mp4`.
- **Web Workstation & Landing Page**: Includes a dual-deck Gradio studio UI (`app.py`) with real-time progress logs and a static HTML/CSS/JS showcase landing page (`index.html`).

---

## System Workflow

```text
Silent Video (.mp4 / .mov)
        │
        ▼
Gemini Vision Analysis ──► Environment + JSON Event List (type, time, material, prompt)
        │
        ▼
Farneback Optical Flow ──► Motion Signal → Smooth & Detrend → Peak Snap (±0.5 s)
        │
        ▼
Dual Sound Synthesis
  ├── MMAudio Engine  : Equal chunks (≤8s) → Remote /predict → 44.1 kHz → 0.15s Crossfade
  └── AudioLDM Engine : 3 seeds per event → Flat filter → Librosa onset trim & timeline sync
        │
        ▼
Reverb Convolution (dry / small_room / large_hall / auto) + Intensity + Peak Limiter (0.95)
        │
        ▼
FFmpeg Audio/Video Mux ──► Final Synchronized MP4 (outputs/)
```

---

## Tech Stack

| Layer | Tools / Libraries |
|---|---|
| **Language & UI** | Python 3.11, Gradio (`app.py`), HTML/CSS/JS (`index.html`) |
| **Video Analysis** | Google Gemini API (`google-genai` — `gemini-2.5-flash` with fallback to `gemini-3.8-flash` / `gemini-3.5-flash`) |
| **Motion & Impact Detection** | OpenCV (Farneback Optical Flow), NumPy, SciPy (`find_peaks`) |
| **Local Audio Synthesis** | PyTorch (CUDA), Diffusers (`cvssp/audioldm-s-full-v2`), Transformers |
| **Remote Audio Synthesis** | MMAudio (`large 44k v2` on Google Colab GPU) via `gradio_client` |
| **Audio DSP & Mastering** | `librosa` (onset detection), `scipy.signal.fftconvolve` (reverb), `soundfile` |
| **Media Processing** | `ffmpeg` |
| **Documentation** | ReportLab (`make_report.py` → `AI_Foley_Studio_Report.pdf`) |

---

## Project Structure

```text
foley-project/
├── app.py                      # Gradio studio web UI (http://127.0.0.1:7860)
├── pipeline.py                 # End-to-end dual-engine Foley pipeline (CLI & UI backend)
├── mmaudio_client.py           # CLI client for remote Colab MMAudio endpoint
├── analyze_video.py            # Standalone Gemini Vision event analyzer
├── detect_impacts.py           # Standalone Farneback optical flow impact detector
├── snap_events.py              # Snaps Gemini events to optical flow impact timestamps
├── gen_auto.py                 # Standalone AudioLDM candidate generator
├── sync_auto.py                # Standalone onset-based audio timeline mixer & muxer
├── index.html                  # Product landing page
├── style.css                   # Landing page stylesheet
├── script.js                   # Landing page interactive demo & animations
├── make_report.py              # Script to generate the PDF project report
├── AI_Foley_Studio_Report.pdf  # Detailed hackathon project report
├── videos/                     # Input sample videos
└── outputs/                    # Per-run artifacts and final generated videos
```

---

## Setup & Installation

### Prerequisites
- **Python 3.11**
- **FFmpeg** installed and available in system `PATH`
- **Google Gemini API Key**
- *(Optional)* **NVIDIA CUDA GPU** for running local `audioldm` mode, or a **Google Colab GPU session** running MMAudio for `mmaudio` mode

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd foley-project
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

pip install opencv-python numpy scipy librosa soundfile google-genai python-dotenv torch diffusers transformers gradio gradio-client reportlab
```

### 3. Configure Environment Variables
Create a `.env` file in the project root (ignored by `.gitignore`) and add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Launch the Gradio Studio UI
```bash
venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:7860` in your browser:
1. Upload a silent video clip.
2. Choose **Sound engine** (`mmaudio` or `audioldm`).
3. If using `mmaudio`, open the **Advanced** accordion and paste your active Colab `gradio.live` share URL.
4. Adjust **Intensity** (`0.2` – `2.0`) and acoustic **Style** (`auto`, `dry`, `small_room`, `large_hall`).
5. Click **Generate Foley Video** to monitor real-time alignment logs and download the synced `.mp4`.

### Run via Command Line (`pipeline.py`)
**Local AudioLDM engine:**
```bash
python pipeline.py videos/test1.mp4 auto 1.0 --engine audioldm
```

**Remote MMAudio engine:**
```bash
python pipeline.py videos/test1.mp4 small_room 1.0 --engine mmaudio --colab-url https://<your-colab-id>.gradio.live
```

### View the Landing Page
Open `index.html` directly in a browser or serve it locally:
```bash
python -m http.server 8080
```
Then visit `http://localhost:8080`.

---

## Key Parameters

| Parameter | Default Value | Purpose |
|---|---|---|
| `TARGET_FLOW_WIDTH` | `320` px | Frame resize width for optical flow |
| `PROMINENCE_FACTOR` | `3.0` | Peak threshold multiplier (`3.0 × std` of detrended flow) |
| `SNAP_WINDOW_SEC` | `±0.5 s` | Window to snap Gemini timestamps to motion peaks |
| `STD_THRESHOLD` | `0.005` | Rejects flat/near-silent AudioLDM generations |
| `MMAUDIO_SR` / `SAMPLE_RATE` | `44100 Hz` / `16000 Hz` | Output sample rates for MMAudio / AudioLDM |
| Reverb `small_room` / `large_hall` | `RT60 0.4s (20% wet)` / `1.5s (35% wet)` | Synthetic convolution reverb settings |
| `PEAK_LIMIT` | `0.95` | Prevents audio clipping before FFmpeg muxing |

---

## Limitations & Notes
- **Colab Dependency (MMAudio)**: `mmaudio` mode relies on an active Colab GPU instance and a temporary Gradio share link.
- **Chunking**: Videos longer than `8 s` are processed in equal chunks (`≤ 8 s`) and joined via `0.15 s` crossfades.
- **Camera Motion**: Optical flow impact snapping performs best when foreground object motion stands out against slow or mostly static camera movement.
- **Licensing / Usage**: MMAudio weights are non-commercial; this project is intended for hackathon demonstration, research, and educational use.

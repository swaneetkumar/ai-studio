"""
app.py
------
AI Foley Studio - Cutting-Edge Multimodal Sound Design Workstation.
Transforms raw silent video into cinematic, physically synchronized Foley sound design.
Theme: Warm Beige & Espresso Brown (matching the landing-page-design visual system).
"""

import os
import sys
import queue
import threading
import gradio as gr
from pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Custom Studio Theme (Warm Beige & Espresso Brown Aesthetic)
# ---------------------------------------------------------------------------
studio_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Manrope"), "system-ui", "-apple-system", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#F7F4EE",
    body_background_fill_dark="#F7F4EE",
    body_text_color="#2A221E",
    body_text_color_dark="#2A221E",
    block_background_fill="#EDE7DE",
    block_background_fill_dark="#EDE7DE",
    block_border_width="1px",
    block_border_color="rgba(42, 34, 30, 0.12)",
    block_border_color_dark="rgba(42, 34, 30, 0.12)",
    block_label_text_color="#6B5C52",
    block_title_text_color="#2A221E",
    input_background_fill="#FAF7F2",
    input_background_fill_dark="#FAF7F2",
    input_border_color="rgba(42, 34, 30, 0.15)",
    input_border_color_dark="rgba(42, 34, 30, 0.15)",
    button_primary_background_fill="#B8542F",
    button_primary_background_fill_dark="#B8542F",
    button_primary_background_fill_hover="#A14421",
    button_primary_background_fill_hover_dark="#A14421",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#EDE7DE",
    button_secondary_background_fill_dark="#EDE7DE",
    button_secondary_text_color="#2A221E",
    button_secondary_text_color_dark="#2A221E",
    slider_color="#B8542F",
    slider_color_dark="#B8542F",
)

# ---------------------------------------------------------------------------
# Custom CSS: Warm Beige & Espresso Brown Visual Styling
# ---------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

/* Main Studio Container */
.gradio-container {
    max-width: 1220px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
    background-color: #F7F4EE !important;
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #2A221E !important;
    min-height: 100vh !important;
}

/* Header & Motion Branding */
.studio-hero {
    text-align: center;
    padding: 16px 0 24px 0;
    margin-bottom: 12px;
    position: relative;
    border-bottom: 1px solid rgba(42, 34, 30, 0.1);
}

.studio-pills-row {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #B8542F;
    background: rgba(184, 84, 47, 0.08);
    border: 1px solid rgba(184, 84, 47, 0.22);
    padding: 5px 14px;
    border-radius: 9999px;
}

.status-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #B8542F;
    box-shadow: 0 0 6px rgba(184, 84, 47, 0.6);
    animation: live-pulse 2s infinite ease-in-out;
}

@keyframes live-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Animated Soundwave Equalizer Bars */
.soundwave-viz {
    display: inline-flex;
    align-items: flex-end;
    gap: 3px;
    height: 16px;
}
.soundwave-viz .wave-bar {
    width: 3px;
    border-radius: 2px;
    background-color: #B8542F;
    animation: wave-anim 1.2s ease-in-out infinite alternate;
}
.soundwave-viz .wb-1 { height: 45%; animation-delay: 0.05s; }
.soundwave-viz .wb-2 { height: 90%; animation-delay: 0.35s; }
.soundwave-viz .wb-3 { height: 60%; animation-delay: 0.15s; }
.soundwave-viz .wb-4 { height: 100%; animation-delay: 0.45s; }
.soundwave-viz .wb-5 { height: 75%; animation-delay: 0.25s; }
.soundwave-viz .wb-6 { height: 85%; animation-delay: 0.5s; }

@keyframes wave-anim {
    0% { transform: scaleY(0.25); opacity: 0.6; }
    100% { transform: scaleY(1.0); opacity: 1; }
}

.studio-main-title {
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin: 0 0 8px 0;
    color: #2A221E;
}
.studio-main-title .gradient-text {
    background: linear-gradient(to right, #2A221E 0%, #6B5C52 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.studio-hero-desc {
    font-size: 15px;
    color: #6B5C52;
    margin: 0 auto;
    max-width: 680px;
    line-height: 1.5;
}

/* 4-Step Pipeline Flow Infographic */
.pipeline-flow-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 16px auto 4px auto;
    padding: 8px 18px;
    background: #EDE7DE;
    border: 1px solid rgba(42, 34, 30, 0.12);
    border-radius: 9999px;
    max-width: 840px;
}
.flow-step {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #6B5C52;
}
.flow-step .step-icon {
    font-size: 14px;
}
.flow-step.active-step {
    color: #B8542F;
}
.flow-arrow {
    color: #8E7D72;
    font-size: 13px;
    animation: flow-beam 2s infinite ease-in-out;
}
@keyframes flow-beam {
    0%, 100% { opacity: 0.4; transform: translateX(0); }
    50% { opacity: 1; color: #B8542F; transform: translateX(2px); }
}

/* Section Deck Header Labels */
.deck-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    margin-bottom: 12px;
    background: #EDE7DE;
    border: 1px solid rgba(42, 34, 30, 0.12);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.deck-banner.deck-input {
    color: #2A221E;
    border-left: 3px solid #B8542F;
}
.deck-banner.deck-output {
    color: #2A221E;
    border-left: 3px solid #6B5C52;
}

/* Panels & Hover Lift */
.gr-panel, .gr-box {
    background-color: #EDE7DE !important;
    border: 1px solid rgba(42, 34, 30, 0.12) !important;
    border-radius: 14px !important;
    transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1) !important;
}
.gr-panel:hover, .gr-box:hover {
    border-color: rgba(42, 34, 30, 0.22) !important;
    box-shadow: 0 8px 24px rgba(42, 34, 30, 0.06) !important;
}

/* Helper Text */
.studio-helper {
    font-size: 12.5px;
    color: #6B5C52;
    margin-top: 4px;
    margin-bottom: 8px;
    line-height: 1.45;
}

/* Primary Action Button (Warm Terracotta / Cinnamon) */
.btn-generate-cinema {
    background-color: #B8542F !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    box-shadow: 0 4px 14px rgba(184, 84, 47, 0.25) !important;
    transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 10px !important;
    padding: 10px 20px !important;
}
.btn-generate-cinema:hover {
    background-color: #A14421 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(184, 84, 47, 0.35) !important;
}
.btn-generate-cinema:active {
    transform: scale(0.98) !important;
    background-color: #8F3B1C !important;
}

/* Studio Log Terminal */
.studio-log textarea {
    font-family: 'JetBrains Mono', Consolas, monospace !important;
    font-size: 12px !important;
    line-height: 1.62 !important;
    background-color: #231C18 !important;
    border: 1px solid rgba(42, 34, 30, 0.15) !important;
    color: #F5EDE2 !important;
    border-radius: 8px !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}

/* Footer */
.studio-footer {
    text-align: center;
    padding: 24px 0 12px 0;
    margin-top: 28px;
    border-top: 1px solid rgba(42, 34, 30, 0.1);
    color: #8E7D72;
    font-size: 12.5px;
    letter-spacing: 0.04em;
    font-weight: 500;
}
.studio-footer strong {
    color: #B8542F;
    font-weight: 600;
}

/* Reduced Motion Mode for Accessibility */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
"""

# ---------------------------------------------------------------------------
# Backend Processing Logic
# ---------------------------------------------------------------------------
def process_video(video_path, engine, colab_url, intensity, style):
    """
    Worker function connected to the Gradio UI.
    Streams progress updates to the status textbox and outputs the final video.
    """
    if not video_path:
        yield None, "Please upload a video file first."
        return

    engine = str(engine).lower().strip()
    if engine == "mmaudio":
        if not colab_url or not str(colab_url).strip():
            yield None, "Please provide a valid Colab URL in the Advanced section to use the MMAudio sound engine."
            return

    msg_queue = queue.Queue()
    logs = []

    def on_progress(msg):
        msg_queue.put(("msg", msg))

    def worker():
        try:
            out_file = run_pipeline(
                video_path=video_path,
                intensity=float(intensity),
                style=str(style),
                progress=on_progress,
                engine=engine,
                colab_url=str(colab_url).strip() if colab_url else None,
            )
            msg_queue.put(("done", out_file))
        except Exception as err:
            msg_queue.put(("error", str(err)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive() or not msg_queue.empty():
        try:
            status, payload = msg_queue.get(timeout=0.25)
            if status == "msg":
                logs.append(payload)
                yield None, "\n".join(logs)
            elif status == "done":
                logs.append(f"\n[Success] Foley generation complete! Video ready: {payload}")
                yield payload, "\n".join(logs)
                return
            elif status == "error":
                friendly_msg = f"\n[Error] An issue occurred during Foley generation: {payload}\nPlease verify your settings and try again."
                logs.append(friendly_msg)
                yield None, "\n".join(logs)
                return
        except queue.Empty:
            pass

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Foley Studio") as demo:
    # Warm Header with Equalizer & Pulse Indicators
    gr.HTML("""
    <div class="studio-hero">
        <div class="studio-pills-row">
            <div class="status-badge">
                <span class="status-pulse-dot"></span>
                <span>Foley Synth Studio</span>
            </div>
            <div class="soundwave-viz">
                <span class="wave-bar wb-1"></span>
                <span class="wave-bar wb-2"></span>
                <span class="wave-bar wb-3"></span>
                <span class="wave-bar wb-4"></span>
                <span class="wave-bar wb-5"></span>
                <span class="wave-bar wb-6"></span>
            </div>
        </div>

        <h1 class="studio-main-title">AI <span class="gradient-text">Foley</span> Studio</h1>
        <p class="studio-hero-desc">Transform raw silent video into cinematic, physically synchronized Foley sound design using vision analysis, optical flow motion impacts, and generative audio.</p>

        <!-- 4-Step Pipeline Infographic -->
        <div class="pipeline-flow-container">
            <div class="flow-step active-step">
                <span class="step-icon">📹</span>
                <span>1. Raw Video In</span>
            </div>
            <span class="flow-arrow">➔</span>
            <div class="flow-step">
                <span class="step-icon">👁️</span>
                <span>2. Gemini Vision</span>
            </div>
            <span class="flow-arrow">➔</span>
            <div class="flow-step">
                <span class="step-icon">⚡</span>
                <span>3. Optical Flow Sync</span>
            </div>
            <span class="flow-arrow">➔</span>
            <div class="flow-step active-step">
                <span class="step-icon">🎬</span>
                <span>4. Master Video Out</span>
            </div>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # Left Column: Deck A (Raw Input & Acoustics Controls)
        with gr.Column(scale=5):
            gr.HTML("""
            <div class="deck-banner deck-input">
                <span>🎛️ Deck A : Raw Input & Synthesis Setup</span>
                <span>44.1kHz / 16kHz</span>
            </div>
            """)

            video_input = gr.Video(
                label="Raw Input Video",
                sources=["upload"],
            )
            gr.HTML("<div class='studio-helper'>Upload silent MP4/MOV footage to generate synchronized Foley sound design.</div>")

            engine_dropdown = gr.Dropdown(
                choices=["audioldm", "mmaudio"],
                value="mmaudio",
                label="Sound engine",
                info="MMAudio (cloud Colab GPU) for full-scene audio, or AudioLDM (local GPU) for impact-by-impact Foley.",
            )

            with gr.Accordion("Advanced", open=False):
                colab_url_input = gr.Textbox(
                    label="Colab URL",
                    placeholder="https://<id>.gradio.live",
                    info="Public Gradio share tunnel URL of your running Colab instance (required for MMAudio).",
                    type="text",
                )

            with gr.Row():
                intensity_slider = gr.Slider(
                    minimum=0.2,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Intensity",
                    info="Master volume multiplier (0.2x to 2.0x).",
                )

                style_dropdown = gr.Dropdown(
                    choices=["auto", "dry", "small_room", "large_hall"],
                    value="auto",
                    label="Style",
                    info="Acoustic spatialization reverb environment.",
                )

            generate_btn = gr.Button(
                "⚡ Generate Foley Video",
                variant="primary",
                size="lg",
                elem_classes=["btn-generate-cinema"],
            )

        # Right Column: Deck B (Master Output Video & Live Telemetry)
        with gr.Column(scale=6):
            gr.HTML("""
            <div class="deck-banner deck-output">
                <span>🎬 Deck B : Master Video & Real-Time Telemetry</span>
                <span>Stereo AAC</span>
            </div>
            """)

            video_output = gr.Video(
                label="Master Video with Synchronized Foley",
            )
            gr.HTML("<div class='studio-helper'>Preview and download your final cinematic video with synchronized Foley audio.</div>")

            status_box = gr.Textbox(
                label="Studio Progress & Alignment Log",
                placeholder="Click 'Generate Foley Video' to begin processing...",
                lines=13,
                max_lines=26,
                autoscroll=True,
                interactive=False,
                elem_classes=["studio-log"],
                info="Real-time log of video analysis, impact snapping, and sound synthesis.",
            )

    # Branded Footer
    gr.HTML("""
    <div class="studio-footer">
        Gemini Vision + optical flow + MMAudio &bull; <strong>AI Foley Studio</strong>
    </div>
    """)

    # Event binding
    generate_btn.click(
        fn=process_video,
        inputs=[video_input, engine_dropdown, colab_url_input, intensity_slider, style_dropdown],
        outputs=[video_output, status_box],
    )

if __name__ == "__main__":
    demo.launch(theme=studio_theme, css=custom_css, inbrowser=True)

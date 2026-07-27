"""
VoxShield AI — Deepfake Voice Detector
A single-file Streamlit app that uses a fine-tuned Wav2Vec2 model
to classify uploaded audio as authentic human speech or AI-generated
(synthetic / deepfake) speech.
"""

import time
import hashlib
import os

import numpy as np
import streamlit as st
import librosa
import plotly.graph_objects as go

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="VoxShield AI — Deepfake Voice Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# Model config
# ==========================================

# Larger Wav2Vec2-XLSR-53 model, specialized for deepfake audio detection
# and used in third-party audio-deepfake-detection benchmarks alongside
# AASIST/RawNet2. ~1.26GB — notably heavier than the previous model, so
# expect slower first-load and a real chance of hitting free-tier RAM
# limits on Streamlit Community Cloud.
MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
MIN_AUDIO_SECONDS = 1.0

REAL_KEYWORDS = ("real", "bona", "genuine", "human", "live")
FAKE_KEYWORDS = ("fake", "spoof", "synthetic", "ai", "clone", "generated")


# ==========================================
# Theme / CSS
# ==========================================

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.html(f"<style>{css}</style>")


def navbar():
    st.markdown(
        """
        <div class="vx-navbar">
            <div class="vx-brand"><span class="dot"></span> VoxShield AI</div>
            <div class="vx-navtag">wav2vec2 · neural voice forensics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero():
    bars = "".join(
        f'<span style="height:{h}%; animation-delay:{d}s;"></span>'
        for h, d in zip(
            [30, 55, 80, 45, 95, 60, 35, 70, 50, 85, 40, 65, 30, 90, 55, 75, 45, 60, 35, 80],
            [i * 0.07 for i in range(20)],
        )
    )
    st.markdown(
        f"""
        <div class="vx-hero">
            <div class="vx-eyebrow">Audio authenticity analysis</div>
            <h1>Is that voice really human?</h1>
            <p>Upload a clip and VoxShield runs it through a Wav2Vec2 model
            fine-tuned to separate genuine human speech from AI-cloned or
            text-to-speech generated audio — segment by segment.</p>
            <div class="vx-scope">{bars}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_cards():
    c1, c2, c3 = st.columns(3)
    cards = [
        ("01 · encode", "Acoustic encoding",
         "Raw audio is resampled to 16kHz and passed through a Wav2Vec2 "
         "backbone that turns the waveform into deep acoustic representations, "
         "not just surface-level pitch or volume."),
        ("02 · classify", "Segment-level scoring",
         "The clip is split into short segments so the model can catch "
         "artifacts that only show up in part of a recording, instead of "
         "judging the whole file as one average."),
        ("03 · verdict", "Aggregated confidence",
         "Per-segment scores are combined into a single real/synthetic "
         "verdict with a confidence score, plus the segment breakdown so "
         "you can see where suspicion is concentrated."),
    ]
    for col, (tag, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""
                <div class="vx-card">
                    <span class="tag">{tag}</span>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def footer():
    st.markdown(
        """
        <div class="vx-footer">
            VoxShield AI is a probabilistic tool, not a certified forensic
            instrument — no detector catches every deepfake, and results can
            be wrong. Treat the verdict as one signal among several, not
            proof, and re-check important cases with additional evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Model + inference
# ==========================================

@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import pipeline
    try:
        return pipeline("audio-classification", model=MODEL_ID, device=-1)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load the model ({MODEL_ID}). If this is an out-of-memory "
            f"error, this model (~1.26GB) may be too large for your current "
            f"hosting plan's free RAM tier — consider Hugging Face Spaces "
            f"(free CPU tier has much more RAM) or a paid Streamlit Cloud tier. "
            f"Original error: {e}"
        )


@st.cache_data(show_spinner=False)
def load_audio(file_bytes: bytes):
    import io
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, mono=True)
    return y


def chunk_audio(y: np.ndarray, sr: int = SAMPLE_RATE, chunk_seconds: int = CHUNK_SECONDS):
    chunk_len = chunk_seconds * sr
    min_len = int(MIN_AUDIO_SECONDS * sr)

    if len(y) < min_len:
        # pad very short clips by looping so the model has enough signal
        reps = int(np.ceil(min_len / max(len(y), 1)))
        y = np.tile(y, reps)[:min_len]

    chunks = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) < min_len:
            if not chunks:
                chunk = np.tile(chunk, int(np.ceil(min_len / max(len(chunk), 1))))[:min_len]
                chunks.append((start / sr, chunk))
            break
        chunks.append((start / sr, chunk))

    return chunks if chunks else [(0.0, y)]


def score_chunk(pipe, chunk: np.ndarray):
    """Returns (real_prob, fake_prob) for one audio chunk."""
    outputs = pipe({"array": chunk, "sampling_rate": SAMPLE_RATE}, top_k=None)

    real_p, fake_p = None, None
    for o in outputs:
        label = o["label"].lower()
        if any(k in label for k in FAKE_KEYWORDS):
            fake_p = o["score"]
        elif any(k in label for k in REAL_KEYWORDS):
            real_p = o["score"]

    # Fallback if label text didn't match known keywords: assume index
    # convention label_0 = real, label_1 = fake (common for these models).
    if real_p is None or fake_p is None:
        outputs_sorted = sorted(outputs, key=lambda o: o["label"])
        if len(outputs_sorted) >= 2:
            real_p = outputs_sorted[0]["score"]
            fake_p = outputs_sorted[1]["score"]
        else:
            fake_p = outputs_sorted[0]["score"]
            real_p = 1 - fake_p

    return real_p, fake_p


def analyze(file_bytes: bytes, status, progress):
    status.info("Loading Wav2Vec2 model...")
    progress.progress(10)
    pipe = load_model()

    status.info("Decoding and resampling audio...")
    progress.progress(30)
    y = load_audio(file_bytes)
    duration = len(y) / SAMPLE_RATE

    # Sanity check: a failed/garbled decode (common cause: missing ffmpeg
    # on the host, which mp3/m4a need) produces near-silent or empty audio.
    # If we score that, the model tends to return one very confident but
    # meaningless label instead of erroring out — so we catch it here.
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    if duration < 0.3 or rms < 1e-4:
        raise ValueError(
            f"Decoded audio looks silent or corrupted (duration={duration:.2f}s, "
            f"rms={rms:.6f}). This usually means the file didn't decode properly "
            f"rather than a real result — try a WAV file, or make sure ffmpeg is "
            f"installed on the server for MP3/M4A support."
        )

    status.info("Splitting into segments...")
    progress.progress(45)
    chunks = chunk_audio(y)

    status.info(f"Scoring {len(chunks)} segment(s)...")
    results = []
    for i, (t, chunk) in enumerate(chunks):
        real_p, fake_p = score_chunk(pipe, chunk)
        results.append({"time": t, "real": real_p, "fake": fake_p})
        progress.progress(45 + int(45 * (i + 1) / len(chunks)))

    status.success("Analysis complete!")
    progress.progress(100)

    mean_fake = float(np.mean([r["fake"] for r in results]))
    max_fake = float(np.max([r["fake"] for r in results]))
    mean_real = 1 - mean_fake
    # Weight peak evidence more than the average: a clip doesn't need to
    # sound synthetic for its whole duration to be a deepfake, so one
    # strongly-flagged segment shouldn't get diluted away by calmer ones.
    combined_fake = 0.4 * mean_fake + 0.6 * max_fake
    return {
        "duration": duration,
        "rms": rms,
        "chunks": results,
        "mean_fake": mean_fake,
        "max_fake": max_fake,
        "mean_real": mean_real,
        "combined_fake": combined_fake,
    }


# ==========================================
# Result rendering
# ==========================================

def render_gauge(confidence_pct: float, color: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={"suffix": "%", "font": {"color": color, "family": "JetBrains Mono", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B93AC", "tickfont": {"color": "#8B93AC"}},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": "rgba(255,255,255,0.04)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E9EDF7"},
    )
    return fig


def render_timeline(chunks, chunk_seconds=CHUNK_SECONDS):
    times = [f"{int(c['time'])}s–{int(c['time']) + chunk_seconds}s" for c in chunks]
    fake_scores = [round(c["fake"] * 100, 1) for c in chunks]
    colors = ["#FF5C7A" if f >= 50 else "#47E0B0" for f in fake_scores]

    fig = go.Figure(
        go.Bar(
            x=times,
            y=fake_scores,
            marker_color=colors,
            text=[f"{f}%" for f in fake_scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Synthetic probability (%)", range=[0, 105], gridcolor="#232C44", color="#8B93AC"),
        xaxis=dict(title="Segment", color="#8B93AC"),
        font={"color": "#E9EDF7", "family": "Inter"},
        showlegend=False,
    )
    return fig


def render_verdict(combined_fake: float, max_fake: float):
    fake_pct = combined_fake * 100
    real_pct = 100 - fake_pct
    peak_pct = max_fake * 100

    if peak_pct >= 50:
        # Even if the overall clip averages out calmer, one segment that's
        # majority-synthetic is meaningful evidence on its own — e.g. a
        # spliced-in synthetic portion — and shouldn't get buried under
        # "Inconclusive" just because the rest of the clip is calmer.
        css_class, label, verdict, color = "fake", "Verdict", "Likely AI-Generated", "#FF5C7A"
        conf = peak_pct
    elif fake_pct >= 65:
        css_class, label, verdict, color = "fake", "Verdict", "Likely AI-Generated", "#FF5C7A"
        conf = fake_pct
    elif fake_pct <= 35:
        css_class, label, verdict, color = "real", "Verdict", "Likely Authentic", "#47E0B0"
        conf = real_pct
    else:
        css_class, label, verdict, color = "warn", "Verdict", "Inconclusive", "#F5B942"
        conf = max(fake_pct, real_pct)

    st.markdown(
        f"""
        <div class="vx-verdict {css_class}">
            <div>
                <div class="label">{label}</div>
                <h2>{verdict}</h2>
            </div>
            <div class="vx-conf">
                <div class="num" style="color:{color}">{conf:.1f}%</div>
                <div class="cap">confidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gauge_value = peak_pct if peak_pct >= 50 else fake_pct
    return color, gauge_value


# ==========================================
# App
# ==========================================

load_css()
navbar()
hero()
feature_cards()

st.markdown('<div class="vx-section-label">Upload audio</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    st.markdown('<div class="vx-audiocard">', unsafe_allow_html=True)
    st.audio(file_bytes)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="vx-muted">FILE</div>', unsafe_allow_html=True)
        st.markdown(f"**{uploaded_file.name}**")
    with col2:
        st.markdown('<div class="vx-muted">SIZE</div>', unsafe_allow_html=True)
        st.markdown(f"**{uploaded_file.size / 1024:.1f} KB**")
    with col3:
        st.markdown('<div class="vx-muted">FORMAT</div>', unsafe_allow_html=True)
        st.markdown(f"**{uploaded_file.type or 'audio'}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Voice")

    if analyze_clicked:
        progress = st.progress(0)
        status = st.empty()

        try:
            result = analyze(file_bytes, status, progress)
        except Exception as e:
            status.error(f"Analysis failed: {e}")
            st.stop()

        time.sleep(0.3)
        status.empty()
        progress.empty()

        st.markdown('<div class="vx-section-label">Result</div>', unsafe_allow_html=True)
        color, gauge_value = render_verdict(result["combined_fake"], result["max_fake"])

        g_col, t_col = st.columns([1, 1.6])
        with g_col:
            st.plotly_chart(
                render_gauge(gauge_value, "#FF5C7A" if gauge_value >= 50 else "#47E0B0"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f'<div class="vx-muted" style="text-align:center">Weighted synthetic-speech score across {len(result["chunks"])} segment(s), {result["duration"]:.1f}s total '
                f'(avg {result["mean_fake"]*100:.0f}%, peak segment {result["max_fake"]*100:.0f}%)</div>',
                unsafe_allow_html=True,
            )
        with t_col:
            st.plotly_chart(render_timeline(result["chunks"]), use_container_width=True, config={"displayModeBar": False})

        with st.expander("Technical details"):
            st.markdown(f"**Model:** `{MODEL_ID}` (Wav2Vec2, fine-tuned for real vs. AI-generated speech)")
            st.markdown(f"**Sample rate:** {SAMPLE_RATE} Hz &nbsp;|&nbsp; **Segment length:** {CHUNK_SECONDS}s")
            st.markdown(f"**Decoded duration:** {result['duration']:.2f}s &nbsp;|&nbsp; **Signal RMS:** {result['rms']:.4f}")
            st.markdown(
                f"**Avg synthetic:** {result['mean_fake']*100:.1f}% &nbsp;|&nbsp; "
                f"**Peak segment:** {result['max_fake']*100:.1f}% &nbsp;|&nbsp; "
                f"**Weighted score:** {result['combined_fake']*100:.1f}%"
            )
            st.markdown("**Per-segment scores:**")
            for c in result["chunks"]:
                st.markdown(
                    f'<span class="mono vx-muted">t={c["time"]:.1f}s → real {c["real"]*100:.1f}% / synthetic {c["fake"]*100:.1f}%</span>',
                    unsafe_allow_html=True,
                )

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Upload an audio file to begin analysis.")

footer()

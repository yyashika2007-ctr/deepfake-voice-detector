"""
VoxShield AI — Deepfake Voice Detector
A single-file Streamlit app that uses a fine-tuned Wav2Vec2 model
to classify uploaded audio as authentic human speech or AI-generated
(synthetic / deepfake) speech.
"""

import time
import hashlib
import textwrap

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

# Fine-tuned Wav2Vec2 model, trained specifically for real-vs-AI speech
# classification (94.6M params — light enough to run on CPU in Streamlit
# Cloud, ~99.7% reported accuracy on its own eval set).
MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
MIN_AUDIO_SECONDS = 1.0

REAL_KEYWORDS = ("real", "bona", "genuine", "human", "live")
FAKE_KEYWORDS = ("fake", "spoof", "synthetic", "ai", "clone", "generated")


# ==========================================
# Theme / CSS
# ==========================================

def load_css():
    st.html(
        textwrap.dedent(
            """\
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <style>

        :root{
            --bg:            #0A0E17;
            --bg-alt:        #0D1220;
            --surface:       #121828;
            --surface-2:     #161D30;
            --border:        #232C44;
            --text:          #E9EDF7;
            --text-muted:    #8B93AC;
            --brand:         #8B7CF6;
            --brand-dim:     #6C5CE0;
            --real:          #47E0B0;
            --real-dim:      rgba(71,224,176,0.14);
            --fake:          #FF5C7A;
            --fake-dim:      rgba(255,92,122,0.14);
            --warn:          #F5B942;
            --warn-dim:      rgba(245,185,66,0.14);
        }

        html, body, [class*="css"]{
            font-family: 'Inter', sans-serif;
        }

        .stApp{
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(139,124,246,0.10), transparent 60%),
                radial-gradient(1000px 500px at 100% 0%, rgba(71,224,176,0.06), transparent 55%),
                var(--bg);
            color: var(--text);
        }

        #MainMenu, header[data-testid="stHeader"], footer{visibility:hidden;}
        .block-container{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }

        h1,h2,h3,h4{
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text) !important;
            letter-spacing: -0.01em;
        }

        .mono{
            font-family: 'JetBrains Mono', monospace;
        }

        /* ---------- Navbar ---------- */
        .vx-navbar{
            display:flex; align-items:center; justify-content:space-between;
            padding: 0.6rem 0 1.6rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2.2rem;
        }
        .vx-brand{
            display:flex; align-items:center; gap:0.6rem;
            font-family:'Space Grotesk', sans-serif;
            font-weight:700; font-size:1.25rem; color:var(--text);
        }
        .vx-brand .dot{
            width:10px; height:10px; border-radius:50%;
            background: var(--real);
            box-shadow: 0 0 12px var(--real);
        }
        .vx-navtag{
            font-family:'JetBrains Mono', monospace;
            font-size:0.72rem; color:var(--text-muted);
            border:1px solid var(--border); padding:0.3rem 0.65rem;
            border-radius:100px; background: var(--surface);
        }

        /* ---------- Hero ---------- */
        .vx-eyebrow{
            font-family:'JetBrains Mono', monospace;
            font-size:0.75rem; letter-spacing:0.12em; text-transform:uppercase;
            color: var(--brand); margin-bottom:0.8rem;
        }
        .vx-hero h1{
            font-size:2.6rem; line-height:1.15; margin-bottom:0.9rem;
        }
        .vx-hero p{
            color: var(--text-muted); font-size:1.05rem; max-width:640px;
            margin-bottom: 0;
        }

        /* Oscilloscope signature element */
        .vx-scope{
            display:flex; align-items:flex-end; gap:4px;
            height:64px; margin: 1.6rem 0 0.4rem 0;
        }
        .vx-scope span{
            display:block; width:4px; border-radius:2px;
            background: linear-gradient(180deg, var(--real), var(--brand));
            animation: vx-bounce 1.4s ease-in-out infinite;
            opacity:0.85;
        }
        @keyframes vx-bounce{
            0%,100%{ transform: scaleY(0.25); }
            50%{ transform: scaleY(1); }
        }

        /* ---------- Feature cards ---------- */
        .vx-card{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.3rem 1.4rem;
            height: 100%;
        }
        .vx-card .tag{
            font-family:'JetBrains Mono', monospace; font-size:0.7rem;
            color: var(--brand); letter-spacing:0.08em; text-transform:uppercase;
            margin-bottom:0.5rem; display:block;
        }
        .vx-card h4{ margin:0 0 0.4rem 0; font-size:1.05rem; }
        .vx-card p{ color:var(--text-muted); font-size:0.9rem; margin:0; line-height:1.5; }

        /* ---------- Upload card ---------- */
        .vx-section-label{
            font-family:'JetBrains Mono', monospace;
            font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase;
            color: var(--text-muted); margin: 2.4rem 0 0.6rem 0;
            display:flex; align-items:center; gap:0.6rem;
        }
        .vx-section-label::after{
            content:""; flex:1; height:1px; background: var(--border);
        }

        [data-testid="stFileUploader"]{
            background: var(--surface);
            border: 1.5px dashed var(--border);
            border-radius: 14px;
            padding: 0.6rem;
        }
        [data-testid="stFileUploader"] section{
            background: transparent;
        }

        .vx-audiocard{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin-top: 1rem;
        }

        /* ---------- Buttons ---------- */
        .stButton > button{
            background: linear-gradient(135deg, var(--brand), var(--brand-dim));
            color: #fff; border: none; border-radius: 10px;
            padding: 0.6rem 1.4rem; font-weight:600;
            font-family:'Space Grotesk', sans-serif;
            box-shadow: 0 6px 20px rgba(139,124,246,0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover{
            transform: translateY(-1px);
            box-shadow: 0 8px 26px rgba(139,124,246,0.4);
        }

        /* ---------- Progress / status ---------- */
        .stProgress > div > div{
            background: linear-gradient(90deg, var(--brand), var(--real));
        }

        /* ---------- Verdict card ---------- */
        .vx-verdict{
            border-radius: 16px; padding: 1.8rem 2rem; margin-top: 1.2rem;
            border: 1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between;
            flex-wrap:wrap; gap:1.2rem;
        }
        .vx-verdict.real{ background: var(--real-dim); border-color: rgba(71,224,176,0.4); }
        .vx-verdict.fake{ background: var(--fake-dim); border-color: rgba(255,92,122,0.4); }
        .vx-verdict.warn{ background: var(--warn-dim); border-color: rgba(245,185,66,0.4); }

        .vx-verdict .label{
            font-family:'JetBrains Mono', monospace; font-size:0.75rem;
            letter-spacing:0.1em; text-transform:uppercase; opacity:0.75;
        }
        .vx-verdict h2{ margin: 0.2rem 0 0 0; font-size:1.9rem; }
        .vx-verdict.real h2{ color: var(--real); }
        .vx-verdict.fake h2{ color: var(--fake); }
        .vx-verdict.warn h2{ color: var(--warn); }

        .vx-conf{ text-align:right; }
        .vx-conf .num{ font-family:'JetBrains Mono', monospace; font-size:2.2rem; font-weight:700; }
        .vx-conf .cap{ font-size:0.75rem; color: var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; }

        /* Misc text */
        .vx-muted{ color: var(--text-muted); font-size:0.88rem; }
        .vx-footer{
            margin-top: 3.5rem; padding-top: 1.4rem; border-top: 1px solid var(--border);
            color: var(--text-muted); font-size: 0.8rem; text-align:center;
        }

        [data-testid="stExpander"]{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        </style>
        """
        )
    )


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
    return pipeline("audio-classification", model=MODEL_ID, device=-1)


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
    mean_real = 1 - mean_fake
    return {
        "duration": duration,
        "chunks": results,
        "mean_fake": mean_fake,
        "mean_real": mean_real,
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


def render_verdict(mean_fake: float):
    fake_pct = mean_fake * 100
    real_pct = 100 - fake_pct

    if fake_pct >= 65:
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
    return color


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
        color = render_verdict(result["mean_fake"])

        g_col, t_col = st.columns([1, 1.6])
        with g_col:
            st.plotly_chart(
                render_gauge(result["mean_fake"] * 100, "#FF5C7A" if result["mean_fake"] >= 0.5 else "#47E0B0"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f'<div class="vx-muted" style="text-align:center">Synthetic-speech probability across {len(result["chunks"])} segment(s), {result["duration"]:.1f}s total</div>',
                unsafe_allow_html=True,
            )
        with t_col:
            st.plotly_chart(render_timeline(result["chunks"]), use_container_width=True, config={"displayModeBar": False})

        with st.expander("Technical details"):
            st.markdown(f"**Model:** `{MODEL_ID}` (Wav2Vec2, fine-tuned for real vs. AI-generated speech)")
            st.markdown(f"**Sample rate:** {SAMPLE_RATE} Hz &nbsp;|&nbsp; **Segment length:** {CHUNK_SECONDS}s")
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

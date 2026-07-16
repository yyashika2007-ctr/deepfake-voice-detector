import streamlit as st
import librosa, numpy as np, joblib, tempfile, os, time

st.set_page_config(page_title='DeepShield', page_icon='🎙️', layout='centered')

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#F8FAFC;
}

/* Main Container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:950px;
}

/* Hide Streamlit Menu */
#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Upload Box */
div[data-testid="stFileUploader"]{

    background:white;

    border:2px dashed #CBD5E1;

    border-radius:18px;

    padding:30px;

    transition:0.3s;
}

div[data-testid="stFileUploader"]:hover{

    border-color:#2563EB;

    background:#F8FBFF;
}

/* Buttons */

.stButton>button{

    background:#2563EB;

    color:white;

    border:none;

    border-radius:12px;

    height:52px;

    width:100%;

    font-size:18px;

    font-weight:600;
}

.stButton>button:hover{

    background:#1D4ED8;

    color:white;
}

/* Success */

div[data-testid="stSuccess"]{

    border-radius:15px;

    border-left:6px solid #22C55E;
}

/* Error */

div[data-testid="stError"]{

    border-radius:15px;

    border-left:6px solid #EF4444;
}

/* Metrics */

div[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    padding:15px;

    box-shadow:0 4px 12px rgba(0,0,0,.05);

    border:1px solid #E5E7EB;
}

/* Progress */

.stProgress>div>div{

    background:#2563EB;
}

hr{

    border:none;

    border-top:1px solid #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('voice_detector.pkl')

model=load_model()

def extract_features(path):
    y,sr=librosa.load(path,sr=16000,mono=True)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)
    return np.mean(mfcc,axis=1).reshape(1,-1)

st.markdown("""
<div style="text-align:center;padding-top:20px;">

<h1 style="font-size:50px;color:#4F9CF9;margin-bottom:0px;">
🛡️ DeepShield
</h1>

<h4 style="color:#D1D5DB;margin-top:5px;">
AI Voice Deepfake Detection System
</h4>

<p style="color:#9CA3AF;font-size:18px;">
Detect AI-generated speech using Machine Learning and MFCC audio analysis.
</p>

</div>
""", unsafe_allow_html=True)

f=st.file_uploader('Upload Audio',type=['wav','mp3','m4a'])
if f:
    st.audio(f)
    if st.button('Analyze Voice'):
        t=time.time()
        with tempfile.NamedTemporaryFile(delete=False,suffix='.wav') as tmp:
            tmp.write(f.read()); p=tmp.name
        try:
            X=extract_features(p)
            pred=model.predict(X)[0]
            proba=model.predict_proba(X)[0]
        finally:
            os.remove(p)
        conf=max(proba)*100
        real=proba[0]*100
        fake=proba[1]*100
        if pred==0:
            st.success(f'REAL VOICE ({conf:.1f}%)')
        else:
            st.error(f'AI GENERATED ({conf:.1f}%)')
        st.progress(int(conf))
        c1,c2=st.columns(2)
        c1.metric('Real',f'{real:.1f}%')
        c2.metric('Fake',f'{fake:.1f}%')
        st.write(f'Prediction Time: {time.time()-t:.2f}s')

import streamlit as st
import tempfile
import os
import time
import subprocess

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import imageio_ffmpeg




st.set_page_config(
    page_title="Bilingual Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Bilingual Speech Emotion Recognition")
st.write("Upload a speech audio file and predict the speaker's emotion.")


# ---------------- SIDEBAR ----------------
with st.sidebar:

    language = st.radio("🌐 Language", ["Kannada", "English"])

    st.header("🤖 Model Information")

    if language == "Kannada":
        st.metric("Test Accuracy", "58.82%")
        st.write("**Architecture**")
        st.write("CNN + BiLSTM + Attention")
        st.write("**Features**")
        st.write("MFCC + Mel + Chroma + ZCR")
        st.write("**Sample Rate**")
        st.write("22050 Hz")
    else:
        st.write("**Architecture**")
        st.write("wav2vec2 + Logistic Regression")
        st.write("**Emotions**")
        st.write("angry, happy, neutral, sad")
        st.write("**Sample Rate**")
        st.write("16000 Hz")


# ---------------- AUDIO CONVERSION ----------------
def convert_to_wav(input_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ).name

    command = [ffmpeg, "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg could not convert the audio:\n\n{error_message}")

    return output_path


# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "🎵 Upload Audio",
    type=["wav", "mp3", "mp4", "m4a", "aac", "flac", "ogg"]
)


if uploaded_file is not None:

    input_path = None
    wav_path = None

    try:

        extension = os.path.splitext(uploaded_file.name)[1]

        input_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        input_file.write(uploaded_file.getbuffer())
        input_file.close()
        input_path = input_file.name

        with st.spinner("Preparing audio..."):
            wav_path = convert_to_wav(input_path)

        y, sr = librosa.load(wav_path, sr=22050, mono=True)

        st.subheader("🔊 Audio")
        st.audio(wav_path, format="audio/wav")

        st.subheader("📈 Waveform")
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if st.button("🔍 Analyze Emotion", type="primary"):

            start_time = time.time()

            if language == "English":
                from ser_english import predict_english   # loads only English model
                emotion, prob_dict = predict_english(wav_path)
                classes = list(prob_dict.keys())
                probabilities = list(prob_dict.values())
            else:
                from inference import predict_emotion, CLASSES   # loads only Kannada model
                emotion, probabilities = predict_emotion(wav_path)
                classes = [str(c) for c in CLASSES]

            latency = time.time() - start_time

            st.subheader("🎯 Prediction")
            st.success(f"Predicted Emotion: **{emotion}**")
            st.metric("Prediction Time", f"{latency:.2f} seconds")

            st.subheader("📊 Emotion Probabilities")
            for class_name, probability in zip(classes, probabilities):
                st.write(f"**{class_name}** — {probability * 100:.2f}%")
                st.progress(float(probability))

    except Exception as e:
        st.error(f"❌ Could not process this audio file:\n\n{e}")

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
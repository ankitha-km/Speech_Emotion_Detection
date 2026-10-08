import streamlit as st
from inference import load_model, predict

st.title("SER — Emotion Prediction")
st.caption("Model trained on train+val+test combined (leakage experiment).")

uploaded = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded:
    model, classes, feat_mean, feat_std = load_model()

    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded.read())

    label, probs = predict("temp_upload.wav", model, classes, feat_mean, feat_std)
    clean = lambda name: name.split("_", 1)[1] if "_" in name else name
    clean_probs = {clean(k): v for k, v in probs.items()}
    confidence = probs[label]

    st.subheader(f"Predicted emotion: {clean(label).capitalize()}")
    st.write(f"Confidence: {confidence * 100:.1f}%")
    st.bar_chart(clean_probs)
    st.audio(uploaded)
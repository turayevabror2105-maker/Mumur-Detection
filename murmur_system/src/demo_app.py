from __future__ import annotations

import io

import librosa
import numpy as np
import pandas as pd
import streamlit as st

from .config import AUDIO_CONFIG, OUTPUT_DIR
from .dsp_features import extract_features, load_audio
from .gap2_quality_gate import evaluate_quality
from .models import load_model


st.set_page_config(page_title="Murmur Detection Demo", layout="centered")

st.title("Murmur Detection Demo")
st.write("Upload a heart sound recording (.wav) to get a baseline prediction.")

uploaded = st.file_uploader("Upload .wav", type=["wav"])

if uploaded is not None:
    try:
        audio_bytes = uploaded.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=AUDIO_CONFIG.sample_rate, mono=True)
        if audio_data.size == 0:
            st.error("Audio file is empty.")
        else:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            target_len = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.duration_sec)
            if audio_data.shape[0] < target_len:
                audio_data = np.pad(audio_data, (0, target_len - audio_data.shape[0]))
            else:
                audio_data = audio_data[:target_len]

            features, feature_names = extract_features(audio_data, sr)
            model, saved_features = load_model()
            if feature_names != saved_features:
                st.warning("Feature mismatch detected. Please retrain the model.")
            prob = model.predict_proba(features.reshape(1, -1))[0, 1]
            pred_label = "murmur" if prob >= 0.5 else "normal"
            confidence = float(max(prob, 1 - prob))

            quality = evaluate_quality(audio_data, uploaded.name)

            st.subheader("Prediction")
            st.write(f"**Label:** {pred_label}")
            st.write(f"**Probability (murmur):** {prob:.3f}")
            st.write(f"**Confidence:** {confidence:.3f}")

            st.subheader("Quality Gate")
            st.write(f"**Quality label:** {quality['quality_label']}")
            st.write(f"**Quality score:** {quality['quality_score']:.3f}")
            st.write(f"**Fail reasons:** {quality['fail_reasons']}")

            st.subheader("Explanation")
            explanation_path = OUTPUT_DIR / "tables" / "explanations.csv"
            if explanation_path.exists():
                explanations = pd.read_csv(explanation_path)
                st.write("Top features (global):")
                st.write(explanations["top_features"].iloc[0])
            else:
                st.write("Run the full pipeline first to generate explanations.")
    except Exception as exc:
        st.error(f"Failed to process audio: {exc}")
else:
    st.info("Upload a .wav file to start.")

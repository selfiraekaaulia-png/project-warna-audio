import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import pickle

# LOAD MODEL
model = load_model("model_ann.h5")

# LOAD SCALER
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# LOAD ENCODER
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# JUDUL WEB
st.title("Klasifikasi Suara Warna")

# UPLOAD AUDIO
uploaded_file = st.file_uploader(
    "Upload suara WAV",
    type=["wav"]
)

# EKSTRAK FITUR
def extract_feature(file_name):

    audio, sample_rate = librosa.load(
        file_name,
        sr=22050,
        mono=True
    )

    mfccs_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_scaled_features = np.mean(
        mfccs_features.T,
        axis=0
    )

    return mfccs_scaled_features

# PREDIKSI
if uploaded_file is not None:

    try:

        features = extract_feature(uploaded_file)

        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)

        predicted_label = encoder.inverse_transform(
            [np.argmax(prediction)]
        )

        st.success(
            f"Hasil Prediksi: {predicted_label[0]}"
        )

    except Exception as e:
        st.error(f"Error: {e}")

import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

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

uploaded_file = st.file_uploader(
    "Upload suara WAV",
    type=["wav"]
)

# EKSTRAK FITUR
def extract_feature(file):

    audio, sample_rate = librosa.load(
        file,
        sr=22050,
        mono=True
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_scaled_features = np.mean(
        mfccs.T,
        axis=0
    )

    return mfccs_scaled_features

# PREDIKSI
if uploaded_file is not None:

    try:

        features = extract_feature(uploaded_file)

        features = scaler.transform([features])

        prediction = model.predict(features)

        predicted_class = np.argmax(prediction)

        label = encoder.inverse_transform(
            [predicted_class]
        )[0]

        st.success(
            f"Hasil Prediksi: {label}"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

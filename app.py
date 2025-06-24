# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from utils import extract_features_dl
import os

# Load model and label encoder
MODEL_PATH = "model/emotion_cnn_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title("🎤 Speech Emotion Classifier")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    try:
        features = extract_features_dl("temp.wav")
        features = np.expand_dims(features, axis=0)  # shape (1, 80, 300, 1)
        
        predictions = model.predict(features)
        predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])
        
        st.success(f"Predicted Emotion: **{predicted_class[0]}**")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

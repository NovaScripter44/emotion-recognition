import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model("model\emotion_cnn_model.h5")
import joblib

label_encoder = joblib.load("model/label_encoder.pkl")


def extract_features(file_path, n_mfcc=40, max_len=300):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    combined = np.vstack((mfcc, delta))
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis]

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dim
    pred = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(pred)])
    return predicted_label[0]

if __name__ == "__main__":
    # Test file path
    test_file = "temp.wav"  # <-- replace with your test file

    if os.path.exists(test_file):
        emotion = predict_emotion(test_file)
        print(f"Predicted emotion for '{test_file}': {emotion}")
    else:
        print("Test audio file not found!")

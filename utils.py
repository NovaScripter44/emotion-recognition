# utils.py
import librosa
import numpy as np

def extract_features_dl(file_path, n_mfcc=40, max_len=300):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    combined = np.vstack((mfcc, delta))
    
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :max_len]
    
    return combined[..., np.newaxis]  # (80, 300, 1)

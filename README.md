# 🌟 Emotion Recognition from Speech using CNNs (RAVDESS Dataset)

This repository showcases a **complete end-to-end pipeline** for emotion recognition from speech using Deep Learning and the **RAVDESS** dataset. The project includes:

* 🎧 Audio processing and MFCC-based feature extraction
* 🤖 CNN-based deep learning model for classifying 8 emotions
* 🌐 Streamlit web application for testing via file upload
* 🧪 CLI script for direct testing

---

## 📚 Project Structure

```bash
emotion-recognition/
├── app.py                      # 🌐 Streamlit app
├── model/
│   ├── emotion_cnn_model.h5    # 🤖 Trained CNN model
│   └── label_encoder.pkl       # 🌿 Fitted LabelEncoder for class decoding
├── utils.py                   # ⚙️ Audio feature extraction & preprocessing
├── requirements.txt           # 📆 Python dependencies
├── test_audio/
│   └── test.wav               # 🎧 Sample test file
├── temp_wav/                  # 🎶 Temporarily stored user recordings
├── audio_emotion_classification.ipynb  # 📓 Model training notebook
└── test_model.py              # 🔮 CLI-based testing script
```

---

## 🚀 Features

* Trained on 8 classes:
  `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`
* Achieved:

  * ✅ Overall accuracy > 85%
  * ✅ Macro F1 score > 0.85
  * ✅ Minimum class accuracy ≥ 75%
* Includes **early stopping**, **class imbalance correction**, and **temporal feature engineering** (MFCC + Delta)

---

## 🔧 Model Details

### 🔊 Feature Extraction (via `utils.py`)

* `librosa` is used to compute:

  * **MFCCs** (Mel-Frequency Cepstral Coefficients)
  * **Delta** features (temporal derivative of MFCC)
* These are stacked to form a 2D input of shape **(80, 300, 1)**
* Padding and truncation ensure fixed-length inputs

### 🧠 CNN Architecture

```python
Sequential([
    Conv2D(32), BatchNorm, MaxPooling2D,
    Conv2D(64), BatchNorm, MaxPooling2D,
    Conv2D(128), BatchNorm, MaxPooling2D,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

* Optimizer: `Adam`
* Loss Function: `categorical_crossentropy`
* Trained with: `EarlyStopping`, `class_weight`

---

## 🚪 Streamlit Web App

 After Cloning the repo,Run the interactive frontend with:

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

### 🔊 App Features

* Upload `.wav` files for emotion detection
* Record live audio (up to 7s) using your browser
* Real-time prediction and result display

---

## 🔮 Testing from Script

Use `test_model.py` to test new samples without the UI:

```bash
python test_model.py
```

* Make sure the test file is in `test_audio/`
* Outputs the predicted emotion and optionally plots the MFCC

---

## 📄 Dataset

[RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)

* 24 professional actors
* 8 emotions (spoken & sung)
* 48 kHz, high-quality WAV audio
* Only **speech** files were used in this project

---

## 📃 Performance Snapshot

![image](https://github.com/user-attachments/assets/eba60231-6258-4b0a-a9c1-24410a7113b0)


---

## 🎓 Future Work

* Add **transformer-based models** for comparison
* Improve robustness for noisy environments
* Add **gender-based emotion analysis**

---

## 🌟 Contributors

* Aditya (NovaScripter44)

---

## 📚 License

This project is released for academic and learning purposes.

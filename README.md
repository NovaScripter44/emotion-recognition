# 🎭 Emotion Recognition from Speech (RAVDESS Dataset)

This project is a complete end-to-end pipeline for **speech/song emotion recognition**, built using:

- Deep Learning (CNNs)
- Audio feature extraction using **MFCCs + Delta**
- Interactive **Streamlit app** to test by uploading audio
- Trained and tested using the **RAVDESS dataset**

---

```
emotion-recognition/
├── app.py # 🌐 Streamlit app
├── model/
│ ├── emotion_cnn_model.h5 # 🤖 Trained CNN model
│ └── label_encoder.pkl # 🏷️ LabelEncoder for emotion classes
├── utils.py # ⚙️ Feature extraction (MFCC, delta), helpers
├── requirements.txt # 📦 Python dependencies
├── test_audio/
│ └── test.wav # 🎙️ Sample test audio
├── temp_wav/ # 🔊 Stores temp audio files
├── emotion-recognition_notebook.ipynb # 📓 Training + experiments notebook
└── test_model.py # 🧪 CLI script for testing model on sample
```



---

## 🚀 Features

- 🎧 Accepts uploaded `.wav` files
- 🎯 Trained on 8 emotion classes:
  `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`
- 📊 Achieved >80% accuracy and macro F1 score and >=75% accuracy in other classes
- 📈 Early stopping, class balancing, and MFCC + Delta stacking used for best performance

---

## 🛠️ How It Works

### 🔉 Feature Extraction

- We use **Librosa** to extract:
  - `MFCCs`: Mel-Frequency Cepstral Coefficients
  - `Delta`: First-order temporal derivative of MFCCs
- For CNN:
  - MFCC + Delta stacked vertically
  - Input shape: `(80, 300, 1)` — 80 features over 300 time steps

### 🧠 Model (CNN)

```python
Sequential([
    Conv2D(32), MaxPooling2D, BatchNorm,
    Conv2D(64), MaxPooling2D, BatchNorm,
    Conv2D(128), MaxPooling2D, BatchNorm,
    GlobalAveragePooling2D,
    Dense(128, relu) + Dropout(0.3),
    Dense(num_classes, softmax)
])
Optimizer: Adam

Loss: categorical_crossentropy

EarlyStopping and class_weight balancing added
```
---

## Streamlit App Usage:
### Activate virtual environment
```
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

### Install dependencies
pip install -r requirements.txt

### Run app
streamlit run app.py
```
---

## Testing using test_model.py :
python test_model.py

## 📌 Dataset Used:
```
https://zenodo.org/records/1188976#.XCx-tc9KhQI
-Only audio files (not video) were used.
-Contains recordings by 24 actors in 8 emotion categories.


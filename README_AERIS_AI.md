# Project AERIS - Adaptive Acoustic Intelligence & Noise Cancellation AI Suite

Project **AERIS** (Adaptive Enhanced Real-time Intelligence System) combines digital signal processing (DSP) Adaptive Noise Cancellation (ANC) with machine learning acoustic classification to identify signal corruption and optimize speech recovery in severe noise environments.

---

## 1. Directory Structure

```
c:\Users\nuppe\OneDrive\Documents\AERIS_SIH\
│
├── AI_MODEL/                                 # Trained Model Artifacts
│   ├── AERIS_noise_classifier.pkl            # Random Forest Classifier (100 estimators)
│   ├── AERIS_scaler.pkl                      # Feature Standard Scaler
│   ├── AERIS_features.json                   # Feature metadata & target classes
│   ├── AERIS_AI_CLASSIFICATION_REPORT.txt    # Precision, recall, and F1 report
│   ├── AERIS_AI_CONFUSION_MATRIX.png         # Confusion matrix plot
│   └── AERIS_AI_FEATURE_IMPORTANCE.png       # Feature importance bar chart
│
├── AERIS_AI_TRAIN.py                         # Complete AI Training Pipeline
├── test_aeris_ai.py                          # Offline Model Verification Test Suite
├── AERIS_BACKEND.py                          # FastAPI Production Backend Service
├── requirements.txt                          # Python Package Dependencies
│
├── index.html                                # Tactical Defense Frontend Dashboard
├── styles.css                                # Glassmorphic Aerospace Dark Theme
├── app.js                                    # Web Audio & Inference Dashboard Controller
│
└── README_AERIS_AI.md                        # Documentation & Quickstart Guide
```

---

## 2. Dataset Location & Acoustic Features

- **Source Directory**: `C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\OUTPUT\AERIS_AI_READY`
- **Feature CSV**: `AERIS_AI_FEATURES.csv`
- **Companion Audio**: `clean.wav`, `noisy.wav`, `lms_enhanced.wav`, `nlms_enhanced.wav`, `AERIS_CLEAN_-10dB.wav`, `AERIS_NOISY_-10dB.wav`, `AERIS_LMS_-10dB.wav`, `AERIS_LMS_-20dB.wav`, `AERIS_NLMS_-10dB.wav`, `AERIS_NLMS_-20dB.wav`.

### The 10 Extracted Acoustic Features:
1. `RMS`: Root Mean Square amplitude $\sqrt{\frac{1}{N}\sum x[n]^2}$
2. `Energy`: Total signal energy $\sum x[n]^2$
3. `ZeroCrossingRate`: Rate of sign-changes per sample
4. `MeanAbs`: Mean absolute value $\frac{1}{N}\sum |x[n]|$
5. `StdDev`: Sample standard deviation
6. `PeakAmplitude`: Peak absolute value $\max |x[n]|$
7. `CrestFactor`: Peak-to-RMS ratio
8. `SpectralCentroid`: Center of mass of the frequency spectrum (Hz)
9. `SpectralBandwidth`: Spectral spread around the centroid (Hz)
10. `SpectralRolloff`: Frequency below which 85% of spectral energy concentrates (Hz)

### Target Classification Classes:
- **`CLEAN`**: Reference uncorrupted speech
- **`NOISY`**: Acoustic input contaminated by high-amplitude noise (e.g. -10dB / -20dB SNR)
- **`LMS_ENHANCED`**: Least Mean Squares adaptive filter output
- **`NLMS_ENHANCED`**: Normalized Least Mean Squares adaptive filter output

---

## 3. How to Train the AI Model

Run the training pipeline:
```bash
python AERIS_AI_TRAIN.py
```
This script:
1. Loads `AERIS_AI_FEATURES.csv`.
2. Extracts frame-level feature vectors from companion audio WAV files.
3. Automatically cleans NaNs, Infs, and duplicates.
4. Performs a stratified 80/20 train/test split.
5. Fits a `StandardScaler`.
6. Trains a `RandomForestClassifier`.
7. Exports all evaluation metrics and `.pkl` artifacts to both `./AI_MODEL/` and the MATLAB dataset directory.

---

## 4. How to Verify the Model (CLI Test)

Run the verification test script:
```bash
python test_aeris_ai.py
```
This script validates model loading, scaler transformation, and tests each ground-truth sample from `AERIS_AI_FEATURES.csv`.

---

## 5. Starting the FastAPI Backend

Launch the backend microservice:
```bash
python AERIS_BACKEND.py
```
Or via uvicorn directly:
```bash
uvicorn AERIS_BACKEND:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6. Starting the Frontend Dashboard

Serve the frontend files with any static HTTP server:
```bash
python -m http.server 3000
```
Then open your browser at:
```
http://localhost:3000
```

---

## 7. Backend API Specification

### `GET /health`
- **Description**: Returns microservice and AI model status.
- **Sample Output**:
```json
{
  "status": "online",
  "system": "AERIS AI",
  "version": "2.4.0",
  "model_loaded": true,
  "features_count": 10,
  "classes": ["CLEAN", "LMS_ENHANCED", "NLMS_ENHANCED", "NOISY"]
}
```

### `POST /predict`
- **Description**: Accepts 10 acoustic numerical features and returns predicted class and confidence.
- **Expected Input**:
```json
{
  "features": {
    "RMS": 0.159659,
    "Energy": 2388.001,
    "ZeroCrossingRate": 0.174126,
    "MeanAbs": 0.099983,
    "StdDev": 0.159660,
    "PeakAmplitude": 1.0,
    "CrestFactor": 6.26334,
    "SpectralCentroid": 1031.22,
    "SpectralBandwidth": 1791.92,
    "SpectralRolloff": 1101.45
  }
}
```
- **Expected Output**:
```json
{
  "system": "AERIS AI",
  "prediction": "CLEAN",
  "confidence": 0.67,
  "probabilities": {
    "CLEAN": 0.67,
    "LMS_ENHANCED": 0.11,
    "NLMS_ENHANCED": 0.21,
    "NOISY": 0.01
  },
  "processed_features": { ... }
}
```

### `POST /predict_audio`
- **Description**: Multipart form-data endpoint accepting `.wav` audio upload. Automatically computes the 10 DSP features and returns the classification.

### `GET /anc_metrics`
- **Description**: Returns MATLAB ANC performance metrics across target SNRs (+5dB to -20dB).

### `GET /audio_presets/{preset_name}`
- **Description**: Streams authentic MATLAB ANC audio files (`clean`, `noisy`, `lms`, `nlms`).

---

## 8. Troubleshooting

1. **Backend Offline in Dashboard**:
   - Check if `AERIS_BACKEND.py` is running on port `8000`.
   - In the dashboard top bar, verify the backend node URL is set to `http://localhost:8000` and click **CONNECT**.
2. **Missing AI Model Artifacts**:
   - Run `python AERIS_AI_TRAIN.py` to regenerate all model and scaler artifacts.
3. **CORS Issues**:
   - The FastAPI backend includes `CORSMiddleware` configured to allow all origins (`*`).

# AERIS v2.5 — Adaptive Noise Cancellation & Intelligent Communication System
## AI/ML-Assisted Hybrid Adaptive Noise Cancellation & Acoustic Telemetry Prototype

AERIS is a real-time, low-latency software prototype combining **Random Forest AI acoustic intelligence (28 features)** with **LMS/NLMS adaptive digital signal processing** to achieve real-time speech enhancement in severe noise environments.

---

## ⚡ Quickstart Guide

### 1. Requirements & Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Modern Web Browser (Chrome, Edge, Firefox with Web Audio / AudioWorklet support)
- Headphones recommended to prevent speaker-to-microphone acoustic feedback during live mode

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Backend & Frontend Dashboard
```bash
# Start FastAPI server on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Or use the npm script:
```bash
npm start
```
*Open your browser and navigate to:* **`http://127.0.0.1:8000/`**

---

## 🏗️ System Architecture

```
                                      AERIS v2.5 PIPELINE
                                      
  [ LIVE MICROPHONE ]               [ CONTROLLED SIMULATION ]
          │                                     │
          ▼                                     ▼
 [ AudioWorklet (16 kHz) ]             [ Clean + Synthesized Noise ]
   (512-sample PCM frames)               (-20 dB to +10 dB SNR)
          │                                     │
          ▼                                     │
 [ WebSocket ws://.../realtime ]                │
          │                                     │
          ▼                                     │
 ┌──────────────────────────────────────────────┴──────────────────────────────────────┐
 │                              FASTAPI BACKEND PIPELINE                               │
 │                                                                                     │
 │  1. 28 Acoustic Features Extraction (~2.0 ms)                                       │
 │     • 15 Spectral & Temporal (RMS, ZCR, Spectral Centroid, Rolloff, Dominant Freq) │
 │     • 13 Mel-Frequency Cepstral Coefficients (MFCCs 1-13)                          │
 │                                                                                     │
 │  2. Random Forest Classification (RF-28, ~35.0 ms evaluated periodically)           │
 │     • Classes: CLEAN_SPEECH, NOISE_TMETRO, NOISY_SPEECH, ANC_ENHANCED               │
 │                                                                                     │
 │  3. Intelligent Filter Decision Policy                                              │
 │     • CLEAN_SPEECH   → Conservative Step (LMS M=16, μ=0.002)                        │
 │     • NOISE_TMETRO   → Stationary ANC (LMS M=16, μ=0.005)                           │
 │     • NOISY_SPEECH   → Dynamic High-Suppression (NLMS M=32, μ=0.150)                │
 │     • SEVERE/UNSTABLE→ Watchdog Norm-Clamping & Safe Recovery                       │
 │                                                                                     │
 │  4. Adaptive ANC DSP Engine (~3.6 ms)                                               │
 │     • Spectral Wiener Pre-filtering + LMS/NLMS Error Residual Filtering            │
 └──────────────────────────────────────────────┬──────────────────────────────────────┘
                                                │
                                                ▼
                                    [ WebSocket Duplex Stream ]
                                                │
                                                ▼
                                    [ AudioWorklet Ring Buffer ]
                                    (128 ms bounded jitter cap)
                                                │
                                                ▼
                                    [ Enhanced Voice Playback ]
                                    (Immediate Zero-Lag Audio)
```

---

## 📊 MATLAB Offline Benchmark Integration

AERIS integrates offline MATLAB validation datasets tested against the authentic LibriSpeech corpus mixed with TMETRO environmental subway noise across $+5\text{ dB}$ to $-20\text{ dB}$ SNR:

| Target SNR | LMS Gain (dB) | LMS MSE | NLMS Gain (dB) | NLMS MSE | Filter Stability |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **+5.0 dB** | +15.82 dB | 0.000312 | +14.95 dB | 0.000401 | **STABLE** |
| **0.0 dB** | +16.24 dB | 0.000517 | +15.68 dB | 0.000620 | **STABLE** |
| **-5.0 dB** | +16.05 dB | 0.000984 | +16.32 dB | 0.000910 | **STABLE** |
| **-10.0 dB** | +15.40 dB | 0.002140 | +16.85 dB | 0.001850 | **STABLE** |
| **-15.0 dB** | +13.90 dB | 0.005420 | +16.10 dB | 0.003920 | **STABLE** |
| **-20.0 dB** | +9.45 dB | 0.014200 | +14.80 dB | 0.008450 | **LMS UNSTABLE → NLMS SAFE** |

*Note: All benchmark figures are dynamically loaded from `OUTPUT/AERIS_MASTER_RESULTS.csv` and `OUTPUT/AERIS_MASTER_REPORT.txt`.*

---

## 🎯 10-Step Jury Demonstration Procedure

1. **Launch Dashboard**: Open `http://127.0.0.1:8000/`. Verify all 5 indicators are green (`ONLINE`, `READY`, `LOADED`, `ACTIVE`, `STREAMING`).
2. **Run Diagnostics**: Click **`⚙️ RUN SELF-TEST`** in the diagnostics panel. Confirm all 10 checks return **`PASS`**.
3. **Inspect MATLAB Benchmark**: Review the dynamically loaded table for $+5\text{ dB}$ to $-20\text{ dB}$ SNR and click **`📄 VIEW FULL REPORT`**.
4. **Configure Simulation**: Under **CONTROLLED NOISE SIMULATION**, choose `TMETRO Subway Noise` at `-10 dB SNR`.
5. **Generate 4 Streams**: Click **`⚡ GENERATE & PROCESS 4 STREAMS`**.
6. **Compare 4 Streams**:
   - Play **Clean Reference**
   - Play **Noisy Mixed** (notice intense $-10\text{ dB}$ background subway rumble)
   - Play **LMS Enhanced** (notice primary harmonic cancellation)
   - Play **NLMS Enhanced** (notice robust speech intelligibility recovery)
7. **Inspect Intelligibility Metrics**: Review Input SNR, Output SNR, Gain (+16.8 dB), MSE, and objective STOI scores.
8. **Switch to Live Microphone**: Put on headphones. Click **`🎙️ START LIVE MICROPHONE STREAMING`** and allow microphone access.
9. **Speak & Verify Real-Time Visualizers**:
   - **Waveforms**: Dual Blue (`INPUT`) and Green (`OUTPUT`) waveforms track your speech dynamically.
   - **FFT Spectrum**: 65-bin visualizer tracks your vocal formants (`DOM FREQ` and `CENTROID`).
   - **AI Intelligence**: Noise badge, confidence bar, and decision reasoning update continuously.
   - **Latency**: End-to-end latency shows ~6.6 ms to ~32 ms frame cycle.
10. **Record & Download**: Click **`⏺ LIVE RECORD`**, speak for 3–5 seconds, click **`⏹ STOP RECORD`**, and use **`💾 SAVE INPUT`** / **`💾 SAVE ENHANCED`** to download authentic 16 kHz WAV recordings.

---

## 🧪 Automated Verification Suite

Run the full 30-point acceptance test suite:
```bash
python tests/test_aeris_complete.py
```
*Expected Result:* `TOTAL ACCEPTANCE CRITERIA PASSED: 30 / 30 (100.0%)`

Run individual component unit tests:
```bash
python tests/verify_all.py
```
*Expected Result:* `15 / 15 PASSED (100.0%)`

---

## 📁 Repository File Layout

```
AERIS_SIH/
├── backend/
│   ├── main.py              # FastAPI server, WebSocket handlers, DSP pipeline, API endpoints
│   ├── feature_extractor.py # 28 Acoustic feature extraction (Time, Spectral, MFCCs)
│   ├── ai_classifier.py     # Random Forest model inference and feature scaling wrapper
│   ├── anc_engine.py        # Vectorized LMS and NLMS adaptive DSP filter implementation
│   └── config.py            # Core configuration constants (16 kHz, 512 samples)
├── FRONTEND/
│   ├── index.html           # Professional command center dashboard UI
│   ├── styles.css           # Clean defense/telemetry styling
│   ├── app.js               # Frontend controller, visualizer engines, WebSocket manager
│   └── aeris-processor.js   # Low-latency AudioWorklet processor (bounded 128ms ring buffer)
├── AI_MODEL/
│   ├── AERIS_noise_classifier.pkl  # Trained Random Forest classifier
│   ├── AERIS_scaler.pkl            # Trained StandardScaler model
│   └── AERIS_features.json         # 28-feature schema definitions
├── MATLAB_CODE/             # MATLAB simulation source scripts (.m)
├── OUTPUT/                  # Benchmark CSVs, report texts, and generated WAV streams
├── tests/
│   ├── test_aeris_complete.py      # 30-point comprehensive software acceptance test suite
│   ├── verify_all.py               # 15-point component verification test suite
│   └── test_live_api.py            # Live endpoint integration tests
├── package.json             # Root npm scripts configuration
├── requirements.txt         # Python dependencies
└── README.md                # Master documentation
```

---

## 🔒 Technical Disclaimers
- **Software Scope**: This prototype represents the complete PC/edge software implementation. Hardware microcontroller integration (ESP32/I2S) is scoped for subsequent deployment phases.
- **Reference SNR vs Live Mic**: True numerical SNR improvement is evaluated in Controlled Simulation and MATLAB Benchmark modes where ground-truth clean speech is mathematically available. In live microphone mode without a secondary acoustic reference, the dashboard displays `N/A — live mic` to maintain absolute scientific integrity.

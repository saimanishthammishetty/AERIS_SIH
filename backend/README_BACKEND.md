# AERIS FASTAPI BACKEND & REAL-TIME WEBSOCKET SERVICE

## 1. Overview
The AERIS backend is an asynchronous, high-performance microservice that interfaces real-time audio input, DSP acoustic feature extraction, Random Forest AI classification, adaptive ANC filtering (LMS/NLMS), and live WebSocket telemetry broadcasting.

---

## 2. API Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns server health, version, model loaded status, and environment tag. |
| `GET` | `/status` | Complete subsystem operational matrix (Mic, AI Model, ANC state, Latency). |
| `POST` | `/predict` | Performs AI acoustic classification on uploaded WAV audio or base64 stream. |
| `GET` | `/metrics` | Returns sub-stage latency breakdown, input/output SNR, and frame statistics. |
| `POST` | `/anc/control` | Starts, stops, resets, or configures ANC algorithms (LMS/NLMS, order, step size). |
| `GET` | `/audio_presets/{name}` | Serves validation preset WAV files for offline/simulation evaluation. |
| `WS` | `/realtime` | Low-latency bidirectional WebSocket connection streaming 20 Hz telemetry packets. |

---

## 3. Real-Time Latency Tracking & Sub-Stages

Every 512-sample frame (32 ms @ 16 kHz) is strictly instrumented across sub-stages:
1. **Audio Capture**: Ingestion from sounddevice or circular buffer.
2. **Feature Extraction**: 28 acoustic DSP, statistical, and MFCC features over a 250ms analysis window.
3. **AI Inference**: Scaled Random Forest evaluation with confidence thresholding.
4. **Adaptive ANC Filtering**: Sample-by-sample LMS/NLMS execution with stability clamping.
5. **Output Stage**: Energy computation and SNR improvement calculation.
6. **Total Latency**: Complete measured end-to-end processing delay.

---

## 4. Decision Logic Mapping

```
AI Noise Category ───> Recommended Mode ───> ANC Parameters
NOISE_TMETRO      ───> Active ANC       ───> LMS (Order: 16, mu: 0.005)
CLEAN_SPEECH      ───> Speech Guard     ───> Low Adaptation (mu: 0.0005)
LOW_CONFIDENCE    ───> Robust Default   ───> LMS (Order: 16, mu: 0.005)
```

---

## 5. Launching the Backend

```bash
cd BACKEND
python AERIS_BACKEND.py
```
*(Runs on `http://localhost:8000`).*

# AERIS (Adaptive Enhanced Real-Time Intelligence System)
### Complete Engineering & Architecture Manual

---

## 1. WHAT AERIS DOES
**AERIS** is an end-to-end **AI-Based Adaptive Noise Cancellation (ANC) & Acoustic Telemetry System**. It intercepts corrupted audio signals in high-noise environments (such as defense cockpits, industrial machinery, and tactical communication channels), classifies the acoustic noise profile using machine learning, and synthesizes phase-inverted anti-noise using adaptive digital filters to cancel background noise in real time while preserving human speech intelligibility.

---

## 2. WHY ACTIVE NOISE CANCELLATION (ANC) IS REQUIRED
Traditional passive noise reduction (earmuffs, acoustic foam, heavy insulation) only blocks high-frequency noise (> 1 kHz). Low-frequency acoustic noise (engine hum, rotor rumble, heavy machinery rumble, 50–500 Hz) has long wavelengths that penetrate physical barriers effortlessly. 
**Active Noise Cancellation (ANC)** uses electronic sensors and loudspeakers to generate an identical acoustic sound wave inverted by $180^\circ$ (anti-noise). When the anti-noise meets the ambient noise in the air, **destructive interference** occurs, cancelling the sound pressure waves mathematically ($1 + (-1) = 0$).

---

## 3. KEY CONCEPTS EXPLAINED IN SIMPLE LANGUAGE

### A. What is LMS (Least Mean Squares)?
The **LMS (Least Mean Squares)** algorithm is an adaptive digital filter that dynamically learns the noise environment. 
- It takes a reference noise input $x[n]$ and adjusts a set of digital weights $\mathbf{w}[n]$ at every single audio sample.
- It computes the anti-noise $y[n] = \mathbf{w}^T \mathbf{x}[n]$ and subtracts it from the noisy speech $d[n]$:
  $$e[n] = d[n] - y[n]$$
- It then updates the weights proportional to the error:
  $$\mathbf{w}[n+1] = \mathbf{w}[n] + 2\mu e[n] \mathbf{x}[n]$$
  where $\mu$ is the **step size** (learning rate).

### B. What is NLMS (Normalized Least Mean Squares)?
**NLMS** is an enhancement over standard LMS. In standard LMS, if the incoming noise suddenly gets very loud, the filter weights can adjust too violently and explode (diverge). NLMS divides the step size by the total energy of the input signal:
$$\mu_{\text{norm}} = \frac{\mu}{\|\mathbf{x}[n]\|^2 + \epsilon}$$
This prevents exploding filter coefficients during sudden acoustic bursts, providing higher mathematical stability at the cost of slightly higher arithmetic division operations.

### C. Why LMS Was Selected for the Primary Prototype
1. **Computational Efficiency**: Standard LMS requires only $2M + 1$ multiply-accumulate (MAC) operations per sample and zero divisions, making it suitable for low-power 32-bit microcontrollers (like ESP32 @ 240 MHz).
2. **Empirical Performance**: In MATLAB benchmark evaluations across stable SNR operating ranges (+5 dB to -10 dB), LMS achieved an average of **~16 dB SNR improvement** (e.g. recovering clean speech from -10 dB up to +5.79 dB).
3. **Safety Protection Guard**: To counteract low-SNR divergence risks, AERIS adds an active safety limiter that clamps weights and dynamically reduces $\mu$ when divergence is detected.

### D. What is SNR (Signal-to-Noise Ratio)?
**Signal-to-Noise Ratio (SNR)** measures the ratio of desired speech signal power to corrupting background noise power:
$$\text{SNR (dB)} = 10 \log_{10}\left(\frac{P_{\text{speech}}}{P_{\text{noise}}}\right)$$
- **+10 dB**: Clean, clear speech.
- **0 dB**: Speech and noise have equal power.
- **-10 dB**: Noise power is 10 times stronger than speech (severe jamming).
- **-20 dB**: Noise power is 100 times stronger than speech (near unintelligible).

### E. What is SNR Improvement?
$$\Delta\text{SNR} = \text{SNR}_{\text{output}} - \text{SNR}_{\text{input}}$$
If input audio enters at **-10.0 dB** and leaves the filter at **+5.79 dB**, the SNR improvement is:
$$\Delta\text{SNR} = +5.79 - (-10.0) = \mathbf{+15.79\text{ dB}}$$
This means the adaptive filter removed ~97% of the noise energy.

---

## 4. WHAT EACH SYSTEM COMPONENT DOES

| Component | Technology | Primary Role in AERIS |
| :--- | :--- | :--- |
| **MATLAB Layer** | MATLAB DSP Simulation | Performs rigorous offline mathematical verification, parameter tuning (order $M$, step size $\mu$), benchmark generation across +5 dB to -20 dB, and creates the baseline dataset. |
| **AI Classifier** | Scikit-Learn Random Forest | Analyzes 10 acoustic DSP features (RMS, ZCR, Spectral Centroid, Rolloff, Energy, Crest Factor) in real time to identify signal state (`CLEAN`, `NOISY`, `LMS_ENHANCED`, `NLMS_ENHANCED`) with confidence scoring. |
| **FastAPI Backend** | Python 3 Asynchronous Service | Microservice providing REST APIs (`/status`, `/metrics`, `/classify`, `/process_audio`), hardware COM port serial interfacing, and live telemetry streaming. |
| **Antigravity UI** | Tactical Engineering Dashboard | Military/defence console providing real-time status matrix, interactive graphs, dual-trace oscilloscope, FFT spectrum visualizer, jury demo selectors, and diagnostic reports. |
| **ESP32 MCU** | Tensilica Xtensa Dual-Core 240MHz | High-speed embedded microcontroller handling DMA-driven I2S digital audio sampling at 16000 Hz, lightweight real-time LMS filtering, and USB serial telemetry. |
| **INMP441 Mic** | Digital I2S MEMS Sensor | Captures acoustic air pressure directly into 24/32-bit digital PCM samples with no analog noise pickup. |
| **MAX98357A DAC** | 3.2W Class-D I2S Amplifier | Converts digital anti-noise samples from the ESP32 into high-efficiency analog speaker output for acoustic cancellation. |

---

## 5. COMPLETE ARCHITECTURE & DATA FLOW

```
                          +-----------------------------------+
                          |      MATLAB SIMULATION ENGINE     |
                          |  - LMS & NLMS Algorithm Design    |
                          |  - Benchmark Tests (+5 to -20 dB) |
                          +-----------------+-----------------+
                                            |
                                            v (AERIS_AI_FEATURES.csv & WAV Presets)
                          +-----------------+-----------------+
                          |    AI & AUDIO BACKEND (PYTHON)    |
                          |  - 10 Acoustic Features DSP Engine|
                          |  - Random Forest Classifier       |
                          |  - LMS/NLMS Stability Protections |
                          +-----------------+-----------------+
                                            |
                                            v (REST APIs / JSON Telemetry)
                          +-----------------+-----------------+
                          |   FASTAPI ASYNC BACKEND (PORT 8000)|
                          +--------+-------------------+------+
                                   |                   |
               (Telemetry & Stream)|                   |(USB Serial Port @ 115200)
                                   v                   v
+------------------------------------+   +------------------------------------+
|     ANTIGRAVITY COMMAND CENTER     |   |      ESP32 HARDWARE PROTOTYPE      |
|  - 8 Live Telemetry Cards          |   |  - INMP441 I2S Digital Mic         |
|  - Dual Oscilloscope & Spectrum    |   |  - Real-Time DMA Audio Loop        |
|  - Input vs Output SNR Graphs      |   |  - Embedded LMS ANC Filter         |
|  - Jury Demonstration Mode         |   |  - MAX98357A DAC & Speaker Output  |
+------------------------------------+   +------------------------------------+
```

---

## 6. HARDWARE WIRING SUMMARY

- **INMP441 I2S Mic → ESP32 DevKit V1**:
  - `VDD` → `3.3V` (Do NOT connect to 5V)
  - `GND` → `GND`
  - `SD` → `GPIO 32`
  - `WS` → `GPIO 15`
  - `SCK` → `GPIO 14`
  - `L/R` → `GND` (Left Channel)
- **MAX98357A I2S DAC → ESP32 DevKit V1**:
  - `VIN` → `5V` (or VIN)
  - `GND` → `GND`
  - `DIN` → `GPIO 22`
  - `BCLK` → `GPIO 26`
  - `LRC` → `GPIO 25`
  - `GAIN` → `GND` (+9 dB gain)
  - `OUT+` / `OUT-` → 4Ω / 8Ω 3W Speaker

> [!NOTE]
> Components such as MPU6050, DS18B20, FC-03, L298N, and 775 DC motor belong to your rotating-machinery project and are **NOT** audio ANC sensors.

---

## 7. HOW TO RUN THE COMPLETE PROJECT

### Option A: One-Click Launcher (Recommended)
1. Double-click **`setup_aeris.bat`** (first time only) to install requirements, verify the Python environment, train the AI model, and check system integrity.
2. Double-click **`run_aeris.bat`** to start the FastAPI backend and automatically open the **AERIS Command Center** in your web browser (`http://localhost:8000`).

### Option B: Manual Command-Line Launch
```bash
# 1. Train the AI model
python AERIS_AI_TRAIN.py

# 2. Start the Backend & Command Center
python AERIS_BACKEND.py
```
Open your browser and navigate to: `http://localhost:8000`

---

## 8. HOW TO DEMONSTRATE TO A JURY

1. **Open the Dashboard**: Show the high-tech defence interface with live clock, ping latency, and subsystem telemetry matrix.
2. **Demonstrate Jury Mode (MATLAB Validation)**:
   - Click through **`+5 dB`**, **`0 dB`**, **`-5 dB`**, **`-10 dB`**, **`-15 dB`**, and **`-20 dB`**.
   - Show how the **Input vs Output SNR** and **SNR Improvement** graphs update dynamically.
   - Point out that at **-10 dB** (nominal operating condition), LMS delivers **+15.79 dB** noise rejection gain.
   - Click **`PLAY`** to let the jury hear the authentic corrupted input vs LMS-enhanced speech.
3. **Demonstrate Safety Protection**:
   - Select **`-20 dB`**.
   - Show how the dashboard triggers **"ANC SAFETY LIMIT ACTIVE"**, explaining that at extreme negative SNR, the system automatically clamps exploding coefficients and scales down step size $\mu$ to prevent audio divergence.
4. **Demonstrate Live Audio Pipeline**:
   - Click **`PROCESS AUDIO`** on any custom or preset WAV file to show real-time end-to-end processing (DSP feature extraction $\to$ AI noise classification $\to$ LMS filtering $\to$ dual waveform display).
5. **Demonstrate Hardware Readiness**:
   - Connect the ESP32 via USB and select the COM port to show the live serial telemetry link.

---

## 9. LIMITATIONS & FUTURE IMPROVEMENTS

### Current Limitations:
- **Acoustic Latency Constraint**: In a free-air acoustic cancellation headphone, electronic latency must be under $150\mu\text{s}$. The ESP32 prototype demonstrates the DSP algorithms and digital filtering pipeline; commercial flight avionics use dedicated sub-microsecond DSP ASICs.
- **Secondary Path Modeling**: Changes in physical ear-cup acoustic fit affect the transfer function $S(z)$.

### Future Roadmap:
- **Filtered-X LMS (FxLMS)**: Integrating offline/online secondary path modeling ($S(z)$) to compensate for speaker and microphone transfer functions.
- **Variable Step-Size LMS (VSS-LMS)**: Automatically tuning $\mu$ dynamically based on instantaneous error gradient.
- **Dedicated DSP ASIC / FPGA**: Porting the verified algorithm to an ultra-low-latency mixed-signal audio processor (e.g. ADAU1787) for sub-10 microsecond flight-grade physical cancellation.

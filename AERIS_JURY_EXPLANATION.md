# PROJECT AERIS - JURY PRESENTATION & DEFENSE GUIDE

---

## 1. THE 30-SECOND ELEVATOR PITCH
> *"Respected Jury members, **AERIS** is an AI-driven Adaptive Noise Cancellation and acoustic telemetry system designed for high-noise defense and mission-critical communications. It combines a 10-feature acoustic AI classifier with an adaptive LMS filter that actively removes up to **16 dB of acoustic jamming and background noise**, recovering intelligible human speech from extreme -10 dB noise environments. We have built an end-to-end architecture connecting MATLAB mathematical validation, a Python FastAPI microservice, an embedded ESP32 hardware prototype with I2S audio peripherals, and a tactical Command Center dashboard with active stability protection against filter divergence."*

---

## 2. THE 1-MINUTE EXECUTIVE SUMMARY
> *"In defense aviation and tactical communications, low-frequency engine, rotor, and machinery noise severely degrades speech intelligibility and voice command accuracy. Traditional passive earmuffs fail at low frequencies.*
>
> ***AERIS solves this through a 4-tier hybrid intelligence architecture:***
> 1. **Mathematical DSP Layer (MATLAB)**: Rigorously benchmarked adaptive LMS and NLMS algorithms across severe noise conditions (+5 dB down to -20 dB), establishing ~16 dB nominal noise reduction.
> 2. **Acoustic AI Intelligence**: A Random Forest classifier operating on 10 extracted DSP features (spectral centroid, rolloff, zero-crossing rate, energy, crest factor) that detects the acoustic noise environment in real time.
> 3. **Embedded Prototype (ESP32)**: Utilizing digital I2S MEMS microphones (INMP441) and I2S DAC amplifiers (MAX98357A) running real-time adaptive filtering with DMA circular buffers.
> 4. **AERIS Command Center**: A live telemetry console displaying real-time dual-trace oscilloscopes, FFT spectrum power density, and an automated safety guard that prevents audio divergence under extreme noise conditions."*

---

## 3. THE 3-MINUTE IN-DEPTH PRESENTATION

### Slide/Section 1: The Operational Problem & Motivation
- **Context**: Military personnel, pilots, and industrial operators operate in environments with noise levels exceeding 95–115 dBA.
- **Challenge**: Passive barriers cannot attenuate low frequencies below 500 Hz due to long acoustic wavelengths.
- **Goal**: Build a digital adaptive system that intercepts noise, extracts acoustic signatures via AI, and cancels noise in real time while preserving speech.

### Slide/Section 2: Mathematical Foundation & Algorithm Selection
- We evaluated both **LMS (Least Mean Squares)** and **NLMS (Normalized LMS)** on 16000 Hz audio.
- **Filter Parameters**: LMS (Filter Order $M = 16$, Step Size $\mu = 0.005$) and NLMS (Order $M = 32$, $\mu = 0.150$).
- **Why LMS for the Prototype**: Requires only $2M + 1$ multiply-accumulate operations per sample and zero divisions, making it exceptionally fast on embedded 32-bit microcontrollers.
- **Benchmark Findings**:
  - At **-10 dB** (severe noise), LMS elevates output SNR to **+5.79 dB** (a **+15.79 dB gain**).
  - At **-20 dB** (extreme stress), LMS can experience weight divergence. To solve this, AERIS implements an **active coefficient clamping & step-size reduction safety guard**.

### Slide/Section 3: AI Noise Classification & Telemetry Hub
- Rather than blindly applying filters, AERIS extracts **10 acoustic features**: RMS, Energy, ZCR, MeanAbs, StdDev, PeakAmplitude, CrestFactor, SpectralCentroid, SpectralBandwidth, and SpectralRolloff.
- A trained **Random Forest classifier** categorizes signal states with high confidence and feeds live telemetry into a **FastAPI backend** running on port 8000.
- The **Antigravity Command Center** provides real-time dual-trace oscilloscopes, live FFT power density, jury demonstration modes, and diagnostic reporting.

### Slide/Section 4: Hardware Implementation & Commercial Path
- We built the hardware demonstrator using an **ESP32 DevKit V1**, an **INMP441 I2S digital microphone**, and a **MAX98357A Class-D I2S amplifier**.
- We distinguish between software algorithmic proof-of-concept, microcontroller prototyping, and the future commercial path using dedicated sub-microsecond DSP ASICs (e.g. ADAU1787) for flight-grade physical over-the-air acoustic phase cancellation.

---

## 4. SYSTEM ARCHITECTURE SUMMARY

```
[Acoustic Environment] ──> [INMP441 I2S Digital Mic]
                                  │
                                  ▼
[ESP32 Microcontroller] ──> Real-Time DMA I2S Stream (16 kHz, 16-bit)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [Embedded LMS Filter]             [USB Serial Telemetry]
                 │                                 │
                 ▼                                 ▼
      [MAX98357A I2S DAC]                [Python FastAPI Backend]
                 │                                 │
                 ▼                                 ▼
         [Speaker Output]               [Random Forest AI Engine]
                                                   │
                                                   ▼
                                      [AERIS COMMAND CENTER DASHBOARD]
```

---

## 5. LIKELY JURY QUESTIONS & MODEL ANSWERS

### Q1: Why did you choose LMS as the primary algorithm instead of deep learning or neural networks for the ANC filtering?
> **Answer**: *"Deep learning neural networks (like RNNs or Wave-U-Net) have significant inference latency (typically 10 to 50 milliseconds) and high power consumption. For active acoustic noise cancellation, sound travels at $343\text{ m/s}$, meaning sound covers a 5 cm headphone distance in only $145\text{ }\mu\text{s}$. Adaptive LMS requires only a few microseconds and minimal energy, making it mathematically capable of real-time sample-by-sample anti-noise synthesis on low-power embedded hardware. We use AI strategically at the supervisory telemetry and classification layer, where millisecond latencies are perfectly acceptable."*

### Q2: What happens at extreme noise levels like -15 dB and -20 dB?
> **Answer**: *"At very low SNRs (-15 dB to -20 dB), the error signal is dominated by overwhelming noise energy, which can cause standard LMS filter coefficients to explode or diverge. In AERIS, we specifically engineered an **Active Safety Protection System**. When the weight vector norm exceeds our stability threshold or when excessive signal amplitude is detected, the system automatically triggers 'ANC SAFETY LIMIT ACTIVE', clamps the coefficient norm, and dynamically reduces the step size $\mu$ to prevent audio corruption and speaker squeal."*

### Q3: What is the purpose of the 10 acoustic DSP features?
> **Answer**: *"The 10 features—RMS, Energy, Zero-Crossing Rate, Mean Absolute Amplitude, Standard Deviation, Peak Amplitude, Crest Factor, Spectral Centroid, Spectral Bandwidth, and Spectral Rolloff—capture both time-domain amplitude dynamics and frequency-domain spectral energy distribution. For example, noisy acoustic jamming concentrates spectral energy at low frequencies (low centroid ~180 Hz), whereas clean speech has rich harmonic distribution across 1 kHz to 4 kHz. This allows our Random Forest classifier to differentiate clean voice, acoustic noise, and adaptively recovered signals without high computational overhead."*

### Q4: Does this prototype achieve complete over-the-air acoustic phase cancellation in a room?
> **Answer**: *"To be scientifically rigorous: our ESP32 prototype demonstrates real-time digital filtering, I2S DMA streaming, adaptive weight updates, stability protection, and telemetry integration. However, true physical over-the-air acoustic phase cancellation in free space requires total electronic latency under $150\mu\text{s}$ to beat the speed of sound. While the ESP32 demonstrates the algorithm and digital signal processing pipeline, commercial flight avionics implement the same LMS equations on dedicated sub-microsecond DSP ASICs (like Analog Devices ADAU1787) or FPGAs. We have designed the AERIS algorithm suite so it is directly portable to such dedicated DSP hardware."*

### Q5: What is the relationship between your MATLAB work and this Python/Web system?
> **Answer**: *"MATLAB is our mathematical design and benchmark validation authority. It performed the offline parameter sweeps, MSE convergence analysis, and generated the baseline empirical dataset across the 6 test SNRs (+5 to -20 dB). Our Python FastAPI backend and Antigravity Command Center load and reference this authentic MATLAB validation data in the 'Jury Demonstration Mode', while providing the live AI inference engine, DSP pipeline, and hardware serial link."*

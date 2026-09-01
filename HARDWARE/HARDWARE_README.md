# AERIS HARDWARE ARCHITECTURE & REAL-TIME EMBEDDED ANC GUIDE

## 1. FEEDFORWARD ADAPTIVE NOISE CANCELLATION PRINCIPLE

Active Noise Cancellation (ANC) relies on the principle of **acoustic superposition and destructive interference**. In a feedforward ANC architecture, the system operates as follows:

```
                          [ Acoustic Noise Source ]
                                  |
               +------------------+------------------+
               | (Acoustic Path)                     | (Primary Acoustic Path P(z))
               v                                     v
     [ Reference Microphone ]              [ Error / Primary Mic ]
               |                                     |
               | Captures:                           | Captures:
               | Noise Reference x[n]                | Speech + Corrupted Noise d[n]
               v                                     |
     +-------------------+                           |
     |  Adaptive Filter  |                           |
     |   (LMS / NLMS)    |                           |
     |    Weights W(z)   |                           |
     +-------------------+                           |
               |                                     |
               | Generates Anti-Noise y[n]           |
               v                                     v
     +-------------------+                  +-----------------+
     | I2S DAC / Speaker | ---------------->| Cancellation    |
     |  (Acoustic Wave)  | (Inverted Phase) | Zone (Ear/Zone) |
     +-------------------+                  +-----------------+
                                                     |
                                                     v
                                            Enhanced Clean Speech
                                               e[n] = d[n] - y[n]
```

### Subsystem Breakdown:
1. **Reference Microphone ($x[n]$)**: Placed facing the exterior acoustic environment to detect incoming acoustic noise before it reaches the listener's ear.
2. **Error / Primary Microphone ($d[n]$)**: Positioned inside the ear cup or near the speaker output to monitor the residual sound ($d[n] = s[n] + n[n]$).
3. **Adaptive Filter ($\mathbf{W}(z)$)**: Models the secondary acoustic transfer function and continuously updates filter coefficients using Least Mean Squares (LMS) or Normalized LMS (NLMS) to minimize residual error power $E[e^2[n]]$.
4. **Audio Output / DAC & Speaker**: Emits the phase-inverted anti-noise wave to destructively cancel ambient acoustic pressure.

---

## 2. LATENCY CONSTRAINTS IN REAL-TIME ACOUSTIC ANC

A practical feedforward ANC system is bound by the **acoustic travel time**:
$$\text{Speed of Sound } c \approx 343\text{ m/s} = 0.343\text{ mm}/\mu\text{s}$$

- For a headphone shell where the distance between reference mic and speaker is $d = 5\text{ cm}$ ($50\text{ mm}$), the acoustic propagation delay is:
$$t_{\text{prop}} = \frac{50\text{ mm}}{0.343\text{ mm}/\mu\text{s}} \approx 145\text{ }\mu\text{s}$$

- The total electronic latency of the cancellation chain:
$$t_{\text{electronic}} = t_{\text{ADC}} + t_{\text{DMA buffer}} + t_{\text{DSP computation}} + t_{\text{DAC}} + t_{\text{Amp}}$$

> [!IMPORTANT]
> To achieve **true physical over-the-air acoustic cancellation**, $t_{\text{electronic}}$ must be less than or equal to $t_{\text{prop}}$ ($\le 145\text{ }\mu\text{s}$). If electronic latency exceeds acoustic propagation time, the anti-noise arrives too late, resulting in constructive interference (amplified noise) rather than cancellation.

---

## 3. CLEAR DISTINCTION: SIMULATION vs PROTOTYPE vs PRODUCTION

To maintain complete scientific and engineering integrity:

| System Layer | Implementation | Purpose & Capabilities | Latency Target |
| :--- | :--- | :--- | :--- |
| **1. MATLAB Simulation** | Offline block DSP simulation (`AERIS_MASTER.m`, `AERIS_LMS_ANC.m`) | Rigorous algorithm verification, mathematical proof-of-concept, baseline SNR optimization (+5 to -20 dB). | Non-real-time |
| **2. ESP32 Prototype** | Dual-core ESP32 @ 240 MHz + INMP441 I2S mic + MAX98357A DAC + Serial Telemetry | Functional hardware demonstrator showing real-time digital filtering, DMA streaming, stability protection, and telemetry integration. | ~1.4 ms DMA buffer |
| **3. Production ANC Headset** | Dedicated Ultra-Low-Power DSP / FPGA / Custom ASIC (e.g., ADI ADAU1787, Cirrus Logic CS47L35) | True low-latency over-the-air acoustic phase cancellation with sub-10 microsecond group delay. | $< 15\text{ }\mu\text{s}$ |

> [!CAUTION]
> **Engineering Reality**: A single development board (ESP32) with a single microphone and standard speaker demonstrates the algorithmic principles and digital signal processing pipeline of ANC. It should **not** be presented as a commercial-grade active noise cancelling headset ready for flight avionics without dedicated sub-microsecond analog/mixed-signal hardware.

---

## 4. SENSOR CLARIFICATION & PROJECT ISOLATION

The following components from your other rotating-machinery project must **NOT** be used for audio ANC:
- **MPU6050** (6-axis IMU accelerometer/gyroscope): Used for mechanical vibration/tilt monitoring, not acoustic audio.
- **DS18B20** (1-wire digital temperature probe): Used for thermal monitoring of bearings/motors.
- **FC-03** (Infrared optocoupler speed sensor): Used for motor RPM measurement.
- **L298N** (Dual H-Bridge Motor Driver): Used for high-current DC motor control.
- **12V 775 DC Motor**: Mechanical drive actuator.

**For AERIS Audio ANC, use only:**
1. ESP32 DevKit V1
2. INMP441 / ICS-43434 I2S Digital MEMS Microphone
3. MAX98357A I2S DAC Class-D Audio Amplifier
4. 4Ω / 8Ω 3W Speaker / Earphones

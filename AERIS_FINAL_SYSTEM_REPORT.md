# AERIS — System Architecture and Integration Notes

## 1. Overview

AERIS (Adaptive Noise Cancellation & Intelligent Communication System) is a prototype exploring AI-assisted acoustic analysis and adaptive DSP for communication in noisy environments.

The repository contains a PC-based software pipeline, MATLAB evaluation material, and an embedded development workstream.

## 2. Software architecture

```text
Audio files / controlled test signals
                |
                v
       Audio preprocessing
                |
                v
      Acoustic feature extraction
                |
                v
       AI classification stage
                |
                v
        LMS / NLMS DSP stage
                |
                v
      Processed signal and metrics
                |
                v
       FastAPI service / dashboard
```

The AI classifier and adaptive filter have distinct roles. Classification can characterize the input or inform processing choices; adaptive ANC still requires a valid reference signal correlated with the noise to be cancelled.

## 3. Embedded workstream

The selected target controller is ESP32. The selected MPU6050 sensor provides acceleration and angular-velocity measurements for inertial sensing.

The MPU6050 is not an audio microphone. It does not provide the acoustic samples needed for conventional audio ANC. The physical audio input/output chain remains a separate design and validation task.

## 4. TinyML status

The TinyML model-training milestone has been completed in the project workflow. This should not be confused with verified ESP32 deployment. Embedded deployment requires:
1. Exporting the trained model to a supported format.
2. Checking model size and RAM/flash requirements.
3. Integrating the inference runtime.
4. Flashing and running the model on the ESP32.
5. Measuring inference latency and validating predictions on-device.

## 5. Validation levels

| Level | Evidence required |
|---|---|
| Offline simulation | Reproducible MATLAB/Python test data and metrics |
| Desktop software | Backend, classifier, DSP, and dashboard tests |
| Embedded sensing | Verified ESP32 firmware and MPU6050 readings |
| Embedded TinyML | On-device model inference and measured resource use |
| Physical acoustic ANC | Valid audio reference/error microphones, output path, latency and acoustic measurements |

Passing a lower-level test does not establish a higher-level capability.

## 6. Limitations and next steps

- Confirm and document the actual ESP32 board and MPU6050 wiring.
- Verify MPU6050 I2C communication and sensor readings.
- Export and validate the TinyML model on the ESP32.
- Design a suitable microphone/reference and audio-output architecture for physical ANC.
- Measure latency, stability, speech intelligibility, and noise reduction under repeatable conditions.
- Keep all hardware and performance claims tied to recorded tests.

## 7. Scope statement

AERIS is a research/prototype project. This repository does not claim field readiness, production-grade acoustic cancellation, or validated deployment in operational environments.

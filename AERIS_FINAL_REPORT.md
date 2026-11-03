# AERIS — Project Status and Verification Notes

> **Status note:** This document summarizes the repository's software prototype and current engineering scope. Older numerical reports in the repository may describe earlier builds and should not be treated as current results without reproducing their test procedures.

## Project

**AERIS — Adaptive Noise Cancellation & Intelligent Communication System**

AERIS explores AI-assisted acoustic analysis and adaptive digital signal processing to improve speech communication in noisy environments.

## Components present in the repository

- Python FastAPI backend.
- Acoustic feature extraction.
- Python Random Forest classifier and associated model artifacts.
- LMS and NLMS adaptive-filter implementations and experiments.
- Browser-based dashboard.
- MATLAB offline simulation and evaluation files.
- ESP32 hardware-development material.
- Software test and verification scripts.

## Embedded direction

The selected embedded controller is ESP32, with an MPU6050 used for inertial sensing. The TinyML model-training milestone has been completed in the project workflow.

Training completion must be distinguished from deployment: on-device inference is only verified after a compatible model has been exported, loaded on the ESP32, and tested with measured memory use, inference latency, and output correctness.

## Current limitations

- MPU6050 is an accelerometer/gyroscope, not an audio microphone.
- The ESP32 + MPU6050 combination alone does not provide the audio reference/error signals required for conventional acoustic ANC.
- Microphone-based physical ANC and its acoustic performance are not validated by the presence of MATLAB/Python simulation results.
- Offline SNR improvements apply only to the reported datasets and test conditions.
- No field or operational deployment is claimed.

## Reproducible software checks

From the repository root, use the test scripts available in the current checkout:

```bash
python tests/verify_all.py
python tests/test_aeris_complete.py
```

Record the actual command output, software versions, dataset, and test conditions when publishing a new verification result.

## Reporting standard

Every performance claim should include:
- Dataset and train/test split.
- Sample rate and frame size.
- Input conditions and SNR calculation method.
- Metric definition and measured output.
- Whether the result came from simulation, desktop software, or embedded hardware.

Do not describe simulation results as hardware measurements.

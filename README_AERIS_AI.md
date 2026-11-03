# AERIS — AI and DSP Software Notes

This document describes the AI and signal-processing components currently present in the AERIS repository. It is a software reference, not proof of embedded or acoustic hardware deployment.

## AI workflow

The repository contains:
- Acoustic feature extraction and dataset-building scripts under `AI/`.
- Random Forest training and inference scripts.
- A trained Random Forest classifier and scaler artifacts under `AI/` and `models/`.
- Evaluation outputs, classification reports, and plots.

The project workflow also records completion of a TinyML model-training milestone. That milestone is distinct from the Python Random Forest artifacts. Before describing TinyML as deployed, verify that the model has been exported to a supported embedded format, optimized if needed, flashed to the ESP32, and tested on-device.

## DSP workflow

The software includes adaptive filtering experiments using:
- LMS (Least Mean Squares)
- NLMS (Normalized Least Mean Squares)

For conventional adaptive ANC, the reference input must be correlated with the noise component in the primary signal. A classifier can help characterize the acoustic environment or select processing parameters, but it does not replace a valid ANC reference signal.

## Embedded target

The selected embedded controller is ESP32. The MPU6050 is the selected inertial sensor for motion/vibration-related measurements.

The MPU6050 is not an audio microphone. It cannot capture speech or airborne noise. A microphone-based physical ANC implementation therefore requires a separately designed and tested audio input/output chain.

## Run the software backend

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser.

## Important validation distinctions

| Item | What it demonstrates |
|---|---|
| MATLAB/Python simulation | Algorithm behavior on the evaluated data |
| Python classifier artifact | Model inference in the software environment |
| TinyML training milestone | Model training completed in the project workflow |
| ESP32 inference | Must be demonstrated with an on-device test |
| Physical acoustic ANC | Must be demonstrated with a valid microphone/reference/output path |

Do not transfer simulation metrics to hardware claims without corresponding measurements.

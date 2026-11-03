# AERIS — Adaptive Noise Cancellation & Intelligent Communication System

AERIS is a research and prototyping project focused on improving voice communication in noisy and challenging environments, with an emphasis on defense and field communication use cases.

The project combines adaptive digital signal processing (DSP), AI-assisted acoustic/noise analysis, and a planned embedded implementation around an ESP32-based platform.

> **Project status:** The software simulation and AI/DSP work are present in this repository. The TinyML model training milestone has been completed in the project workflow. ESP32 deployment and end-to-end hardware validation remain in progress. This repository does not claim a field-ready or production ANC headset.

## Project objectives

- Improve speech intelligibility under environmental noise.
- Use adaptive filtering techniques such as LMS and NLMS for signal processing.
- Use a trained machine-learning model to support noise/signal classification and filter decisions.
- Develop an embedded prototype using an ESP32.
- Explore sensor-assisted monitoring and telemetry for reliable communication.

## Current architecture

```text
             AUDIO DATA / TEST WAV FILES
                       |
                       v
             Signal preprocessing
                       |
                       v
          Acoustic feature extraction
                       |
                       v
           AI noise/signal classifier
                       |
                       v
          Adaptive DSP (LMS / NLMS)
                       |
                       v
           Enhanced signal + metrics
                       |
                       v
             FastAPI / Dashboard

        Planned embedded integration:
              ESP32 + MPU6050
                       |
                Sensor telemetry
```

The Python software pipeline supports controlled audio processing and evaluation. The repository also contains a Random Forest classifier artifact used by the current software implementation. The separately completed TinyML training work should be treated as a model-development milestone; ESP32 conversion, quantization (if required), and on-device inference must be validated before claiming embedded deployment.

## Hardware decisions and scope

- **Selected embedded controller:** ESP32.
- **Selected sensor in the current hardware plan:** MPU6050.
- The hardware plan is intentionally kept separate from the PC-based audio simulation.
- An MPU6050 measures acceleration and angular velocity; it is **not an audio microphone** and cannot by itself provide the acoustic reference/error signals required for conventional audio ANC.
- No microphone-based physical ANC setup is claimed as completed. Audio capture, reference-signal design, output path, and real-time acoustic validation remain open engineering tasks.
- The embedded design should only be described as operational after the required audio input/output hardware is available and the complete signal path is tested.

## Software stack

- Python
- FastAPI
- NumPy / SciPy
- scikit-learn
- LMS / NLMS adaptive filtering
- HTML, CSS, JavaScript dashboard
- MATLAB scripts and offline benchmark outputs

## Repository structure

```text
AERIS_SIH/
├── backend/       # FastAPI service, feature extraction, classifier, ANC DSP
├── FRONTEND/      # Dashboard and browser audio interface
├── AI/            # AI training, inference, evaluation and dataset utilities
├── AI_MODEL/      # Model artifacts from the software workflow
├── models/        # Classifier, scaler and feature metadata
├── MATLAB_CODE/   # Offline DSP simulation and benchmark scripts
├── OUTPUT/        # Generated evaluation outputs and reports
├── REALTIME/      # Real-time processing experiments
├── HARDWARE/      # Embedded design notes and hardware status
├── tests/         # Software verification tests
├── requirements.txt
├── package.json
└── README.md
```

## Run the software prototype

### 1. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Start the FastAPI application

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Or use the npm script:

```bash
npm start
```

### 3. Open the dashboard

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Testing

Run the repository's available verification scripts:

```bash
python tests/verify_all.py
python tests/test_aeris_complete.py
```

Run the scripts supported by the current checkout; test availability and results may vary as the project evolves.

## Engineering limitations

1. **Simulation is not hardware validation.** Offline MATLAB/Python SNR results demonstrate algorithm behavior on the evaluated data, not guaranteed real-world or battlefield performance.
2. **Sensor limitation.** The MPU6050 is an inertial sensor, not an audio input device.
3. **Physical ANC requires an audio path.** A practical acoustic ANC system needs suitable microphone reference/error signals, an audio output transducer, and careful secondary-path/latency handling.
4. **TinyML deployment is a separate milestone.** Training completion does not by itself prove that the model has been converted, optimized, flashed, or tested on the ESP32.
5. **No field-readiness claim.** The system remains a prototype/research effort until hardware, latency, stability, intelligibility, and safety are validated.

## Development status

| Area | Status |
|---|---|
| Software audio-processing pipeline | Present in repository |
| LMS / NLMS DSP experiments | Present in repository |
| Python AI classifier | Present in repository |
| TinyML model training milestone | Completed in project workflow |
| ESP32 as target controller | Selected |
| MPU6050 sensor integration | Planned / under development |
| ESP32 on-device TinyML inference | Pending integration validation |
| Microphone-based physical ANC | Pending suitable audio hardware and testing |
| Field deployment | Not validated |

---

**AERIS** — Adaptive Noise Cancellation & Intelligent Communication System.

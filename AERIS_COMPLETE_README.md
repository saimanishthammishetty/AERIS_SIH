# AERIS — Engineering and Architecture Notes

**AERIS** stands for **Adaptive Noise Cancellation & Intelligent Communication System**. The project explores AI-assisted signal analysis and adaptive DSP for improving communication quality in noisy environments, with a defense-communication use case.

## 1. Core signal-processing concepts

### LMS — Least Mean Squares

LMS is an adaptive filtering algorithm that updates filter coefficients to minimize the error between a desired signal and the filter output.

```text
y[n] = w[n]^T x[n]
e[n] = d[n] - y[n]
w[n+1] = w[n] + 2 μ e[n] x[n]
```

### NLMS — Normalized Least Mean Squares

NLMS normalizes the adaptation step using input-signal energy. This can improve stability when input amplitude varies.

```text
w[n+1] = w[n] + (μ / (ε + ||x[n]||²)) e[n] x[n]
```

The correct adaptive ANC topology requires a meaningful reference signal correlated with the noise to be cancelled. A single arbitrary signal or a synthetic copy of the primary microphone signal is not a valid substitute for a properly designed reference path.

## 2. AI / TinyML work

The project includes acoustic feature extraction and an AI classification pipeline in the software repository. The TinyML model training milestone has been completed in the project workflow.

The repository also contains a Python/scikit-learn Random Forest model artifact used by the current software prototype. These should not be conflated: ESP32 deployment of the TinyML model requires a compatible exported model, any required quantization, memory analysis, and on-device inference testing.

## 3. Embedded hardware direction

The selected embedded platform is ESP32, with an MPU6050 used for inertial sensing.

**Important:** MPU6050 is an accelerometer/gyroscope, not an audio microphone. It cannot directly capture speech or ambient acoustic noise. Physical audio ANC therefore remains a separate hardware integration task until suitable audio input and output components are available and validated.

## 4. Software architecture

The current software prototype contains:

- Python / FastAPI backend
- Acoustic feature extraction
- AI classification
- LMS / NLMS adaptive DSP experiments
- Browser-based dashboard
- MATLAB offline simulation and evaluation material

## 5. Validation and reporting

Clearly distinguish the following evidence:

1. **Offline simulation:** results from MATLAB/Python test signals.
2. **Software integration:** successful execution of backend, classifier, DSP, and dashboard.
3. **Embedded validation:** model and sensor execution on the ESP32.
4. **Physical ANC validation:** measured acoustic performance using a real audio input/output path.

A result at one level must not be presented as proof of a higher level. Report dataset, test conditions, sample rate, frame size, metric definitions, and limitations alongside numerical results.

## 6. Current limitations

- The MPU6050 does not provide audio samples.
- Physical microphone-based ANC is not validated.
- TinyML training completion does not by itself establish ESP32 deployment.
- Offline SNR improvements do not guarantee performance in real environments.
- Field or operational deployment has not been validated.

## 7. Next engineering milestones

1. Finalize the ESP32 + MPU6050 wiring and verify sensor readings.
2. Export the trained TinyML model into a format supported by the selected ESP32 inference runtime.
3. Measure model memory, inference time, and stability on-device.
4. Define the required audio reference/error signal architecture.
5. Integrate and validate the audio input/output chain before claiming physical ANC.
6. Run repeatable tests and publish measured results with their setup and limitations.

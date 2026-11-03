# AERIS Hardware Architecture — Current Project Scope

## 1. Selected embedded platform

The current embedded direction for AERIS is:

- **ESP32** as the target microcontroller.
- **MPU6050** as the selected inertial sensor for motion/vibration-related measurements.
- The trained TinyML model is intended to be integrated into the embedded workflow after the model format, memory use, and inference performance are verified on the target board.

The current repository also contains a PC-based audio-processing prototype. Do not confuse the software simulation with a completed physical ANC device.

## 2. Important sensor limitation

The MPU6050 contains a 3-axis accelerometer and a 3-axis gyroscope. It does not capture airborne sound and cannot replace an audio microphone.

Conventional audio ANC needs an audio signal path, including appropriate reference and/or error microphone signals, a DSP path, and an output transducer. Since a microphone-based hardware setup has not been established in the current project status, physical acoustic cancellation is **not claimed as implemented or validated**.

The ESP32 + MPU6050 setup can support embedded sensing and telemetry experiments. It must not be described as a complete microphone-based ANC headset.

## 3. System-level architecture

```text
            PC / SOFTWARE PROTOTYPE
     Audio files or controlled test signals
                     |
                     v
        Feature extraction and AI analysis
                     |
                     v
              LMS / NLMS DSP
                     |
                     v
          Dashboard and evaluation metrics


             EMBEDDED WORKSTREAM
                 ESP32
                   |
          +--------+---------+
          |                  |
       MPU6050         TinyML inference
    inertial data       (integration/
          |             validation step)
          +--------+---------+
                   |
             Serial telemetry
```

The software and embedded workstreams may exchange telemetry, but the diagram does not imply that the MPU6050 supplies audio samples.

## 4. Current implementation status

| Subsystem | Status |
|---|---|
| ESP32 selected as embedded controller | Confirmed project decision |
| MPU6050 selected for sensor work | Confirmed project decision |
| TinyML model training | Completed in project workflow |
| TinyML model flashed and verified on ESP32 | Pending documented hardware test |
| MPU6050 wiring and sensor readout | To be documented after the actual board setup is tested |
| Microphone-based ANC input/output chain | Not yet established |
| Physical acoustic noise cancellation | Not validated |
| PC-based audio simulation and DSP | Available in repository |

## 5. Hardware documentation policy

Only publish pin numbers, wiring diagrams, measured latency, or hardware performance after they have been checked against the actual assembled circuit and a repeatable test.

Do not reuse microphone/DAC pinouts from an earlier experimental design as if they were the current AERIS wiring. Keep unverified hardware details marked as **planned** or **under test**.

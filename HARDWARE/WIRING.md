# AERIS HARDWARE PROTOTYPE - WIRING & PINOUT GUIDE

This document details the exact physical wiring between the **ESP32 DevKit V1**, the **INMP441 I2S Digital MEMS Microphone**, and the **MAX98357A I2S DAC Class-D Audio Amplifier**.

---

## 1. COMPONENT SPECIFICATIONS

| Component | Model | Function | Operating Voltage | Interface |
| :--- | :--- | :--- | :--- | :--- |
| **Microcontroller** | ESP32 DevKit V1 (30/36-pin) | Real-Time DSP & Serial Telemetry | 5V (USB) / 3.3V Logic | I2S0, I2S1, UART |
| **Audio Input** | INMP441 / ICS-43434 | Digital MEMS Reference/Error Mic | **3.3V ONLY** | I2S (RX) |
| **Audio Output** | MAX98357A | 3.2W Class-D Mono DAC Amplifier | **5V (VIN)** / 3.3V | I2S (TX) |
| **Speaker** | 4Ω or 8Ω, 3W | Anti-Noise Acoustic Emitter | Driven by MAX98357A | Analog Differential |

> [!WARNING]
> **Component Notice**: Do **NOT** connect MPU6050, DS18B20, FC-03, L298N, or the 775 DC motor to this audio ANC system. Those components belong to your rotating-machinery condition monitoring project and are **NOT** acoustic ANC sensors.

---

## 2. PIN-TO-PIN CONNECTION TABLE

### A. I2S Digital Microphone (INMP441) → ESP32 DevKit V1

| INMP441 Pin | ESP32 Pin | Function / Signal Description |
| :--- | :--- | :--- |
| **VDD** | **3.3V** | Power Supply (Do NOT connect to 5V; will damage MEMS sensor) |
| **GND** | **GND** | Common Ground |
| **SD** | **GPIO 32** | Serial Data (I2S0 Data Input) |
| **WS / LRCK**| **GPIO 15** | Word Select / Left-Right Clock (16 kHz Frame Sync) |
| **SCK / BCLK**| **GPIO 14** | Continuous Bit Clock (16 kHz × 32 bits × 2 = 1.024 MHz) |
| **L/R** | **GND** | Channel Select (Tied to GND for Left Channel) |

---

### B. I2S Audio Amplifier (MAX98357A) → ESP32 DevKit V1

| MAX98357A Pin | ESP32 Pin | Function / Signal Description |
| :--- | :--- | :--- |
| **VIN** | **VIN / 5V** | Power Supply (5V recommended for full 3.2W output power) |
| **GND** | **GND** | Common Ground |
| **DIN** | **GPIO 22** | Serial Data Out (I2S1 Data Output) |
| **BCLK** | **GPIO 26** | Bit Clock (I2S1 TX Bit Clock) |
| **LRC** | **GPIO 25** | Left/Right Clock (I2S1 Word Select) |
| **GAIN** | **GND** | Gain Selection (Tied to GND for +9 dB nominal gain) |
| **SD_MODE** | **Unconnected** | Left + Right / 2 Mono Mix (leave floating with pullup) |

---

### C. MAX98357A Output → Speaker

| MAX98357A Terminal | Speaker Terminal |
| :--- | :--- |
| **OUT+** | Speaker Positive (+) terminal |
| **OUT-** | Speaker Negative (-) terminal |

---

## 3. ASCII CIRCUIT SCHEMATIC

```
                +------------------------------------+
                |          ESP32 DevKit V1           |
                |                                    |
  [3.3V] <------+ 3V3                            5V  +------> [VIN] MAX98357A
  [GND]  <------+ GND                            GND +------> [GND] MAX98357A
                |                                    |
                | [I2S0 Input - Microphone]          |
  [SD]   <------+ GPIO 32                            |
  [WS]   <------+ GPIO 15                            |
  [SCK]  <------+ GPIO 14                            |
                |                                    |
                | [I2S1 Output - DAC Amplifier]      |
                |                            GPIO 22 +------> [DIN]  MAX98357A
                |                            GPIO 26 +------> [BCLK] MAX98357A
                |                            GPIO 25 +------> [LRC]  MAX98357A
                +------------------------------------+

     INMP441 Microphone                         MAX98357A DAC Amplifier
   +--------------------+                     +--------------------------+
   | VDD   -> 3.3V      |                     | VIN     -> 5V            |
   | GND   -> GND       |                     | GND     -> GND           |
   | SD    -> GPIO 32   |                     | DIN     -> GPIO 22       |
   | WS    -> GPIO 15   |                     | BCLK    -> GPIO 26       |
   | SCK   -> GPIO 14   |                     | LRC     -> GPIO 25       |
   | L/R   -> GND       |                     | GAIN    -> GND (+9dB)    |
   +--------------------+                     | OUT+    -> Speaker (+)   |
                                              | OUT-    -> Speaker (-)   |
                                              +--------------------------+
```

---

## 4. HARDWARE ASSEMBLY CHECKLIST

1. **Breadboard Layout**: Keep wire lengths as short as possible (< 15 cm) for high-speed I2S clocks (`SCK`/`BCLK`) to prevent capacitive loading and clock jitter.
2. **Ground Planes**: Tie all grounds (`INMP441 GND`, `ESP32 GND`, `MAX98357A GND`) together to a single common breadboard power rail.
3. **Power Decoupling**: If you experience high-frequency buzzing from the speaker, add a $100\mu\text{F}$ electrolytic capacitor across `VIN` and `GND` right beside the MAX98357A breakout board.
4. **USB Connection**: Connect the ESP32 micro-USB cable directly to your PC. The Python FastAPI backend and AERIS Dashboard will automatically discover and link to the COM port.

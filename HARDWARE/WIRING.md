# AERIS Hardware Wiring — Verification Status

## Current selected hardware

- ESP32 development board
- MPU6050 inertial measurement unit

## Wiring status

A verified pin-to-pin wiring table is not published yet. The final ESP32 board variant, I2C pin assignment, power arrangement, and physical connections must be checked against the actual hardware before documenting fixed GPIO numbers.

Do not assume that older microphone or audio-amplifier wiring diagrams describe the current AERIS setup.

## MPU6050 interface notes

The MPU6050 communicates over I2C. A typical connection requires:

- VCC — supply compatible with the specific breakout board
- GND — common ground
- SDA — ESP32 I2C data
- SCL — ESP32 I2C clock

Confirm the breakout-board voltage requirements and the selected ESP32 I2C pins before powering the circuit. Do not treat these generic interface notes as a verified project pinout.

## Critical limitation

The MPU6050 measures acceleration and angular velocity. It is not a microphone and does not provide the acoustic reference/error audio signals needed for conventional audio ANC.

The current project therefore does not claim physical microphone-based ANC through the ESP32 + MPU6050 combination alone. A suitable audio input/output chain and its wiring must be designed, assembled, and tested before physical ANC can be demonstrated.

## Verification checklist

- [ ] Confirm exact ESP32 board variant.
- [ ] Confirm MPU6050 breakout supply voltage.
- [ ] Confirm SDA/SCL pin assignments.
- [ ] Run an I2C device scan and verify the sensor address.
- [ ] Read and validate accelerometer and gyroscope data.
- [ ] Record the tested wiring and firmware version.
- [ ] Validate TinyML inference on the ESP32 separately.
- [ ] Design and validate the audio input/output path before claiming physical ANC.

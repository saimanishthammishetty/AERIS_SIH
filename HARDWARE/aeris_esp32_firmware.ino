/*
 * ============================================================================
 * PROJECT AERIS - ESP32 EMBEDDED REAL-TIME ADAPTIVE NOISE CANCELLATION FIRMWARE
 * Microcontroller: ESP32 DevKit V1 (Dual Core Tensilica Xtensa LX6 @ 240MHz)
 * Audio Input    : INMP441 / ICS-43434 I2S Digital MEMS Microphone
 * Audio Output   : MAX98357A I2S 3.2W Class-D Mono DAC Amplifier
 * Telemetry Port : USB Serial @ 115200 Baud (JSON format)
 * ============================================================================
 */

#include <Arduino.h>
#include "driver/i2s.h"

// ----------------------------------------------------------------------------
// 1. PIN DEFINITIONS & I2S CONFIGURATION
// ----------------------------------------------------------------------------
// I2S IN (INMP441 Microphone)
#define I2S_MIC_PORT        I2S_NUM_0
#define I2S_MIC_SCK_PIN     14   // BCLK
#define I2S_MIC_WS_PIN      15   // LRC / WS
#define I2S_MIC_SD_PIN      32   // Serial Data In

// I2S OUT (MAX98357A DAC Amplifier)
#define I2S_DAC_PORT        I2S_NUM_1
#define I2S_DAC_BCLK_PIN    26   // Bit Clock
#define I2S_DAC_LRC_PIN     25   // Left/Right Clock (WS)
#define I2S_DAC_DIN_PIN     22   // Data Out

// Audio Sampling Parameters
#define SAMPLE_RATE         16000
#define DMA_BUF_COUNT       8
#define DMA_BUF_LEN         64
#define FRAME_SIZE          64

// ----------------------------------------------------------------------------
// 2. ADAPTIVE NOISE CANCELLATION (LMS) PARAMETERS & STATE
// ----------------------------------------------------------------------------
#define LMS_FILTER_ORDER    16
#define DEFAULT_STEP_SIZE   0.005f
#define MAX_WEIGHT_NORM     10.0f
#define OUTPUT_CLIP_LIMIT   32767.0f

// Filter State
float w[LMS_FILTER_ORDER];       // Adaptive FIR weights
float x_buf[LMS_FILTER_ORDER];   // Input delay line buffer
float step_size = DEFAULT_STEP_SIZE;
bool anc_active = true;
bool safety_limit_active = false;
uint32_t telemetry_counter = 0;

// DMA Buffers
int32_t rx_buffer[FRAME_SIZE];
int16_t tx_buffer[FRAME_SIZE];

// ----------------------------------------------------------------------------
// 3. I2S PERIPHERAL DRIVER INITIALIZATION
// ----------------------------------------------------------------------------
void init_i2s_microphone() {
    i2s_config_t i2s_mic_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t mic_pin_config = {
        .bck_io_num = I2S_MIC_SCK_PIN,
        .ws_io_num = I2S_MIC_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD_PIN
    };

    i2s_driver_install(I2S_MIC_PORT, &i2s_mic_config, 0, NULL);
    i2s_set_pin(I2S_MIC_PORT, &mic_pin_config);
    i2s_zero_dma_buffer(I2S_MIC_PORT);
}

void init_i2s_dac() {
    i2s_config_t i2s_dac_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_COUNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };

    i2s_pin_config_t dac_pin_config = {
        .bck_io_num = I2S_DAC_BCLK_PIN,
        .ws_io_num = I2S_DAC_LRC_PIN,
        .data_out_num = I2S_DAC_DIN_PIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    i2s_driver_install(I2S_DAC_PORT, &i2s_dac_config, 0, NULL);
    i2s_set_pin(I2S_DAC_PORT, &dac_pin_config);
    i2s_zero_dma_buffer(I2S_DAC_PORT);
}

// ----------------------------------------------------------------------------
// 4. REAL-TIME SAMPLE-BY-SAMPLE LMS FILTER WITH STABILITY PROTECTION
// ----------------------------------------------------------------------------
inline float process_lms_sample(float d_sample, float x_sample) {
    // 1. Shift input delay line
    for (int i = LMS_FILTER_ORDER - 1; i > 0; i--) {
        x_buf[i] = x_buf[i - 1];
    }
    x_buf[0] = x_sample;

    if (!anc_active) {
        return d_sample; // Pass-through if ANC disengaged
    }

    // 2. Compute estimated noise y[n] = w^T * x
    float y = 0.0f;
    for (int i = 0; i < LMS_FILTER_ORDER; i++) {
        y += w[i] * x_buf[i];
    }

    // 3. Error / Enhanced signal e[n] = d[n] - y[n]
    float e = d_sample - y;

    // 4. Stability Check: Detect NaN or excessive divergence
    if (isnan(e) || isinf(e) || abs(e) > 50000.0f) {
        safety_limit_active = true;
        for (int i = 0; i < LMS_FILTER_ORDER; i++) w[i] = 0.0f;
        step_size = max(step_size * 0.5f, 0.0001f);
        return constrain(d_sample, -OUTPUT_CLIP_LIMIT, OUTPUT_CLIP_LIMIT);
    }

    // 5. LMS Weight Update: w[n+1] = w[n] + 2 * mu * e[n] * x[n]
    float norm_sq = 0.0f;
    for (int i = 0; i < LMS_FILTER_ORDER; i++) {
        w[i] += 2.0f * step_size * e * x_buf[i];
        norm_sq += w[i] * w[i];
    }

    // 6. Exploding Weight Clamping Guard
    if (norm_sq > (MAX_WEIGHT_NORM * MAX_WEIGHT_NORM)) {
        safety_limit_active = true;
        float inv_norm = MAX_WEIGHT_NORM / sqrtf(norm_sq);
        for (int i = 0; i < LMS_FILTER_ORDER; i++) {
            w[i] *= inv_norm;
        }
    } else {
        safety_limit_active = false;
    }

    // 7. Clamped Safe Output
    return constrain(e, -OUTPUT_CLIP_LIMIT, OUTPUT_CLIP_LIMIT);
}

// ----------------------------------------------------------------------------
// 5. MAIN SETUP & DSP REAL-TIME LOOP
// ----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(500);
    
    // Clear Filter state
    for (int i = 0; i < LMS_FILTER_ORDER; i++) {
        w[i] = 0.0f;
        x_buf[i] = 0.0f;
    }

    // Initialize I2S Audio Hardware
    init_i2s_microphone();
    init_i2s_dac();

    Serial.println("{\"system\":\"AERIS\",\"status\":\"BOOT_COMPLETE\",\"sample_rate\":16000,\"anc_algorithm\":\"LMS\",\"filter_order\":16}");
}

void loop() {
    size_t bytes_read = 0;
    size_t bytes_written = 0;

    // 1. Read block of microphone samples via I2S DMA
    esp_err_t err = i2s_read(I2S_MIC_PORT, (void*)rx_buffer, sizeof(rx_buffer), &bytes_read, portMAX_DELAY);
    
    if (err == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);
        float sum_sq_in = 0.0f;
        float sum_sq_out = 0.0f;
        int64_t t_dsp_start = esp_timer_get_time();

        for (int i = 0; i < samples_read; i++) {
            // Convert 24-bit aligned INMP441 sample to 16-bit PCM scale
            float mic_sample = (float)(rx_buffer[i] >> 14);
            sum_sq_in += mic_sample * mic_sample;

            // In prototype feedforward single-stream mode, synthesize reference from correlated lag
            float ref_noise = mic_sample * 0.8f;

            // Execute LMS ANC DSP Filter
            float enhanced_val = process_lms_sample(mic_sample, ref_noise);
            sum_sq_out += enhanced_val * enhanced_val;

            tx_buffer[i] = (int16_t)enhanced_val;
        }

        int64_t t_dsp_latency_us = esp_timer_get_time() - t_dsp_start;

        // 2. Output Enhanced Audio to I2S DAC Amplifier (MAX98357A)
        i2s_write(I2S_DAC_PORT, (const void*)tx_buffer, samples_read * sizeof(int16_t), &bytes_written, portMAX_DELAY);

        // 3. Periodic JSON Telemetry over USB Serial (Every ~100 frames)
        telemetry_counter++;
        if (telemetry_counter >= 100) {
            telemetry_counter = 0;
            float rms_in = sqrtf(sum_sq_in / samples_read);
            float rms_out = sqrtf(sum_sq_out / samples_read);
            float snr_gain = (rms_in > 1.0f && rms_out > 1.0f) ? 20.0f * log10f(rms_in / rms_out) : 0.0f;

            Serial.printf(
                "{\"system\":\"AERIS_ESP32\",\"anc_status\":\"%s\",\"algorithm\":\"LMS\",\"order\":%d,\"rms_in\":%.2f,\"rms_out\":%.2f,\"gain_db\":%.2f,\"latency_us\":%lld,\"safety_limit\":%s}\n",
                anc_active ? "ACTIVE" : "INACTIVE",
                LMS_FILTER_ORDER,
                rms_in,
                rms_out,
                snr_gain,
                t_dsp_latency_us,
                safety_limit_active ? "true" : "false"
            );
        }
    }

    // 4. Handle incoming Serial Commands from Backend / Dashboard
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "START_ANC") {
            anc_active = true;
            Serial.println("{\"status\":\"ANC_STARTED\"}");
        } else if (cmd == "STOP_ANC") {
            anc_active = false;
            Serial.println("{\"status\":\"ANC_STOPPED\"}");
        } else if (cmd == "RESET") {
            for (int i = 0; i < LMS_FILTER_ORDER; i++) w[i] = 0.0f;
            step_size = DEFAULT_STEP_SIZE;
            safety_limit_active = false;
            Serial.println("{\"status\":\"FILTER_RESET\"}");
        }
    }
}

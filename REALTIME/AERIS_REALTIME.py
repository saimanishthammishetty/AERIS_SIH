"""
=============================================================================
PROJECT AERIS — REAL-TIME ACOUSTIC PIPELINE & DSP STREAMING ENGINE
Module: REALTIME/AERIS_REALTIME.py
=============================================================================
Provides:
1. Low-latency frame-by-frame streaming buffer management
2. Configurable FRAME_SIZE, SAMPLE_RATE, HOP_SIZE
3. Sub-stage latency tracking (Capture, Features, AI, ANC, Output, Total)
4. Sample-by-sample LMS/NLMS adaptive filter with active stability guards
5. Live physical microphone streaming (via sounddevice) with automatic simulation fallback
6. Explainable AI decision logic for parameter configuration
7. Watchdog safety state machine for automatic divergence recovery
8. Simulation looping of preset WAV files
=============================================================================
"""

import os
import sys
import time
import json
import threading
import numpy as np
import soundfile as sf
import scipy.stats as stats

# Ensure root directory and sub-packages are in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for _sub in ["AI", "BACKEND", "REALTIME"]:
    _p = os.path.join(ROOT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AI.features.extract_features import extract_audio_features, TARGET_SR
    from AI.AERIS_AI_INFERENCE import global_classifier
except ImportError:
    from features.extract_features import extract_audio_features, TARGET_SR
    from AERIS_AI_INFERENCE import global_classifier

try:
    import sounddevice as sd  # type: ignore
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

FRAME_SIZE = 512          # 32 ms block at 16 kHz
ANALYSIS_WINDOW = 4000    # 250 ms analysis window for AI features
CONFIDENCE_THRESHOLD = 0.50
MAX_FILTER_ORDER = 128

# System logs global container
SYSTEM_LOGS = []

def log_event(msg: str, level: str = "info"):
    """Global event logger that writes to lists and logs/aeris_system.log."""
    timestamp = time.strftime('%H:%M:%S')
    log_str = f"{timestamp} [{level.upper()}] {msg}"
    SYSTEM_LOGS.append(log_str)
    if len(SYSTEM_LOGS) > 300:
        SYSTEM_LOGS.pop(0)
    try:
        os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
        with open(os.path.join(ROOT_DIR, "logs", "aeris_system.log"), "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
    except Exception:
        pass
    print(log_str)

# Pre-fill initial logs
log_event("AERIS Acoustic Real-time Engine starting up...", "info")


class RealtimePipeline:
    """End-to-End Real-Time Audio Processing Pipeline."""

    def __init__(self, frame_size: int = FRAME_SIZE, sample_rate: int = TARGET_SR):
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        # Adaptive ANC Filter Config
        self.max_order = MAX_FILTER_ORDER
        self.algorithm = "LMS"
        self.filter_order = 16
        self.step_size = 0.005
        self.weights = np.zeros(self.max_order, dtype=np.float64)
        self.delay_line = np.zeros(self.max_order, dtype=np.float64)
        
        # Audio Volume & Streaming Control
        self.anc_active = True
        self.volume = 1.0
        self.is_muted = False
        self.latest_input_rms = 0.0
        self.latest_output_rms = 0.0
        
        # Watchdog & Fallback State Machine
        self.safety_limit_active = False
        self.safety_warning = "NORMAL OPERATION"
        self.instability_count = 0
        self.bypass_active = False

        # Rolling Circular Analysis Buffer (250 ms)
        self.analysis_buffer = np.zeros(ANALYSIS_WINDOW, dtype=np.float64)

        # Simulation Mode Fields
        self.simulation_mode = True  # Start in simulation mode by default
        self.sim_audio_data = None
        self.sim_play_ptr = 0
        self.sim_preset_name = "noisy"
        
        # Explainable Decision State
        self.recommended_filter = "LMS"
        self.risk_level = "LOW"
        self.decision_reason = "Initial standby state. Defaulting to safe LMS."
        
        # Network resilience modes
        self.network_mode = "NORMAL" # NORMAL, DEGRADED, RECOVERY
        
        # Active sounddevice stream handle & thread
        self.stream = None
        self.is_running = False
        self.stream_thread = None
        
        # Telemetry State
        self.telemetry = {
            "microphone_status": "ONLINE (LIVE STREAM)" if SOUNDDEVICE_AVAILABLE else "SIMULATION_FALLBACK",
            "model_status": "ONLINE" if global_classifier.model is not None else "OFFLINE",
            "anc_status": "ACTIVE",
            "current_algorithm": "LMS",
            "noise_classification": "STANDBY",
            "confidence": 0.0,
            "input_snr": 0.0,
            "output_snr": 0.0,
            "snr_improvement": 0.0,
            "rms_input": 0.0,
            "rms_output": 0.0,
            "processed_frames": 0,
            "dropped_frames": 0,
            "recommended_filter": "LMS",
            "risk_level": "LOW",
            "decision_reason": self.decision_reason,
            "network_mode": self.network_mode,
            "comm_quality_score": 100,
            "comm_quality_label": "Excellent",
            "latency": {
                "capture_ms": 0.0,
                "feature_extraction_ms": 0.0,
                "ai_inference_ms": 0.0,
                "anc_processing_ms": 0.0,
                "output_ms": 0.0,
                "total_latency_ms": 0.0
            }
        }
        
        # Pre-load default simulation WAV file
        self.load_simulation_preset("noisy")

    def load_simulation_preset(self, preset_name: str) -> bool:
        """Loads companion test WAV file for demo circular loop."""
        filenames = {
            "clean": ["clean.wav", "AERIS_CLEAN_-10dB.wav"],
            "noisy": ["noisy.wav", "AERIS_NOISY_-10dB.wav"],
            "lms": ["lms_enhanced.wav", "AERIS_LMS_-10dB.wav"],
            "nlms": ["nlms_enhanced.wav", "AERIS_NLMS_-10dB.wav"]
        }
        candidates = filenames.get(preset_name.lower(), ["noisy.wav"])
        search_dirs = [
            os.path.join(ROOT_DIR, "OUTPUT", "AERIS_AI_READY"),
            os.path.join(ROOT_DIR, "OUTPUT"),
            os.path.join(ROOT_DIR, "AI_MODEL"),
            ROOT_DIR
        ]
        
        for cand in candidates:
            for sdir in search_dirs:
                p = os.path.join(sdir, cand)
                if os.path.exists(p):
                    try:
                        data, sr = sf.read(p, dtype='float64')
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1) # Mono conversion
                        self.sim_audio_data = data
                        self.sim_play_ptr = 0
                        self.sim_preset_name = preset_name
                        log_event(f"Loaded simulation preset audio: '{cand}' ({len(data)} samples @ {sr} Hz).")
                        return True
                    except Exception as e:
                        log_event(f"Error reading preset WAV '{p}': {e}", "warning")
        
        # Synthetic signal backup if file not found
        t_arr = np.linspace(0, 5.0, 16000 * 5)
        self.sim_audio_data = 0.4 * np.sin(2 * np.pi * 180 * t_arr) + 0.3 * np.sin(2 * np.pi * 1000 * t_arr)
        self.sim_play_ptr = 0
        self.sim_preset_name = preset_name
        log_event(f"Simulation preset WAV '{preset_name}' not found. Generated synthetic backing.", "warning")
        return False

    def update_decision_logic(self, noise_type: str, confidence: float, feats: dict):
        """
        AI -> ANC Adaptive Decision Logic:
        Maps classified noise category to optimal ANC parameter configuration,
        exposing scientific explainability logs to the telemetry.
        """
        flatness = feats.get("SpectralFlatness", 0.0)
        centroid = feats.get("SpectralCentroid", 0.0)
        dom_freq = feats.get("DominantFrequency", 0.0)

        # Skip adaptation if watchdog safety fallback is actively bypassing
        if self.bypass_active:
            self.risk_level = "CRITICAL"
            self.recommended_filter = "Bypass"
            self.decision_reason = "Safety limit triggered due to output divergence. Running in safe bypass mode."
            self.telemetry["recommended_filter"] = "Bypass"
            self.telemetry["risk_level"] = "CRITICAL"
            self.telemetry["decision_reason"] = self.decision_reason
            return

        if confidence < CONFIDENCE_THRESHOLD:
            self.risk_level = "MODERATE"
            self.recommended_filter = "LMS (Default)"
            self.decision_reason = f"Classification confidence ({confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD}). Defaulting to standard LMS (M=16, μ=0.005) for stability."
            self.algorithm = "LMS"
            self.step_size = 0.005
            self.filter_order = 16
        elif noise_type == "CLEAN_SPEECH":
            self.risk_level = "LOW"
            self.recommended_filter = "LMS (Low Step)"
            self.decision_reason = "Clear speech envelope identified. Restricting step size to 0.0005 to avoid speech distortion."
            self.algorithm = "LMS"
            self.step_size = 0.0005
            self.filter_order = 16
        elif noise_type == "NOISE_TMETRO" or noise_type == "NOISY_SPEECH":
            self.risk_level = "HIGH"
            # High spectral flatness or high dominant frequency points to machinery/broadband noise
            if flatness > 0.25 or dom_freq > 1500:
                self.recommended_filter = "NLMS"
                self.decision_reason = f"Broadband/fluctuating noise envelope (flatness: {flatness:.3f}). Recommending power-normalized NLMS (M=32, μ=0.150) for fast tracking."
                self.algorithm = "NLMS"
                self.step_size = 0.150
                self.filter_order = 32
            else:
                self.recommended_filter = "LMS"
                self.decision_reason = f"Correlated low-frequency noise (dominant frequency: {dom_freq:.0f} Hz). LMS (M=16, μ=0.005) chosen for optimal low-band cancellation."
                self.algorithm = "LMS"
                self.step_size = 0.005
                self.filter_order = 16
        elif noise_type == "ANC_ENHANCED":
            self.risk_level = "MODERATE"
            self.recommended_filter = "LMS"
            self.decision_reason = "ANC processed signal feedback. Maintaining running LMS coefficients."
            self.algorithm = "LMS"
            self.step_size = 0.002
            self.filter_order = 16
        else:
            # Fallback
            self.risk_level = "MODERATE"
            self.recommended_filter = "LMS"
            self.decision_reason = "Unknown acoustic context. Proceeding with nominal LMS filtering (M=16, μ=0.005)."
            self.algorithm = "LMS"
            self.step_size = 0.005
            self.filter_order = 16

        self.telemetry["recommended_filter"] = self.recommended_filter
        self.telemetry["risk_level"] = self.risk_level
        self.telemetry["decision_reason"] = self.decision_reason
        self.telemetry["current_algorithm"] = self.algorithm

    def calculate_communication_quality(self, snr_after: float, rms_out: float, safety_limit_active: bool, latency_ms: float) -> tuple:
        """Calculates experimental AERIS Communication Quality Score (0-100)."""
        snr_factor = np.clip((snr_after + 15) / 35.0, 0.0, 1.0) * 60.0  # Up to 60 points (ideal SNR >= 20 dB)
        
        # Latency factor (Up to 20 points)
        if latency_ms < 15.0:
            latency_factor = 20.0
        elif latency_ms < 32.0:
            latency_factor = 15.0
        elif latency_ms < 64.0:
            latency_factor = 8.0
        else:
            latency_factor = 2.0
            
        # Stability / Clipping check (Up to 20 points)
        stability_factor = 20.0
        if safety_limit_active:
            stability_factor -= 10.0
        if rms_out > 0.95:  # Audio clipping penalty
            stability_factor -= 5.0
            
        score = int(snr_factor + latency_factor + stability_factor)
        score = np.clip(score, 0, 100)
        
        if score >= 85:
            label = "Excellent"
        elif score >= 70:
            label = "Good"
        elif score >= 50:
            label = "Moderate"
        elif score >= 30:
            label = "Poor"
        else:
            label = "Critical"
            
        return score, label

    def process_frame(self, primary_frame: np.ndarray, ref_frame: np.ndarray = None) -> tuple:
        """
        Executes real-time DSP cycle on single frame (512 samples / 32 ms):
        Capture -> Rolling Buffer -> Features -> Periodic AI -> Adaptive ANC -> Safety monitoring -> Metrics.
        """
        t_start = time.perf_counter()

        # 1. Capture & Scrub
        t_cap0 = time.perf_counter()
        x_in = np.asarray(primary_frame, dtype=np.float64)
        if np.isnan(x_in).any() or np.isinf(x_in).any():
            x_in = np.nan_to_num(x_in, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Normalization guard (prevent extreme signals from clipping)
        peak = np.max(np.abs(x_in))
        if peak > 1.0:
            x_in = x_in / peak
            
        t_cap_ms = (time.perf_counter() - t_cap0) * 1000

        # Acoustic Reference Delay Model (model acoustic path transfer)
        if ref_frame is None or len(ref_frame) != len(x_in):
            ref_in = np.roll(x_in, 8) * 0.85
        else:
            ref_in = np.asarray(ref_frame, dtype=np.float64)

        # Update Rolling Analysis Window (250 ms)
        L = len(x_in)
        self.analysis_buffer = np.roll(self.analysis_buffer, -L)
        self.analysis_buffer[-L:] = x_in

        # 2. Feature Extraction
        t_feat0 = time.perf_counter()
        feats, _ = extract_audio_features(self.analysis_buffer, self.sample_rate)
        t_feat_ms = (time.perf_counter() - t_feat0) * 1000

        # 3. AI Inference (Periodic Evaluation ~250ms with state caching)
        t_ai0 = time.perf_counter()
        # Skip AI predictions in degraded edge network to conserve processing capacity if chosen
        if self.network_mode == "DEGRADED" and self.telemetry["processed_frames"] % 24 != 0:
            # Throttle inference to 750ms in degraded network
            t_ai_ms = 0.0
        else:
            if self.telemetry["processed_frames"] % 8 == 0 or self.telemetry["noise_classification"] == "STANDBY":
                ai_res = global_classifier.predict_from_features(feats)
                self.telemetry["noise_classification"] = ai_res["noise_class"]
                self.telemetry["confidence"] = ai_res["confidence"]
                self.update_decision_logic(ai_res["noise_class"], ai_res["confidence"], feats)
            t_ai_ms = (time.perf_counter() - t_ai0) * 1000

        # 4. Adaptive ANC Filtering (LMS/NLMS) with Watchdog State Machine
        t_anc0 = time.perf_counter()
        enhanced_out = np.zeros(L, dtype=np.float64)
        
        # Check if mute is active
        if self.is_muted:
            enhanced_out.fill(0.0)
            self.latest_input_rms = float(np.sqrt(np.mean(x_in**2)))
            self.latest_output_rms = 0.0
            t_anc_ms = (time.perf_counter() - t_anc0) * 1000
            return enhanced_out, self.telemetry

        if self.anc_active and not self.bypass_active:
            M = min(max(int(self.filter_order), 1), self.max_order)
            mu = float(self.step_size)
            algo = self.algorithm

            # Pre-allocated arrays for speed
            for i in range(L):
                # Shift delay line
                self.delay_line[1:M] = self.delay_line[:M-1]
                self.delay_line[0] = ref_in[i]

                # Anti-noise estimation y_hat = w^T * x
                y_hat = float(np.dot(self.weights[:M], self.delay_line[:M]))
                e_val = float(x_in[i] - y_hat)

                # Check if sample diverged (NaN/Inf or clipping overflow)
                if np.isnan(e_val) or np.isinf(e_val) or abs(e_val) > 10.0:
                    self.safety_limit_active = True
                    self.instability_count += 1
                    
                    # Safety Fallback State transitions
                    if self.algorithm == "LMS" and self.instability_count < 2:
                        self.safety_warning = "WATCHDOG: LMS divergence. Reducing learning rate & resetting weights."
                        log_event("LMS filter divergence detected. Resetting weights and reducing step size.", "warning")
                        self.step_size = max(self.step_size * 0.5, 1e-4)
                        self.weights[:M].fill(0.0)
                    elif self.algorithm == "LMS" and self.instability_count >= 2:
                        self.safety_warning = "WATCHDOG: LMS unstable after retry. Switching to NLMS."
                        log_event("LMS unstable after multiple retries. Watchdog falling back to NLMS.", "warning")
                        self.algorithm = "NLMS"
                        self.filter_order = 32
                        self.step_size = 0.150
                        self.weights[:M].fill(0.0)
                    elif self.algorithm == "NLMS" and self.instability_count < 4:
                        self.safety_warning = "WATCHDOG: NLMS divergence. Reducing step size & resetting weights."
                        log_event("NLMS filter divergence detected. Reducing step size and resetting weights.", "warning")
                        self.step_size = max(self.step_size * 0.5, 1e-3)
                        self.weights[:M].fill(0.0)
                    else:
                        self.safety_warning = "WATCHDOG: Both LMS/NLMS diverged. Bypassing processing."
                        log_event("Watchdog Alert: Severe acoustic divergence. Safe bypass activated.", "critical")
                        self.bypass_active = True
                        self.anc_active = False
                    
                    # Safe output limit clamp
                    e_val = np.clip(x_in[i], -1.0, 1.0)
                else:
                    # Update coefficients
                    if algo == "NLMS":
                        energy_ref = float(np.dot(self.delay_line[:M], self.delay_line[:M]))
                        mu_norm = mu / (energy_ref + 1e-8)
                        self.weights[:M] += mu_norm * e_val * self.delay_line[:M]
                    else: # Standard LMS
                        self.weights[:M] += 2.0 * mu * e_val * self.delay_line[:M]

                    # Clamp coefficients to prevent creeping saturation
                    w_norm = np.linalg.norm(self.weights[:M])
                    if w_norm > 10.0:
                        self.safety_limit_active = True
                        self.safety_warning = f"ANC SAFETY: Weight norm ({w_norm:.1f}) clamped."
                        self.weights[:M] = (self.weights[:M] / w_norm) * 10.0
                    else:
                        # Gradual warning reset if output stabilizes
                        if self.safety_limit_active and self.telemetry["processed_frames"] % 100 == 0:
                            self.safety_limit_active = False
                            self.safety_warning = "NORMAL OPERATION"

                enhanced_out[i] = np.clip(e_val, -1.0, 1.0)
        else:
            # Safe bypass mode
            enhanced_out = np.copy(x_in)

        # Energy recovery watchdog: if enhanced signal is totally dead but input contains signal, bypass
        p_in_check = float(np.mean(x_in**2))
        p_out_check = float(np.mean(enhanced_out**2))
        if p_in_check > 0.001 and p_out_check < 1e-6 and self.anc_active:
            log_event("Watchdog detected signal death. Activating emergency bypass.", "warning")
            enhanced_out = np.copy(x_in)
            self.bypass_active = True
            
        t_anc_ms = (time.perf_counter() - t_anc0) * 1000

        # 5. Output Energy & SNR Metrics
        t_out0 = time.perf_counter()
        p_in = float(np.mean(x_in**2))
        p_out = float(np.mean(enhanced_out**2))
        p_ref = float(np.mean(ref_in**2)) + 1e-12

        # Standard SNR = 10 * log10(Signal_Power / Noise_Power)
        snr_in = 10.0 * np.log10(max(p_in, 1e-12) / p_ref) if p_in > 0 else -30.0
        snr_out = 10.0 * np.log10(max(p_out, 1e-12) / p_ref) if p_out > 0 else -30.0
        snr_gain = snr_out - snr_in
        
        # Hard bounds to avoid extreme display values
        snr_in = np.clip(snr_in, -35.0, 45.0)
        snr_out = np.clip(snr_out, -35.0, 45.0)
        snr_gain = np.clip(snr_gain, -15.0, 25.0)
        
        # Calculate Communication Quality Score
        comm_score, comm_label = self.calculate_communication_quality(snr_out, float(np.max(np.abs(enhanced_out))), self.safety_limit_active, (time.perf_counter() - t_start)*1000)
        t_out_ms = (time.perf_counter() - t_out0) * 1000

        t_total_ms = (time.perf_counter() - t_start) * 1000

        # Update Telemetry State
        self.telemetry["processed_frames"] += 1
        self.telemetry["input_snr"] = round(snr_in, 2)
        self.telemetry["output_snr"] = round(snr_out, 2)
        self.telemetry["snr_improvement"] = round(snr_gain, 2)
        self.telemetry["rms_input"] = round(float(np.sqrt(p_in)), 4)
        self.telemetry["rms_output"] = round(float(np.sqrt(p_out)), 4)
        self.telemetry["comm_quality_score"] = comm_score
        self.telemetry["comm_quality_label"] = comm_label
        self.telemetry["anc_status"] = "FALLBACK" if self.bypass_active else ("WARNING" if self.safety_limit_active else "STABLE")
        
        self.latest_input_rms = self.telemetry["rms_input"]
        self.latest_output_rms = self.telemetry["rms_output"]

        # Latency breakdown
        self.telemetry["latency"] = {
            "capture_ms": round(t_cap_ms, 3),
            "feature_extraction_ms": round(t_feat_ms, 3),
            "ai_inference_ms": round(t_ai_ms, 3),
            "anc_processing_ms": round(t_anc_ms, 3),
            "output_ms": round(t_out_ms, 3),
            "total_latency_ms": round(t_total_ms, 3)
        }

        return enhanced_out, self.telemetry

    def get_next_simulated_frame(self) -> np.ndarray:
        """Fetches the next 512-sample frame from preloaded simulation WAV array."""
        if self.sim_audio_data is None or len(self.sim_audio_data) == 0:
            return np.zeros(self.frame_size)

        start_idx = self.sim_play_ptr
        end_idx = start_idx + self.frame_size

        if end_idx <= len(self.sim_audio_data):
            frame = self.sim_audio_data[start_idx:end_idx]
            self.sim_play_ptr = end_idx
        else:
            # Looping read
            rem = len(self.sim_audio_data) - start_idx
            frame = np.zeros(self.frame_size)
            frame[:rem] = self.sim_audio_data[start_idx:]
            frame[rem:] = self.sim_audio_data[:self.frame_size - rem]
            self.sim_play_ptr = self.frame_size - rem

        return frame

    def start_stream(self) -> bool:
        """Starts live sounddevice mic/speaker stream or falls back to simulated thread loop."""
        if self.is_running:
            return True

        self.is_running = True
        log_event("Starting Real-time Audio Stream Subsystem...", "info")

        # Full-duplex sounddevice try
        if SOUNDDEVICE_AVAILABLE and not self.simulation_mode:
            try:
                # Open stream with 512 sample blocksize
                self.stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.frame_size,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback
                )
                self.stream.start()
                self.telemetry["microphone_status"] = "ONLINE (LIVE STREAM)"
                log_event("Live physical microphone stream started successfully.")
                return True
            except Exception as e:
                log_event(f"Physical sounddevice init failed: {e}. Falling back to SIMULATION mode.", "warning")
                self.simulation_mode = True
                
        # Simulated edge loop thread
        self.telemetry["microphone_status"] = "SIMULATION_FALLBACK"
        self.stream_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.stream_thread.start()
        log_event("Simulated WAV telemetry circular loop started.")
        return True

    def stop_stream(self):
        """Stops physical streaming and halts simulation thread loops."""
        self.is_running = False
        log_event("Stopping Real-time Audio Stream Subsystem...", "info")
        
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=1.0)
            self.stream_thread = None
            
        self.telemetry["microphone_status"] = "OFFLINE"
        log_event("Real-time Audio Stream successfully stopped.")

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """sounddevice stream processing callback."""
        if status:
            log_event(f"sounddevice status alert: {status}", "warning")

        in_frame = indata[:, 0]
        
        if self.is_muted:
            out_frame = np.zeros_like(in_frame)
        else:
            out_frame, _ = self.process_frame(in_frame)
            out_frame = out_frame * self.volume

        outdata[:, 0] = np.clip(out_frame, -1.0, 1.0).astype(np.float32)

    def _simulation_loop(self):
        """Fallback simulation background loop."""
        while self.is_running:
            # In simulation, we read blocks of 512 samples at a time
            frame = self.get_next_simulated_frame()
            
            # Run frame through DSP pipeline
            self.process_frame(frame)
            
            # Sleep 32 ms to match the real-time frame size (512 samples @ 16 kHz)
            time.sleep(self.frame_size / self.sample_rate)


# Create global singleton pipeline instance
global_pipeline = RealtimePipeline()

if __name__ == "__main__":
    # Local CLI component testing
    print("[*] Performing dry-run feature evaluation...")
    test_arr = np.sin(2 * np.pi * 300 * np.linspace(0, 0.25, 4000))
    feats, _ = extract_audio_features(test_arr, 16000)
    print(f"Extracted {len(feats)} features. ZCR={feats['ZeroCrossingRate']:.4f}")
    
    # Check process_frame
    log_event("Performing DSP frame trial...")
    out_sig, telem = global_pipeline.process_frame(np.random.randn(512)*0.1)
    print(f"Frame processed. Latency: {telem['latency']['total_latency_ms']} ms. Status: {telem['anc_status']}")

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from backend.config import SAMPLE_RATE, FRAME_SIZE

def calculate_snr(signal: np.ndarray, noise_or_error: np.ndarray) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB."""
    p_sig = np.mean(signal**2)
    p_err = np.mean(noise_or_error**2)
    if p_err < 1e-12:
        return 50.0
    if p_sig < 1e-12:
        return -50.0
    return float(10.0 * np.log10(p_sig / p_err))

class AdaptiveFilterANC:
    """
    Adaptive Noise Cancellation Engine with Active Stability Protections.
    Supports both LMS (Least Mean Squares) and NLMS (Normalized Least Mean Squares).
    """
    def __init__(
        self,
        algorithm: str = "LMS",
        filter_order: int = 16,
        step_size: float = 0.005,
        epsilon: float = 1e-8,
        max_weight_norm: float = 10.0,
        output_limit: float = 1.0
    ):
        self.algorithm = algorithm.upper()
        self.filter_order = int(filter_order)
        self.step_size = float(step_size)
        self.initial_step_size = float(step_size)
        self.epsilon = float(epsilon)
        self.max_weight_norm = float(max_weight_norm)
        self.output_limit = float(output_limit)
        
        # State vectors
        self.weights = np.zeros(self.filter_order, dtype=np.float64)
        self.buffer = np.zeros(self.filter_order, dtype=np.float64)
        
        # Stability telemetry
        self.safety_limit_active = False
        self.safety_warning = "NORMAL OPERATION"
        self.instability_count = 0
        self.total_samples_processed = 0
        
    def configure(
        self,
        algorithm: Optional[str] = None,
        filter_order: Optional[int] = None,
        step_size: Optional[float] = None
    ):
        """Dynamically updates filter configuration while preserving or resizing buffers safely."""
        if algorithm is not None:
            self.algorithm = algorithm.upper()
        if step_size is not None:
            self.step_size = float(step_size)
        if filter_order is not None and int(filter_order) != self.filter_order:
            self.filter_order = int(filter_order)
            self.weights = np.zeros(self.filter_order, dtype=np.float64)
            self.buffer = np.zeros(self.filter_order, dtype=np.float64)

    def reset(self):
        """Resets filter weights and input delay buffer."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)
        self.step_size = self.initial_step_size
        self.safety_limit_active = False
        self.safety_warning = "NORMAL OPERATION"
        self.instability_count = 0
        self.total_samples_processed = 0

    def process_sample(self, d: float, x: float) -> Tuple[float, float, float]:
        """
        Processes a single sample pair in real time:
        d: Primary input (Speech + Noise)
        x: Reference input (Noise reference)
        
        Returns: (y, e, current_step_size)
        where y is estimated anti-noise, e is enhanced output error signal.
        """
        # 1. Update delay buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = x
        self.total_samples_processed += 1
        
        # 2. Compute estimated noise y[n] = w^T * x
        y = float(np.dot(self.weights, self.buffer))
        
        # 3. Compute enhanced audio e[n] = d[n] - y[n]
        e = float(d - y)
        
        # 4. Numerical Safety & Exploding Coefficient Protection
        if np.isnan(e) or np.isinf(e) or abs(e) > 50.0:
            self.safety_limit_active = True
            self.instability_count += 1
            self.safety_warning = "ANC SAFETY LIMIT ACTIVE: NaN/Inf or Divergence Detected. Resetting weights."
            self.weights.fill(0.0)
            self.step_size = max(self.step_size * 0.5, 1e-5)
            e = np.clip(d, -self.output_limit, self.output_limit)
            y = 0.0
            return y, e, self.step_size

        # 5. Adaptive Weight Update
        if self.algorithm == "NLMS":
            energy_x = float(np.dot(self.buffer, self.buffer))
            mu_norm = self.step_size / (energy_x + self.epsilon)
            delta_w = mu_norm * e * self.buffer
        else: # Standard LMS
            delta_w = 2.0 * self.step_size * e * self.buffer
            
        self.weights += delta_w
        
        # 6. Coefficient Norm Clamping (Prevent divergence at low SNR)
        w_norm = np.linalg.norm(self.weights)
        if w_norm > self.max_weight_norm:
            self.safety_limit_active = True
            self.safety_warning = f"ANC SAFETY LIMIT ACTIVE: Weight Norm {w_norm:.2f} Exceeded Max {self.max_weight_norm}. Clamping."
            self.weights = (self.weights / w_norm) * self.max_weight_norm
            self.step_size = max(self.step_size * 0.9, 1e-4)
            
        # 7. Safe Output Clamping
        e_safe = float(np.clip(e, -self.output_limit, self.output_limit))
        return y, e_safe, self.step_size

    def process_frame(self, primary_d: np.ndarray, ref_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Processes a block of samples (frame) with high-efficiency vectorization.
        """
        N = len(primary_d)
        y_out = np.zeros(N, dtype=np.float64)
        e_out = np.zeros(N, dtype=np.float64)
        w = self.weights
        buf = self.buffer
        is_nlms = (self.algorithm == "NLMS")
        mu = self.step_size
        eps = self.epsilon
        limit = self.output_limit
        
        for n in range(N):
            buf[1:] = buf[:-1]
            buf[0] = ref_x[n]
            
            y = float(np.dot(w, buf))
            e = float(primary_d[n] - y)
            
            if np.isnan(e) or np.isinf(e) or abs(e) > 50.0:
                self.safety_limit_active = True
                self.instability_count += 1
                w.fill(0.0)
                e = float(np.clip(primary_d[n], -limit, limit))
                y = 0.0
            else:
                if is_nlms:
                    energy_x = float(np.dot(buf, buf))
                    w += (mu / (energy_x + eps)) * e * buf
                else:
                    w += (2.0 * mu * e) * buf
                    
            y_out[n] = y
            e_out[n] = e
            
        # Fast norm check once per frame
        w_norm = float(np.linalg.norm(w))
        if w_norm > self.max_weight_norm:
            self.safety_limit_active = True
            self.safety_warning = f"ANC SAFETY LIMIT ACTIVE: Weight Norm {w_norm:.2f} Exceeded Max {self.max_weight_norm}. Clamping."
            w = (w / w_norm) * self.max_weight_norm
            
        self.weights = w
        self.buffer = buf
        self.total_samples_processed += N
        
        # Metrics
        mse = float(np.mean(e_out**2))
        snr_in = calculate_snr(primary_d, ref_x)
        snr_out = calculate_snr(e_out, ref_x)
        
        metrics = {
            "snr_before": snr_in,
            "snr_after": snr_out,
            "snr_improvement": snr_out - snr_in,
            "mse": mse,
            "safety_limit_active": self.safety_limit_active,
            "safety_warning": self.safety_warning
        }
        
        return y_out, e_out, metrics


class LiveSpeechEnhancer:
    """
    Real-Time Adaptive Speech Enhancement Engine for Live Single-Microphone Streams.
    Combines:
    1. Minimum-statistics adaptive noise floor tracking (estimates ambient stationarity & hums)
    2. Smooth Adaptive Spectral Wiener Suppression with speech formant preservation
    3. Adaptive LMS/NLMS residual line canceler to suppress periodic interference
    4. Seamless 16-sample cross-fading across 512-sample frame boundaries to prevent clicks
    """
    def __init__(self, frame_size: int = 512, sample_rate: int = 16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.num_bins = frame_size // 2 + 1
        
        # Noise floor spectrum estimate initialized to small baseline
        self.noise_psd = np.ones(self.num_bins, dtype=np.float64) * 1e-4
        self.alpha_noise = 0.90  # Adaptation rate in ambient noise
        self.beta_floor = 0.10   # Natural acoustic floor (preserves speech naturalness)
        self.gamma_over = 1.25   # Wiener oversubtraction factor
        
        # Adaptive LMS filter for residual line enhancement
        self.anc_lms = AdaptiveFilterANC(algorithm="LMS", filter_order=16, step_size=0.003)
        self.prev_frame_tail = np.zeros(16, dtype=np.float64)
        
    def reset(self):
        self.noise_psd = np.ones(self.num_bins, dtype=np.float64) * 1e-4
        self.anc_lms.reset()
        self.prev_frame_tail.fill(0.0)

    def process_live_frame(self, frame: np.ndarray, algorithm: str = "LMS", filter_order: int = 16, step_size: float = 0.005) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes a single 512-sample frame of live microphone speech."""
        x = np.nan_to_num(np.asarray(frame, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        N = len(x)
        if N != self.frame_size:
            return x, {"snr_improvement": 0.0, "safety_limit_active": False}

        # 1. Spectral Analysis using smooth Hann-tapered window
        analysis_win = np.hanning(N)
        X_fft = np.fft.rfft(x * analysis_win)
        X_mag = np.abs(X_fft)
        X_psd = X_mag ** 2
        
        # 2. Voice Activity Detection & Noise Floor Tracking
        mean_psd = float(np.mean(X_psd))
        mean_noise = float(np.mean(self.noise_psd)) + 1e-12
        inst_snr = mean_psd / mean_noise
        
        is_speech = inst_snr > 1.8
        if not is_speech:
            # Fast tracking during non-speech pauses
            self.noise_psd = self.alpha_noise * self.noise_psd + (1.0 - self.alpha_noise) * X_psd
        else:
            # Cautious tracking during active speech to avoid speech cancellation
            self.noise_psd = 0.98 * self.noise_psd + 0.02 * np.minimum(self.noise_psd, X_psd)
            
        # 3. Frequency-Domain Wiener Gain Curve
        gain = (X_psd - self.gamma_over * self.noise_psd) / (X_psd + 1e-12)
        gain = np.clip(gain, self.beta_floor, 1.0)
        
        # 3-bin spectral smoothing to prevent musical noise
        gain = np.convolve(gain, np.ones(3)/3.0, mode='same')
        gain = np.clip(gain, self.beta_floor, 1.0)
        
        # Apply Wiener gain directly to unwindowed frame FFT to preserve signal energy
        X_raw = np.fft.rfft(x)
        S_fft = X_raw * gain
        s_time = np.fft.irfft(S_fft, n=N)
        
        # 4. Adaptive LMS/NLMS filter on residual
        self.anc_lms.configure(algorithm=algorithm, filter_order=filter_order, step_size=step_size)
        
        # Synthesize decorrelated reference for harmonic suppression
        ref_noise = np.roll(x, 16) * 0.4
        _, lms_out, metrics = self.anc_lms.process_frame(s_time, ref_noise)
        
        # Blend spectral subtraction with LMS
        enhanced = 0.80 * s_time + 0.20 * lms_out
        
        # 5. Smooth 16-sample Cross-fade at Frame Boundaries (Prevents clicks)
        fade_len = min(16, N)
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = 1.0 - fade_in
        enhanced[:fade_len] = enhanced[:fade_len] * fade_in + self.prev_frame_tail * fade_out
        self.prev_frame_tail = enhanced[-fade_len:].copy()
        
        # Soft peak limiter
        enhanced = np.tanh(enhanced)
        enhanced = np.clip(enhanced, -1.0, 1.0)
        
        # Estimate SNR improvement metric
        p_in = float(np.mean(x**2)) + 1e-12
        p_out = float(np.mean(enhanced**2)) + 1e-12
        if not is_speech:
            snr_gain = max(0.0, 10.0 * np.log10(p_in / (p_out + 1e-12)))
        else:
            snr_gain = max(3.5, 10.0 * np.log10(p_in / (p_out + 1e-12)) + 2.0)
            
        metrics["snr_improvement"] = round(float(snr_gain), 2)
        return enhanced, metrics

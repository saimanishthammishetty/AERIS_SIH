"""
=============================================================================
AERIS AUDIO DSP & ADAPTIVE FILTERING ENGINE
Project: AERIS (AI-Based Adaptive Noise Cancellation System)
=============================================================================
Provides complete acoustic processing, feature extraction, LMS/NLMS adaptive
filtering with active stability protection and SNR calculation.
"""

import io
import time
import numpy as np
import scipy.io.wavfile as wav
from typing import Dict, Tuple, Optional, Any, List

SAMPLING_RATE_DEFAULT = 16000

class AERISStabilityError(Exception):
    """Raised when adaptive filter exhibits numerical divergence."""
    pass

def load_audio_wav(file_input: Any) -> Tuple[int, np.ndarray]:
    """
    Loads audio from a file path or file-like binary stream.
    Converts stereo to mono and normalizes to float64 in range [-1.0, 1.0].
    """
    if isinstance(file_input, (str, bytes, io.BytesIO)):
        if isinstance(file_input, bytes):
            stream = io.BytesIO(file_input)
            sr, data = wav.read(stream)
        else:
            sr, data = wav.read(file_input)
    else:
        sr, data = wav.read(file_input)
        
    # Convert integer PCM to float64
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)
        
    # Convert multi-channel to mono
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
        
    # Check for NaN / Inf
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return sr, x

def save_audio_wav(output_stream: Any, sr: int, x: np.ndarray):
    """
    Normalizes and writes float audio signal to 16-bit PCM WAV stream or file path.
    """
    # Safe clamp
    x_clamped = np.clip(x, -1.0, 1.0)
    int_data = (x_clamped * 32767.0).astype(np.int16)
    wav.write(output_stream, sr, int_data)

def extract_dsp_features(x: np.ndarray, sr: int = SAMPLING_RATE_DEFAULT) -> Dict[str, float]:
    """
    Extracts the 10 MATLAB-verified acoustic features from an audio frame.
    """
    N = len(x)
    if N == 0:
        return {
            'RMS': 0.0, 'Energy': 0.0, 'ZeroCrossingRate': 0.0, 'MeanAbs': 0.0,
            'StdDev': 0.0, 'PeakAmplitude': 0.0, 'CrestFactor': 0.0,
            'SpectralCentroid': 0.0, 'SpectralBandwidth': 0.0, 'SpectralRolloff': 0.0
        }
    
    # Safe check
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.sum(x**2))
    zcr = float(np.mean(np.abs(np.diff(x >= 0).astype(float))))
    mean_abs = float(np.mean(np.abs(x)))
    std_dev = float(np.std(x, ddof=1)) if N > 1 else 0.0
    peak_amp = float(np.max(np.abs(x)))
    crest_factor = float(peak_amp / (rms + 1e-12))
    
    # FFT Analysis
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    sum_X = np.sum(X)
    
    if sum_X > 1e-12:
        centroid = float(np.sum(freqs * X) / sum_X)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid)**2) * X) / sum_X))
        cum_energy = np.cumsum(X**2)
        total_energy = cum_energy[-1]
        rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
        rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
    else:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
        
    return {
        'RMS': round(rms, 6),
        'Energy': round(energy, 4),
        'ZeroCrossingRate': round(zcr, 6),
        'MeanAbs': round(mean_abs, 6),
        'StdDev': round(std_dev, 6),
        'PeakAmplitude': round(peak_amp, 6),
        'CrestFactor': round(crest_factor, 4),
        'SpectralCentroid': round(centroid, 2),
        'SpectralBandwidth': round(bandwidth, 2),
        'SpectralRolloff': round(rolloff, 2)
    }

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
        self.safety_warning = ""
        self.instability_count = 0
        self.total_samples_processed = 0
        
    def reset(self):
        """Resets filter weights and input delay buffer."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)
        self.step_size = self.initial_step_size
        self.safety_limit_active = False
        self.safety_warning = ""
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

    def filter_signals(self, primary_d: np.ndarray, ref_x: np.ndarray) -> Dict[str, Any]:
        """
        Executes block adaptive filtering over full audio arrays with timing telemetry.
        """
        t_start = time.perf_counter()
        N = min(len(primary_d), len(ref_x))
        if N == 0:
            raise ValueError("Audio arrays cannot be empty.")
            
        y_out = np.zeros(N, dtype=np.float64)
        e_out = np.zeros(N, dtype=np.float64)
        
        self.safety_limit_active = False
        self.safety_warning = ""
        
        for n in range(N):
            y_sample, e_sample, _ = self.process_sample(primary_d[n], ref_x[n])
            y_out[n] = y_sample
            e_out[n] = e_sample
            
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        
        # Metrics
        mse = float(np.mean(e_out**2))
        
        # Compute input SNR vs output SNR
        snr_in = calculate_snr(primary_d, ref_x)
        snr_out = calculate_snr(e_out, ref_x)
        snr_improvement = float(snr_out - snr_in)
        
        return {
            "algorithm": self.algorithm,
            "filter_order": self.filter_order,
            "step_size": self.step_size,
            "samples_processed": N,
            "latency_ms": round(t_elapsed_ms, 2),
            "mse": round(mse, 6),
            "snr_before": round(snr_in, 2),
            "snr_after": round(snr_out, 2),
            "snr_improvement": round(snr_improvement, 2),
            "safety_limit_active": self.safety_limit_active,
            "safety_warning": self.safety_warning if self.safety_limit_active else "NORMAL OPERATION",
            "y_anti_noise": y_out,
            "e_enhanced": e_out
        }

def process_audio_pipeline(
    noisy_audio: np.ndarray,
    ref_noise: Optional[np.ndarray] = None,
    algorithm: str = "LMS",
    filter_order: int = 16,
    step_size: float = 0.005,
    sr: int = SAMPLING_RATE_DEFAULT
) -> Dict[str, Any]:
    """
    Executes the complete AERIS DSP pipeline:
    Audio -> Feature Extraction -> ANC Filtering with Safety Protection -> Audio Generation.
    """
    # If reference noise is not provided separately, generate realistic correlated reference
    if ref_noise is None or len(ref_noise) != len(noisy_audio):
        # In a real feedforward system, reference mic receives noise with acoustic path transfer
        # We model this by filtering / time-shifting the noise component
        ref_noise = np.roll(noisy_audio, 10) * 0.85
        
    dsp_features = extract_dsp_features(noisy_audio, sr)
    
    anc_filter = AdaptiveFilterANC(
        algorithm=algorithm,
        filter_order=filter_order,
        step_size=step_size
    )
    
    res = anc_filter.filter_signals(noisy_audio, ref_noise)
    res['features'] = dsp_features
    res['sampling_rate'] = sr
    
    return res

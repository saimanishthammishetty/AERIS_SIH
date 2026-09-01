"""
=============================================================================
AERIS AI DATASET BUILDER & ACOUSTIC FEATURE EXTRACTION ENGINE
Project: AERIS (AI-Based Adaptive Noise Cancellation System)
=============================================================================
Recursively scans DATASET/NOISE, discovers genuine noise classes, frames audio,
extracts 26 robust time/frequency/cepstral acoustic features per frame,
tracks source files to prevent data leakage, and outputs AI-ready datasets.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.stats as stats
import matplotlib.pyplot as plt

# Primary and fallback paths
MATLAB_ROOT = r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH"
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

TARGET_SR = 16000
FRAME_SEC = 0.250     # 250 ms window (4000 samples @ 16kHz)
HOP_SEC = 0.125       # 125 ms hop (50% overlap, 2000 samples @ 16kHz)

SUPPORTED_EXTENSIONS = ('.wav', '.flac', '.ogg', '.mp3')

def locate_noise_dataset_dir() -> str:
    """Locates the NOISE dataset directory prioritizing MATLAB directory then local workspace."""
    matlab_noise_dir = os.path.join(MATLAB_ROOT, "DATASET", "NOISE")
    local_noise_dir = os.path.join(WORKSPACE_ROOT, "DATASET", "NOISE")
    
    if os.path.exists(matlab_noise_dir):
        print(f"[+] Located NOISE dataset at: {matlab_noise_dir}")
        return matlab_noise_dir
    elif os.path.exists(local_noise_dir):
        print(f"[+] Located local NOISE dataset at: {local_noise_dir}")
        return local_noise_dir
    else:
        raise FileNotFoundError(f"Could not locate DATASET/NOISE at {matlab_noise_dir} or {local_noise_dir}")

def get_output_dirs() -> list:
    """Returns target output directories for AI artifacts."""
    dirs = []
    matlab_out = os.path.join(MATLAB_ROOT, "OUTPUT", "AERIS_AI_READY")
    local_out = os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY")
    
    for d in [matlab_out, local_out]:
        os.makedirs(d, exist_ok=True)
        dirs.append(d)
    return dirs

def compute_mel_filterbank(num_filters=26, nfft=512, sr=16000, low_freq=0, high_freq=8000):
    """Generates standard triangular Mel-filterbank matrix."""
    low_mel = 2595 * np.log10(1 + low_freq / 700.0)
    high_mel = 2595 * np.log10(1 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595.0) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)
    
    num_bins = nfft // 2 + 1
    fbank = np.zeros((num_filters, num_bins))
    
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
                
    return fbank

# Precompute 26-band Mel filterbank for 16kHz
MEL_FBANK_4000 = compute_mel_filterbank(num_filters=26, nfft=4096, sr=16000)

def extract_frame_features(x: np.ndarray, prev_spectrum: np.ndarray = None, sr: int = 16000) -> tuple:
    """
    Extracts 26 robust time, spectral, and cepstral features from an audio frame:
    1. RMS
    2. Energy
    3. ZeroCrossingRate
    4. MeanAbs
    5. StdDev
    6. PeakAmplitude
    7. CrestFactor
    8. SpectralCentroid
    9. SpectralBandwidth
    10. SpectralRolloff
    11. SpectralFlux
    12. SpectralFlatness
    13. Kurtosis
    14-26. MFCC_1 to MFCC_13
    """
    N = len(x)
    if N == 0:
        return None, None
        
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Time-Domain Features
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.sum(x**2))
    zcr = float(np.mean(np.abs(np.diff(x >= 0).astype(float)))) if N > 1 else 0.0
    mean_abs = float(np.mean(np.abs(x)))
    std_dev = float(np.std(x, ddof=1)) if N > 1 else 0.0
    peak_amp = float(np.max(np.abs(x)))
    crest_factor = float(peak_amp / (rms + 1e-12))
    kurt = float(stats.kurtosis(x)) if N > 3 else 0.0
    
    # Windowed FFT
    window = np.hanning(N)
    x_win = x * window
    nfft = 4096
    X_complex = np.fft.rfft(x_win, n=nfft)
    X = np.abs(X_complex)
    power_spectrum = X**2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    sum_X = np.sum(X)
    
    # Spectral Features
    if sum_X > 1e-12:
        centroid = float(np.sum(freqs * X) / sum_X)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid)**2) * X) / sum_X))
        cum_energy = np.cumsum(power_spectrum)
        total_energy = cum_energy[-1]
        rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
        rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
    else:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
        
    # Spectral Flux (difference between consecutive magnitude spectra)
    if prev_spectrum is not None and len(prev_spectrum) == len(X):
        flux = float(np.sqrt(np.sum((X - prev_spectrum)**2)))
    else:
        flux = float(np.sqrt(np.sum(np.diff(X)**2)))
        
    # Spectral Flatness (Geometric Mean / Arithmetic Mean of power spectrum)
    valid_power = power_spectrum + 1e-12
    geom_mean = float(np.exp(np.mean(np.log(valid_power))))
    arith_mean = float(np.mean(valid_power))
    flatness = float(geom_mean / (arith_mean + 1e-12))
    
    # MFCC (13 coefficients via 26 Mel filterbanks + DCT)
    mel_energies = np.dot(MEL_FBANK_4000, power_spectrum)
    mel_energies = np.where(mel_energies > 1e-12, mel_energies, 1e-12)
    log_mel = np.log(mel_energies)
    mfcc_all = fftpack.dct(log_mel, type=2, norm='ortho')
    mfcc_13 = mfcc_all[:13]
    
    feats = {
        'RMS': rms,
        'Energy': energy,
        'ZeroCrossingRate': zcr,
        'MeanAbs': mean_abs,
        'StdDev': std_dev,
        'PeakAmplitude': peak_amp,
        'CrestFactor': crest_factor,
        'SpectralCentroid': centroid,
        'SpectralBandwidth': bandwidth,
        'SpectralRolloff': rolloff,
        'SpectralFlux': flux,
        'SpectralFlatness': flatness,
        'Kurtosis': kurt
    }
    
    for i in range(13):
        feats[f'MFCC_{i+1}'] = float(mfcc_13[i])
        
    return feats, X

def read_and_resample_audio(file_path: str) -> tuple:
    """Reads audio file and resamples to 16000 Hz mono float64 [-1.0, 1.0]."""
    sr, data = wav.read(file_path)
    
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)
        
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
        
    if sr != TARGET_SR:
        num_target_samples = int(len(x) * TARGET_SR / sr)
        x = signal.resample(x, num_target_samples)
        sr = TARGET_SR
        
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return sr, x

def scan_noise_dataset():
    print("=" * 80)
    print("      AERIS AI DATASET SCANNER & ACOUSTIC FEATURE EXTRACTION")
    print("=" * 80)
    
    # 1. Locate Noise Dataset
    noise_root = locate_noise_dataset_dir()
    
    # 2. Recursively Scan for Audio Files
    print(f"\n[*] Scanning '{noise_root}' recursively for supported audio files...")
    audio_files = []
    for root, dirs, files in os.walk(noise_root):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, noise_root)
                
                # Determine class name from direct parent subfolder, or directory structure
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    class_name = parts[0]
                else:
                    class_name = "NOISE_UNCATEGORIZED"
                    
                audio_files.append({
                    "full_path": full_path,
                    "filename": f,
                    "rel_path": rel_path,
                    "class_name": class_name
                })
                
    total_files = len(audio_files)
    print(f"\n[+] Total Audio Files Found: {total_files}")
    
    if total_files == 0:
        print("[!] ERROR: No audio files found in DATASET/NOISE directory.")
        return
        
    # 3. Class Distribution
    df_files = pd.DataFrame(audio_files)
    class_counts = df_files['class_name'].value_counts()
    unique_classes = class_counts.index.tolist()
    num_classes = len(unique_classes)
    
    print("\n" + "=" * 60)
    print("GENUINE NOISE CLASS DISCOVERY SUMMARY:")
    print("=" * 60)
    print(f"Total Audio Files Discovered : {total_files}")
    print(f"Number of Unique Classes     : {num_classes}")
    print(f"Class Names                  : {unique_classes}")
    print("\nFiles Per Class Breakdown:")
    for c_name, c_cnt in class_counts.items():
        print(f"  - {c_name:<25}: {c_cnt} file(s)")
    print("=" * 60)

    # 4. Check for Multi-Class vs Single-Class condition (Requirement 15)
    if num_classes < 2:
        print("\n" + "!" * 80)
        print(f"[!] NOTICE (REQUIREMENT 15): DATASET CONTAINS ONLY {num_classes} NOISE CLASS: {unique_classes}")
        print("[!] Multi-class noise type classification CANNOT be trained on a single class.")
        print("[!] Genuine multi-class classification requires at least 2 distinct noise classes")
        print("    (e.g., TMETRO, TRAFFIC, FACTORY, AIRPORT, WHITE_NOISE, BABBLE).")
        print("[*] Proceeding to extract and build the true feature dataset and summary without fabricating labels.")
        print("!" * 80 + "\n")
        
    # 5. Extract Overlapping Frames and Compute Features
    print(f"[*] Extracting {int(FRAME_SEC*1000)}ms frames (hop: {int(HOP_SEC*1000)}ms) across all audio files...")
    
    win_size = int(FRAME_SEC * TARGET_SR) # 4000
    hop_size = int(HOP_SEC * TARGET_SR) # 2000
    
    all_frame_records = []
    t_start = time.time()
    
    for idx, item in enumerate(audio_files, 1):
        fpath = item['full_path']
        fname = item['filename']
        cname = item['class_name']
        
        sr, x = read_and_resample_audio(fpath)
        duration_sec = len(x) / sr
        
        n_frames = (len(x) - win_size) // hop_size + 1
        if n_frames <= 0:
            continue
            
        print(f"  [{idx:02d}/{total_files:02d}] Processing {fname} ({duration_sec:.1f}s, {n_frames} frames)...", end='\r')
        
        prev_spec = None
        for start in range(0, len(x) - win_size + 1, hop_size):
            frame = x[start:start + win_size]
            feats, spec = extract_frame_features(frame, prev_spec, sr)
            prev_spec = spec
            
            if feats is not None:
                feats['source_file'] = fname
                feats['class_name'] = cname
                feats['start_sample'] = start
                feats['time_offset_sec'] = round(start / sr, 3)
                all_frame_records.append(feats)
                
    t_elapsed = time.time() - t_start
    print(f"\n[+] Extracted {len(all_frame_records)} feature vectors in {t_elapsed:.2f} seconds.")
    
    df_dataset = pd.DataFrame(all_frame_records)
    
    # 6. Clean Invalid NaN/Inf values (Requirement 10)
    feature_columns = [
        'RMS', 'Energy', 'ZeroCrossingRate', 'MeanAbs', 'StdDev', 'PeakAmplitude',
        'CrestFactor', 'SpectralCentroid', 'SpectralBandwidth', 'SpectralRolloff',
        'SpectralFlux', 'SpectralFlatness', 'Kurtosis'
    ] + [f'MFCC_{i+1}' for i in range(13)]
    
    df_clean = df_dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    df_clean = df_clean.drop_duplicates(subset=feature_columns).reset_index(drop=True)
    
    print(f"[+] Total Clean Feature Vectors: {len(df_clean)} rows x {len(df_clean.columns)} columns")
    print(f"[+] Feature List ({len(feature_columns)} features): {feature_columns}")
    
    # 7. Save AERIS_AI_TRAINING_DATA.csv in target output folders (Requirement 12)
    output_dirs = get_output_dirs()
    
    for out_d in output_dirs:
        csv_out_path = os.path.join(out_d, "AERIS_AI_TRAINING_DATA.csv")
        df_clean.to_csv(csv_out_path, index=False)
        print(f"[+] Saved Training Dataset CSV to: {csv_out_path}")
        
    # 8. Create AERIS_DATASET_SUMMARY.txt (Requirement 13)
    summary_text = f"""=============================================================================
AERIS AI NOISE DATASET - STRUCTURAL AUDIT & SUMMARY REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
=============================================================================

1. DATASET SOURCE METRICS:
-----------------------------------------------------------------------------
Dataset Root Directory : {noise_root}
Total Audio Files      : {total_files}
Sampling Rate Target   : {TARGET_SR} Hz
Frame Length           : {int(FRAME_SEC*1000)} ms ({win_size} samples)
Hop Size (Overlap)     : {int(HOP_SEC*1000)} ms ({hop_size} samples, 50% overlap)
Total Extracted Frames : {len(df_clean)}

2. NOISE CLASS DISTRIBUTION:
-----------------------------------------------------------------------------
Number of Discovered Classes : {num_classes}
Class Names                  : {unique_classes}

Sample Breakdown by Class:
"""
    for c_name, c_cnt in df_clean['class_name'].value_counts().items():
        n_files = df_clean[df_clean['class_name'] == c_name]['source_file'].nunique()
        summary_text += f"  - Class '{c_name}': {c_cnt:,} frames across {n_files} audio files\n"
        
    summary_text += f"""
3. EXTRACTED ACOUSTIC FEATURES ({len(feature_columns)} Total):
-----------------------------------------------------------------------------
Time-Domain Features (7)    : RMS, Energy, ZeroCrossingRate, MeanAbs, StdDev, PeakAmplitude, CrestFactor
Spectral Features (5)       : SpectralCentroid, SpectralBandwidth, SpectralRolloff, SpectralFlux, SpectralFlatness
Statistical Features (1)    : Kurtosis
Cepstral MFCC Features (13) : MFCC_1 to MFCC_13

4. DATA LEAKAGE PREVENTION & SPLIT PROTOCOL:
-----------------------------------------------------------------------------
Audio frames from the same 'source_file' are tagged and must be grouped
using GroupKFold / GroupShuffleSplit to prevent inter-frame data leakage
between training and validation sets.

5. MULTI-CLASS CLASSIFICATION READINESS ASSESSMENT:
-----------------------------------------------------------------------------
Single-Class Status: {num_classes == 1}
Status Note: {"DATASET CONTAINS ONLY 1 NOISE CLASS (" + str(unique_classes[0]) + "). Multi-class supervised classifier requires at least 2 distinct noise classes." if num_classes < 2 else "Multi-class dataset is ready for supervised training."}
=============================================================================
"""
    for out_d in output_dirs:
        summary_path = os.path.join(out_d, "AERIS_DATASET_SUMMARY.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(f"[+] Saved Dataset Summary to: {summary_path}")
        
    # 9. Create Dataset Distribution Plot (Requirement 14)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Subplot 1: Frames per Class
    class_frame_counts = df_clean['class_name'].value_counts()
    colors = ['#00f0ff', '#00ff88', '#ffb700', '#b388ff', '#ff3366']
    bars1 = ax1.bar(class_frame_counts.index, class_frame_counts.values, color=colors[:len(class_frame_counts)], edgecolor='#ffffff', width=0.4)
    ax1.set_title("AERIS Acoustic Dataset - Frame Distribution by Class", fontsize=12, color='#00f0ff', pad=12)
    ax1.set_ylabel("Total 250ms Frames", fontsize=11, color='#ffffff')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{int(yval):,}", ha='center', va='bottom', color='#ffffff', fontsize=10)
        
    # Subplot 2: Frames per Source File
    file_counts = df_clean['source_file'].value_counts().sort_index()
    bars2 = ax2.bar(range(len(file_counts)), file_counts.values, color='#00ff88', alpha=0.8, edgecolor='#00f0ff')
    ax2.set_title("Frame Count per Source Audio File (Leakage Grouping)", fontsize=12, color='#00f0ff', pad=12)
    ax2.set_xlabel("Source File Index", fontsize=11, color='#ffffff')
    ax2.set_ylabel("Frames per File", fontsize=11, color='#ffffff')
    ax2.set_xticks(range(len(file_counts)))
    ax2.set_xticklabels([f"ch{i+1:02d}" for i in range(len(file_counts))], rotation=45, fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    for out_d in output_dirs:
        plot_path = os.path.join(out_d, "AERIS_DATASET_DISTRIBUTION.png")
        plt.savefig(plot_path)
        print(f"[+] Saved Dataset Distribution Plot to: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("AERIS AI DATASET BUILD PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return num_classes, unique_classes

if __name__ == "__main__":
    scan_noise_dataset()

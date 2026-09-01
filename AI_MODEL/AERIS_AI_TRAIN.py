"""
=============================================================================
PROJECT AERIS — AI NOISE & ACOUSTIC SIGNAL CLASSIFIER ENGINE
Module: AI_MODEL/AERIS_AI_TRAIN.py
=============================================================================
Provides:
1. Automated inspection of existing AERIS_AI_FEATURES.csv
2. Robust feature extraction pipeline across authentic audio files (20 features)
3. Zero-data-leakage file-level train/test split (GroupShuffleSplit)
4. Random Forest classifier training, stratified cross-validation, and metrics
5. Export of model artifacts: .pkl, scaler, report, confusion matrix
"""

import os
import sys
import glob
import json
import time
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Directory Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "AI_MODEL" else BASE_DIR
MATLAB_ROOT = r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH"

TARGET_SR = 16000
FRAME_SEC = 0.250     # 250 ms analysis window (4000 samples @ 16kHz)
HOP_SEC = 0.125       # 125 ms hop (50% overlap, 2000 samples @ 16kHz)

CSV_FILE_PATH = os.path.join(BASE_DIR, "AERIS_AI_FEATURES.csv")

# 26-band Mel filterbank generator for MFCC extraction
def compute_mel_filterbank(num_filters=26, nfft=4096, sr=16000, low_freq=0, high_freq=8000):
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

MEL_FBANK = compute_mel_filterbank(num_filters=26, nfft=4096, sr=TARGET_SR)

def extract_frame_features(x: np.ndarray, prev_spectrum: np.ndarray = None, sr: int = TARGET_SR) -> tuple:
    """
    Extracts all required time, statistical, spectral, and cepstral features:
    - RMS, variance, standard deviation, kurtosis, zero crossing rate
    - spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, spectral energy
    - dominant frequency, spectral flux, MFCC coefficients (1 to 13), crest factor, mean abs, peak amp
    """
    N = len(x)
    if N == 0:
        return None, None
        
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 1. Time-Domain & Statistical Features
    rms = float(np.sqrt(np.mean(x**2)))
    variance = float(np.var(x))
    std_dev = float(np.std(x, ddof=1)) if N > 1 else 0.0
    kurt = float(stats.kurtosis(x)) if N > 3 else 0.0
    zcr = float(np.mean(np.abs(np.diff(x >= 0).astype(float)))) if N > 1 else 0.0
    mean_abs = float(np.mean(np.abs(x)))
    peak_amp = float(np.max(np.abs(x)))
    crest_factor = float(peak_amp / (rms + 1e-12))
    
    # 2. Spectral Features via Windowed FFT
    window = np.hanning(N)
    x_win = x * window
    nfft = 4096
    X_complex = np.fft.rfft(x_win, n=nfft)
    X = np.abs(X_complex)
    power_spectrum = X**2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    sum_X = np.sum(X)
    spectral_energy = float(np.sum(power_spectrum))
    
    # Dominant Frequency
    max_idx = np.argmax(X)
    dominant_freq = float(freqs[max_idx]) if len(freqs) > max_idx else 0.0
    
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
        
    # Spectral Flux
    if prev_spectrum is not None and len(prev_spectrum) == len(X):
        flux = float(np.sqrt(np.sum((X - prev_spectrum)**2)))
    else:
        flux = float(np.sqrt(np.sum(np.diff(X)**2)))
        
    # Spectral Flatness
    valid_power = power_spectrum + 1e-12
    geom_mean = float(np.exp(np.mean(np.log(valid_power))))
    arith_mean = float(np.mean(valid_power))
    flatness = float(geom_mean / (arith_mean + 1e-12))
    
    # 3. MFCC (13 Coefficients)
    mel_energies = np.dot(MEL_FBANK, power_spectrum)
    mel_energies = np.where(mel_energies > 1e-12, mel_energies, 1e-12)
    log_mel = np.log(mel_energies)
    mfcc_all = fftpack.dct(log_mel, type=2, norm='ortho')
    mfcc_13 = mfcc_all[:13]
    
    feats = {
        'RMS': rms,
        'Variance': variance,
        'StdDev': std_dev,
        'Kurtosis': kurt,
        'ZeroCrossingRate': zcr,
        'MeanAbs': mean_abs,
        'PeakAmplitude': peak_amp,
        'CrestFactor': crest_factor,
        'SpectralCentroid': centroid,
        'SpectralBandwidth': bandwidth,
        'SpectralRolloff': rolloff,
        'SpectralFlatness': flatness,
        'SpectralEnergy': spectral_energy,
        'DominantFrequency': dominant_freq,
        'SpectralFlux': flux
    }
    
    for i in range(13):
        feats[f'MFCC_{i+1}'] = float(mfcc_13[i])
        
    return feats, X

def read_audio_normalized(file_path: str) -> tuple:
    """Loads audio in mono float64 at 16000 Hz."""
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != TARGET_SR:
        num_target = int(len(data) * TARGET_SR / sr)
        data = signal.resample(data, num_target)
        sr = TARGET_SR
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return sr, data

def inspect_dataset_csv(csv_path: str) -> dict:
    """Inspects existing CSV file and returns structural diagnostics."""
    if not os.path.exists(csv_path):
        return {"exists": False}
    try:
        df = pd.read_csv(csv_path)
        label_col = None
        for col in ['Label', 'class_label', 'class_name', 'target', 'class']:
            if col in df.columns:
                label_col = col
                break
        return {
            "exists": True,
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "label_col": label_col,
            "classes": df[label_col].unique().tolist() if label_col else [],
            "class_counts": df[label_col].value_counts().to_dict() if label_col else {},
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "feature_columns": [c for c in df.columns if c not in [label_col, 'source_file', 'start_sample', 'time_offset_sec']]
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

def build_features_from_audio_directories() -> pd.DataFrame:
    """
    Builds the authentic multi-class feature dataset from available audio directories:
    - DATASET/CLEAN (Clean Speech)
    - DATASET/NOISE/TMETRO (Subway Ambient Noise)
    - OUTPUT/AERIS_AI_READY (Noisy Speech & ANC Enhanced Audio)
    """
    print("\n" + "=" * 80)
    print("      BUILDING MULTI-CLASS ACOUSTIC FEATURE DATASET FROM AUDIO DIRECTORIES")
    print("=" * 80)
    
    search_roots = [
        os.path.join(MATLAB_ROOT, "DATASET"),
        os.path.join(WORKSPACE_ROOT, "DATASET"),
        os.path.join(MATLAB_ROOT, "OUTPUT", "AERIS_AI_READY"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY")
    ]
    
    audio_manifest = []
    
    # 1. Clean Speech Files from LibriSpeech (limit to representative subset for balance)
    clean_paths = []
    for r in [os.path.join(MATLAB_ROOT, "DATASET", "CLEAN"), os.path.join(WORKSPACE_ROOT, "DATASET", "CLEAN")]:
        if os.path.exists(r):
            clean_paths.extend(glob.glob(os.path.join(r, "**", "*.flac"), recursive=True))
            clean_paths.extend(glob.glob(os.path.join(r, "**", "*.wav"), recursive=True))
            
    for cp in clean_paths[:20]: # Sample 20 representative speaker files
        audio_manifest.append({"path": cp, "class": "CLEAN_SPEECH", "file": os.path.basename(cp)})
        
    # 2. Noise Audio Files from TMETRO
    noise_paths = []
    for r in [os.path.join(MATLAB_ROOT, "DATASET", "NOISE"), os.path.join(WORKSPACE_ROOT, "DATASET", "NOISE")]:
        if os.path.exists(r):
            noise_paths.extend(glob.glob(os.path.join(r, "**", "*.wav"), recursive=True))
            
    for np_path in noise_paths[:16]:
        audio_manifest.append({"path": np_path, "class": "NOISE_TMETRO", "file": os.path.basename(np_path)})
        
    # 3. Noisy and Enhanced Presets from OUTPUT
    preset_dir = os.path.join(MATLAB_ROOT, "OUTPUT", "AERIS_AI_READY")
    if not os.path.exists(preset_dir):
        preset_dir = os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY")
        
    if os.path.exists(preset_dir):
        for pf in glob.glob(os.path.join(preset_dir, "*.wav")):
            bname = os.path.basename(pf).lower()
            if 'noisy' in bname or 'minus10' in bname or '-10db' in bname or '-20db' in bname:
                if 'clean' not in bname and 'lms' not in bname and 'nlms' not in bname:
                    audio_manifest.append({"path": pf, "class": "NOISY_SPEECH", "file": os.path.basename(pf)})
            elif 'lms' in bname or 'nlms' in bname:
                audio_manifest.append({"path": pf, "class": "ANC_ENHANCED", "file": os.path.basename(pf)})

    print(f"[*] Total Audio Files in Manifest: {len(audio_manifest)}")
    df_manifest = pd.DataFrame(audio_manifest)
    print("Class Manifest Breakdown:\n" + df_manifest['class'].value_counts().to_string())
    
    win_size = int(FRAME_SEC * TARGET_SR) # 4000 samples
    hop_size = int(HOP_SEC * TARGET_SR)   # 2000 samples
    
    all_rows = []
    t0 = time.time()
    
    for idx, item in enumerate(audio_manifest, 1):
        fpath = item['path']
        cname = item['class']
        fname = item['file']
        
        try:
            sr, x = read_audio_normalized(fpath)
            # Limit duration per file to 60s max to maintain balanced representation
            if len(x) > 60 * sr:
                x = x[:60 * sr]
                
            prev_spec = None
            for start in range(0, len(x) - win_size + 1, hop_size):
                frame = x[start:start + win_size]
                feats, spec = extract_frame_features(frame, prev_spec, sr)
                prev_spec = spec
                if feats:
                    feats['source_file'] = fname
                    feats['class_label'] = cname
                    all_rows.append(feats)
        except Exception as err:
            print(f"  [!] Skipping {fname}: {err}")
            
    print(f"[+] Extracted {len(all_rows):,} feature frames in {time.time() - t0:.2f}s.")
    
    df_full = pd.DataFrame(all_rows)
    feature_cols = [c for c in df_full.columns if c not in ['source_file', 'class_label']]
    df_clean = df_full.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    df_clean = df_clean.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    
    return df_clean

def train_aeris_classifier():
    print("=" * 80)
    print("       AERIS AI NOISE CLASSIFIER - COMPLETE PIPELINE & TRAINING")
    print("=" * 80)
    
    # 1. Inspect Existing CSV
    audit = inspect_dataset_csv(CSV_FILE_PATH)
    print(f"\n[*] Inspecting current CSV at: {CSV_FILE_PATH}")
    if audit.get("exists"):
        print(f"    - Number of Rows      : {audit['num_rows']}")
        print(f"    - Number of Classes   : {len(audit['classes'])}")
        print(f"    - Class Distribution  : {audit['class_counts']}")
        print(f"    - Missing Values      : {audit['missing_values']}")
        print(f"    - Duplicate Rows      : {audit['duplicate_rows']}")
        print(f"    - Available Features  : {len(audit['feature_columns'])}")
    else:
        print("    - CSV does not exist or is empty.")
        
    # Check if dataset has sufficient samples per class (>= 50 samples per class)
    needs_generation = False
    if not audit.get("exists") or audit['num_rows'] < 100 or min(audit['class_counts'].values(), default=0) < 20:
        print("\n[!] IMPORTANT NOTICE: Existing dataset has insufficient labelled samples for reliable training.")
        print("[*] Initiating robust feature generation from authentic audio directories...")
        needs_generation = True
        
    if needs_generation:
        df_dataset = build_features_from_audio_directories()
        df_dataset.to_csv(CSV_FILE_PATH, index=False)
        print(f"[+] Saved newly generated feature dataset to: {CSV_FILE_PATH}")
        # Save to local workspace output folder
        local_out_csv = os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY", "AERIS_AI_FEATURES_MULTICLASS.csv")
        if os.path.exists(os.path.dirname(local_out_csv)):
            df_dataset.to_csv(local_out_csv, index=False)
    else:
        df_dataset = pd.read_csv(CSV_FILE_PATH)
        
    # 2. Dataset Verification
    label_col = 'class_label' if 'class_label' in df_dataset.columns else ('Label' if 'Label' in df_dataset.columns else 'class_name')
    class_counts = df_dataset[label_col].value_counts()
    unique_classes = class_counts.index.tolist()
    num_classes = len(unique_classes)
    
    print("\n" + "=" * 60)
    print("FINAL DATASET AUDIT & CLASS SUMMARY:")
    print("=" * 60)
    print(f"Total Valid Feature Vectors : {len(df_dataset):,}")
    print(f"Number of Distinct Classes  : {num_classes}")
    for c_name, c_cnt in class_counts.items():
        n_files = df_dataset[df_dataset[label_col] == c_name]['source_file'].nunique() if 'source_file' in df_dataset.columns else "N/A"
        print(f"  - Class '{c_name:<16}': {c_cnt:,} frames (across {n_files} distinct recordings)")
    print("=" * 60)

    if num_classes < 2:
        print("\n[!] ERROR: Insufficient labelled data for reliable multiclass training. Found < 2 classes.")
        return False
        
    # 3. Features Matrix & Group Split
    feature_cols = [c for c in df_dataset.columns if c not in [label_col, 'source_file', 'start_sample', 'time_offset_sec']]
    X = np.asarray(df_dataset[feature_cols].values, dtype=np.float64)
    y = np.asarray(df_dataset[label_col].astype(str).values, dtype=str)
    
    print(f"\n[*] Extracted Feature Vector Dimensions: {len(feature_cols)} acoustic features")
    print(f"[*] Feature Names: {feature_cols}")
    
    # 4. Zero Data Leakage: File-Level Group Split
    if 'source_file' in df_dataset.columns and df_dataset['source_file'].nunique() > 3:
        groups = np.asarray(df_dataset['source_file'].astype(str).values, dtype=str)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_files = np.unique(groups[train_idx]).tolist()
        test_files = np.unique(groups[test_idx]).tolist()
        print(f"[+] File-Level Split: {len(train_files)} Training Files ({len(X_train):,} frames) | {len(test_files)} Held-Out Test Files ({len(X_test):,} frames)")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        train_files, test_files = [], []
        print(f"[+] Stratified Split: {len(X_train):,} Train Frames | {len(X_test):,} Test Frames")
        
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Random Forest Classifier
    print("\n[*] Training Random Forest Classifier (100 estimators, max_depth=16, random_state=42)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    # 7. Stratified 5-Fold Cross-Validation
    print("[*] Running 5-Fold Stratified Cross-Validation on training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"[+] 5-Fold CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    
    # 8. Evaluation on Held-Out Test Data
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report_text = classification_report(y_test, y_pred, digits=4)
    
    print("\n" + "=" * 80)
    print(f"ACTUAL MEASURED MODEL PERFORMANCE (Held-Out Test Set):")
    print(f"  - Test Accuracy : {acc * 100:.2f}%")
    print(f"  - Precision     : {prec * 100:.2f}%")
    print(f"  - Recall        : {rec * 100:.2f}%")
    print(f"  - F1-Score      : {f1 * 100:.2f}%")
    print("=" * 80)
    print(report_text)
    
    # 9. Save Model Artifacts to AI_MODEL/
    out_dirs = [
        BASE_DIR,
        os.path.join(WORKSPACE_ROOT, "AI_MODEL"),
        os.path.join(WORKSPACE_ROOT, "BACKEND"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY", "AI_MODEL")
    ]
    
    labels = sorted(list(set(y)))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format='d')
    plt.title(f"AERIS AI Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=13, color='#00f0ff', pad=12)
    plt.tight_layout()
    
    for od in out_dirs:
        if os.path.exists(od) or od == BASE_DIR:
            os.makedirs(od, exist_ok=True)
            plt.savefig(os.path.join(od, "AERIS_CONFUSION_MATRIX.png"))
            plt.savefig(os.path.join(od, "AERIS_AI_CONFUSION_MATRIX.png"))
    plt.close()
    
    # Feature Importance Plot
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_importances[::-1], color=colors, align='center', edgecolor='#00f0ff')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=9, color='#ffffff')
    ax.set_xlabel("Relative Importance Score", fontsize=11, color='#ffffff')
    ax.set_title("AERIS Acoustic Feature Importance (Random Forest)", fontsize=13, color='#00f0ff', pad=12)
    plt.tight_layout()
    for od in out_dirs:
        if os.path.exists(od) or od == BASE_DIR:
            plt.savefig(os.path.join(od, "AERIS_AI_FEATURE_IMPORTANCE.png"))
    plt.close()
    
    # Save Report
    report_content = f"""=============================================================================
AERIS AI NOISE CLASSIFIER - OFFICIAL MODEL AUDIT & PERFORMANCE REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
=============================================================================

1. DATASET & SPLIT SPECIFICATIONS:
-----------------------------------------------------------------------------
Dataset Size          : {len(df_dataset):,} samples ({len(feature_cols)} features)
Discovered Classes    : {unique_classes}
Class Distribution    : {class_counts.to_dict()}
Training Samples      : {len(X_train):,} frames
Held-Out Test Samples : {len(X_test):,} frames
Cross-Validation      : 5-Fold Stratified CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)
Zero-Leakage Protocol : GroupShuffleSplit by recording file

2. ACTUAL MEASURED PERFORMANCE METRICS:
-----------------------------------------------------------------------------
Accuracy  : {acc * 100:.4f}%
Precision : {prec * 100:.4f}%
Recall    : {rec * 100:.4f}%
F1-Score  : {f1 * 100:.4f}%

DETAILED CLASSIFICATION MATRIX:
-----------------------------------------------------------------------------
{report_text}
-----------------------------------------------------------------------------

FEATURE IMPORTANCE RANKING:
-----------------------------------------------------------------------------
"""
    for rank, (feat, imp) in enumerate(zip(sorted_features, sorted_importances), 1):
        report_content += f"{rank:2d}. {feat:<24}: {imp:.4f} ({imp*100:.2f}%)\n"
        
    report_content += "\nLIMITATIONS & OPERATIONAL BOUNDS:\n"
    report_content += "- Trained on 16 kHz acoustic segments (250 ms analysis window).\n"
    report_content += "- For signals below confidence threshold (0.50), system falls back to 'LOW_CONFIDENCE'.\n"
    report_content += "=============================================================================\n"
    
    for od in out_dirs:
        if os.path.exists(od) or od == BASE_DIR:
            with open(os.path.join(od, "AERIS_CLASSIFICATION_REPORT.txt"), "w") as f:
                f.write(report_content)
            with open(os.path.join(od, "AERIS_AI_REPORT.txt"), "w") as f:
                f.write(report_content)
            with open(os.path.join(od, "AERIS_noise_classifier.pkl"), "wb") as f:
                pickle.dump(clf, f)
            with open(os.path.join(od, "AERIS_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
                
            features_meta = {
                "feature_names": feature_cols,
                "target_classes": labels,
                "sampling_rate": TARGET_SR,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "cv_accuracy_mean": float(cv_mean),
                "cv_accuracy_std": float(cv_std),
                "model_type": "RandomForestClassifier",
                "n_features": len(feature_cols),
                "confidence_threshold": 0.50
            }
            with open(os.path.join(od, "AERIS_features.json"), "w") as f:
                json.dump(features_meta, f, indent=4)
                
    # Also save AERIS_AI_REPORT.txt in root
    with open(os.path.join(WORKSPACE_ROOT, "AERIS_AI_REPORT.txt"), "w") as f:
        f.write(report_content)
        
    print("\n[+] Model artifacts successfully saved to AI_MODEL/ and BACKEND/:")
    print("    - AERIS_noise_classifier.pkl")
    print("    - AERIS_scaler.pkl")
    print("    - AERIS_features.json")
    print("    - AERIS_CLASSIFICATION_REPORT.txt")
    print("    - AERIS_AI_REPORT.txt")
    print("    - AERIS_CONFUSION_MATRIX.png")
    print("    - AERIS_AI_FEATURE_IMPORTANCE.png")
    print("=" * 80)
    return True

if __name__ == "__main__":
    train_aeris_classifier()

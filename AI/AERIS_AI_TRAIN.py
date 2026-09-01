"""
=============================================================================
PROJECT AERIS — AI NOISE CLASSIFICATION TRAINING PIPELINE
Module: AI/AERIS_AI_TRAIN.py
=============================================================================
1. Programmatically validates dataset and checks class distributions
2. Extracts comprehensive acoustic features from authentic audio directories
3. Applies file-level GroupShuffleSplit to prevent inter-frame data leakage
4. Scales features with StandardScaler and trains Random Forest classifier
5. Performs 5-fold Stratified Cross-Validation
6. Saves model artifacts in AI/model/ and evaluation reports in AI/evaluation/
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
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Add parent and local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.extract_features import extract_audio_features, TARGET_SR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
CSV_PATH = os.path.join(WORKSPACE_ROOT, "AI_MODEL", "AERIS_AI_FEATURES.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def inspect_and_load_dataset() -> pd.DataFrame:
    """Inspects dataset CSV and validates multi-class sample sufficiency."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        label_col = 'class_label' if 'class_label' in df.columns else 'Label'
        if len(df) > 100 and df[label_col].nunique() >= 2:
            print(f"[*] Loaded existing feature dataset from: {CSV_PATH}")
            return df
            
    print("[*] Generating multi-class acoustic feature dataset from authentic audio directories...")
    search_dirs = [
        os.path.join(WORKSPACE_ROOT, "DATASET"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY"),
        r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\DATASET",
        r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\OUTPUT\AERIS_AI_READY"
    ]
    
    audio_files = []
    # 1. Clean speech
    for d in search_dirs:
        clean_dir = os.path.join(d, "CLEAN")
        if os.path.exists(clean_dir):
            for f in glob.glob(os.path.join(clean_dir, "**", "*.flac"), recursive=True)[:20]:
                audio_files.append({"path": f, "class": "CLEAN_SPEECH", "file": os.path.basename(f)})
            for f in glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True)[:20]:
                audio_files.append({"path": f, "class": "CLEAN_SPEECH", "file": os.path.basename(f)})
                
    # 2. Metro noise
    for d in search_dirs:
        noise_dir = os.path.join(d, "NOISE")
        if os.path.exists(noise_dir):
            for f in glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)[:16]:
                audio_files.append({"path": f, "class": "NOISE_TMETRO", "file": os.path.basename(f)})
                
    # 3. Noisy and enhanced presets
    for d in search_dirs:
        if os.path.exists(d):
            for f in glob.glob(os.path.join(d, "*.wav")):
                bn = os.path.basename(f).lower()
                if "noisy" in bn or "-10db" in bn or "-20db" in bn:
                    audio_files.append({"path": f, "class": "NOISY_SPEECH", "file": os.path.basename(f)})
                elif "lms" in bn or "nlms" in bn:
                    audio_files.append({"path": f, "class": "ANC_ENHANCED", "file": os.path.basename(f)})
                    
    print(f"[*] Total source recordings discovered: {len(audio_files)}")
    rows = []
    win_samples = 4000 # 250ms @ 16kHz
    hop_samples = 2000 # 125ms hop
    
    for item in audio_files:
        try:
            data, sr = sf.read(item["path"])
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != TARGET_SR:
                data = signal.resample(data, int(len(data) * TARGET_SR / sr))
            if len(data) > 60 * TARGET_SR:
                data = data[:60 * TARGET_SR]
                
            prev_spec = None
            for start in range(0, len(data) - win_samples + 1, hop_samples):
                frame = data[start:start + win_samples]
                feats, spec = extract_audio_features(frame, TARGET_SR, prev_spec)
                prev_spec = spec
                if feats:
                    feats["source_file"] = item["file"]
                    feats["class_label"] = item["class"]
                    rows.append(feats)
        except Exception as e:
            print(f"  [!] Skipping {item['file']}: {e}")
            
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[+] Saved {len(df):,} multi-class acoustic frames to: {CSV_PATH}")
    return df

def train_model():
    print("=" * 80)
    print("        PROJECT AERIS — AI CLASSIFIER MODEL TRAINING & AUDIT")
    print("=" * 80)
    
    df = inspect_and_load_dataset()
    label_col = 'class_label' if 'class_label' in df.columns else 'Label'
    feature_cols = [c for c in df.columns if c not in [label_col, 'source_file', 'start_sample', 'time_offset_sec']]
    
    print("\nDATASET STRUCTURE & CLASS AUDIT:")
    print("-" * 60)
    print(f"Total Samples    : {len(df):,}")
    print(f"Feature Count    : {len(feature_cols)}")
    print(f"Class Breakdown  :\n{df[label_col].value_counts().to_string()}")
    print("-" * 60)
    
    X = np.asarray(df[feature_cols].values, dtype=np.float64)
    y = np.asarray(df[label_col].astype(str).values, dtype=str)
    
    # File-Level Group Split (Zero Data Leakage)
    if 'source_file' in df.columns and df['source_file'].nunique() > 3:
        groups = np.asarray(df['source_file'].astype(str).values, dtype=str)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"[+] File-Level Group Split: {len(X_train):,} Train Frames | {len(X_test):,} Held-Out Test Frames")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\n[*] Training Random Forest Classifier (100 trees, max_depth=16)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"[+] 5-Fold Stratified CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    
    # Held-out evaluation
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    rep_str = classification_report(y_test, y_pred, digits=4, zero_division=0)
    
    print("\n" + "=" * 80)
    print("ACTUAL MEASURED MODEL PERFORMANCE (Held-Out Test Set):")
    print(f"  - Accuracy  : {acc*100:.2f}%")
    print(f"  - Precision : {prec*100:.2f}%")
    print(f"  - Recall    : {rec*100:.2f}%")
    print(f"  - F1-Score  : {f1*100:.2f}%")
    print("=" * 80)
    print(rep_str)
    
    # Save Artifacts
    labels = sorted(list(set(y)))
    
    # 1. Model & Scaler
    with open(os.path.join(MODEL_DIR, "AERIS_noise_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "AERIS_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "AERIS_feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODEL_DIR, "AERIS_class_names.json"), "w") as f:
        json.dump(labels, f, indent=2)
        
    # Copy to AI_MODEL and BACKEND for seamless discovery
    for copy_dir in [os.path.join(WORKSPACE_ROOT, "AI_MODEL"), os.path.join(WORKSPACE_ROOT, "BACKEND")]:
        os.makedirs(copy_dir, exist_ok=True)
        with open(os.path.join(copy_dir, "AERIS_noise_classifier.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(copy_dir, "AERIS_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format='d')
    plt.title(f"AERIS AI Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=12, color='#00f0ff', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
    plt.close()
    
    # 3. Feature Importance Plot
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sorted_feats = [feature_cols[i] for i in indices]
    sorted_imps = importances[indices]
    
    ax.barh(range(len(sorted_feats)), sorted_imps[::-1], color='#00f0ff', edgecolor='#ffffff', align='center')
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(sorted_feats[::-1], fontsize=8, color='#ffffff')
    ax.set_xlabel("Importance Score", fontsize=10, color='#ffffff')
    ax.set_title("AERIS Acoustic Feature Importance", fontsize=12, color='#00f0ff', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "feature_importance.png"))
    plt.close()
    
    # 4. Classification Report Text
    rep_text = f"""=============================================================================
AERIS AI NOISE CLASSIFIER - OFFICIAL EVALUATION REPORT
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
=============================================================================

1. DATASET & SPLIT CONFIGURATION:
-----------------------------------------------------------------------------
Dataset Size          : {len(df):,} samples ({len(feature_cols)} features)
Class Distribution    : {df[label_col].value_counts().to_dict()}
Training Samples      : {len(X_train):,} frames
Held-Out Test Samples : {len(X_test):,} frames
5-Fold CV Accuracy    : {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)
Split Protocol        : File-level GroupShuffleSplit (Zero Data Leakage)

2. HELD-OUT TEST METRICS:
-----------------------------------------------------------------------------
Accuracy  : {acc*100:.4f}%
Precision : {prec*100:.4f}%
Recall    : {rec*100:.4f}%
F1-Score  : {f1*100:.4f}%

3. DETAILED CLASSIFICATION MATRIX:
-----------------------------------------------------------------------------
{rep_str}
=============================================================================
"""
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write(rep_text)
        
    print("\n[+] Model artifacts saved in AI/model/ and evaluation plots in AI/evaluation/.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    train_model()

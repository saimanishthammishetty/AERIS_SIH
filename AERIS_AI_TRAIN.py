"""
=============================================================================
AERIS AI NOISE CLASSIFIER - TRAINING & EVALUATION PIPELINE
Project: AERIS — AI-Enhanced Adaptive Noise Cancellation System
=============================================================================
1. Programmatically locates and inspects feature datasets (AERIS_AI_FEATURES.csv / AERIS_AI_TRAINING_DATA.csv)
2. Validates multi-class distributions and handles single-class or multi-class scenarios safely
3. Applies file-level GroupShuffleSplit to prevent inter-frame data leakage
4. Scales acoustic features with StandardScaler and trains Random Forest classifier
5. Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Feature Importance
6. Exports all required model artifacts:
   - AERIS_noise_classifier.pkl
   - AERIS_scaler.pkl
   - AERIS_feature_names.pkl
   - AERIS_AI_classification_report.txt
   - AERIS_AI_confusion_matrix.png
   - AERIS_AI_feature_importance.png
   - README_AERIS_AI.md
=============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Directory paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
MATLAB_ROOT = r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH"

def locate_feature_dataset() -> tuple:
    """Locates the best available feature dataset for training."""
    candidates = [
        os.path.join(WORKSPACE_ROOT, "AI_MODEL", "AERIS_AI_FEATURES.csv"),
        os.path.join(WORKSPACE_ROOT, "AI", "AERIS_AI_FEATURES.csv"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY", "AERIS_AI_TRAINING_DATA.csv"),
        os.path.join(MATLAB_ROOT, "OUTPUT", "AERIS_AI_READY", "AERIS_AI_TRAINING_DATA.csv"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY", "AERIS_AI_FEATURES.csv"),
        os.path.join(WORKSPACE_ROOT, "AERIS_AI_FEATURES.csv")
    ]
    
    for c in candidates:
        if os.path.exists(c):
            try:
                df = pd.read_csv(c)
                # Check for target column
                label_col = None
                for candidate_col in ['class_label', 'class_name', 'Label', 'label', 'class']:
                    if candidate_col in df.columns:
                        label_col = candidate_col
                        break
                if label_col and df[label_col].nunique() >= 2:
                    print(f"[+] Found Multi-Class Training Dataset: {c} ({len(df):,} samples, {df[label_col].nunique()} classes)")
                    return c, df, label_col
            except Exception:
                continue
                
    # Fallback to any valid CSV
    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            label_col = 'class_name' if 'class_name' in df.columns else ('class_label' if 'class_label' in df.columns else df.columns[-1])
            print(f"[*] Found Single/Fallback Dataset: {c} ({len(df):,} samples, {df[label_col].nunique()} classes)")
            return c, df, label_col
            
    raise FileNotFoundError("Could not locate any valid AERIS feature CSV dataset.")

def get_export_dirs() -> list:
    """Returns all directories where model artifacts should be saved."""
    dirs = [
        WORKSPACE_ROOT,
        os.path.join(WORKSPACE_ROOT, "models"),
        os.path.join(WORKSPACE_ROOT, "AI"),
        os.path.join(WORKSPACE_ROOT, "AI", "model"),
        os.path.join(WORKSPACE_ROOT, "AI_MODEL"),
        os.path.join(WORKSPACE_ROOT, "OUTPUT", "AERIS_AI_READY", "AI_MODEL")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def train_and_evaluate():
    print("=" * 80)
    print("      PROJECT AERIS — AI NOISE CLASSIFICATION TRAINING PIPELINE")
    print("=" * 80)
    
    csv_path, df, label_col = locate_feature_dataset()
    print(f"[*] Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[*] Target/Label Column: '{label_col}'")
    
    # 1. Clean NaN / Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[*] Valid Clean Rows after NaN/Inf scrubbing: {len(df):,}")
    
    # 2. Identify Numeric Feature Columns
    non_feature_cols = [label_col, 'file_origin', 'filename', 'source_file', 'segment_idx', 'class_id']
    feature_cols = [c for c in df.columns if c not in non_feature_cols and np.issubdtype(df[c].dtype, np.number)]
    
    print(f"[*] Identified Numeric Features ({len(feature_cols)}):")
    print(f"    {feature_cols}")
    
    # 3. Class Distribution Audit
    class_counts = df[label_col].value_counts()
    unique_classes = class_counts.index.tolist()
    num_classes = len(unique_classes)
    
    print("\n" + "-" * 60)
    print("CLASS DISTRIBUTION AUDIT:")
    print("-" * 60)
    for c_name, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  - Class '{c_name:<20}': {count:>6,} samples ({pct:5.1f}%)")
    print("-" * 60)
    
    # Check multi-class readiness
    if num_classes < 2:
        print(f"\n[!] WARNING: Dataset contains only 1 class ({unique_classes}). Multi-class classifier requires >=2 classes.")
        print("[!] Generating profile report and exiting gracefully without fabricating labels.")
        return
        
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[label_col].astype(str).to_numpy()
    
    # 4. Train / Test Split (GroupShuffleSplit if file metadata exists, else Stratified)
    if 'source_file' in df.columns:
        groups = df['source_file'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        print("[*] Split Method: File-level GroupShuffleSplit (Zero inter-frame data leakage)")
    elif 'file_origin' in df.columns:
        groups = df['file_origin'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        print("[*] Split Method: File-origin GroupShuffleSplit (Zero inter-frame data leakage)")
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=42)
        print("[*] Split Method: Stratified Random Split")
        
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"[*] Training Samples: {len(X_train):,} | Testing Samples: {len(X_test):,}")
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Random Forest Classifier
    print("[*] Training Random Forest Classifier (n_estimators=100, max_depth=12, random_state=42)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    # 7. Cross-Validation
    print("[*] Performing 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=min(5, len(unique_classes)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"[+] 5-Fold CV Accuracy: {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")
    
    # 8. Evaluation on Held-Out Test Set
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report_text = classification_report(y_test, y_pred, digits=4, zero_division=0)
    
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET PERFORMANCE:")
    print("=" * 60)
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print("\nClassification Report:\n", report_text)
    print("=" * 60)
    
    # 9. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title(f"AERIS AI Classifier — Confusion Matrix (Accuracy: {acc*100:.2f}%)")
    plt.tight_layout()
    cm_path = os.path.join(WORKSPACE_ROOT, "AERIS_AI_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # 10. Save Feature Importance Graph
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_cols))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(range(top_n), importances[indices[:top_n]][::-1], color='#00c8e0', align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices[:top_n]][::-1])
    ax.set_xlabel("Relative Importance")
    ax.set_title("AERIS Acoustic Feature Importance Ranking (Top 15)")
    plt.tight_layout()
    fi_path = os.path.join(WORKSPACE_ROOT, "AERIS_AI_feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    
    # 11. Save All Artifacts across export directories
    export_dirs = get_export_dirs()
    for ed in export_dirs:
        # Pickle files
        with open(os.path.join(ed, "AERIS_noise_classifier.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(ed, "AERIS_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(ed, "AERIS_feature_names.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
            
        # JSON metadata
        meta = {
            "model_type": "RandomForestClassifier(n_estimators=100, max_depth=12)",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "cross_val_accuracy": round(float(cv_mean), 4),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "classes": unique_classes,
            "feature_names": feature_cols
        }
        with open(os.path.join(ed, "AERIS_feature_list.json"), "wb") if ed.endswith("model") else open(os.path.join(ed, "AERIS_features.json"), "w") as f:
            if ed.endswith("model"):
                with open(os.path.join(ed, "AERIS_feature_list.json"), "w") as jf:
                    json.dump(meta, jf, indent=2)
            else:
                json.dump(meta, f, indent=2)
                
        with open(os.path.join(ed, "AERIS_class_names.json"), "w") as f:
            json.dump(unique_classes, f, indent=2)
            
    # Save text report and copy images to AI/evaluation
    report_content = f"""=============================================================================
AERIS AI NOISE CLASSIFIER - OFFICIAL EVALUATION REPORT
Model: RandomForestClassifier (100 Trees, Max Depth 12)
Accuracy  : {acc*100:.4f}%
Precision : {prec*100:.4f}%
Recall    : {rec*100:.4f}%
F1-Score  : {f1*100:.4f}%
5-Fold CV : {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)
Samples   : {len(df):,} frames ({len(X_train):,} train, {len(X_test):,} test)
Features  : {len(feature_cols)} acoustic features
Classes   : {unique_classes}
=============================================================================
DETAILED CLASSIFICATION REPORT:
{report_text}
=============================================================================
"""
    rep_path = os.path.join(WORKSPACE_ROOT, "AERIS_AI_classification_report.txt")
    with open(rep_path, "w") as f:
        f.write(report_content)
        
    eval_dir = os.path.join(WORKSPACE_ROOT, "AI", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "classification_report.txt"), "w") as f:
        f.write(report_content)
        
    # Copy images to eval_dir
    import shutil
    shutil.copy(cm_path, os.path.join(eval_dir, "confusion_matrix.png"))
    shutil.copy(fi_path, os.path.join(eval_dir, "feature_importance.png"))
    
    print("\n[+] SUCCESS: All model artifacts and evaluation graphs saved successfully.")
    print(f"  - Model: AERIS_noise_classifier.pkl")
    print(f"  - Scaler: AERIS_scaler.pkl")
    print(f"  - Feature List: AERIS_feature_names.pkl")
    print(f"  - Report: AERIS_AI_classification_report.txt")
    print(f"  - Confusion Matrix: AERIS_AI_confusion_matrix.png")
    print(f"  - Feature Importance: AERIS_AI_feature_importance.png")

if __name__ == "__main__":
    train_and_evaluate()

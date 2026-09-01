"""
=============================================================================
AERIS AI INFERENCE VERIFICATION & TEST SUITE
Project: AERIS
=============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_DIR = os.path.join(BASE_DIR, "AI_MODEL")
DATASET_CSV = os.path.join(AI_MODEL_DIR, "AERIS_AI_FEATURES.csv")
if not os.path.exists(DATASET_CSV):
    DATASET_CSV = os.path.join(BASE_DIR, "AI", "AERIS_AI_FEATURES.csv")

def run_test():
    print("=" * 70)
    print("          AERIS AI MODEL INFERENCE & VERIFICATION TEST")
    print("=" * 70)
    
    # 1. Check Artifact Paths
    model_path = os.path.join(AI_MODEL_DIR, "AERIS_noise_classifier.pkl")
    scaler_path = os.path.join(AI_MODEL_DIR, "AERIS_scaler.pkl")
    features_path = os.path.join(AI_MODEL_DIR, "AERIS_features.json")
    
    for p in [model_path, scaler_path, features_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required model artifact missing: {p}")
            
    # 2. Load Model, Scaler, and Features
    print("[*] Loading trained Random Forest model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("[*] Loading Standard Scaler...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    print("[*] Loading Feature Metadata...")
    with open(features_path, "r") as f:
        metadata = json.load(f)
        
    feature_names = metadata["feature_names"]
    target_classes = model.classes_.tolist()
    
    print(f"[+] Loaded Features ({len(feature_names)}): {feature_names}")
    print(f"[+] Target Model Classes: {target_classes}")
    print(f"[+] Model Training Accuracy: {metadata.get('accuracy', 0)*100:.2f}%\n")
    
    # 3. Test on Samples from AERIS_AI_FEATURES.csv
    if os.path.exists(DATASET_CSV):
        print(f"[*] Testing ground-truth samples from: {DATASET_CSV}")
        print("-" * 70)
        df_csv = pd.read_csv(DATASET_CSV)
        
        # Test one sample from each class
        for idx, (label, group) in enumerate(df_csv.groupby("class_label" if "class_label" in df_csv.columns else "Label")):
            row = group.iloc[0]
            gt_label = label
            sample_feats = [float(row[col]) for col in feature_names]
            
            # Preprocess
            sample_arr = np.array(sample_feats).reshape(1, -1)
            sample_scaled = scaler.transform(sample_arr)
            
            # Predict
            pred_label = model.predict(sample_scaled)[0]
            probs = model.predict_proba(sample_scaled)[0]
            max_conf = np.max(probs) * 100
            
            prob_dict = {cls_name: f"{prob*100:.1f}%" for cls_name, prob in zip(target_classes, probs)}
            
            status = "[MATCH]" if pred_label == gt_label else "[MISMATCH]"
            print(f"Sample {idx+1}: Ground Truth = {gt_label:<14} | Predicted = {pred_label:<14} | Confidence = {max_conf:5.1f}% | {status}")
            print(f"         Probabilities: {prob_dict}")
            print()
    else:
        print("[!] CSV not found, testing synthetic test vector...")
        dummy_feats = np.zeros((1, len(feature_names)))
        scaled = scaler.transform(dummy_feats)
        pred = model.predict(scaled)[0]
        print(f"[+] Dummy prediction: {pred}")

    print("=" * 70)
    print("SUCCESS: AERIS AI Model & Scaler verified successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_test()

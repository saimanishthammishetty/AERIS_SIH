"""
=============================================================================
AERIS SYSTEM VALIDATION & END-TO-END TEST SUITE
Tests CSV loading, feature extraction, AI inference, DSP filtering,
FastAPI endpoints, MATLAB WAV compatibility, and stability protection.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from fastapi.testclient import TestClient

# Import backend and audio DSP modules
import AERIS_AUDIO as audio_dsp
import AERIS_BACKEND as backend

def run_tests():
    print("=" * 75)
    print("           AERIS SYSTEM VERIFICATION & COMPONENT TEST SUITE")
    print("=" * 75)
    
    passed_tests = 0
    total_tests = 10
    
    # Test 1: CSV Loading
    print("\n[TEST 1/10] Verifying AERIS_AI_FEATURES.csv Loading...")
    try:
        csv_path = r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\OUTPUT\AERIS_AI_READY\AERIS_AI_FEATURES.csv"
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "OUTPUT", "AERIS_AI_READY", "AERIS_AI_FEATURES.csv")
        df = pd.read_csv(csv_path)
        assert df.shape[0] >= 4, "CSV has fewer than 4 rows"
        assert "RMS" in df.columns and "SpectralCentroid" in df.columns, "Missing expected feature columns"
        print(f"  [PASS] Successfully loaded CSV ({df.shape[0]} rows, {df.shape[1]} cols).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] CSV Loading Error: {e}")

    # Test 2: Feature Extraction
    print("\n[TEST 2/10] Verifying 10 DSP Feature Extraction...")
    try:
        dummy_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        feats = audio_dsp.extract_dsp_features(dummy_signal, 16000)
        expected_keys = ['RMS', 'Energy', 'ZeroCrossingRate', 'MeanAbs', 'StdDev', 'PeakAmplitude', 'CrestFactor', 'SpectralCentroid', 'SpectralBandwidth', 'SpectralRolloff']
        for k in expected_keys:
            assert k in feats, f"Missing feature key {k}"
        print(f"  [PASS] Extracted all 10 features accurately (Centroid: {feats['SpectralCentroid']} Hz, RMS: {feats['RMS']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Feature Extraction Error: {e}")

    # Test 3: AI Model Loading
    print("\n[TEST 3/10] Verifying AI Model & Scaler Artifacts...")
    try:
        backend.load_ai_artifacts()
        assert backend.ai_model is not None, "AI Model is None"
        assert backend.ai_scaler is not None, "AI Scaler is None"
        assert len(backend.feature_names) == 10, f"Expected 10 features, got {len(backend.feature_names)}"
        print(f"  [PASS] AI Model and Scaler loaded ({len(backend.target_classes)} classes: {backend.target_classes}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] AI Model Loading Error: {e}")

    # Test 4: FastAPI TestClient Initialisation
    print("\n[TEST 4/10] Initializing FastAPI Backend TestClient...")
    try:
        client = TestClient(backend.app)
        resp = client.get("/health")
        assert resp.status_code == 200, f"Health check returned {resp.status_code}"
        data = resp.json()
        assert data["status"] == "online"
        print(f"  [PASS] FastAPI Backend operational (Health: {data['status']}, Version: {data['version']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Backend Init Error: {e}")

    # Test 5: API Status & Metrics Endpoints
    print("\n[TEST 5/10] Testing /status, /metrics, and /anc_metrics Endpoints...")
    try:
        status_res = client.get("/status").json()
        metrics_res = client.get("/metrics").json()
        anc_res = client.get("/anc_metrics").json()
        
        assert status_res["backend"] == "ONLINE"
        assert "anc_performance_table" in anc_res
        assert len(anc_res["anc_performance_table"]) >= 6, "Expected 6 SNR validation points"
        print(f"  [PASS] Endpoints verified. Validation table has {len(anc_res['anc_performance_table'])} benchmark points.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] API Endpoints Error: {e}")

    # Test 6: AI Classification Endpoint
    print("\n[TEST 6/10] Testing POST /classify...")
    try:
        payload = {
            "features": {
                "RMS": 0.159659, "Energy": 2388.0, "ZeroCrossingRate": 0.174126,
                "MeanAbs": 0.099983, "StdDev": 0.159660, "PeakAmplitude": 1.0,
                "CrestFactor": 6.26334, "SpectralCentroid": 1031.22,
                "SpectralBandwidth": 1791.92, "SpectralRolloff": 1101.45
            }
        }
        res = client.post("/classify", json=payload)
        assert res.status_code == 200, f"Classify returned status {res.status_code}"
        res_data = res.json()
        assert "noise_type" in res_data and "confidence" in res_data
        print(f"  [PASS] Classification successful (Detected: {res_data['noise_type']}, Confidence: {res_data['confidence']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Classification Error: {e}")

    # Test 7: MATLAB WAV File Compatibility
    print("\n[TEST 7/10] Verifying MATLAB WAV Compatibility...")
    try:
        wav_path = r"C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\OUTPUT\AERIS_AI_READY\noisy.wav"
        if not os.path.exists(wav_path):
            wav_path = os.path.join(os.path.dirname(__file__), "OUTPUT", "AERIS_AI_READY", "noisy.wav")
        sr, audio_data = audio_dsp.load_audio_wav(wav_path)
        assert sr == 16000, f"Expected 16000 Hz, got {sr}"
        assert len(audio_data) > 0, "Empty audio buffer"
        assert not np.isnan(audio_data).any(), "Found NaN in audio data"
        print(f"  [PASS] MATLAB WAV loaded cleanly ({len(audio_data)} samples @ {sr} Hz).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] WAV Compatibility Error: {e}")

    # Test 8: Adaptive ANC Filtering Pipeline
    print("\n[TEST 8/10] Verifying LMS & NLMS Adaptive Filtering...")
    try:
        noisy_sig = audio_data[:3200]
        lms_res = audio_dsp.process_audio_pipeline(noisy_sig, algorithm="LMS", filter_order=16, step_size=0.005)
        nlms_res = audio_dsp.process_audio_pipeline(noisy_sig, algorithm="NLMS", filter_order=32, step_size=0.150)
        
        assert len(lms_res["e_enhanced"]) == len(noisy_sig)
        assert "snr_improvement" in lms_res
        assert not np.isnan(lms_res["e_enhanced"]).any()
        print(f"  [PASS] LMS Filter Executed: SNR Gain = {lms_res['snr_improvement']} dB (Latency: {lms_res['latency_ms']} ms).")
        print(f"  [PASS] NLMS Filter Executed: SNR Gain = {nlms_res['snr_improvement']} dB (Latency: {nlms_res['latency_ms']} ms).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] ANC Pipeline Error: {e}")

    # Test 9: Stability Protection Test (Extreme low SNR divergence guard)
    print("\n[TEST 9/10] Verifying Active Stability & Divergence Protection...")
    try:
        # Create exploding signal to trigger safety guard
        diverging_sig = np.random.randn(2000) * 100.0
        filter_inst = audio_dsp.AdaptiveFilterANC(algorithm="LMS", filter_order=16, step_size=0.5, max_weight_norm=5.0)
        res = filter_inst.filter_signals(diverging_sig, diverging_sig)
        assert filter_inst.safety_limit_active == True, "Safety limit failed to trigger on diverging signal"
        assert not np.isnan(res["e_enhanced"]).any(), "NaN leaked into audio output!"
        assert np.max(np.abs(res["e_enhanced"])) <= 1.01, "Audio output exceeded safe bounds!"
        print(f"  [PASS] Stability Guard Triggered: '{filter_inst.safety_warning}'. Clamped output safely.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Stability Protection Error: {e}")

    # Test 10: End-to-End POST /process_audio Endpoint
    print("\n[TEST 10/10] Testing POST /process_audio Full Pipeline...")
    try:
        resp = client.post("/process_audio", data={"preset": "noisy_10db", "algorithm": "LMS", "step_size": 0.005, "filter_order": 16})
        assert resp.status_code == 200, f"Process audio returned status {resp.status_code}"
        res_json = resp.json()
        assert res_json["status"] == "success"
        assert "noise_type" in res_json
        assert "snr_before" in res_json
        assert "improvement_db" in res_json
        assert "audio_base64" in res_json
        print(f"  [PASS] Full Pipeline JSON: Noise={res_json['noise_type']}, SNR In={res_json['snr_before']} dB, Gain=+{res_json['improvement_db']} dB, Audio Data={len(res_json['audio_base64'])} bytes.")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] /process_audio Error: {e}")

    print("\n" + "=" * 75)
    print(f"TOTAL TEST RESULTS: {passed_tests} / {total_tests} TESTS PASSED ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()

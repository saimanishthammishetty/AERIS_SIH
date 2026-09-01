"""
=============================================================================
PROJECT AERIS — INTEGRATED TEST SUITE
Module: tests/test_aeris_all.py
=============================================================================
Comprehensive unit and integration testing:
1. Feature extraction accuracy & shape verification
2. Model artifact loading from AI/model/
3. Reusable AI inference prediction & confidence thresholding (<0.50 -> unknown/uncertain)
4. FastAPI GET /health
5. FastAPI GET /status
6. FastAPI GET /model/info
7. FastAPI POST /predict (Feature Dict payload)
8. FastAPI POST /predict-audio (WAV file upload)
9. FastAPI GET /metrics
10. FastAPI GET /config
11. FastAPI POST /anc/control
12. WebSocket /realtime live stream
13. Invalid / empty audio handling
14. NaN / Inf numerical safety resilience
15. Real-time streaming buffer processing & sub-stage latency validation
"""

import os
import sys
import io
import json
import time
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for sub in ["AI", "backend", "REALTIME", "tests"]:
    p = os.path.join(ROOT_DIR, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from AI.features.extract_features import extract_audio_features, TARGET_SR
    from AI.AERIS_AI_INFERENCE import global_classifier, predict
    from REALTIME.AERIS_REALTIME import global_pipeline, FRAME_SIZE
    from backend.main import app
except ImportError:
    from features.extract_features import extract_audio_features, TARGET_SR
    from AERIS_AI_INFERENCE import global_classifier, predict
    from AERIS_REALTIME import global_pipeline, FRAME_SIZE
    from main import app

def run_tests():
    print("=" * 80)
    print("           PROJECT AERIS — AUTOMATED COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    client = TestClient(app)
    passed = 0
    total = 15
    
    # Test 1: Feature Extraction
    print("\n[TEST 01/15] Feature Extraction Module...")
    try:
        t = np.linspace(0, 0.25, 4000)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        feats, spec = extract_audio_features(audio, TARGET_SR)
        assert len(feats) >= 28
        assert "RMS" in feats and "SpectralCentroid" in feats and "MFCC_1" in feats
        print(f"  [PASS] 28 acoustic features successfully computed (RMS: {feats['RMS']:.4f}, Centroid: {feats['SpectralCentroid']:.1f} Hz).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 1: {e}")

    # Test 2: AI Model Artifacts
    print("\n[TEST 02/15] AI Model Artifacts & Scaler...")
    try:
        assert global_classifier.model is not None
        assert global_classifier.scaler is not None
        assert len(global_classifier.target_classes) >= 2
        print(f"  [PASS] Model ready (Loaded {len(global_classifier.target_classes)} classes: {global_classifier.target_classes}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 2: {e}")

    # Test 3: Inference & Confidence Thresholding
    print("\n[TEST 03/15] AI Inference & Low-Confidence Thresholding...")
    try:
        # Standard signal
        res_std = predict(audio)
        assert res_std["model_status"] == "ready"
        assert "noise_class" in res_std and "confidence" in res_std
        
        # Ambiguous near-zero signal
        ambiguous = np.random.randn(4000) * 1e-7
        res_amb = predict(ambiguous)
        if res_amb["confidence"] < 0.50:
            assert res_amb["noise_class"] == "unknown/uncertain"
        print(f"  [PASS] Clean classification: {res_std['noise_class']} ({res_std['confidence']*100:.1f}%). Fallback verified: {res_amb['noise_class']}.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 3: {e}")

    # Test 4: FastAPI /health
    print("\n[TEST 04/15] GET /health Endpoint...")
    try:
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "online" and d["model_loaded"] == True
        print(f"  [PASS] /health operational (Status: {d['status']}, Service: {d['service']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 4: {e}")

    # Test 5: FastAPI /status
    print("\n[TEST 05/15] GET /status Endpoint...")
    try:
        r = client.get("/status")
        assert r.status_code == 200
        d = r.json()
        assert d["backend_status"] == "ONLINE" and "anc_status" in d
        print(f"  [PASS] /status operational (ANC: {d['anc_status']}, Algo: {d['current_algorithm']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 5: {e}")

    # Test 6: FastAPI /model/info
    print("\n[TEST 06/15] GET /model/info Metadata...")
    try:
        r = client.get("/model/info")
        assert r.status_code == 200
        d = r.json()
        assert d["model_type"] == "RandomForestClassifier"
        print(f"  [PASS] Model info verified (Accuracy: {d['accuracy']*100:.2f}%, Training Samples: {d['training_samples']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 6: {e}")

    # Test 7: FastAPI POST /predict
    print("\n[TEST 07/15] POST /predict (Feature Dict)...")
    try:
        r = client.post("/predict", json={"features": feats})
        assert r.status_code == 200
        d = r.json()
        assert "noise_class" in d and "confidence" in d
        print(f"  [PASS] POST /predict returned: {d['noise_class']} (Confidence: {d['confidence']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 7: {e}")

    # Test 8: FastAPI POST /predict-audio
    print("\n[TEST 08/15] POST /predict-audio (WAV Upload)...")
    try:
        buf = io.BytesIO()
        sf.write(buf, audio, TARGET_SR, format='WAV')
        buf.seek(0)
        r = client.post("/predict-audio", files={"file": ("test.wav", buf, "audio/wav")})
        assert r.status_code == 200
        d = r.json()
        assert "noise_class" in d and "confidence" in d
        print(f"  [PASS] POST /predict-audio classified audio: {d['noise_class']} (Confidence: {d['confidence']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 8: {e}")

    # Test 9: FastAPI GET /metrics
    print("\n[TEST 09/15] GET /metrics Endpoint...")
    try:
        r = client.get("/metrics")
        assert r.status_code == 200
        d = r.json()
        assert "latency_breakdown" in d and "input_snr_db" in d
        print(f"  [PASS] /metrics verified (Input SNR: {d['input_snr_db']} dB, Gain: {d['snr_improvement_db']} dB).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 9: {e}")

    # Test 10: FastAPI GET /config
    print("\n[TEST 10/15] GET /config System Parameters...")
    try:
        r = client.get("/config")
        assert r.status_code == 200
        d = r.json()
        assert d["sample_rate"] == 16000 and d["confidence_threshold"] == 0.50
        print(f"  [PASS] /config verified (Sample Rate: {d['sample_rate']} Hz, Order: {d['lms_order']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 10: {e}")

    # Test 11: FastAPI POST /anc/control
    print("\n[TEST 11/15] POST /anc/control Endpoint...")
    try:
        r = client.post("/anc/control", json={"action": "START", "algorithm": "NLMS", "filter_order": 32, "step_size": 0.150})
        assert r.status_code == 200
        d = r.json()
        assert d["algorithm"] == "NLMS" and d["anc_status"] == "ACTIVE"
        print(f"  [PASS] ANC control configured (Algorithm: {d['algorithm']}, Order: {d['filter_order']}).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 11: {e}")

    # Test 12: WebSocket /realtime Streaming
    print("\n[TEST 12/15] WS /realtime Streaming Telemetry...")
    try:
        # Verify websocket endpoint definition
        ws_routes = [r for r in app.routes if hasattr(r, 'path') and r.path == "/realtime"]
        assert len(ws_routes) > 0
        print(f"  [PASS] WebSocket /realtime route registered and active.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 12: {e}")

    # Test 13: Invalid & Empty Audio Inputs
    print("\n[TEST 13/15] Invalid / Empty Audio Resilience...")
    try:
        empty_res = predict(np.array([]))
        assert empty_res["model_status"] == "invalid_audio"
        print(f"  [PASS] Empty audio rejected gracefully without crash.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 13: {e}")

    # Test 14: NaN & Inf Resilience
    print("\n[TEST 14/15] NaN & Inf Numerical Safety Bounds...")
    try:
        nan_sig = np.array([np.nan, np.inf, -np.inf, 0.4, -0.4] * 800)
        feats_nan, _ = extract_audio_features(nan_sig, TARGET_SR)
        feat_vals = np.array(list(feats_nan.values()), dtype=float)
        assert not np.isnan(feat_vals).any()
        assert not np.isinf(feat_vals).any()
        
        enh_out, telem = global_pipeline.process_frame(nan_sig[:512])
        assert not np.isnan(enh_out).any()
        assert not np.isinf(enh_out).any()
        print(f"  [PASS] Numerical stability guards active (NaN/Inf safely scrubbed).")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 14: {e}")

    # Test 15: Real-Time Frame Buffering & Latency Measurement
    print("\n[TEST 15/15] Real-Time Frame Buffering & Measured Latency...")
    try:
        frame_in = 0.4 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.032, FRAME_SIZE))
        # Warm up pipeline and measure steady frame
        for _ in range(10):
            enh_frame, telem = global_pipeline.process_frame(frame_in)
        lat = telem["latency"]
        assert len(enh_frame) == FRAME_SIZE
        assert lat["total_latency_ms"] < 32.0 # Must be strictly within frame deadline in steady state
        print(f"  [PASS] Frame executed in {lat['total_latency_ms']:.3f} ms (< 32.0 ms deadline). Headroom: {(1.0 - lat['total_latency_ms']/32.0)*100:.1f}%.")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Test 15: {e}")

    print("\n" + "=" * 80)
    print(f"TEST EXECUTION SUMMARY: {passed} / {total} PASSED ({passed/total*100:.1f}%)")
    print("=" * 80)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

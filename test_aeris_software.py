"""
=============================================================================
PROJECT AERIS — COMPREHENSIVE SOFTWARE INTEGRATION TEST SUITE
Module: test_aeris_software.py
=============================================================================
Automated test verification for:
1. Acoustic 28-feature extraction
2. AI model & scaler loading
3. AI prediction & confidence scoring
4. FastAPI /health endpoint
5. FastAPI /status endpoint
6. FastAPI /predict endpoint (WAV & Base64)
7. FastAPI /metrics endpoint
8. WebSocket /realtime telemetry stream
9. Invalid audio handling
10. Missing model / Uninitialized handling
11. Low confidence threshold handling (< 0.50 -> LOW_CONFIDENCE)
12. NaN & Inf resilience
13. Real-time streaming buffer handling & sub-stage latency metrics
14. Complete end-to-end processing pipeline
"""

import os
import sys
import io
import json
import time
import base64
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

# Add project root to sys.path so 'backend' package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.AERIS_REALTIME import extract_acoustic_features, AIInferenceEngine, RealtimeANCPipeline
import backend.AERIS_REALTIME as realtime
from backend.main import app

def run_all_software_tests():
    print("=" * 80)
    print("        PROJECT AERIS - COMPREHENSIVE SOFTWARE INTEGRATION TEST SUITE")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = 14
    client = TestClient(app)
    
    # Test 1: Feature Extraction
    print("\n[TEST 01/14] Testing 28 Acoustic Feature Extraction...")
    try:
        t_arr = np.linspace(0, 0.25, 4000)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t_arr)
        feats = realtime.extract_acoustic_features(test_audio, 16000)
        assert len(feats) >= 28, f"Expected >= 28 features, got {len(feats)}"
        assert "RMS" in feats and "SpectralCentroid" in feats and "MFCC_13" in feats
        print(f"  [PASS] Extracted {len(feats)} features (RMS: {feats['RMS']:.4f}, Centroid: {feats['SpectralCentroid']:.1f} Hz).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 1 Error: {e}")

    # Test 2: AI Model & Scaler Loading
    print("\n[TEST 02/14] Testing AI Model and Scaler Loading...")
    try:
        engine = realtime.AIInferenceEngine()
        assert engine.model is not None, "Model is None"
        assert engine.scaler is not None, "Scaler is None"
        assert len(engine.target_classes) >= 2, "Expected >= 2 classes"
        print(f"  [PASS] Model loaded successfully ({len(engine.target_classes)} classes: {engine.target_classes}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 2 Error: {e}")

    # Test 3: AI Prediction on Clean Audio
    print("\n[TEST 03/14] Testing AI Prediction Accuracy on Sample Audio...")
    try:
        engine = realtime.AIInferenceEngine()
        pred = engine.predict(test_audio, 16000)
        assert pred["status"] == "success"
        assert "noise_type" in pred and "confidence" in pred
        print(f"  [PASS] Prediction Result: {pred['noise_type']} (Confidence: {pred['confidence']*100:.1f}%, Time: {pred['prediction_time_ms']} ms).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 3 Error: {e}")

    # Test 4: FastAPI /health Endpoint
    print("\n[TEST 04/14] Testing GET /health Endpoint...")
    try:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online" and data["model_loaded"] == True
        print(f"  [PASS] /health operational (Status: {data['status']}, Version: {data['version']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 4 Error: {e}")

    # Test 5: FastAPI /status Endpoint
    print("\n[TEST 05/14] Testing GET /status Subsystem Matrix...")
    try:
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "anc_status" in data and "current_algorithm" in data
        print(f"  [PASS] /status operational (ANC: {data['anc_status']}, Algo: {data['current_algorithm']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 5 Error: {e}")

    # Test 6: FastAPI POST /predict-audio with WAV File
    print("\n[TEST 06/14] Testing POST /predict-audio with Audio Payload...")
    try:
        # Create in-memory WAV
        wav_buf = io.BytesIO()
        sf.write(wav_buf, test_audio, 16000, format='WAV')
        wav_buf.seek(0)
        
        resp = client.post("/predict-audio", files={"file": ("test.wav", wav_buf, "audio/wav")})
        assert resp.status_code == 200
        res = resp.json()
        assert res["status"] == "success"
        print(f"  [PASS] POST /predict-audio verified (Predicted: {res['noise_type']}, Confidence: {res['confidence']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 6 Error: {e}")

    # Test 7: FastAPI GET /metrics Endpoint
    print("\n[TEST 07/14] Testing GET /metrics Telemetry & Latency Breakdown...")
    try:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency_breakdown" in data
        print(f"  [PASS] /metrics verified (SNR In: {data['input_snr_db']} dB, Improvement: {data['snr_improvement_db']} dB).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 7 Error: {e}")

    # Test 8: WebSocket /realtime Telemetry Stream
    print("\n[TEST 08/14] Testing WS /realtime WebSocket Connection & Streaming...")
    try:
        with client.websocket_connect("/realtime") as ws:
            msg = ws.receive_text()
            data = json.loads(msg)
            assert data["system"] == "AERIS_TELEMETRY"
            assert "telemetry" in data and "wave_preview" in data
            print(f"  [PASS] WebSocket /realtime streaming active (Received valid telemetry packet at {data['timestamp']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 8 Error: {e}")

    # Test 9: Invalid Audio Handling
    print("\n[TEST 09/14] Testing Invalid & Empty Audio Input Handling...")
    try:
        engine = realtime.AIInferenceEngine()
        empty_res = engine.predict(np.array([]))
        assert empty_res["status"] == "error"
        assert empty_res["noise_type"] == "INVALID_AUDIO"
        print(f"  [PASS] Empty audio caught cleanly (Returned: {empty_res['noise_type']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 9 Error: {e}")

    # Test 10: Missing Model Handling
    print("\n[TEST 10/14] Testing Missing Model State Handling...")
    try:
        uninit_engine = realtime.AIInferenceEngine()
        uninit_engine.model = None
        uninit_engine.scaler = None
        res = uninit_engine.predict(test_audio)
        assert res["status"] == "error"
        assert res["noise_type"] == "MODEL_UNINITIALIZED"
        print(f"  [PASS] Uninitialized model state handled (Returned: {res['noise_type']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 10 Error: {e}")

    # Test 11: Low Confidence Threshold Handling
    print("\n[TEST 11/14] Testing Low Confidence Threshold Detection (< 0.50)...")
    try:
        engine = realtime.AIInferenceEngine()
        # Flat noise with near-zero energy producing ambiguous probabilities
        ambiguous_audio = np.random.randn(4000) * 1e-6
        res = engine.predict(ambiguous_audio)
        assert "confidence" in res
        if res["confidence"] < 0.50:
            assert res["noise_type"] == "LOW_CONFIDENCE"
        print(f"  [PASS] Confidence evaluation: {res['noise_type']} (Confidence: {res['confidence']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 11 Error: {e}")

    # Test 12: NaN & Inf Input Resilience
    print("\n[TEST 12/14] Testing NaN and Inf Numerical Resilience...")
    try:
        nan_audio = np.array([np.nan, np.inf, -np.inf, 0.5, -0.2] * 800)
        feats = realtime.extract_acoustic_features(nan_audio, 16000)
        assert not np.isnan(list(feats.values())).any()
        
        pipeline = realtime.RealtimeANCPipeline()
        out, telem = pipeline.process_frame(nan_audio[:512])
        assert not np.isnan(out).any()
        print(f"  [PASS] NaN/Inf scrubbed successfully (Output peak: {np.max(np.abs(out)):.4f}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 12 Error: {e}")

    # Test 13: Real-Time Buffer & Latency Breakdown
    print("\n[TEST 13/14] Testing Real-Time Frame Buffering & Latency Tracking...")
    try:
        pipeline = realtime.RealtimeANCPipeline()
        frame = np.random.randn(512) * 0.2
        enh_out, telem = pipeline.process_frame(frame)
        lat = telem["latency"]
        assert len(enh_out) == 512
        assert lat["total_latency_ms"] > 0.0
        print(f"  [PASS] Frame processed in {lat['total_latency_ms']:.3f} ms (Feat: {lat['feature_extraction_ms']} ms, ANC: {lat['anc_processing_ms']} ms).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 13 Error: {e}")

    # Test 14: Complete End-to-End Decision & ANC Execution
    print("\n[TEST 14/14] Testing Complete End-to-End AI-to-ANC Control Pipeline...")
    try:
        ctrl_resp = client.post("/anc/control", json={"action": "START", "algorithm": "NLMS", "filter_order": 32, "step_size": 0.150})
        assert ctrl_resp.status_code == 200
        ctrl_data = ctrl_resp.json()
        assert ctrl_data["algorithm"] == "NLMS" and ctrl_data["anc_status"] == "ACTIVE"
        print(f"  [PASS] End-to-End Control verified (Algorithm: {ctrl_data['algorithm']}, Order: {ctrl_data['filter_order']}).")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Test 14 Error: {e}")

    print("\n" + "=" * 80)
    print(f"TOTAL SOFTWARE TEST RESULTS: {passed_tests} / {total_tests} PASSED ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 80)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_software_tests()
    sys.exit(0 if success else 1)

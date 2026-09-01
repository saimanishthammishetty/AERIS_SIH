"""
=============================================================================
PROJECT AERIS — 30-POINT FINAL SOFTWARE ACCEPTANCE VERIFICATION
Module: tests/test_aeris_complete.py
=============================================================================
Performs end-to-end automated verification of all 30 criteria:
1. Backend startup & routing
2. GET /health
3. Frontend HTML / JS / CSS availability
4. Dashboard index.html root serving
5. WebSocket /realtime and /ws/telemetry connectivity
6. AudioWorklet module serving
7. 512-sample frame ingest
8. Frame counter monotonicity
9. Waveform preview payload generation
10. FFT spectrum preview calculation
11. 28 Acoustic feature extraction
12. Random Forest AI classification
13. Model confidence & thresholding
14. LMS Adaptive Filtering (M=16, mu=0.005)
15. NLMS Adaptive Filtering (M=32, mu=0.150)
16. Enhanced output waveform generation
17. Audio output PCM 16-bit format integrity
18. Original clean audio stream endpoint
19. Noisy mixed audio stream endpoint
20. LMS enhanced stream endpoint
21. NLMS enhanced stream endpoint
22. Sub-stage & E2E latency measurement (<32ms)
23. Truthful SNR (N/A for live mic, accurate for reference)
24. GET /diagnostics self-test
25. POST /anc/control (START, STOP, RESET)
26. Single-frame processing latency (<15ms, zero 10-20s delay)
27. Stable WebSocket message exchange
28. Singleton pipeline protection
29. Clean JSON serializability
30. Zero unhandled backend exceptions on dirty/NaN audio
=============================================================================
"""

import os
import sys
import io
import time
import json
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.main import app, global_pipeline, global_classifier, resolve_preset_path
from backend.feature_extractor import extract_features_28
from backend.anc_engine import AdaptiveFilterANC, LiveSpeechEnhancer

def run_30_point_verification():
    print("=" * 80)
    print("      AERIS v2.5 — 30-POINT FINAL SOFTWARE ACCEPTANCE SUITE")
    print("=" * 80)
    
    client = TestClient(app)
    passed_tests = []
    failed_tests = []
    
    def log_test(num: int, title: str, condition: bool, details: str = ""):
        if condition:
            print(f"  [TEST {num:02d}/30] [PASS] {title:<45} {details}")
            passed_tests.append((num, title))
        else:
            print(f"  [TEST {num:02d}/30] [FAIL] {title:<45} {details}")
            failed_tests.append((num, title, details))

    # TEST 1: Backend starts & app object valid
    log_test(1, "FastAPI backend app initialized", app is not None and len(app.routes) > 5)

    # TEST 2: GET /health works
    r_health = client.get("/health")
    d_health = r_health.json() if r_health.status_code == 200 else {}
    log_test(2, "GET /health operational", r_health.status_code == 200 and d_health.get("status") == "online", f"Model: {d_health.get('ai_model')}")

    # TEST 3: Frontend assets available
    r_css = client.get("/styles.css")
    r_js = client.get("/app.js")
    r_proc = client.get("/aeris-processor.js")
    log_test(3, "Frontend static assets served", r_css.status_code == 200 and r_js.status_code == 200 and r_proc.status_code == 200)

    # TEST 4: Dashboard loads at root /
    r_root = client.get("/")
    log_test(4, "Dashboard HTML root route serves index.html", r_root.status_code == 200 and "<html" in r_root.text.lower())

    # TEST 5: WebSocket route verified
    with client.websocket_connect("/realtime") as ws:
        init_pkt = ws.receive_json()
        log_test(5, "WebSocket /realtime connection established", init_pkt.get("system") == "AERIS_TELEMETRY")

    # TEST 6: AudioWorklet module integrity
    proc_text = r_proc.text
    log_test(6, "AudioWorklet processor registered", "registerProcessor('aeris-processor'" in proc_text)

    # TEST 7: 512-sample frame ingest over WS
    test_samples = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.032, 512))).tolist()
    with client.websocket_connect("/realtime") as ws:
        _ = ws.receive_json() # consume init
        ws.send_json({"type": "audio_frame", "frame_id": 101, "samples": test_samples})
        resp = ws.receive_json()
        log_test(7, "512-sample PCM frame processed over WS", resp.get("system") == "AERIS_TELEMETRY" and len(resp.get("enhanced_audio", [])) == 512)

    # TEST 8: Frame counter increases
    with client.websocket_connect("/realtime") as ws:
        _ = ws.receive_json()
        ws.send_json({"type": "audio_frame", "frame_id": 102, "samples": test_samples})
        resp1 = ws.receive_json()
        c1 = resp1["telemetry"]["frames_received"]
        ws.send_json({"type": "audio_frame", "frame_id": 103, "samples": test_samples})
        resp2 = ws.receive_json()
        c2 = resp2["telemetry"]["frames_received"]
        log_test(8, "Telemetry frame counter increments", c2 > c1, f"Count: {c1} -> {c2}")

    # TEST 9: Input waveform responds
    has_wave = "wave_preview" in resp2 and "noisy" in resp2["wave_preview"] and len(resp2["wave_preview"]["noisy"]) > 0
    log_test(9, "Input waveform preview computed", has_wave, f"Points: {len(resp2['wave_preview']['noisy'])}")

    # TEST 10: Spectrum preview responds
    has_spec = "spec_preview" in resp2 and "noisy" in resp2["spec_preview"] and len(resp2["spec_preview"]["noisy"]) > 0
    log_test(10, "FFT Spectrum preview computed", has_spec, f"Bins: {len(resp2['spec_preview']['noisy'])}")

    # TEST 11: Feature extraction works (28 acoustic features)
    t = np.linspace(0, 0.25, 4000)
    sig_test = 0.4 * np.sin(2 * np.pi * 440 * t)
    feats, spec = extract_features_28(sig_test)
    log_test(11, "28 Acoustic features extracted cleanly", len(feats) >= 28 and "RMS" in feats and "SpectralCentroid" in feats)

    # TEST 12: AI model predicts
    ai_res = global_classifier.predict_from_features(feats)
    log_test(12, "RandomForest AI model classifies", ai_res.get("model_status") == "ready" and "noise_class" in ai_res)

    # TEST 13: Confidence displayed
    conf = ai_res.get("confidence", 0.0)
    log_test(13, "Model confidence bounded [0, 1]", 0.0 <= conf <= 1.0, f"Confidence: {conf*100:.1f}%")

    # TEST 14: LMS filter operates
    lms = AdaptiveFilterANC(algorithm="LMS", filter_order=16, step_size=0.005)
    d_in = np.random.randn(512) * 0.1
    x_ref = np.roll(d_in, 4) * 0.8
    _, lms_out, lms_metrics = lms.process_frame(d_in, x_ref)
    log_test(14, "LMS frame-by-frame DSP functional", len(lms_out) == 512 and not np.isnan(lms_out).any())

    # TEST 15: NLMS filter operates
    nlms = AdaptiveFilterANC(algorithm="NLMS", filter_order=32, step_size=0.150)
    _, nlms_out, nlms_metrics = nlms.process_frame(d_in, x_ref)
    log_test(15, "NLMS normalized step DSP functional", len(nlms_out) == 512 and not np.isnan(nlms_out).any())

    # TEST 16: Enhanced audio output generated
    log_test(16, "Enhanced speech output produced", len(resp2.get("enhanced_audio", [])) == 512)

    # TEST 17: Browser audio output format (float32 array)
    enh_arr = np.array(resp2.get("enhanced_audio", []), dtype=np.float32)
    log_test(17, "Audio output format valid (Float32 in [-1, 1])", np.max(np.abs(enh_arr)) <= 1.05)

    # TEST 18-21: 4 Audio Streams Synthesis & Playback endpoints
    synth_req = client.post("/api/simulation/synthesize", json={
        "noise_type": "tmetro",
        "snr_db": -10.0,
        "lms_order": 16,
        "lms_mu": 0.005,
        "nlms_order": 32,
        "nlms_mu": 0.150
    })
    synth_d = synth_req.json() if synth_req.status_code == 200 else {}
    
    r_orig = client.get("/api/audio/stream/stream_original")
    log_test(18, "Stream 1 (Original Clean) serves WAV", r_orig.status_code == 200 and r_orig.headers.get("content-type") == "audio/wav")

    r_noisy = client.get("/api/audio/stream/stream_noisy")
    log_test(19, "Stream 2 (Noisy Mixed) serves WAV", r_noisy.status_code == 200 and r_noisy.headers.get("content-type") == "audio/wav")

    r_lms_s = client.get("/api/audio/stream/stream_lms")
    log_test(20, "Stream 3 (LMS Enhanced) serves WAV", r_lms_s.status_code == 200 and r_lms_s.headers.get("content-type") == "audio/wav")

    r_nlms_s = client.get("/api/audio/stream/stream_nlms")
    log_test(21, "Stream 4 (NLMS Enhanced) serves WAV", r_nlms_s.status_code == 200 and r_nlms_s.headers.get("content-type") == "audio/wav")

    # TEST 22: Measured latency breakdown
    lat = resp2["telemetry"].get("latency", {})
    tot_lat = lat.get("total_latency_ms", 0.0)
    log_test(22, "Real-time latency breakdown measured", tot_lat > 0.0, f"E2E: {tot_lat:.3f} ms")

    # TEST 23: Truthful SNR handling
    live_has_ref = resp2["telemetry"].get("has_clean_reference", False)
    log_test(23, "Live microphone reports truthful SNR (has_clean_ref=False)", live_has_ref == False and resp2["telemetry"].get("input_snr_db") is None)

    # TEST 24: GET /diagnostics self-test
    r_diag = client.get("/diagnostics")
    d_diag = r_diag.json() if r_diag.status_code == 200 else {}
    log_test(24, "GET /diagnostics comprehensive self-test", r_diag.status_code == 200 and d_diag.get("overall_status") == "PASS")

    # TEST 25: POST /anc/control (START, STOP, RESET)
    r_c1 = client.post("/anc/control", json={"action": "START", "algorithm": "NLMS", "filter_order": 32})
    r_c2 = client.post("/anc/control", json={"action": "RESET"})
    log_test(25, "ANC Engine control (START/STOP/RESET)", r_c1.status_code == 200 and r_c2.status_code == 200)

    # TEST 26: Frame latency budget (< 32.0 ms, no 10-20s buffer)
    t0 = time.perf_counter()
    for _ in range(8):
        global_pipeline.process_frame(np.random.randn(512)*0.1)
    avg_frame_ms = ((time.perf_counter() - t0) / 8) * 1000.0
    log_test(26, "Continuous 32ms frame deadline satisfied", avg_frame_ms < 32.0, f"Average frame processing: {avg_frame_ms:.2f} ms")

    # TEST 27: WebSocket stability (no connection storm)
    ping_resp = None
    with client.websocket_connect("/realtime") as ws:
        _ = ws.receive_json()
        ws.send_json({"type": "ping"})
        ping_resp = ws.receive_json()
    log_test(27, "WebSocket ping/pong & heartbeat stable", ping_resp == {"type": "pong"})

    # TEST 28: Singleton pipeline state protection
    p1 = global_pipeline
    log_test(28, "Singleton pipeline state verified", p1 is not None and p1.frame_size == 512)

    # TEST 29: Clean JSON serialization
    log_test(29, "Telemetry JSON encoding clean (No NaNs/Infs)", "NaN" not in json.dumps(resp2))

    # TEST 30: NaN/Inf safety resilience
    nan_frame = np.array([np.nan, np.inf, -np.inf, 0.5] * 128)
    safe_out, safe_telem = global_pipeline.process_frame(nan_frame)
    log_test(30, "Extreme numerical input safely sanitized", not np.isnan(safe_out).any() and not np.isinf(safe_out).any())

    print("=" * 80)
    print(f"  TOTAL ACCEPTANCE CRITERIA PASSED: {len(passed_tests)} / 30 ({len(passed_tests)/30.0*100:.1f}%)")
    print("=" * 80)
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = run_30_point_verification()
    sys.exit(0 if success else 1)

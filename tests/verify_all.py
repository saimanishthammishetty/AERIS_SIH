"""Quick verification of all core AERIS modules."""
import sys, os, numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'AI')
sys.path.insert(0, 'backend')
sys.path.insert(0, 'REALTIME')

results = {}

# Test 1: Feature extraction
try:
    try:
        from AI.features.extract_features import extract_audio_features, TARGET_SR
    except ImportError:
        from features.extract_features import extract_audio_features, TARGET_SR
    t = np.linspace(0, 0.25, 4000)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    feats, spec = extract_audio_features(audio, TARGET_SR)
    assert len(feats) >= 28
    results["feature_extraction"] = f"PASS ({len(feats)} features)"
except Exception as e:
    results["feature_extraction"] = f"FAIL: {e}"

# Test 2: AI model loading
try:
    try:
        from AI.AERIS_AI_INFERENCE import global_classifier
    except ImportError:
        from AERIS_AI_INFERENCE import global_classifier
    assert global_classifier.model is not None
    results["model_loading"] = f"PASS (classes: {global_classifier.target_classes})"
except Exception as e:
    results["model_loading"] = f"FAIL: {e}"

# Test 3: AI prediction
try:
    res = global_classifier.predict_from_audio(audio)
    assert "noise_class" in res
    results["prediction"] = f"PASS (class={res['noise_class']}, conf={res['confidence']:.3f})"
except Exception as e:
    results["prediction"] = f"FAIL: {e}"

# Test 4: Low-confidence fallback
try:
    tiny = np.random.randn(4000) * 1e-7
    res2 = global_classifier.predict_from_audio(tiny)
    if res2["confidence"] < 0.50:
        assert res2["noise_class"] == "unknown/uncertain"
    results["low_confidence"] = f"PASS (noise_class={res2['noise_class']}, conf={res2['confidence']:.3f})"
except Exception as e:
    results["low_confidence"] = f"FAIL: {e}"

# Test 5: Real-time pipeline
try:
    from AERIS_REALTIME import global_pipeline, FRAME_SIZE
    frame = np.random.randn(FRAME_SIZE) * 0.2
    out, telem = global_pipeline.process_frame(frame)
    assert len(out) == FRAME_SIZE
    lat = telem["latency"]["total_latency_ms"]
    results["realtime_pipeline"] = f"PASS (latency={lat:.3f}ms)"
except Exception as e:
    results["realtime_pipeline"] = f"FAIL: {e}"

# Test 6: NaN/Inf safety
try:
    nan_sig = np.array([np.nan, np.inf, -np.inf, 0.4] * 128)
    out2, _ = global_pipeline.process_frame(nan_sig[:FRAME_SIZE])
    assert not np.isnan(out2).any() and not np.isinf(out2).any()
    results["nan_safety"] = "PASS"
except Exception as e:
    results["nan_safety"] = f"FAIL: {e}"

# Test 7: BACKEND import (without blocking)
try:
    from main import app
    results["backend_import"] = "PASS (FastAPI app importable)"
except Exception as e:
    results["backend_import"] = f"FAIL: {e}"

# Test 8: REST endpoints (no WebSocket)
try:
    from fastapi.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=True)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "online"
    results["endpoint_health"] = "PASS"

    r2 = client.get("/status")
    assert r2.status_code == 200
    results["endpoint_status"] = "PASS"

    r3 = client.get("/model/info")
    assert r3.status_code == 200
    d3 = r3.json()
    results["endpoint_model_info"] = f"PASS (accuracy={d3['accuracy']*100:.2f}%)"

    r4 = client.post("/predict", json={"features": feats})
    assert r4.status_code == 200
    results["endpoint_predict"] = f"PASS (class={r4.json()['noise_class']})"

    import io, soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SR, format='WAV')
    buf.seek(0)
    r5 = client.post("/predict-audio", files={"file": ("test.wav", buf, "audio/wav")})
    assert r5.status_code == 200
    results["endpoint_predict_audio"] = f"PASS (class={r5.json()['noise_class']})"

    r6 = client.get("/metrics")
    assert r6.status_code == 200
    results["endpoint_metrics"] = "PASS"

    r7 = client.get("/config")
    assert r7.status_code == 200
    results["endpoint_config"] = "PASS"

    r8 = client.post("/anc/control", json={"action": "START", "algorithm": "LMS"})
    assert r8.status_code == 200
    results["endpoint_anc_control"] = "PASS"

except Exception as e:
    results["backend_api"] = f"FAIL: {e}"

print("\n" + "="*70)
print("  AERIS END-TO-END COMPONENT VERIFICATION REPORT")
print("="*70)
for k, v in results.items():
    status = "PASS" if v.startswith("PASS") else "FAIL"
    icon = "[+]" if status == "PASS" else "[X]"
    print(f"  {icon}  {k:<28}: {v}")
print("="*70)

passed = sum(1 for v in results.values() if v.startswith("PASS"))
total = len(results)
print(f"\n  RESULT: {passed}/{total} components verified.")

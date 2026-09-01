import time
import numpy as np
from backend.feature_extractor import extract_features_28
from backend.ai_classifier import global_classifier, AERISClassifier
from backend.anc_engine import AdaptiveFilterANC
from backend.config import SAMPLE_RATE, FRAME_SIZE

def extract_acoustic_features(audio_data: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compatibility wrapper for 28 acoustic features extraction."""
    feats, _ = extract_features_28(audio_data, sr)
    return feats

class AIInferenceEngine:
    """Compatibility adapter for AIInferenceEngine used in software test suites."""
    
    def __init__(self):
        self.classifier = global_classifier
        self.model = self.classifier.model
        self.scaler = self.classifier.scaler
        self.target_classes = self.classifier.target_classes
        
    def predict(self, audio_data: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        t0 = time.perf_counter()
        
        # Test 10: Missing model handling
        if self.model is None or self.scaler is None:
            return {
                "status": "error",
                "noise_type": "MODEL_UNINITIALIZED",
                "message": "AI model or scaler is not loaded."
            }
            
        # Test 9: Empty/invalid audio input handling
        if audio_data is None or len(audio_data) == 0:
            return {
                "status": "error",
                "noise_type": "INVALID_AUDIO",
                "message": "Audio input is empty."
            }
            
        try:
            feats, _ = extract_features_28(audio_data, sr)
            # Reorder features
            vec = []
            for fname in self.classifier.feature_names:
                val = feats.get(fname, 0.0)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                vec.append(val)
                
            vec_arr = np.array(vec, dtype=np.float64).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec_arr)
            
            raw_class = str(self.model.predict(vec_scaled)[0])
            probs = self.model.predict_proba(vec_scaled)[0]
            max_conf = float(np.max(probs))
            
            # Test 11: Low confidence threshold handling (< 0.50 -> LOW_CONFIDENCE)
            if max_conf < 0.50:
                final_class = "LOW_CONFIDENCE"
            else:
                final_class = raw_class
                
            t_proc_ms = (time.perf_counter() - t0) * 1000
            
            return {
                "status": "success",
                "noise_type": final_class,
                "confidence": round(max_conf, 4),
                "prediction_time_ms": round(t_proc_ms, 2)
            }
        except Exception as e:
            return {
                "status": "error",
                "noise_type": "ERROR",
                "message": str(e)
            }

class RealtimeANCPipeline:
    """Compatibility adapter for RealtimeANCPipeline used in software test suites."""
    
    def __init__(self):
        self.filter = AdaptiveFilterANC(algorithm="LMS", filter_order=16, step_size=0.005)
        self.telemetry = {
            "latency": {
                "capture_ms": 0.0,
                "feature_extraction_ms": 0.0,
                "ai_inference_ms": 0.0,
                "anc_processing_ms": 0.0,
                "output_ms": 0.0,
                "total_latency_ms": 0.0
            }
        }
        
    def process_frame(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        
        # Test 12: NaN/Inf scrubbing
        frame_clean = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=-1.0)
        frame_clean = np.clip(frame_clean, -1.0, 1.0)
        
        ref_noise = np.roll(frame_clean, 8) * 0.85
        
        t_anc0 = time.perf_counter()
        _, enhanced, metrics = self.filter.process_frame(frame_clean, ref_noise)
        t_anc_ms = (time.perf_counter() - t_anc0) * 1000.0
        
        total_lat_ms = (time.perf_counter() - t0) * 1000.0
        
        self.telemetry["latency"] = {
            "capture_ms": 0.0,
            "feature_extraction_ms": 0.0,
            "ai_inference_ms": 0.0,
            "anc_processing_ms": round(t_anc_ms, 2),
            "output_ms": 0.0,
            "total_latency_ms": round(total_lat_ms, 2)
        }
        
        return enhanced, self.telemetry

import os
import json
import pickle
import time
import numpy as np
from typing import Dict, Any, Optional
from backend.config import MODEL_SEARCH_DIRS, CONFIDENCE_THRESHOLD, SAMPLE_RATE
from backend.feature_extractor import extract_features_28

class AERISClassifier:
    """AERIS Acoustic Noise Classifier Inference Engine using the real trained RF model."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.target_classes = []
        self.status = "uninitialized"
        self.load_model()

    def load_model(self) -> bool:
        for sdir in MODEL_SEARCH_DIRS:
            m_path = os.path.join(sdir, "AERIS_noise_classifier.pkl")
            s_path = os.path.join(sdir, "AERIS_scaler.pkl")
            f_path = os.path.join(sdir, "AERIS_features.json")
            if not os.path.exists(f_path):
                f_path = os.path.join(sdir, "AERIS_feature_list.json")
            c_path = os.path.join(sdir, "AERIS_class_names.json")

            if os.path.exists(m_path) and os.path.exists(s_path):
                try:
                    with open(m_path, "rb") as f:
                        self.model = pickle.load(f)
                    with open(s_path, "rb") as f:
                        self.scaler = pickle.load(f)
                    
                    if os.path.exists(f_path):
                        with open(f_path, "r") as f:
                            data = json.load(f)
                            self.feature_names = data.get("feature_names", []) if isinstance(data, dict) else data
                    else:
                        # Fallback default feature names
                        self.feature_names = [
                            "RMS", "Variance", "StdDev", "Kurtosis", "ZeroCrossingRate", "MeanAbs",
                            "PeakAmplitude", "CrestFactor", "SpectralCentroid", "SpectralBandwidth",
                            "SpectralRolloff", "SpectralFlatness", "SpectralEnergy", "DominantFrequency",
                            "SpectralFlux"
                        ] + [f"MFCC_{i+1}" for i in range(13)]
                        
                    if os.path.exists(c_path):
                        with open(c_path, "r") as f:
                            self.target_classes = json.load(f)
                    elif hasattr(self.model, "classes_"):
                        self.target_classes = self.model.classes_.tolist()
                    else:
                        self.target_classes = ["NOISE_TMETRO", "CLEAN_SPEECH", "ANC_ENHANCED", "NOISY_SPEECH"]

                    self.status = "ready"
                    print(f"[+] AERISClassifier loaded model successfully from: {sdir}")
                    return True
                except Exception as e:
                    print(f"[!] Error loading model from {sdir}: {e}")

        self.status = "model_missing"
        print("[!] Warning: AERIS model artifacts could not be located.")
        return False

    def predict_from_audio(self, audio_data: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
        """Classifies an audio array."""
        t0 = time.perf_counter()

        if self.status != "ready" or self.model is None:
            return {
                "noise_class": "unknown/uncertain",
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model_status": "model_missing",
                "message": "AI model is not loaded."
            }

        if len(audio_data) == 0 or np.isnan(audio_data).all():
            return {
                "noise_class": "unknown/uncertain",
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model_status": "invalid_audio",
                "message": "Audio input is empty or contains all NaNs."
            }

        try:
            feats, _ = extract_features_28(audio_data, sr)
            return self.predict_from_features(feats, t0)
        except Exception as e:
            return {
                "noise_class": "unknown/uncertain",
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model_status": "error",
                "message": str(e)
            }

    def predict_from_features(self, features_dict: Dict[str, float], t_start: Optional[float] = None) -> Dict[str, Any]:
        """Classifies a dictionary of pre-extracted acoustic features."""
        t0 = t_start if t_start is not None else time.perf_counter()

        if self.status != "ready" or self.model is None:
            return {
                "noise_class": "unknown/uncertain",
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model_status": "model_missing"
            }

        try:
            # Build feature vector matching trained feature ordering
            vec = []
            for fname in self.feature_names:
                val = features_dict.get(fname, 0.0)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                vec.append(val)

            vec_arr = np.array(vec, dtype=np.float64).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec_arr)

            probs = self.model.predict_proba(vec_scaled)[0]
            max_idx = int(np.argmax(probs))
            raw_class = str(self.model.classes_[max_idx]) if hasattr(self.model, "classes_") else str(self.target_classes[max_idx])
            max_conf = float(probs[max_idx])

            prob_dict = {
                cls_name: round(float(p), 4)
                for cls_name, p in zip(self.target_classes, probs)
            }

            # Confidence Threshold Handling
            if max_conf < CONFIDENCE_THRESHOLD:
                final_class = "unknown/uncertain"
            else:
                final_class = raw_class

            t_proc_ms = (time.perf_counter() - t0) * 1000

            return {
                "noise_class": final_class,
                "raw_predicted_class": raw_class,
                "confidence": round(max_conf, 4),
                "is_confident": max_conf >= CONFIDENCE_THRESHOLD,
                "class_probabilities": prob_dict,
                "processing_time_ms": round(t_proc_ms, 2),
                "model_status": "ready"
            }
        except Exception as e:
            return {
                "noise_class": "unknown/uncertain",
                "confidence": 0.0,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model_status": "error",
                "message": str(e)
            }

global_classifier = AERISClassifier()

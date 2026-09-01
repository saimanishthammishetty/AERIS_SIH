"""
=============================================================================
PROJECT AERIS — AI ACOUSTIC INFERENCE ENGINE
Module: AI/AERIS_AI_INFERENCE.py
=============================================================================
Provides clean, reusable inference functions for predicting noise categories
from raw audio signals or extracted feature vectors with confidence thresholding.
=============================================================================
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional

# Ensure root directory and sub-packages are in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
AI_DIR = os.path.dirname(os.path.abspath(__file__))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

try:
    from AI.features.extract_features import extract_audio_features, TARGET_SR
except ImportError:
    from features.extract_features import extract_audio_features, TARGET_SR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CONFIDENCE_THRESHOLD = 0.50


class AERISClassifier:
    """AERIS Acoustic Noise Classifier Inference Engine."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.target_classes = []
        self.status = "uninitialized"
        self.load_model()

    def load_model(self) -> bool:
        search_dirs = [
            self.model_dir,
            os.path.join(os.path.dirname(BASE_DIR), "models"),
            os.path.join(os.path.dirname(BASE_DIR), "AI_MODEL"),
            os.path.join(os.path.dirname(BASE_DIR), "BACKEND"),
            BASE_DIR
        ]

        for sdir in search_dirs:
            m_path = os.path.join(sdir, "AERIS_noise_classifier.pkl")
            s_path = os.path.join(sdir, "AERIS_scaler.pkl")
            f_path = os.path.join(sdir, "AERIS_feature_list.json")
            if not os.path.exists(f_path):
                f_path = os.path.join(sdir, "AERIS_features.json")

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
                    if os.path.exists(c_path):
                        with open(c_path, "r") as f:
                            self.target_classes = json.load(f)
                    elif hasattr(self.model, "classes_"):
                        self.target_classes = self.model.classes_.tolist()

                    self.status = "ready"
                    print(f"[+] AERISClassifier loaded successfully from: {sdir}")
                    return True
                except Exception as e:
                    print(f"[!] Error loading model from {sdir}: {e}")

        self.status = "model_missing"
        print("[!] Warning: AERIS model artifacts could not be located.")
        return False

    def predict_from_audio(self, audio_data: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
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
            feats, _ = extract_audio_features(audio_data, sr)
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
            # Build feature vector according to trained feature ordering
            vec = []
            for fname in self.feature_names:
                val = features_dict.get(fname, 0.0)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                vec.append(val)

            vec_arr = np.array(vec, dtype=np.float64).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec_arr)

            raw_class = str(self.model.predict(vec_scaled)[0])
            probs = self.model.predict_proba(vec_scaled)[0]
            max_conf = float(np.max(probs))

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


def predict(input_data: Any) -> Dict[str, Any]:
    """Unified entry point for inference."""
    if isinstance(input_data, dict):
        return global_classifier.predict_from_features(input_data)
    elif isinstance(input_data, (np.ndarray, list)):
        arr = np.asarray(input_data, dtype=np.float32)
        return global_classifier.predict_from_audio(arr)
    else:
        return {
            "noise_class": "unknown/uncertain",
            "confidence": 0.0,
            "model_status": "invalid_input",
            "message": f"Unsupported input type: {type(input_data)}"
        }

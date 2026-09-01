import os

# Base Directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

# DSP Constants
SAMPLE_RATE = 16000
FRAME_SIZE = 512
TARGET_LATENCY_MS = 32.0

# AI Classification Parameters
CONFIDENCE_THRESHOLD = 0.50
FEATURE_COUNT = 28

# Model Paths search priority
MODEL_SEARCH_DIRS = [
    os.path.join(ROOT_DIR, "AI_MODEL"),
    os.path.join(ROOT_DIR, "OUTPUT", "AERIS_AI_READY", "AI_MODEL"),
    os.path.join(ROOT_DIR, "AI", "model"),
    ROOT_DIR
]

# Audio presets search paths
AUDIO_PRESET_SEARCH_DIRS = [
    os.path.join(ROOT_DIR, "OUTPUT", "AERIS_AI_READY"),
    os.path.join(ROOT_DIR, "OUTPUT"),
    ROOT_DIR
]

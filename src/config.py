"""Configuration settings for the inference pipeline."""

import os
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.parent / "RoadBuddy" / "traffic_buddy_train+public_test"
PUBLIC_TEST_DIR = DATA_DIR / "public_test"
PUBLIC_TEST_JSON = PUBLIC_TEST_DIR / "public_test.json"
VIDEOS_DIR = PUBLIC_TEST_DIR / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Generate timestamp for unique output file
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUBMISSION_FILE = OUTPUT_DIR / f"submission_{TIMESTAMP}.csv"

# Model configuration
# Supported models:
# - "YannQi/R-4B"
# - "OpenGVLab/InternVL3-8B"
# - "Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME = "YannQi/R-4B"
DEVICE = "cuda"  # or "cpu"
TRUST_REMOTE_CODE = True

# Video processing configuration
FRAME_SAMPLE_RATE = 1.0  # Extract 1 frame per second
MAX_FRAMES_PER_VIDEO = 20  # Maximum number of frames to extract
USE_MID_FRAME_ONLY = False  # If True, only extract the middle frame

# Inference configuration
# Note: THINKING_MODE is not supported by most vision-language models
# It's only available for specific models like some Claude models
# THINKING_MODE = "auto"  # "auto", "explicit", or "non-thinking"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for more deterministic outputs
DO_SAMPLE = False  # Use greedy decoding for consistent answers

# Batch processing
BATCH_SIZE = 1  # Process one question at a time (model handles one image-question pair)
CACHE_FRAMES = True  # Cache extracted frames when multiple questions use same video

# Prompt template
PROMPT_TEMPLATE = """Dựa vào video này, hãy trả lời câu hỏi sau:

Câu hỏi: {question}

Các lựa chọn:
{choices}

Hãy chọn đáp án đúng (chỉ trả lời A, B, C, hoặc D)."""

# Answer parsing
VALID_ANSWERS = ["A", "B", "C", "D"]

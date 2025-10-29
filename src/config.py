"""Configuration settings for the inference pipeline."""

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
MAX_NEW_TOKENS = 50  # Allow some buffer for response format
TEMPERATURE = 0.1  # Low temperature for more deterministic outputs
DO_SAMPLE = False  # Use greedy decoding for consistent answers

# Batch processing
BATCH_SIZE = 1  # Process one question at a time (model handles one image-question pair)
CACHE_FRAMES = True  # Cache extracted frames when multiple questions use same video

# Prompt template
PROMPT_TEMPLATE = """Watch the video and answer this question:

{question}

{choices}

IMPORTANT: Respond with ONLY one letter (A, B, C, or D). No explanations, no extra text, no Chinese characters. Just the letter.

Your answer:"""

# Answer parsing
VALID_ANSWERS = ["A", "B", "C", "D"]

# ============================================================================
# Traffic Inference Configuration (InternVL3 Multi-Frame)
# ============================================================================
# Configuration for the enhanced traffic video inference pipeline using
# InternVL3-8B with multi-frame processing and traffic-optimized prompts

# Enable/disable traffic inference features
TRAFFIC_INFERENCE_ENABLED = True

# InternVL3-specific configuration
TRAFFIC_MODEL_NAME = "OpenGVLab/InternVL3-8B"
TRAFFIC_MAX_NUM = 24  # InternVL max patches per frame (higher = more detail, more memory)
TRAFFIC_PRECISION = "float32"  # Use float32 for highest accuracy

# Frame extraction configuration
TRAFFIC_MIN_FRAMES = 6  # Minimum frames for short videos (5-7 seconds)
TRAFFIC_MAX_FRAMES = 12  # Maximum frames for long videos (13-15 seconds)
TRAFFIC_USE_SUPPORT_FRAMES = True  # Use support_frames timestamps when available
TRAFFIC_CONTEXT_WINDOW = 0.5  # Seconds of context around support frames

# Prompt configuration
TRAFFIC_SIMPLE_PROMPTS = False  # If True, use shorter/simpler prompt templates
TRAFFIC_AUTO_DETECT_LANGUAGE = True  # Auto-detect Vietnamese vs English

# Performance configuration
TRAFFIC_CPU_WARNING = True  # Warn user about CPU memory/speed limitations
TRAFFIC_CLEAR_CUDA_CACHE = True  # Clear CUDA cache after each inference

# Output configuration
TRAFFIC_OUTPUT_FILE = OUTPUT_DIR / f"traffic_inference_{TIMESTAMP}.csv"
TRAFFIC_VERBOSE = False  # Print detailed information during inference

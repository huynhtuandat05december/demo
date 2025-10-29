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
# InternVL3 Multi-Frame Inference Configuration
# ============================================================================
# Configuration for the enhanced video inference pipeline using
# InternVL3-8B with multi-frame processing and optimized prompts

# Enable/disable inference features
INFERENCE_ENABLED = True

# InternVL3-specific configuration
INTERNVL_MODEL_NAME = "OpenGVLab/InternVL3-8B"
INTERNVL_MAX_NUM = 6  # InternVL max patches per frame (REDUCED: 6=fits token limit, 12=may exceed)
INTERNVL_LOAD_IN_8BIT = False  # Use 8-bit quantization (~50% memory reduction) - DISABLED due to dtype issues
INTERNVL_LOAD_IN_4BIT = True  # Use 4-bit quantization (~75% memory reduction) - ENABLED for stability

# Frame extraction configuration
MIN_FRAMES = 4  # Minimum frames for short videos (REDUCED to fit token limit)
MAX_FRAMES = 6  # Maximum frames for long videos (REDUCED: 6 frames * 6 patches = ~36 patches)
USE_SUPPORT_FRAMES = True  # Use support_frames timestamps when available
CONTEXT_WINDOW = 0.5  # Seconds of context around support frames

# Prompt configuration
SIMPLE_PROMPTS = False  # If True, use shorter/simpler prompt templates
AUTO_DETECT_LANGUAGE = True  # Auto-detect Vietnamese vs English

# Performance configuration
CPU_WARNING = True  # Warn user about CPU memory/speed limitations
CLEAR_CUDA_CACHE = True  # Clear CUDA cache after each inference

# Output configuration
INFERENCE_OUTPUT_FILE = OUTPUT_DIR / f"inference_{TIMESTAMP}.csv"
VERBOSE = False  # Print detailed information during inference

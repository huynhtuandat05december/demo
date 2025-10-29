"""
Training configuration for InternVL3-8B on RoadBuddy traffic dataset.
"""
from pathlib import Path

# ============================================================================
# Data Configuration
# ============================================================================

# Data paths
# Use relative path from this file's location
# train_config.py -> internvl3_8B -> src -> road_buddy -> zaloAI -> RoadBuddy
DATA_ROOT = Path(__file__).parent.parent.parent.parent / "RoadBuddy" / "traffic_buddy_train+public_test"
TRAIN_JSON = DATA_ROOT / "train" / "train.json"
TRAIN_VIDEO_DIR = DATA_ROOT / "train" / "videos"

# Data split
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME = "OpenGVLab/InternVL3-8B"
CACHE_DIR = None  # Set to a path if you want to cache the model

# Video processing
MIN_FRAMES = 4
MAX_FRAMES = 6
USE_SUPPORT_FRAMES = True
CONTEXT_WINDOW = 0.5  # seconds around support frames

# Frame extraction settings
MAX_NUM_PATCHES = 6  # Number of patches per frame
DYNAMIC_IMAGE_SIZE = True  # Use dynamic tiling for better quality

# ============================================================================
# LoRA Configuration
# ============================================================================

# PEFT settings
USE_LORA = True
LORA_CONFIG = {
    "r": 32,  # LoRA rank
    "lora_alpha": 64,  # LoRA alpha (typically 2x rank)
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # Target modules for InternVL3 (attention layers)
    "target_modules": [
        "qkv_proj",  # Query, Key, Value projection
        "out_proj",  # Output projection
    ],
}

# ============================================================================
# Training Configuration
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 4  # Per device batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4  # Standard learning rate for LoRA
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1  # 10% warmup steps
WEIGHT_DECAY = 0.01

# Mixed precision
FP16 = True  # Use FP16 mixed precision training
BF16 = False  # Use BF16 if your GPU supports it (A100, H100)

# Optimizer
OPTIM = "adamw_torch"  # or "adamw_bnb_8bit" for memory efficiency

# Learning rate scheduler
LR_SCHEDULER_TYPE = "cosine"  # Cosine annealing with warmup

# ============================================================================
# Checkpointing and Logging
# ============================================================================

# Output directory
# Use relative path from this file's location
# train_config.py -> internvl3_8B -> src -> road_buddy -> output
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "internvl3_8B_lora"

# Checkpoint settings
SAVE_STRATEGY = "epoch"  # Save checkpoint every epoch
SAVE_TOTAL_LIMIT = 3  # Keep only the 3 best checkpoints
LOAD_BEST_MODEL_AT_END = True  # Load best model when training finishes

# Evaluation settings
EVALUATION_STRATEGY = "epoch"  # Evaluate every epoch
METRIC_FOR_BEST_MODEL = "accuracy"  # Use accuracy to determine best model
GREATER_IS_BETTER = True  # Higher accuracy is better

# Logging settings
LOGGING_STRATEGY = "steps"
LOGGING_STEPS = 10  # Log every 10 steps
REPORT_TO = ["tensorboard"]  # or ["wandb"] if you use W&B

# ============================================================================
# System Configuration
# ============================================================================

# Device settings
DEVICE = "cuda"  # or "cpu" for CPU-only training
DATALOADER_NUM_WORKERS = 4  # Number of workers for data loading
DATALOADER_PIN_MEMORY = True

# Reproducibility
SET_SEED = True
SEED = RANDOM_SEED

# Gradient settings
MAX_GRAD_NORM = 1.0  # Gradient clipping

# Early stopping (optional)
EARLY_STOPPING_PATIENCE = None  # Set to int (e.g., 3) to enable early stopping

# ============================================================================
# Advanced Settings
# ============================================================================

# Gradient checkpointing (reduces memory at cost of speed)
GRADIENT_CHECKPOINTING = False

# DeepSpeed config (if you want to use DeepSpeed)
DEEPSPEED_CONFIG = None  # Path to DeepSpeed config JSON

# Flash Attention 2 (if installed)
USE_FLASH_ATTENTION_2 = False  # Set to True if flash-attn is installed

# ============================================================================
# Prompt Configuration
# ============================================================================

# Language settings
USE_VIETNAMESE_PROMPTS = True
PROMPT_VARIANT = "multi_frame"  # or "single_frame"

# Answer format
ANSWER_CHOICES = ["A", "B", "C", "D"]

# ============================================================================
# Validation and Testing
# ============================================================================

# Maximum samples for quick validation (None = use all)
MAX_VAL_SAMPLES = None

# Generation config for inference
GENERATION_CONFIG = {
    "max_new_tokens": 10,  # Only need 1-2 tokens for A/B/C/D
    "do_sample": False,  # Deterministic generation
    "temperature": 0.0,
}

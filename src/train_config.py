"""Training configuration settings."""

from src import config

# Data paths
TRAIN_JSON = config.DATA_DIR / "train" / "train.json"
TRAIN_VIDEOS_DIR = config.DATA_DIR / "train" / "videos"

# Model configuration
MODEL_NAME = config.MODEL_NAME  # Default from config
DEVICE = config.DEVICE
TRUST_REMOTE_CODE = config.TRUST_REMOTE_CODE

# Training hyperparameters
BATCH_SIZE = 2  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-5  # Lower learning rate for fine-tuning
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1  # 10% of training steps for warmup
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0  # Gradient clipping

# Optimizer settings
OPTIMIZER = "adamw"  # "adamw" or "sgd"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# Learning rate scheduler
LR_SCHEDULER_TYPE = "cosine"  # "linear", "cosine", "constant"

# Data settings
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation
MAX_SEQ_LENGTH = 512  # Maximum sequence length for text
USE_SUPPORT_FRAMES = True  # Use support_frames timestamps if available
SHUFFLE_TRAIN = True
NUM_WORKERS = 4  # DataLoader workers

# Checkpoint and logging
OUTPUT_DIR = config.PROJECT_ROOT / "checkpoints"
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs
SAVE_BEST_MODEL = True  # Save the best model based on validation accuracy
LOGGING_STEPS = 10  # Log metrics every N steps
EVAL_STEPS = 100  # Evaluate on validation set every N steps

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N evaluations
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement to count

# Mixed precision training
USE_FP16 = True  # Use mixed precision training (requires CUDA)
USE_BF16 = False  # Use bfloat16 (better for newer GPUs)

# Model freezing (for faster training)
FREEZE_VISION_ENCODER = False  # Freeze vision encoder weights
FREEZE_LLM_LAYERS = 0  # Number of LLM layers to freeze from the bottom

# LoRA configuration (efficient fine-tuning)
USE_LORA = True  # Use LoRA for parameter-efficient fine-tuning
LORA_R = 8  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Modules to apply LoRA to

# Debugging
DEBUG_MODE = False  # Use small subset of data for debugging
DEBUG_SAMPLES = 100  # Number of samples to use in debug mode
DETERMINISTIC = False  # Use deterministic training (slower but reproducible)

# Seed for reproducibility
RANDOM_SEED = 42

# Resume training
RESUME_FROM_CHECKPOINT = None  # Path to checkpoint to resume from (None to start fresh)

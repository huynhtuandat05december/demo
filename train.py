#!/usr/bin/env python3
"""
Training script for traffic video question answering.

Usage:
    # Use default configuration
    python train.py

    # Specify a model
    python train.py --model YannQi/R-4B

    # Adjust hyperparameters
    python train.py --learning-rate 1e-5 --batch-size 4 --epochs 5

    # Use CPU instead of GPU
    python train.py --device cpu

    # Enable/disable LoRA
    python train.py --use-lora
    python train.py --no-lora

    # Debug mode (small dataset)
    python train.py --debug

    # Resume from checkpoint
    python train.py --resume checkpoints/checkpoint_epoch_2.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import train_config
from src.train import Trainer


def generate_checkpoint_dir(model_name: str, base_dir: Path) -> Path:
    """
    Generate checkpoint directory name based on model and timestamp.

    Args:
        model_name: Name of the model
        base_dir: Base directory for checkpoints

    Returns:
        Path to checkpoint directory
    """
    # Extract model name
    if "/" in model_name:
        model_short = model_name.split("/")[-1]
    else:
        model_short = model_name

    # Remove special characters
    model_short = model_short.replace("-", "_").replace(".", "_")

    # Get current date (without time for training runs)
    date_str = datetime.now().strftime("%Y%m%d")

    # Create directory name
    dir_name = f"{model_short}_{date_str}"

    return base_dir / dir_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Traffic Video Question Answering Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=train_config.MODEL_NAME,
        help="HuggingFace model name to fine-tune",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=train_config.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to train on",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_config.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=train_config.GRADIENT_ACCUMULATION_STEPS,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=train_config.LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=train_config.WARMUP_RATIO,
        help="Warmup ratio for learning rate scheduler",
    )

    # LoRA configuration
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=train_config.USE_LORA,
        help="Use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--no-lora",
        action="store_false",
        dest="use_lora",
        help="Disable LoRA",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=train_config.LORA_R,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=train_config.LORA_ALPHA,
        help="LoRA alpha",
    )

    # Data configuration
    parser.add_argument(
        "--train-json",
        type=Path,
        default=train_config.TRAIN_JSON,
        help="Path to training JSON file",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=train_config.TRAIN_VAL_SPLIT,
        help="Train/validation split ratio",
    )
    parser.add_argument(
        "--use-support-frames",
        action="store_true",
        default=train_config.USE_SUPPORT_FRAMES,
        help="Use support_frames timestamps for frame extraction",
    )

    # Checkpoint and output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints (default: auto-generate based on model and date)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use default checkpoint directory without timestamp",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=train_config.RESUME_FROM_CHECKPOINT,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=train_config.SAVE_EVERY_N_EPOCHS,
        help="Save checkpoint every N epochs",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=train_config.USE_EARLY_STOPPING,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=train_config.EARLY_STOPPING_PATIENCE,
        help="Early stopping patience",
    )

    # Mixed precision
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=train_config.USE_FP16,
        help="Use mixed precision training (FP16)",
    )

    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with small dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=train_config.RANDOM_SEED,
        help="Random seed for reproducibility",
    )

    # Logging
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=train_config.LOGGING_STEPS,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=train_config.EVAL_STEPS,
        help="Evaluate every N steps",
    )

    return parser.parse_args()


def update_config_from_args(args):
    """Update training configuration from command line arguments."""
    # Model
    train_config.MODEL_NAME = args.model
    train_config.DEVICE = args.device

    # Training hyperparameters
    train_config.BATCH_SIZE = args.batch_size
    train_config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    train_config.LEARNING_RATE = args.learning_rate
    train_config.NUM_EPOCHS = args.epochs
    train_config.WARMUP_RATIO = args.warmup_ratio

    # LoRA
    train_config.USE_LORA = args.use_lora
    train_config.LORA_R = args.lora_r
    train_config.LORA_ALPHA = args.lora_alpha

    # Data
    train_config.TRAIN_JSON = args.train_json
    train_config.TRAIN_VAL_SPLIT = args.train_val_split
    train_config.USE_SUPPORT_FRAMES = args.use_support_frames

    # Output
    if args.output_dir is None:
        if args.no_timestamp:
            train_config.OUTPUT_DIR = Path(__file__).parent / "checkpoints"
        else:
            # Auto-generate checkpoint directory with model name and date
            base_checkpoints_dir = Path(__file__).parent / "checkpoints"
            train_config.OUTPUT_DIR = generate_checkpoint_dir(args.model, base_checkpoints_dir)
    else:
        train_config.OUTPUT_DIR = args.output_dir

    train_config.RESUME_FROM_CHECKPOINT = args.resume
    train_config.SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs

    # Early stopping
    train_config.USE_EARLY_STOPPING = args.early_stopping
    train_config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    # Mixed precision
    train_config.USE_FP16 = args.fp16

    # Debug
    if args.debug:
        train_config.DEBUG_MODE = True
        train_config.DEBUG_SAMPLES = 100
        train_config.NUM_EPOCHS = 2
        print("\n⚠️  DEBUG MODE ENABLED - Using small dataset and 2 epochs")

    # Seed
    train_config.RANDOM_SEED = args.seed

    # Logging
    train_config.LOGGING_STEPS = args.logging_steps
    train_config.EVAL_STEPS = args.eval_steps


def main():
    """Main entry point."""
    args = parse_args()

    # Update config
    update_config_from_args(args)

    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {train_config.MODEL_NAME}")
    print(f"Device: {train_config.DEVICE}")
    print(f"Batch size: {train_config.BATCH_SIZE}")
    print(f"Gradient accumulation steps: {train_config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {train_config.BATCH_SIZE * train_config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {train_config.LEARNING_RATE}")
    print(f"Epochs: {train_config.NUM_EPOCHS}")
    print(f"LoRA: {'Enabled' if train_config.USE_LORA else 'Disabled'}")
    if train_config.USE_LORA:
        print(f"  - Rank: {train_config.LORA_R}")
        print(f"  - Alpha: {train_config.LORA_ALPHA}")
    print(f"Mixed precision (FP16): {'Enabled' if train_config.USE_FP16 else 'Disabled'}")
    print(f"Early stopping: {'Enabled' if train_config.USE_EARLY_STOPPING else 'Disabled'}")
    print(f"Output directory: {train_config.OUTPUT_DIR}")
    print(f"Random seed: {train_config.RANDOM_SEED}")
    print("=" * 60)

    # Check if training data exists
    if not train_config.TRAIN_JSON.exists():
        print(f"\n❌ Error: Training data not found at {train_config.TRAIN_JSON}")
        sys.exit(1)

    # Initialize trainer
    try:
        trainer = Trainer(
            model_name=train_config.MODEL_NAME,
            device=train_config.DEVICE,
            config=train_config,
        )
    except Exception as e:
        print(f"\n❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Start training
    try:
        trainer.train()
        print(f"\n✅ Training completed successfully!")
        print(f"Best model saved to: {train_config.OUTPUT_DIR / 'best_model_hf'}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test training script for a small subset of data.

This script tests the training pipeline on a small subset to verify everything works.

Usage:
    python test_train.py
    python test_train.py --model YannQi/R-4B
    python test_train.py --device cpu
    python test_train.py --samples 5  # Train on 5 samples
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import train_config
from src.train import Trainer


def test_training():
    """Test training on a small subset."""
    parser = argparse.ArgumentParser(description="Test training on a small subset")
    parser.add_argument("--model", type=str, default=train_config.MODEL_NAME, help="Model to test")
    parser.add_argument("--device", type=str, default=train_config.DEVICE, choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--no-lora", action="store_false", dest="use_lora", help="Disable LoRA")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    print("=" * 60 + "\n")

    # Check if training data exists
    if not train_config.TRAIN_JSON.exists():
        print(f"❌ Error: Training data not found at {train_config.TRAIN_JSON}")
        print(f"\nPlease ensure the training data exists:")
        print(f"  - {train_config.TRAIN_JSON}")
        print(f"  - {train_config.TRAIN_VIDEOS_DIR}/")
        sys.exit(1)

    # Update config for testing
    train_config.MODEL_NAME = args.model
    train_config.DEVICE = args.device
    train_config.DEBUG_MODE = True
    train_config.DEBUG_SAMPLES = args.samples
    train_config.NUM_EPOCHS = args.epochs
    train_config.BATCH_SIZE = args.batch_size
    train_config.USE_LORA = args.use_lora
    train_config.SAVE_EVERY_N_EPOCHS = 1
    train_config.SAVE_BEST_MODEL = True
    train_config.USE_EARLY_STOPPING = False  # Disable for testing
    train_config.LOGGING_STEPS = 1  # Log every step for testing
    train_config.EVAL_STEPS = 0  # Disable mid-epoch eval for speed
    train_config.OUTPUT_DIR = Path(__file__).parent / "test_checkpoints"
    train_config.NUM_WORKERS = 0  # Disable multiprocessing for simpler debugging

    print("⚠️  TEST MODE: Training on small subset")
    print(f"  - Only {args.samples} samples will be used")
    print(f"  - Only {args.epochs} epoch(s)")
    print(f"  - Checkpoints will be saved to: {train_config.OUTPUT_DIR}")
    print()

    # Initialize trainer
    print("🔄 Initializing trainer...")
    try:
        trainer = Trainer(
            model_name=args.model,
            device=args.device,
            config=train_config,
        )
    except Exception as e:
        print(f"\n❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run training
    print("\n🔄 Starting training test...")
    try:
        trainer.train()
        print(f"\n✅ Training test completed!")
        print(f"\n📊 RESULTS:")
        print(f"  Best validation accuracy: {trainer.best_val_accuracy:.4f} ({trainer.best_val_accuracy * 100:.2f}%)")
        print(f"  Checkpoints saved to: {train_config.OUTPUT_DIR}")

        # Check if best model was saved
        best_model_path = train_config.OUTPUT_DIR / "best_model_hf"
        if best_model_path.exists():
            print(f"\n✅ Best model saved successfully!")
            print(f"  Location: {best_model_path}")
            print(f"\n  You can use this model for inference:")
            print(f"    python run_multi_model.py --model {best_model_path}")
        else:
            print(f"\n⚠️  Best model not found at {best_model_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe training pipeline is working correctly.")
    print("You can now run full training with:")
    print(f"  python train.py --model {args.model}")
    if args.use_lora:
        print(f"  python train.py --use-lora --batch-size 2 --epochs 3")


if __name__ == "__main__":
    test_training()

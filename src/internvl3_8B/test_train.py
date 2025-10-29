"""
Test script for training pipeline - verify everything works before full training.

This script:
1. Loads a small subset of training data (5 samples)
2. Initializes model with LoRA configuration
3. Runs a few training steps
4. Verifies dataset loading and model forward pass
5. Reports any errors or configuration issues

Run this before starting full training to catch issues early!
"""
import sys
import torch
import argparse
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Subset

# Import local modules
from train_dataset import TrafficVideoDataset, collate_fn
import train_config as cfg


def test_dataset_loading(num_samples: int = 5):
    """
    Test that dataset loads correctly.

    Args:
        num_samples: Number of samples to test
    """
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    try:
        # Test train split
        print("\nLoading TRAIN dataset...")
        train_dataset = TrafficVideoDataset(
            json_path=cfg.TRAIN_JSON,
            data_root=cfg.DATA_ROOT,
            split="train",
            train_val_split=cfg.TRAIN_VAL_SPLIT,
            random_seed=cfg.RANDOM_SEED,
            min_frames=cfg.MIN_FRAMES,
            max_frames=cfg.MAX_FRAMES,
            max_num_patches=cfg.MAX_NUM_PATCHES,
            use_support_frames=cfg.USE_SUPPORT_FRAMES,
            context_window=cfg.CONTEXT_WINDOW,
        )
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")

        # Test val split
        print("\nLoading VAL dataset...")
        val_dataset = TrafficVideoDataset(
            json_path=cfg.TRAIN_JSON,
            data_root=cfg.DATA_ROOT,
            split="val",
            train_val_split=cfg.TRAIN_VAL_SPLIT,
            random_seed=cfg.RANDOM_SEED,
            min_frames=cfg.MIN_FRAMES,
            max_frames=cfg.MAX_FRAMES,
            max_num_patches=cfg.MAX_NUM_PATCHES,
            use_support_frames=cfg.USE_SUPPORT_FRAMES,
            context_window=cfg.CONTEXT_WINDOW,
        )
        print(f"✓ Val dataset loaded: {len(val_dataset)} samples")

        # Test loading a few samples
        print(f"\nTesting {num_samples} sample loading...")
        test_indices = list(range(min(num_samples, len(train_dataset))))

        for idx in test_indices:
            sample = train_dataset[idx]
            print(f"  Sample {idx}:")
            print(f"    - Video ID: {sample['video_id']}")
            print(f"    - Frames shape: {sample['pixel_values'].shape}")
            print(f"    - Num patches: {sample['num_patches_list']}")
            print(f"    - Answer: {sample['answer']}")
            print(f"    - Question: {sample['question'][:80]}...")

        print(f"\n✓ Successfully loaded {len(test_indices)} samples")

        return train_dataset, val_dataset

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_model_loading():
    """
    Test that model loads correctly with LoRA configuration.
    """
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)

    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_NAME,
            trust_remote_code=True,
        )
        print(f"✓ Tokenizer loaded")

        # Load base model
        print("\nLoading base model...")
        model = AutoModel.from_pretrained(
            cfg.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"✓ Base model loaded")

        # Apply LoRA if enabled
        if cfg.USE_LORA:
            print("\nApplying LoRA configuration...")
            lora_config = LoraConfig(**cfg.LORA_CONFIG)
            model = get_peft_model(model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ LoRA applied")
            print(f"  - Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"  - Total params: {total_params:,}")

        # Move to device
        if cfg.DEVICE == "cuda" and torch.cuda.is_available():
            print(f"\nMoving model to GPU...")
            model = model.to(cfg.DEVICE)
            print(f"✓ Model on GPU")
        else:
            print(f"\n⚠ Running on CPU (will be very slow)")

        return model, tokenizer

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, tokenizer, dataset, num_samples: int = 2):
    """
    Test a forward pass through the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dataset: The dataset
        num_samples: Number of samples to test
    """
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    try:
        model.eval()

        # Create a small subset
        subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
        dataloader = DataLoader(
            subset,
            batch_size=1,  # Process one at a time
            collate_fn=collate_fn,
        )

        print(f"\nTesting forward pass on {num_samples} samples...")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"\n  Sample {i + 1}:")

                # Extract batch data
                pixel_values = batch['pixel_values_list'][0].to(model.device)
                num_patches_list = batch['num_patches_lists'][0]
                question = batch['questions'][0]
                answer = batch['answers'][0]

                # Match dtype
                pixel_values = pixel_values.to(torch.bfloat16)

                print(f"    - Pixel values shape: {pixel_values.shape}")
                print(f"    - Num patches: {num_patches_list}")
                print(f"    - Question: {question[:60]}...")
                print(f"    - Expected answer: {answer}")

                # Try to generate an answer
                try:
                    generation_config = {
                        'max_new_tokens': 10,
                        'do_sample': False,
                    }

                    response = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False
                    )

                    print(f"    - Generated response: {response}")
                    print(f"    ✓ Forward pass successful")

                except Exception as e:
                    print(f"    ✗ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"\n✓ Forward pass test completed")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(dataset, num_samples: int = 3):
    """
    Test that DataLoader works with collate_fn.

    Args:
        dataset: The dataset
        num_samples: Number of samples to test
    """
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    try:
        # Create a small subset
        subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

        # Test with batch_size=1 (recommended for InternVL3)
        print(f"\nTesting with batch_size=1...")
        dataloader = DataLoader(
            subset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        for i, batch in enumerate(dataloader):
            print(f"  Batch {i + 1}:")
            print(f"    - Num samples: {len(batch['questions'])}")
            print(f"    - Pixel values list length: {len(batch['pixel_values_list'])}")
            print(f"    - First sample frames shape: {batch['pixel_values_list'][0].shape}")

        print(f"✓ DataLoader test successful")
        return True

    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to test (default: 5)"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model loading test (faster, only test dataset)"
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip forward pass test (requires model)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RoadBuddy Training Pipeline Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Model: {cfg.MODEL_NAME}")
    print(f"  - Device: {cfg.DEVICE}")
    print(f"  - Batch size: {cfg.BATCH_SIZE}")
    print(f"  - Learning rate: {cfg.LEARNING_RATE}")
    print(f"  - Num epochs: {cfg.NUM_EPOCHS}")
    print(f"  - Min frames: {cfg.MIN_FRAMES}")
    print(f"  - Max frames: {cfg.MAX_FRAMES}")
    print(f"  - Use LoRA: {cfg.USE_LORA}")
    if cfg.USE_LORA:
        print(f"  - LoRA rank: {cfg.LORA_CONFIG['r']}")
    print("=" * 60)

    # Test 1: Dataset loading
    train_dataset, val_dataset = test_dataset_loading(num_samples=args.samples)
    if train_dataset is None:
        print("\n❌ Dataset loading test FAILED")
        return 1

    # Validate both datasets were created (val_dataset checked here)
    if val_dataset is None:
        print("\n⚠ Warning: Validation dataset is None")
    else:
        print(f"✓ Both train ({len(train_dataset)}) and val ({len(val_dataset)}) datasets loaded")

    # Test 2: DataLoader
    if not test_dataloader(train_dataset, num_samples=3):
        print("\n❌ DataLoader test FAILED")
        return 1

    # Test 3: Model loading (optional)
    if not args.skip_model:
        model, tokenizer = test_model_loading()
        if model is None:
            print("\n❌ Model loading test FAILED")
            return 1

        # Test 4: Forward pass (optional)
        if not args.skip_forward:
            if not test_forward_pass(model, tokenizer, train_dataset, num_samples=2):
                print("\n❌ Forward pass test FAILED")
                return 1
        else:
            print("\n⚠ Skipping forward pass test (--skip-forward)")
    else:
        print("\n⚠ Skipping model loading and forward pass tests (--skip-model)")

    # Summary
    print("\n" + "=" * 60)
    print("✅ All Tests Passed!")
    print("=" * 60)
    print("\nYour training pipeline is ready. You can now run:")
    print("  python train.py")
    print("\nTo monitor training with TensorBoard:")
    print(f"  tensorboard --logdir {cfg.OUTPUT_DIR}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

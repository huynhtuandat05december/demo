#!/usr/bin/env python3
"""
Test inference script for a single example.

This script tests the inference pipeline on one example to verify everything works.

Usage:
    python test_inference.py
    python test_inference.py --model YannQi/R-4B
    python test_inference.py --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.inference import R4BInferencePipeline


def generate_test_output_filename(model_name: str) -> Path:
    """Generate test output filename based on model name and timestamp."""
    # Extract model name
    if "/" in model_name:
        model_short = model_name.split("/")[-1]
    else:
        model_short = model_name

    # Remove special characters
    model_short = model_short.replace("-", "_").replace(".", "_")

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"test_result_{model_short}_{timestamp}.txt"

    return Path(__file__).parent / "output" / filename


def test_single_example():
    """Test inference on a single example."""
    parser = argparse.ArgumentParser(description="Test inference on a single example")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Model to test")
    parser.add_argument("--device", type=str, default=config.DEVICE, choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--test-json", type=Path, default=config.PUBLIC_TEST_JSON, help="Path to test JSON")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TESTING INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Test file: {args.test_json}")
    print("=" * 60 + "\n")

    # Check if test file exists
    if not args.test_json.exists():
        print(f"❌ Error: Test file not found at {args.test_json}")
        print(f"\nPlease ensure the test data exists:")
        print(f"  - {args.test_json}")
        sys.exit(1)

    # Load test data
    print("Loading test data...")
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if not test_data.get("data"):
        print("❌ Error: No data found in test file")
        sys.exit(1)

    # Get first example
    example = test_data["data"][0]
    print(f"\n📝 Testing with example:")
    print(f"  ID: {example['id']}")
    print(f"  Question: {example['question'][:80]}...")
    print(f"  Choices: {len(example['choices'])} options")
    print(f"  Video: {example['video_path']}")

    # Check if video exists
    video_path = config.DATA_DIR / example["video_path"]
    if not video_path.exists():
        print(f"\n❌ Error: Video file not found at {video_path}")
        sys.exit(1)

    # Initialize pipeline
    print("\n🔄 Initializing inference pipeline...")
    try:
        pipeline = R4BInferencePipeline(
            model_name=args.model,
            device=args.device,
        )
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run inference on single example
    print("\n🔄 Running inference...")
    try:
        answer = pipeline.run_inference(example)
        print(f"\n✅ Inference completed!")
        print(f"\n📊 RESULT:")
        print(f"  Question: {example['question']}")
        print(f"  Predicted Answer: {answer}")
        print(f"  Choices:")
        for choice in example["choices"]:
            marker = "  👉" if choice.startswith(answer) else "    "
            print(f"{marker} {choice}")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe inference pipeline is working correctly.")
    print("You can now run full inference with:")
    print(f"  python run.py")
    print(f"  python run_multi_model.py --model {args.model}")


if __name__ == "__main__":
    test_single_example()

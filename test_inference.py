#!/usr/bin/env python3
"""
Test inference script for a single or multiple examples.

This script tests the inference pipeline on one or more examples to verify everything works.

Usage:
    python test_inference.py
    python test_inference.py --model YannQi/R-4B
    python test_inference.py --device cpu
    python test_inference.py --samples 5  # Test on 5 samples
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
    """Test inference on one or more examples."""
    parser = argparse.ArgumentParser(description="Test inference on one or more examples")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Model to test")
    parser.add_argument("--device", type=str, default=config.DEVICE, choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--test-json", type=Path, default=config.PUBLIC_TEST_JSON, help="Path to test JSON")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to test (default: 1)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TESTING INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Test file: {args.test_json}")
    print(f"Samples: {args.samples}")
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

    # Get samples to test
    total_available = len(test_data["data"])
    num_samples = min(args.samples, total_available)

    if num_samples < args.samples:
        print(f"⚠️  Warning: Only {total_available} samples available, will test on {num_samples} samples")

    examples = test_data["data"][:num_samples]
    print(f"\n📝 Testing on {num_samples} sample(s)")

    # Check if videos exist
    missing_videos = []
    for example in examples:
        video_path = config.DATA_DIR / example["video_path"]
        if not video_path.exists():
            missing_videos.append(video_path)

    if missing_videos:
        print(f"\n❌ Error: {len(missing_videos)} video file(s) not found:")
        for video_path in missing_videos[:5]:  # Show first 5
            print(f"  - {video_path}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos) - 5} more")
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

    # Run inference on samples
    print(f"\n🔄 Running inference on {num_samples} sample(s)...")
    results = []
    errors = []

    for idx, example in enumerate(examples, 1):
        print(f"\n{'─' * 60}")
        print(f"Sample {idx}/{num_samples}")
        print(f"{'─' * 60}")
        print(f"  ID: {example['id']}")
        print(f"  Question: {example['question'][:80]}{'...' if len(example['question']) > 80 else ''}")
        print(f"  Video: {example['video_path']}")

        try:
            answer = pipeline.run_inference(example)
            results.append({
                "id": example["id"],
                "question": example["question"],
                "answer": answer,
                "choices": example["choices"],
                "has_ground_truth": "answer" in example,
                "ground_truth": example.get("answer", None),
                "correct": example.get("answer", "").startswith(answer) if "answer" in example else None
            })

            print(f"  ✅ Predicted: {answer}")

            # Show choices with marker for predicted answer
            print(f"  Choices:")
            for choice in example["choices"]:
                marker = "  👉" if choice.startswith(answer) else "    "
                # Also mark ground truth if available
                if "answer" in example and choice == example["answer"]:
                    marker += " ⭐"
                print(f"{marker} {choice}")

            # Show if correct (if ground truth available)
            if "answer" in example:
                is_correct = example["answer"].startswith(answer)
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"  {status} (Ground truth: {example['answer']})")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append({
                "id": example["id"],
                "error": str(e)
            })
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total samples: {num_samples}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")

    if results and any(r["has_ground_truth"] for r in results):
        # Calculate accuracy if ground truth is available
        correct_count = sum(1 for r in results if r["correct"] is True)
        total_with_truth = sum(1 for r in results if r["has_ground_truth"])
        accuracy = (correct_count / total_with_truth * 100) if total_with_truth > 0 else 0
        print(f"\n📈 Accuracy: {correct_count}/{total_with_truth} ({accuracy:.2f}%)")

    if errors:
        print(f"\n❌ Failed samples:")
        for error in errors:
            print(f"  - {error['id']}: {error['error']}")

    if len(errors) > 0:
        print("\n⚠️  Some samples failed during inference")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nThe inference pipeline is working correctly on {len(results)} sample(s).")
    print("You can now run full inference with:")
    print(f"  python run.py")
    print(f"  python run_multi_model.py --model {args.model}")
    if num_samples == 1:
        print("\nTip: Test with more samples using --samples N:")
        print(f"  python test_inference.py --samples 5")


if __name__ == "__main__":
    test_single_example()

#!/usr/bin/env python3
"""
Test script for InternVL3.5-8B Traffic Inference Pipeline.

Tests the InternVL3.5-8B inference with Flash Attention on:
- Multiple video samples
- Memory monitoring (optimized for 24GB VRAM)
- Accuracy metrics (when ground truth available)
- Flash Attention efficiency

Usage:
    python test_inference.py
    python test_inference.py --samples 5
    python test_inference.py --device cpu --samples 3
    python test_inference.py --samples 10
    python test_inference.py --load-in-8bit --samples 5
    python test_inference.py --no-flash-attn  # Not recommended for 24GB
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.internvl3_5_8B.inference import InternVL35Inference


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_gpu_memory_info() -> str:
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return "N/A (No CUDA)"

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    total_gb = total / 1024**3

    return f"{allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free_gb:.2f} GB free / {total_gb:.2f} GB total"


def test_inference():
    """Test the inference pipeline on sample videos."""
    parser = argparse.ArgumentParser(
        description="Test InternVL3.5-8B traffic inference on sample videos"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3_5-8B",
        help="Model to use (default: OpenGVLab/InternVL3_5-8B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples to test (default: 3)"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Minimum frames to extract (default: 8)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum frames to extract (default: 32)"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=12,
        help="Max number of tiles per frame (default: 12)"
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=config.PUBLIC_TEST_JSON,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--no-support-frames",
        action="store_true",
        help="Disable using support_frames timestamps"
    )
    parser.add_argument(
        "--simple-prompts",
        action="store_true",
        help="Use simpler/shorter prompt templates"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization to reduce memory usage (~50%% reduction)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization to reduce memory usage (~75%% reduction)"
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention (not recommended for 24GB VRAM)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each sample"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("INTERNVL3.5-8B TRAFFIC INFERENCE TEST")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.samples}")
    print(f"Frame range: {args.min_frames}-{args.max_frames} (adaptive)")
    print(f"Max patches per frame: {args.max_num}")
    print(f"Support frames: {'Disabled' if args.no_support_frames else 'Enabled'}")
    print(f"Flash Attention: {'Disabled' if args.no_flash_attn else 'Enabled (~30% memory savings)'}")
    print(f"Prompts: {'Simple' if args.simple_prompts else 'Detailed'}")
    if args.load_in_4bit:
        print(f"Quantization: 4-bit (memory optimized)")
    elif args.load_in_8bit:
        print(f"Quantization: 8-bit (memory optimized)")
    else:
        print(f"Quantization: None (full precision)")
    print(f"Test file: {args.test_json}")
    print("=" * 70 + "\n")

    # Check if test file exists
    if not args.test_json.exists():
        print(f"❌ Error: Test file not found at {args.test_json}")
        sys.exit(1)

    # Load test data
    print("Loading test data...")
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if not test_data.get("data"):
        print("❌ Error: No data found in test file")
        sys.exit(1)

    # Get samples
    total_available = len(test_data["data"])
    num_samples = min(args.samples, total_available)

    if num_samples < args.samples:
        print(f"⚠️  Warning: Only {total_available} samples available, testing {num_samples}\n")

    examples = test_data["data"][:num_samples]

    # Check if videos exist
    missing_videos = []
    for example in examples:
        video_path = config.DATA_DIR / example["video_path"]
        if not video_path.exists():
            missing_videos.append(video_path)

    if missing_videos:
        print(f"❌ Error: {len(missing_videos)} video file(s) not found:")
        for video_path in missing_videos[:5]:
            print(f"  - {video_path}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos) - 5} more")
        sys.exit(1)

    # Clear CUDA cache
    if args.device == "cuda" and torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU Memory: {get_gpu_memory_info()}\n")

    # Initialize pipeline
    print("Initializing inference pipeline...\n")
    start_time = time.time()

    try:
        pipeline = InternVL35Inference(
            model_name=args.model,
            device=args.device,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            max_num=args.max_num,
            use_support_frames=not args.no_support_frames,
            simple_prompts=args.simple_prompts,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            use_flash_attn=not args.no_flash_attn,
        )
        init_time = time.time() - start_time
        print(f"✓ Initialization complete in {format_duration(init_time)}\n")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print(f"\n💡 Suggestions for 24GB VRAM:")
            print(f"   1. Use 8-bit quantization: --load-in-8bit (recommended, ~50% memory reduction)")
            print(f"   2. Use 4-bit quantization: --load-in-4bit (~75% memory reduction)")
            print(f"   3. Keep Flash Attention enabled (remove --no-flash-attn if present)")
            print(f"   4. Reduce max_num: --max-num 8")
            print(f"   5. Reduce frame count: --min-frames 8 --max-frames 24")
            print(f"   6. Try running on CPU: --device cpu")
            print(f"   7. Close other GPU applications")
            if torch.cuda.is_available():
                print(f"\n📊 GPU Memory: {get_gpu_memory_info()}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run inference on samples
    print(f"{'='*70}")
    print(f"RUNNING INFERENCE ON {num_samples} SAMPLE(S)")
    print(f"{'='*70}\n")

    results = []
    errors = []
    total_inference_time = 0

    for idx, example in enumerate(examples, 1):
        print(f"{'─'*70}")
        print(f"Sample {idx}/{num_samples}")
        print(f"{'─'*70}")
        print(f"  ID: {example['id']}")
        print(f"  Question: {example['question'][:80]}{'...' if len(example['question']) > 80 else ''}")
        print(f"  Video: {example['video_path']}")

        if "support_frames" in example and example["support_frames"]:
            print(f"  Support frames: {example['support_frames']}")

        try:
            # Prepare inputs
            video_path = config.DATA_DIR / example["video_path"]
            question = example["question"]
            choices = example["choices"]
            support_frames = example.get("support_frames", None)

            # Run inference
            inference_start = time.time()
            answer = pipeline.infer(
                video_path=video_path,
                question=question,
                choices=choices,
                support_frames=support_frames,
                verbose=args.verbose
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # Store result
            results.append({
                "id": example["id"],
                "question": question,
                "answer": answer,
                "choices": choices,
                "has_ground_truth": "answer" in example,
                "ground_truth": example.get("answer", None),
                "correct": example.get("answer", "").startswith(answer) if "answer" in example else None,
                "inference_time": inference_time
            })

            print(f"  ✅ Predicted: {answer} (in {format_duration(inference_time)})")

            # Show choices
            print(f"  Choices:")
            for choice in choices:
                marker = "  👉" if choice.startswith(answer) else "    "
                if "answer" in example and choice == example["answer"]:
                    marker += " ⭐"
                print(f"{marker} {choice}")

            # Show correctness
            if "answer" in example:
                is_correct = example["answer"].startswith(answer)
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"  {status} (Ground truth: {example['answer']})")

            # GPU memory info
            if args.device == "cuda" and torch.cuda.is_available() and (idx % 5 == 0 or idx == num_samples):
                print(f"  GPU Memory: {get_gpu_memory_info()}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append({
                "id": example["id"],
                "error": str(e)
            })
            import traceback
            traceback.print_exc()

        print()  # Blank line between samples

        # Clear CUDA cache
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final cleanup
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print(f"{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {num_samples}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")

    if total_inference_time > 0:
        avg_time = total_inference_time / len(results) if results else 0
        print(f"\n⏱️  Timing:")
        print(f"  Total inference time: {format_duration(total_inference_time)}")
        print(f"  Average per video: {format_duration(avg_time)}")
        print(f"  Initialization time: {format_duration(init_time)}")

    if results and any(r["has_ground_truth"] for r in results):
        correct_count = sum(1 for r in results if r["correct"] is True)
        total_with_truth = sum(1 for r in results if r["has_ground_truth"])
        accuracy = (correct_count / total_with_truth * 100) if total_with_truth > 0 else 0
        print(f"\n📈 Accuracy: {correct_count}/{total_with_truth} ({accuracy:.2f}%)")

    if args.device == "cuda" and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n💾 Peak GPU Memory: {peak_memory:.2f} GB")
        print(f"💾 Final GPU Memory: {get_gpu_memory_info()}")

        # 24GB VRAM check
        if peak_memory <= 24:
            print(f"✅ Fits in 24GB VRAM (peak: {peak_memory:.2f} GB)")
        else:
            print(f"⚠️  Exceeded 24GB VRAM (peak: {peak_memory:.2f} GB)")

    if errors:
        print(f"\n❌ Failed samples:")
        for error in errors:
            print(f"  - {error['id']}: {error['error']}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nThe inference pipeline is working correctly on {len(results)} sample(s).")
    print("\nYou can now run full inference with:")
    print(f"  python road_buddy/src/internvl3_5_8B/run_inference.py")
    print()


if __name__ == "__main__":
    test_inference()

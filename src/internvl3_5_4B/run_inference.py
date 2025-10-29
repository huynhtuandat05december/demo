#!/usr/bin/env python3
"""
Run InternVL3.5-4B Traffic Inference on Full Test Dataset.

This script runs the InternVL3.5-4B inference pipeline with Flash Attention
on the complete test dataset and generates a CSV submission file.

Optimized for 12GB+ VRAM with Flash Attention enabled by default.

Usage:
    # Basic usage with defaults (recommended for 12GB+ VRAM)
    python run_inference.py

    # With 8-bit quantization for lower-end GPUs
    python run_inference.py --load-in-8bit

    # Custom configuration
    python run_inference.py --device cuda --min-frames 8 --max-frames 32

    # CPU mode (very slow)
    python run_inference.py --device cpu

    # Custom output path
    python run_inference.py --output my_results.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.internvl3_5_4B.inference import InternVL35_4BInference


def main():
    """Run traffic inference on full test dataset."""
    parser = argparse.ArgumentParser(
        description="Run InternVL3.5-4B traffic inference on full test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (CUDA, 8-32 frames, Flash Attention)
  # Recommended for 12GB+ VRAM
  python run_inference.py

  # With 8-bit quantization for lower-end GPUs
  python run_inference.py --load-in-8bit

  # Run on CPU (very slow)
  python run_inference.py --device cpu

  # Custom frame configuration
  python run_inference.py --min-frames 16 --max-frames 64 --max-num 8

  # Custom output file
  python run_inference.py --output results/my_submission.csv

  # Disable flash attention
  python run_inference.py --no-flash-attn
        """
    )

    # Model and device configuration
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3_5-4B",
        help="Model to use (default: OpenGVLab/InternVL3_5-4B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    # Frame extraction configuration
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
        help="Maximum frames to extract, max 512 (default: 32)"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=12,
        help="Max number of tiles per frame (default: 12)"
    )

    # Data paths
    parser.add_argument(
        "--test-json",
        type=Path,
        default=config.PUBLIC_TEST_JSON,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Base directory for video paths"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: auto-generated with timestamp)"
    )

    # Feature flags
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
        "--no-flash-attn",
        action="store_true",
        help="Disable Flash Attention"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each video"
    )

    # Quantization options
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit mode (reduces memory ~50%%)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit mode (reduces memory ~75%%)"
    )

    args = parser.parse_args()

    # Validate frame limits
    if args.max_frames > 512:
        print(f"⚠️  Warning: max-frames ({args.max_frames}) exceeds InternVL3.5 limit of 512")
        print("   Setting max-frames to 512")
        args.max_frames = 512

    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = "InternVL3_5_4B"
        args.output = config.OUTPUT_DIR / f"inference_{model_short}_{timestamp}.csv"
    else:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("INTERNVL3.5-4B TRAFFIC INFERENCE - FULL TEST DATASET")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Frame range: {args.min_frames}-{args.max_frames} (adaptive, max 512)")
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

    # Estimate VRAM usage
    if args.device == "cuda":
        base_vram = 8  # 4B model in bfloat16
        if args.load_in_4bit:
            base_vram = 2
        elif args.load_in_8bit:
            base_vram = 4
        if not args.no_flash_attn:
            base_vram *= 0.7  # Flash attention saves ~30%
        frame_vram = 2 + (args.max_frames / 32) * 4  # Rough estimate
        total_vram = base_vram + frame_vram
        print(f"\n💾 Estimated VRAM: ~{total_vram:.1f} GB")
        if total_vram > 16:
            print(f"⚠️  WARNING: Estimated VRAM exceeds 16GB!")
            print("   Consider using --load-in-8bit or reducing frames")
        elif total_vram > 14:
            print(f"⚠️  CAUTION: Close to limit, may cause OOM")
        else:
            print(f"✓ Should fit comfortably in available VRAM")

    print(f"\nInput:")
    print(f"  Test JSON: {args.test_json}")
    print(f"  Data directory: {args.data_dir}")
    print(f"\nOutput:")
    print(f"  CSV file: {args.output}")
    print("=" * 70 + "\n")

    # Validate paths
    if not args.test_json.exists():
        print(f"❌ Error: Test JSON file not found: {args.test_json}")
        sys.exit(1)

    if not args.data_dir.exists():
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Warning for CPU
    if args.device == "cpu":
        print("⚠️  WARNING: Running on CPU")
        print("   This will be VERY SLOW for full dataset (~2-5 minutes per video)")
        print("   Estimated total time: several hours to days")
        print("   Consider using GPU or reducing test samples\n")
        response = input("Continue with CPU? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            sys.exit(0)
        print()

    # Warning for disabled flash attention
    if args.no_flash_attn and args.device == "cuda" and not (args.load_in_8bit or args.load_in_4bit):
        print("⚠️  WARNING: Flash Attention is disabled")
        print("   Without flash attention, memory usage increases ~30%")
        print("   This may cause OOM")
        print("   Consider keeping it enabled or using --load-in-8bit\n")
        response = input("Continue without flash attention? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted. Run again without --no-flash-attn flag.")
            sys.exit(0)
        print()

    # Initialize pipeline
    try:
        print("Initializing InternVL3.5-4B inference pipeline...\n")
        pipeline = InternVL35_4BInference(
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
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run inference
    try:
        print("Starting inference on full test dataset...\n")
        pipeline.run_pipeline(
            test_json_path=args.test_json,
            data_dir=args.data_dir,
            output_csv_path=args.output,
            verbose=args.verbose
        )

        print(f"\n{'='*70}")
        print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {args.output}")
        print(f"You can now submit this CSV file.\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
        print("Partial results may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

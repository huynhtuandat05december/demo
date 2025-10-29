#!/usr/bin/env python3
"""
Run InternVideo2.5_Chat_8B Traffic Inference on Full Test Dataset.

This script runs the InternVideo2.5_Chat_8B multi-frame traffic inference pipeline on the
complete test dataset and generates a CSV submission file.

Usage:
    # Basic usage with defaults
    python run_inference.py

    # Custom configuration
    python run_inference.py --device cuda --min-frames 8 --max-frames 32

    # CPU mode
    python run_inference.py --device cpu

    # Custom output path
    python run_inference.py --output my_results.csv

    # With quantization for memory optimization
    python run_inference.py --load-in-8bit
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.internlvideo2_5_chat_8B.inference import InternVideo2Inference


def main():
    """Run traffic inference on full test dataset."""
    parser = argparse.ArgumentParser(
        description="Run InternVideo2.5_Chat_8B traffic inference on full test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (CUDA, 8-32 frames, max_num=12)
  python run_inference.py

  # Run on CPU
  python run_inference.py --device cpu

  # Custom frame configuration
  python run_inference.py --min-frames 16 --max-frames 64 --max-num 8

  # Custom output file
  python run_inference.py --output results/my_submission.csv

  # Memory optimized (8-bit quantization)
  python run_inference.py --load-in-8bit

  # Maximum quality (more frames)
  python run_inference.py --min-frames 32 --max-frames 128
        """
    )

    # Model and device configuration
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVideo2_5_Chat_8B",
        help="Model to use (default: OpenGVLab/InternVideo2_5_Chat_8B)"
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
        "--use-duration-frames",
        action="store_true",
        help="Calculate frames based on video duration (4 sec/segment, 128-512 frames)"
    )
    parser.add_argument(
        "--simple-prompts",
        action="store_true",
        help="Use simpler/shorter prompt templates"
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
        print(f"⚠️  Warning: max-frames ({args.max_frames}) exceeds InternVideo2.5 limit of 512")
        print("   Setting max-frames to 512")
        args.max_frames = 512

    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = "InternVideo2_5_Chat_8B"
        args.output = config.OUTPUT_DIR / f"inference_{model_short}_{timestamp}.csv"
    else:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("INTERNVIDEO2.5_CHAT_8B TRAFFIC INFERENCE - FULL TEST DATASET")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.use_duration_frames:
        print(f"Frame mode: Duration-based (128-512 frames, 4 sec/segment)")
    else:
        print(f"Frame range: {args.min_frames}-{args.max_frames} (adaptive, max 512)")
    print(f"Max patches per frame: {args.max_num}")
    print(f"Support frames: {'Disabled' if args.no_support_frames else 'Enabled'}")
    print(f"Prompts: {'Simple' if args.simple_prompts else 'Detailed'}")
    if args.load_in_4bit:
        print(f"Quantization: 4-bit (memory optimized)")
    elif args.load_in_8bit:
        print(f"Quantization: 8-bit (memory optimized)")
    else:
        print(f"Quantization: None (full precision)")
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

    # Initialize pipeline
    try:
        print("Initializing InternVideo2.5_Chat_8B inference pipeline...\n")
        pipeline = InternVideo2Inference(
            model_name=args.model,
            device=args.device,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            max_num=args.max_num,
            use_support_frames=not args.no_support_frames,
            simple_prompts=args.simple_prompts,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            use_duration_based_frames=args.use_duration_frames,
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

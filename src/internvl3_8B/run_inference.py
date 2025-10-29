#!/usr/bin/env python3
"""
Run InternVL3 Traffic Inference on Full Test Dataset.

This script runs the enhanced multi-frame traffic inference pipeline on the
complete test dataset and generates a CSV submission file.

Usage:
    # Basic usage with defaults
    python run_inference.py

    # Custom configuration
    python run_inference.py --device cuda --min-frames 8 --max-frames 10

    # CPU mode
    python run_inference.py --device cpu

    # Custom output path
    python run_inference.py --output my_results.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.internvl3_8B.inference import InternVL3Inference


def main():
    """Run traffic inference on full test dataset."""
    parser = argparse.ArgumentParser(
        description="Run InternVL3 traffic inference on full test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (CUDA, 6-12 frames, max_num=24)
  python run_inference.py

  # Run on CPU
  python run_inference.py --device cpu

  # Custom frame configuration
  python run_inference.py --min-frames 8 --max-frames 10 --max-num 16

  # Custom output file
  python run_inference.py --output results/my_submission.csv

  # Balanced configuration (faster, less memory)
  python run_inference.py --min-frames 6 --max-frames 8 --max-num 12
        """
    )

    # Model and device configuration
    parser.add_argument(
        "--model",
        type=str,
        default=config.INTERNVL_MODEL_NAME,
        help="Model to use (default: OpenGVLab/InternVL3-8B)"
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
        default=config.MIN_FRAMES,
        help=f"Minimum frames to extract (default: {config.MIN_FRAMES})"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=config.MAX_FRAMES,
        help=f"Maximum frames to extract (default: {config.MAX_FRAMES})"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=config.INTERNVL_MAX_NUM,
        help=f"InternVL max_num parameter (default: {config.INTERNVL_MAX_NUM})"
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
        "--verbose",
        action="store_true",
        help="Print detailed information for each video"
    )

    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1].replace("-", "_")
        args.output = config.OUTPUT_DIR / f"inference_{model_short}_{timestamp}.csv"
    else:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("INTERNVL3 TRAFFIC INFERENCE - FULL TEST DATASET")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Frame range: {args.min_frames}-{args.max_frames} (adaptive)")
    print(f"Max patches per frame: {args.max_num}")
    print(f"Support frames: {'Disabled' if args.no_support_frames else 'Enabled'}")
    print(f"Prompts: {'Simple' if args.simple_prompts else 'Detailed'}")
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
        print("Initializing InternVL3 inference pipeline...\n")
        pipeline = InternVL3Inference(
            model_name=args.model,
            device=args.device,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            max_num=args.max_num,
            use_support_frames=not args.no_support_frames,
            simple_prompts=args.simple_prompts,
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

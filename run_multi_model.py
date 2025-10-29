#!/usr/bin/env python3
"""
Flexible runner script for multi-model inference pipeline.

Usage:
    # Use default model from config
    python run_multi_model.py

    # Specify a model
    python run_multi_model.py --model YannQi/R-4B
    python run_multi_model.py --model OpenGVLab/InternVL3-8B
    python run_multi_model.py --model Qwen/Qwen3-VL-8B-Instruct

    # Use CPU instead of GPU
    python run_multi_model.py --device cpu

    # Verbose output
    python run_multi_model.py --verbose

    # Custom data paths
    python run_multi_model.py --test-json path/to/test.json --output path/to/output.csv
"""

import argparse
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.inference_multi_model import MultiModelInferencePipeline


def generate_output_filename(model_name: str, output_dir: Path) -> Path:
    """
    Generate output filename based on model name and current date/time.

    Args:
        model_name: Name of the model (e.g., "YannQi/R-4B")
        output_dir: Directory to save the output file

    Returns:
        Path to output file with format: submission_{model}_{date}_{time}.csv
    """
    # Extract model name (handle HuggingFace format like "YannQi/R-4B")
    if "/" in model_name:
        model_short = model_name.split("/")[-1]  # "R-4B"
    else:
        model_short = model_name

    # Remove special characters and replace with underscore
    model_short = model_short.replace("-", "_").replace(".", "_")

    # Get current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"submission_{model_short}_{timestamp}.csv"

    return output_dir / filename


SUPPORTED_MODELS = [
    "YannQi/R-4B",
    "OpenGVLab/InternVL3-8B",
    "Qwen/Qwen3-VL-8B-Instruct",
]


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Video Question Answering with Multiple Model Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Model to use. Supported: {', '.join(SUPPORTED_MODELS)}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available, else cpu)",
    )

    parser.add_argument(
        "--test-json",
        type=Path,
        default=config.PUBLIC_TEST_JSON,
        help="Path to test JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save output CSV (default: auto-generate based on model and date)",
    )

    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use default filename without timestamp",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported models",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("Supported models:")
        for model in SUPPORTED_MODELS:
            print(f"  - {model}")
        return

    # Validate model
    model_found = any(
        supported.lower() in args.model.lower() for supported in SUPPORTED_MODELS
    )
    if not model_found:
        print(f"Warning: Model '{args.model}' may not be supported.")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Generate output filename if not specified
    if args.output is None:
        if args.no_timestamp:
            args.output = config.SUBMISSION_FILE
        else:
            args.output = generate_output_filename(args.model, config.OUTPUT_DIR)
            print(f"\n📝 Auto-generated output file: {args.output.name}")

    # Print configuration
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print('='*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Test JSON: {args.test_json}")
    print(f"Output: {args.output}")
    print(f"Verbose: {args.verbose}")
    print('='*60)

    # Check if test file exists
    if not args.test_json.exists():
        print(f"\nError: Test JSON file not found: {args.test_json}")
        sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = MultiModelInferencePipeline(
            model_name=args.model,
            device=args.device,
        )
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        sys.exit(1)

    # Run inference
    try:
        pipeline.run_pipeline(
            test_json_path=args.test_json,
            output_csv_path=args.output,
            verbose=args.verbose,
        )
        print(f"\n✓ Inference completed successfully!")
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

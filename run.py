#!/usr/bin/env python3
"""
Runner script for the inference pipeline.

This runs inference with the default configuration from src/config.py.
Output will be saved with auto-generated filename based on model and timestamp.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.inference import R4BInferencePipeline


def generate_output_filename(model_name: str, output_dir: Path) -> Path:
    """Generate output filename based on model name and current date/time."""
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
    filename = f"submission_{model_short}_{timestamp}.csv"

    return output_dir / filename


def main():
    """Main entry point for the inference pipeline."""
    # Generate output filename with timestamp
    output_file = generate_output_filename(config.MODEL_NAME, config.OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("RUNNING INFERENCE")
    print('='*60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Output: {output_file}")
    print('='*60 + '\n')

    # Initialize pipeline
    pipeline = R4BInferencePipeline()

    # Run inference on public test data
    pipeline.run_pipeline(
        test_json_path=config.PUBLIC_TEST_JSON,
        output_csv_path=output_file,
    )


if __name__ == "__main__":
    main()

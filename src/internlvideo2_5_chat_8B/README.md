# InternVideo2.5_Chat_8B Inference for RoadBuddy Traffic Dataset

This module implements inference for the RoadBuddy traffic dataset using the **OpenGVLab/InternVideo2_5_Chat_8B** model.

## Model Information

- **Model**: OpenGVLab/InternVideo2_5_Chat_8B
- **Type**: Video Understanding Model
- **Max Frames**: 512 frames per video
- **HuggingFace Page**: https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B

## Installation

Install required dependencies:

```bash
pip install transformers==4.40.1 av imageio decord opencv-python pandas tqdm
pip install flash-attn --no-build-isolation

# For quantization support (optional)
pip install bitsandbytes
```

## Usage

### Basic Usage

Run inference with default settings:

```bash
python road_buddy/src/internlvideo2_5_chat_8B/run_inference.py
```

### Custom Configuration

```bash
# Use different frame counts
python run_inference.py --min-frames 16 --max-frames 64

# Use duration-based frame calculation (official HuggingFace example)
python run_inference.py --use-duration-frames

# Use 8-bit quantization to save memory
python run_inference.py --load-in-8bit

# Custom output path
python run_inference.py --output results/my_submission.csv

# Run on CPU (slow)
python run_inference.py --device cpu
```

### Command-Line Options

```
--model MODEL              Model to use (default: OpenGVLab/InternVideo2_5_Chat_8B)
--device {cuda,cpu}        Device to use (default: cuda)
--min-frames N             Minimum frames to extract (default: 8)
--max-frames N             Maximum frames to extract, max 512 (default: 32)
--max-num N                Max number of tiles per frame (default: 12)
--test-json PATH           Path to test JSON file
--data-dir PATH            Base directory for video paths
--output PATH              Output CSV path
--no-support-frames        Disable using support_frames timestamps
--use-duration-frames      Calculate frames based on video duration (4 sec/segment, 128-512 frames)
--simple-prompts           Use simpler/shorter prompt templates
--load-in-8bit            Load model in 8-bit mode (~50% memory reduction)
--load-in-4bit            Load model in 4-bit mode (~75% memory reduction)
--verbose                 Print detailed information for each video
```

## Configuration Recommendations

### Duration-Based (official HuggingFace example)
Uses automatic frame calculation based on video duration (4 seconds per segment, 128-512 frames):
```bash
python run_inference.py --use-duration-frames
```

### High Quality (if you have 24GB+ VRAM)
```bash
python run_inference.py --min-frames 32 --max-frames 128 --max-num 12
```

### Balanced (recommended for most cases)
```bash
python run_inference.py --min-frames 8 --max-frames 32 --max-num 12
```

### Memory Optimized (for limited VRAM)
```bash
python run_inference.py --min-frames 8 --max-frames 16 --max-num 8 --load-in-8bit
```

## Output

The inference pipeline generates a CSV file with the following format:

```csv
id,answer
testa_0001,A
testa_0002,B
...
```

This CSV can be directly submitted to the competition.

## Features

- **Multi-frame processing**: Extracts 8-512 frames per video for comprehensive analysis
- **Adaptive frame extraction**: Automatically adjusts frame count based on video content
- **Support frames**: Uses temporal anchoring for better context
- **Quantization support**: 4-bit/8-bit quantization for memory efficiency
- **Multilingual**: Handles Vietnamese, English, and Chinese responses
- **Progress tracking**: Real-time progress bar during inference

## Implementation Details

The inference pipeline:
1. Loads videos using decord with temporal sampling
2. Preprocesses frames with dynamic tiling for high resolution
3. Generates prompts optimized for traffic scenarios
4. Uses the model's chat interface for generation
5. Parses multilingual responses to extract answers

## Troubleshooting

### Out of Memory Errors
- Reduce `--max-frames` and `--max-num` parameters
- Use `--load-in-8bit` or `--load-in-4bit` for quantization
- Process fewer videos at once

### Slow Inference
- Ensure you're using GPU (`--device cuda`)
- Reduce number of frames
- Check that CUDA is properly installed

### Import Errors
- Make sure all dependencies are installed
- Verify transformers version is 4.40.1
- Install flash-attn if needed

## Files

- `inference.py`: Main inference pipeline implementation
- `run_inference.py`: Command-line interface for running inference
- `README.md`: This documentation file

## Credits

Based on the official InternVideo2.5 implementation from OpenGVLab.

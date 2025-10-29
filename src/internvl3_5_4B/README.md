# InternVL3.5-4B Inference for RoadBuddy Traffic Dataset

This module implements inference for the RoadBuddy traffic dataset using **OpenGVLab/InternVL3_5-4B** with Flash Attention for memory efficiency.

## Model Information

- **Model**: OpenGVLab/InternVL3_5-4B
- **Type**: Multimodal Video Understanding Model
- **Key Feature**: Flash Attention (~30% memory savings)
- **Max Frames**: 512 frames per video
- **Optimized For**: 12GB+ VRAM
- **HuggingFace Page**: https://huggingface.co/OpenGVLab/InternVL3_5-4B

## 12GB+ VRAM Compatibility

### ✅ **Recommended Configurations**

#### 1. Default Configuration (9-12GB) - RECOMMENDED
```bash
python road_buddy/src/internvl3_5_4B/run_inference.py
```
- Full precision (bfloat16)
- Flash Attention enabled
- 8-32 frames adaptive
- **Estimated VRAM**: ~9-12GB ✓

#### 2. With 8-bit Quantization (6-8GB) - For Lower-end GPUs
```bash
python run_inference.py --load-in-8bit
```
- 8-bit quantization + Flash Attention
- 50% model weight reduction
- **Estimated VRAM**: ~6-8GB ✓
- **Recommended for 8GB GPUs**

#### 3. Higher Frame Count (10-14GB)
```bash
python run_inference.py --min-frames 16 --max-frames 64
```
- More frames for better accuracy
- **Estimated VRAM**: ~10-14GB ✓
- Recommended for 16GB+ GPUs

#### 4. Maximum Safety - 4-bit (4-6GB)
```bash
python run_inference.py --load-in-4bit
```
- 4-bit quantization
- 75% model weight reduction
- **Estimated VRAM**: ~4-6GB ✓
- May slightly reduce accuracy
- Perfect for 6-8GB GPUs

### ⚠️ **Risky Configurations (May OOM on 12GB)**

```bash
# Without Flash Attention - NOT RECOMMENDED
python run_inference.py --no-flash-attn
# Estimated: ~12-15GB ⚠️ May exceed 12GB!

# Very high frames without quantization
python run_inference.py --min-frames 64 --max-frames 128
# Estimated: ~14-18GB ⚠️ May exceed 12GB!
```

## Installation

Install required dependencies:

```bash
pip install transformers==4.40.1 av imageio decord opencv-python pandas tqdm
pip install flash-attn --no-build-isolation  # Required for Flash Attention

# For quantization support (optional but recommended)
pip install bitsandbytes
```

## Usage

### Basic Usage (Recommended for 12GB+)

```bash
# Run with default settings - optimized for 12GB+ VRAM
python road_buddy/src/internvl3_5_4B/run_inference.py
```

### With Quantization (For Lower-end GPUs)

```bash
# 8-bit quantization (recommended for 8GB GPUs)
python run_inference.py --load-in-8bit

# 4-bit quantization (for 6-8GB GPUs)
python run_inference.py --load-in-4bit
```

### Custom Configuration

```bash
# Custom frame range
python run_inference.py --min-frames 16 --max-frames 48

# Custom output path
python run_inference.py --output results/my_submission.csv

# Combine options
python run_inference.py --load-in-8bit --min-frames 16 --max-frames 64
```

## Command-Line Options

```
--model MODEL              Model to use (default: OpenGVLab/InternVL3_5-4B)
--device {cuda,cpu}        Device to use (default: cuda)
--min-frames N             Minimum frames to extract (default: 8)
--max-frames N             Maximum frames to extract, max 512 (default: 32)
--max-num N                Max number of tiles per frame (default: 12)
--test-json PATH           Path to test JSON file
--data-dir PATH            Base directory for video paths
--output PATH              Output CSV path
--no-support-frames        Disable using support_frames timestamps
--no-flash-attn           Disable Flash Attention
--simple-prompts           Use simpler/shorter prompt templates
--load-in-8bit            Load model in 8-bit mode (~50% memory reduction)
--load-in-4bit            Load model in 4-bit mode (~75% memory reduction)
--verbose                 Print detailed information for each video
```

## VRAM Usage Guide

### Memory Breakdown

| Configuration | Model Weights | Frames & Activations | Flash Attn Savings | Total VRAM |
|--------------|---------------|---------------------|-------------------|-----------|
| Full Precision + Flash | ~8GB | ~2-6GB | -30% (~2-3GB) | **~9-12GB** ✓ |
| 8-bit + Flash | ~4GB | ~2-6GB | -30% (~1GB) | **~6-8GB** ✓ |
| 4-bit + Flash | ~2GB | ~2-6GB | -30% (~0.5GB) | **~4-6GB** ✓ |
| Full Precision (no Flash) | ~8GB | ~2-6GB | None | **~12-15GB** ⚠️ |

### Frame Count Impact

- **8-32 frames**: Base memory (~2-4GB for frames)
- **32-64 frames**: +2-3GB additional
- **64-128 frames**: +4-6GB additional
- **128+ frames**: +6-8GB additional

## Configuration Recommendations by Use Case

### Maximum Quality (requires 16GB+ VRAM)
```bash
python run_inference.py --min-frames 16 --max-frames 64 --max-num 12
# Expected: ~12-15GB VRAM
```

### Balanced Quality & Safety (RECOMMENDED for 12GB)
```bash
python run_inference.py
# Default: 8-32 frames, Flash Attention
# Expected: ~9-12GB VRAM
```

### For 8GB GPUs
```bash
python run_inference.py --load-in-8bit
# Expected: ~6-8GB VRAM
```

### For 6GB GPUs
```bash
python run_inference.py --load-in-4bit --max-frames 24
# Expected: ~4-6GB VRAM
```

## Testing

Before running full inference, test on a few samples:

```bash
# Test with default settings
python road_buddy/src/internvl3_5_4B/test_inference.py

# Test with 5 samples
python test_inference.py --samples 5

# Test with quantization
python test_inference.py --load-in-8bit --samples 3
```

## Flash Attention

Flash Attention is **enabled by default** and provides:
- ~30% memory reduction
- Faster inference
- Same accuracy

To disable (not recommended):
```bash
python run_inference.py --no-flash-attn
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

- **Flash Attention**: Automatic 30% memory reduction
- **Quantization**: 8-bit/4-bit options for lower-end GPUs
- **Adaptive Frames**: Automatically adjusts 8-32 frames based on content
- **Support Frames**: Uses temporal anchoring for better context
- **Multilingual**: Handles Vietnamese, English, and Chinese responses
- **Progress Tracking**: Real-time progress bar during inference
- **VRAM Estimation**: Shows estimated VRAM before starting

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. **Use 8-bit quantization** (recommended for 8GB GPUs):
   ```bash
   python run_inference.py --load-in-8bit
   ```

2. **Use 4-bit quantization** (for 6-8GB GPUs):
   ```bash
   python run_inference.py --load-in-4bit
   ```

3. **Reduce frame count**:
   ```bash
   python run_inference.py --max-frames 24
   ```

4. **Reduce tiles per frame**:
   ```bash
   python run_inference.py --max-num 8
   ```

5. **Combine all optimizations**:
   ```bash
   python run_inference.py --load-in-4bit --max-frames 24 --max-num 8
   ```

### Slow Inference

- Ensure you're using GPU: `--device cuda`
- Check CUDA is properly installed
- Verify Flash Attention is enabled (default)
- Monitor GPU utilization with `nvidia-smi`

### Flash Attention Installation Issues

If flash-attn installation fails:
```bash
# Install from source
pip install flash-attn --no-build-isolation

# Or disable it (will use more VRAM)
python run_inference.py --no-flash-attn --load-in-8bit
```

### Import Errors

- Ensure all dependencies are installed
- Verify transformers version: `pip install transformers==4.40.1`
- Check CUDA version compatibility with PyTorch

## Comparison with Other Models

| Model | VRAM (default) | VRAM (8-bit) | Speed | Quality |
|-------|---------------|-------------|-------|---------|
| InternVL3.5-4B | ~9-12GB | ~6-8GB | Fast | High |
| InternVL3.5-8B | ~18-22GB | ~12-16GB | Medium | Highest |
| InternVL3-8B | ~20-24GB | ~14-18GB | Medium | High |

**InternVL3.5-4B is the best option for GPUs with 12GB or less VRAM.**

## Performance Tips

1. **Use default settings first** - optimized for 12GB+ VRAM
2. **Add --load-in-8bit if you have 8GB** - minimal quality loss
3. **Add --load-in-4bit if you have 6-8GB** - some quality loss
4. **Keep Flash Attention enabled** - free 30% memory savings
5. **Monitor VRAM usage**: `nvidia-smi -l 1`
6. **Close other GPU applications** before running

## Files

- `inference.py`: Main inference pipeline with Flash Attention
- `run_inference.py`: Command-line interface with VRAM estimation
- `test_inference.py`: Testing script for validating setup
- `README.md`: This documentation file

## Credits

Based on the official InternVL3.5 implementation from OpenGVLab with optimizations for 12GB+ VRAM usage.

## Support

For issues related to:
- **Model**: Check https://huggingface.co/OpenGVLab/InternVL3_5-4B
- **Flash Attention**: Check https://github.com/Dao-AILab/flash-attention
- **Memory optimization**: Refer to VRAM Usage Guide above

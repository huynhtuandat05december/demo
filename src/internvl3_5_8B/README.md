# InternVL3.5-8B Inference for RoadBuddy Traffic Dataset

This module implements inference for the RoadBuddy traffic dataset using **OpenGVLab/InternVL3_5-8B** with Flash Attention for memory efficiency.

## Model Information

- **Model**: OpenGVLab/InternVL3_5-8B
- **Type**: Multimodal Video Understanding Model
- **Key Feature**: Flash Attention (~30% memory savings)
- **Max Frames**: 512 frames per video
- **Optimized For**: 24GB VRAM
- **HuggingFace Page**: https://huggingface.co/OpenGVLab/InternVL3_5-8B

## 24GB VRAM Compatibility

### ✅ **Recommended Configurations (Fits 24GB)**

#### 1. Default Configuration (18-22GB) - RECOMMENDED
```bash
python road_buddy/src/internvl3_5_8B/run_inference.py
```
- Full precision (bfloat16)
- Flash Attention enabled
- 8-32 frames adaptive
- **Estimated VRAM**: ~18-22GB ✓

#### 2. With 8-bit Quantization (12-16GB) - SAFEST
```bash
python run_inference.py --load-in-8bit
```
- 8-bit quantization + Flash Attention
- 50% model weight reduction
- **Estimated VRAM**: ~12-16GB ✓
- **Recommended for safety margin**

#### 3. Higher Frame Count + 8-bit (16-20GB)
```bash
python run_inference.py --load-in-8bit --min-frames 16 --max-frames 64
```
- More frames for better accuracy
- 8-bit keeps memory manageable
- **Estimated VRAM**: ~16-20GB ✓

#### 4. Maximum Safety - 4-bit (8-12GB)
```bash
python run_inference.py --load-in-4bit
```
- 4-bit quantization
- 75% model weight reduction
- **Estimated VRAM**: ~8-12GB ✓
- May slightly reduce accuracy

### ⚠️ **Risky Configurations (May OOM on 24GB)**

```bash
# Without Flash Attention - NOT RECOMMENDED
python run_inference.py --no-flash-attn
# Estimated: ~24-28GB ⚠️ May exceed 24GB!

# High frames without quantization
python run_inference.py --min-frames 32 --max-frames 128
# Estimated: ~24-30GB ⚠️ May exceed 24GB!
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

### Basic Usage (Recommended for 24GB)

```bash
# Run with default settings - optimized for 24GB VRAM
python road_buddy/src/internvl3_5_8B/run_inference.py
```

### With Quantization (Extra Safety)

```bash
# 8-bit quantization (recommended)
python run_inference.py --load-in-8bit

# 4-bit quantization (maximum safety)
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
--model MODEL              Model to use (default: OpenGVLab/InternVL3_5-8B)
--device {cuda,cpu}        Device to use (default: cuda)
--min-frames N             Minimum frames to extract (default: 8)
--max-frames N             Maximum frames to extract, max 512 (default: 32)
--max-num N                Max number of tiles per frame (default: 12)
--test-json PATH           Path to test JSON file
--data-dir PATH            Base directory for video paths
--output PATH              Output CSV path
--no-support-frames        Disable using support_frames timestamps
--no-flash-attn           Disable Flash Attention (NOT RECOMMENDED for 24GB)
--simple-prompts           Use simpler/shorter prompt templates
--load-in-8bit            Load model in 8-bit mode (~50% memory reduction)
--load-in-4bit            Load model in 4-bit mode (~75% memory reduction)
--verbose                 Print detailed information for each video
```

## VRAM Usage Guide

### Memory Breakdown

| Configuration | Model Weights | Frames & Activations | Flash Attn Savings | Total VRAM |
|--------------|---------------|---------------------|-------------------|-----------|
| Full Precision + Flash | ~16GB | ~2-6GB | -30% (~5GB) | **~18-22GB** ✓ |
| 8-bit + Flash | ~8GB | ~2-6GB | -30% (~2GB) | **~12-16GB** ✓ |
| 4-bit + Flash | ~4GB | ~2-6GB | -30% (~1GB) | **~8-12GB** ✓ |
| Full Precision (no Flash) | ~16GB | ~2-6GB | None | **~24-28GB** ⚠️ |

### Frame Count Impact

- **8-32 frames**: Base memory (~2-4GB for frames)
- **32-64 frames**: +2-3GB additional
- **64-128 frames**: +4-6GB additional
- **128+ frames**: +6-8GB additional

## Configuration Recommendations by Use Case

### Maximum Quality (requires good 24GB card with headroom)
```bash
python run_inference.py --min-frames 16 --max-frames 48 --max-num 12
# Expected: ~20-23GB VRAM
```

### Balanced Quality & Safety (RECOMMENDED)
```bash
python run_inference.py
# Default: 8-32 frames, Flash Attention
# Expected: ~18-22GB VRAM
```

### Maximum Safety (guaranteed to fit)
```bash
python run_inference.py --load-in-8bit
# Expected: ~12-16GB VRAM
```

### Conservative (for older/thermal throttled cards)
```bash
python run_inference.py --load-in-8bit --min-frames 8 --max-frames 24
# Expected: ~10-14GB VRAM
```

## Flash Attention

Flash Attention is **enabled by default** and provides:
- ~30% memory reduction
- Faster inference
- Same accuracy

To disable (not recommended on 24GB):
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
- **Quantization**: 8-bit/4-bit options for extra safety
- **Adaptive Frames**: Automatically adjusts 8-32 frames based on content
- **Support Frames**: Uses temporal anchoring for better context
- **Multilingual**: Handles Vietnamese, English, and Chinese responses
- **Progress Tracking**: Real-time progress bar during inference
- **VRAM Estimation**: Shows estimated VRAM before starting

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors on 24GB VRAM:

1. **Use 8-bit quantization** (recommended):
   ```bash
   python run_inference.py --load-in-8bit
   ```

2. **Reduce frame count**:
   ```bash
   python run_inference.py --max-frames 24
   ```

3. **Reduce tiles per frame**:
   ```bash
   python run_inference.py --max-num 8
   ```

4. **Combine all optimizations**:
   ```bash
   python run_inference.py --load-in-8bit --max-frames 24 --max-num 8
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

| Model | Flash Attn | VRAM (default) | VRAM (8-bit) | Accuracy |
|-------|-----------|---------------|-------------|----------|
| InternVL3.5-8B | ✓ Yes | ~18-22GB | ~12-16GB | Highest |
| InternVL3-8B | ✗ No | ~20-24GB | ~14-18GB | High |
| InternVideo2.5-8B | ✗ No | ~22-26GB | ~15-19GB | High |

**InternVL3.5-8B is the most memory-efficient option for 24GB VRAM.**

## Performance Tips

1. **Use default settings first** - optimized for 24GB
2. **Add --load-in-8bit if OOM** - minimal quality loss
3. **Keep Flash Attention enabled** - free 30% memory savings
4. **Monitor VRAM usage**: `nvidia-smi -l 1`
5. **Close other GPU applications** before running

## Files

- `inference.py`: Main inference pipeline with Flash Attention
- `run_inference.py`: Command-line interface with VRAM estimation
- `README.md`: This documentation file

## Credits

Based on the official InternVL3.5 implementation from OpenGVLab with optimizations for 24GB VRAM usage.

## Support

For issues related to:
- **Model**: Check https://huggingface.co/OpenGVLab/InternVL3_5-8B
- **Flash Attention**: Check https://github.com/Dao-AILab/flash-attention
- **24GB VRAM optimization**: Refer to VRAM Usage Guide above

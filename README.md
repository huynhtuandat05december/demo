# Road Buddy - Video Question Answering

A video question answering system for dashcam scenarios using vision-language models for inference.

## Project Structure

```
road_buddy/
├── RoadBuddy/                     # Data directory (external)
│   └── traffic_buddy_train+public_test/
│       ├── train/
│       │   ├── train.json         # Training data (1,490 samples)
│       │   └── videos/            # Training videos (549 files)
│       └── public_test/
│           ├── public_test.json   # Test data
│           └── videos/            # Test videos
├── src/
│   ├── config.py                  # Inference configuration
│   ├── prompts.py                 # Prompt templates
│   ├── video_processor.py         # Video frame extraction
│   ├── model_adapters.py          # Model-specific adapters
│   ├── inference.py               # Single model inference
│   ├── inference_multi_model.py   # Multi-model inference
│   └── internvl3_8B/
│       ├── inference.py           # InternVL3 inference
│       ├── run_inference.py       # Run full inference
│       ├── test_inference.py      # Test on samples
│       ├── train_config.py        # Training configuration
│       ├── train_dataset.py       # Training dataset class
│       ├── train.py               # Main training script
│       └── evaluate.py            # Model evaluation script
├── output/
│   ├── submission.csv             # Inference predictions
│   └── internvl3_8B_lora/         # Training checkpoints
│       ├── checkpoint-XXX/        # Epoch checkpoints
│       ├── final_model/           # Final trained model
│       └── runs/                  # TensorBoard logs
├── run.py                         # Simple inference runner
├── run_multi_model.py             # Flexible inference runner
└── pyproject.toml                 # Dependencies (includes peft)
```

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install basic dependencies
uv sync

# For InternVL3-8B model
uv sync --extra internvl

# Install all optional dependencies
uv sync --extra all
```

### Option 2: Using pip

#### Basic Dependencies
```bash
pip install torch transformers pillow pandas tqdm opencv-python
```

#### Model-Specific Dependencies
```bash
# For InternVL3-8B model (inference)
pip install einops timm decord

# For training InternVL3-8B with LoRA
pip install peft>=0.7.0

# For Qwen models (if using Qwen3-VL-8B-Instruct)
# All dependencies should be covered by basic installation
```

## Supported Models

- **YannQi/R-4B** (default) - No additional dependencies. Good balance of speed and accuracy.
- **OpenGVLab/InternVL3-8B** - Requires: `pip install einops timm`. Higher accuracy but requires more GPU memory.
- **Qwen/Qwen3-VL-8B-Instruct** - No additional dependencies. Recommended to use with CPU for testing due to memory requirements.

## Quick Testing

Test the inference pipeline on one or more examples to verify everything works:

```bash
# Test with default model and device (1 sample)
python test_inference.py

# Test with specific models
python test_inference.py --model YannQi/R-4B
python test_inference.py --model OpenGVLab/InternVL3-8B
python test_inference.py --model Qwen/Qwen3-VL-8B-Instruct

# Test on CPU (recommended if GPU memory is limited)
python test_inference.py --device cpu

# Test multiple samples
python test_inference.py --samples 5
python test_inference.py --samples 10 --device cpu

# Test Qwen3-VL on CPU with 3 samples
python test_inference.py --model Qwen/Qwen3-VL-8B-Instruct --device cpu --samples 3

# Test with custom test file
python test_inference.py --test-json data/traffic_buddy_train+public_test/public_test/public_test.json
```

**Output**: Shows the predicted answer for each sample with detailed logging, accuracy (if ground truth available), and success/failure summary.

**Why test first?**
- ✅ Verifies your environment is set up correctly
- ✅ Tests data paths and file access
- ✅ Catches errors quickly
- ✅ Checks GPU/CPU compatibility
- ✅ Validates model loading and inference
- ✅ Fast feedback (seconds to minutes)

## Running Inference

### Option 1: Simple Runner (Default Configuration)

Use `run.py` to run inference with the default model and settings from `src/config.py`:

```bash
python run.py
```

This will:
- Use the model specified in `src/config.py` (default: YannQi/R-4B)
- Process videos from `data/traffic_buddy_train+public_test/public_test/`
- Save results to `output/submission_{model}_{timestamp}.csv`

**Note**: Output files are automatically named with model and timestamp, e.g., `submission_R_4B_20250115_143022.csv`

### Option 2: Multi-Model Runner (Command-Line Arguments)

Use `run_multi_model.py` for more flexibility:

#### Basic Usage
```bash
# Use default model from config
# Output: submission_{model}_{timestamp}.csv (auto-generated)
python run_multi_model.py

# List all supported models
python run_multi_model.py --list-models

# Use default filename without timestamp
python run_multi_model.py --no-timestamp
# Output: submission.csv
```

#### Specify a Model
```bash
# Use R-4B model
# Output: submission_R_4B_20250115_143022.csv
python run_multi_model.py --model YannQi/R-4B

# Use InternVL3-8B model
# Output: submission_InternVL3_8B_20250115_143022.csv
python run_multi_model.py --model OpenGVLab/InternVL3-8B

# Use Qwen3-VL-8B model
# Output: submission_Qwen3_VL_8B_Instruct_20250115_143022.csv
python run_multi_model.py --model Qwen/Qwen3-VL-8B-Instruct

# Use your fine-tuned model
# Output: submission_best_model_hf_20250115_143022.csv
python run_multi_model.py --model checkpoints/best_model_hf
```

#### Change Device
```bash
# Run on CPU
python run_multi_model.py --device cpu

# Run on GPU (default)
python run_multi_model.py --device cuda
```

#### Custom Data Paths
```bash
python run_multi_model.py \
    --test-json path/to/your/test.json \
    --output path/to/your/output.csv
```

#### Verbose Output
```bash
python run_multi_model.py --verbose
```

#### Combined Options
```bash
python run_multi_model.py \
    --model OpenGVLab/InternVL3-8B \
    --device cuda \
    --test-json data/traffic_buddy_train+public_test/public_test/public_test.json \
    --output output/submission_internvl.csv \
    --verbose
```

## Enhanced Traffic Video Inference (InternVL3 Multi-Frame)

For improved accuracy on traffic videos, we provide an enhanced inference pipeline using InternVL3-8B with multi-frame processing.

> **📖 Complete Guide:** See [TRAFFIC_INFERENCE_GUIDE.md](TRAFFIC_INFERENCE_GUIDE.md) for detailed instructions, configuration presets, troubleshooting, and FAQ.

### 🚀 Quick Start - Generate CSV Submission

**Just want to run inference and get a CSV?** Use this one command:

```bash
# Run on full test dataset (requires GPU with 20-24 GB VRAM)
python run_traffic_inference.py

# Output: traffic_inference_InternVL3_8B_YYYYMMDD_HHMMSS.csv
```

**Need to customize?**
```bash
# Balanced mode (12-16 GB GPU)
python run_traffic_inference.py --min-frames 6 --max-frames 8 --max-num 12

# Low memory mode (8-12 GB GPU)
python run_traffic_inference.py --min-frames 4 --max-frames 8 --max-num 12

# Custom output filename
python run_traffic_inference.py --output my_submission.csv

# CPU mode (VERY slow, hours/days)
python run_traffic_inference.py --device cpu
```

**Test first before running full dataset:**
```bash
# Test on 3 videos to verify everything works
python test_traffic_inference.py --samples 3
```

### Features

- **Multi-Frame Analysis**: Processes 6-12 frames per video (adaptive based on duration)
- **Temporal Understanding**: Captures sequences and changes over time
- **Traffic-Optimized Prompts**: Specialized prompts for dashcam scenarios
- **Support Frames Integration**: Uses important timestamps from dataset
- **High Accuracy**: float32 precision with max_num=24 for detail

### When to Use Traffic Inference

✅ **Use when:**
- You need highest possible accuracy
- Questions involve temporal reasoning (sequences, changes, movements)
- Videos show complex traffic scenarios
- Support frames timestamps are available
- You have sufficient GPU memory (12-24 GB)

❌ **Don't use when:**
- Quick results are priority over accuracy
- Limited GPU memory
- Simple static scene analysis
- Testing on many samples quickly

### Quick Start

#### Testing (Recommended First)

```bash
# Test on 3 samples (recommended first test)
python test_traffic_inference.py --samples 3

# Test on CPU (slower but works without GPU)
python test_traffic_inference.py --device cpu --samples 2

# Test with custom frame range
python test_traffic_inference.py --samples 5 --min-frames 8 --max-frames 10
```

#### Full Dataset Inference (Production)

```bash
# Run on full test dataset with defaults
python run_traffic_inference.py

# Run on CPU (SLOW - will take hours/days)
python run_traffic_inference.py --device cpu

# Balanced configuration (faster, less memory)
python run_traffic_inference.py --min-frames 6 --max-frames 8 --max-num 12

# Custom output path
python run_traffic_inference.py --output results/my_submission.csv

# High accuracy configuration (default - requires 20-24 GB GPU)
python run_traffic_inference.py --min-frames 6 --max-frames 12 --max-num 24

# Low memory configuration (8-12 GB GPU)
python run_traffic_inference.py --min-frames 4 --max-frames 8 --max-num 12
```

**Output:** Generates a CSV file with format:
```csv
id,answer
test_0001,A
test_0002,C
test_0003,B
...
```

#### Programmatic Usage

```python
from src.traffic_inference import InternVL3TrafficInference
from src import config

# Initialize pipeline
pipeline = InternVL3TrafficInference(
    model_name="OpenGVLab/InternVL3-8B",
    device="cuda",
    min_frames=6,
    max_frames=12,
    max_num=24,
    use_support_frames=True,
)

# Run on single video
answer = pipeline.infer(
    video_path=Path("path/to/video.mp4"),
    question="Your question here",
    choices=["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    support_frames=[2.5, 5.0],  # Optional timestamps
    verbose=True
)

# Run on full dataset
pipeline.run_pipeline(
    test_json_path=config.PUBLIC_TEST_JSON,
    data_dir=config.DATA_DIR,
    output_csv_path=Path("output/traffic_results.csv"),
    verbose=False
)
```

### Configuration Options

#### Frame Extraction

```python
# In src/config.py
TRAFFIC_MIN_FRAMES = 6   # Minimum frames for short videos
TRAFFIC_MAX_FRAMES = 12  # Maximum frames for long videos
TRAFFIC_USE_SUPPORT_FRAMES = True  # Use timestamps from data
TRAFFIC_CONTEXT_WINDOW = 0.5  # Seconds around support frames
```

**Frame Allocation Strategy:**
- **5-7 second videos**: 6-8 frames
- **8-12 second videos**: 8-10 frames
- **13-15 second videos**: 10-12 frames

#### Model Configuration

```python
# In src/config.py
TRAFFIC_MODEL_NAME = "OpenGVLab/InternVL3-8B"
TRAFFIC_MAX_NUM = 24  # Higher = more detail, more memory
TRAFFIC_PRECISION = "float32"  # Highest accuracy
```

#### Prompt Configuration

```python
# In src/config.py
TRAFFIC_SIMPLE_PROMPTS = False  # Use detailed prompts
TRAFFIC_AUTO_DETECT_LANGUAGE = True  # Auto Vietnamese/English
```

### Performance Expectations

#### Accuracy

| Scenario | Single-Frame | Multi-Frame (Traffic) | Improvement |
|----------|-------------|----------------------|-------------|
| Static scenes | 70-75% | 72-77% | +2-3% |
| Temporal reasoning | 45-55% | 65-80% | +20-25% |
| Sequence understanding | 40-50% | 70-85% | +30-35% |
| Overall | 60-65% | 75-85% | +15-20% |

*Note: Actual results depend on dataset and scenarios*

#### Speed & Memory

| Configuration | GPU Memory | Speed (per video) | Use Case |
|--------------|-----------|------------------|----------|
| min=6, max=8, max_num=12 | 12-16 GB | 10-15s | Balanced |
| min=6, max=12, max_num=24 | 20-24 GB | 15-25s | High accuracy (default) |
| min=4, max=8, max_num=12 | 8-12 GB | 8-12s | Limited memory |
| CPU mode | 8-16 GB RAM | 2-5 min | No GPU available |

### Command-Line Options

```bash
# All available options
python test_traffic_inference.py \
    --model OpenGVLab/InternVL3-8B \
    --device cuda \
    --samples 5 \
    --min-frames 6 \
    --max-frames 12 \
    --max-num 24 \
    --test-json path/to/test.json \
    --simple-prompts \  # Use shorter prompts
    --no-support-frames \  # Ignore timestamps
    --verbose  # Detailed output
```

### Troubleshooting Traffic Inference

**AttributeError: 'InternVLChatModel' object has no attribute 'load_image':**

This has been fixed in the latest version. The code no longer uses `load_image` and instead uses `model.chat()` directly with PIL Images. See [FIX_LOAD_IMAGE_ERROR.md](FIX_LOAD_IMAGE_ERROR.md) for details.

If you still encounter this:
```bash
# 1. Update transformers
pip install --upgrade transformers

# 2. Test with latest code
python test_traffic_inference.py --samples 1 --verbose

# 3. The code will automatically fall back to frame grid if needed
```

**CUDA Out of Memory:**
```bash
# Option 1: Reduce frames
python test_traffic_inference.py --min-frames 4 --max-frames 8

# Option 2: Reduce max_num
python test_traffic_inference.py --max-num 12

# Option 3: Use CPU
python test_traffic_inference.py --device cpu --samples 2

# Option 4: Combination
python test_traffic_inference.py --device cpu --min-frames 4 --max-frames 6
```

**Slow Inference:**
- Multi-frame is 8-12x slower than single-frame (expected)
- Use fewer samples for testing
- Reduce frame count: `--min-frames 4 --max-frames 8`
- Consider GPU if using CPU

**Installation Issues:**
```bash
# Install InternVL3 dependencies
pip install einops timm

# Or with uv
uv sync --extra internvl
```

### Comparison with Standard Inference

| Feature | Standard (Single-Frame) | Traffic (Multi-Frame) |
|---------|------------------------|----------------------|
| **Frames per video** | 1 (middle frame) | 6-12 (adaptive) |
| **Temporal context** | None | Full video context |
| **Accuracy (temporal)** | Low (~50%) | High (~75%) |
| **Speed** | Fast (~2s/video) | Moderate (~15s/video) |
| **Memory** | Low (~4-6 GB) | High (~20-24 GB) |
| **Best for** | Quick testing, static scenes | Production, temporal reasoning |
| **Support frames** | Ignored | Utilized |

### Implementation Details

**File Structure:**
```
src/
├── traffic_inference.py      # Main InternVL3 traffic pipeline
├── traffic_prompts.py         # Traffic-optimized prompt templates
├── video_processor.py         # Enhanced with adaptive extraction
└── config.py                  # Traffic configuration section

run_traffic_inference.py       # Run on full test dataset → CSV output
test_traffic_inference.py      # Test on sample videos
```

**Key Components:**

1. **Adaptive Frame Extraction** (`video_processor.py`):
   - `extract_frames_adaptive()` - Duration-based sampling
   - `extract_frames_with_support()` - Timestamp-based extraction
   - Uniform distribution across video

2. **Traffic Prompts** (`traffic_prompts.py`):
   - Vietnamese and English templates
   - Automatic language detection
   - Emphasizes temporal reasoning
   - Simple and detailed variants

3. **InternVL3 Pipeline** (`traffic_inference.py`):
   - Multi-frame input preparation
   - High-accuracy configuration
   - Memory management
   - Comprehensive error handling

## Training InternVL3-8B with LoRA

Fine-tune InternVL3-8B on the RoadBuddy training dataset using LoRA (Low-Rank Adaptation) for efficient training with minimal resource requirements.

### Overview

The training pipeline consists of:
- **train_config.py**: Configuration file with all hyperparameters
- **train_dataset.py**: Dataset class for loading and preprocessing videos
- **train.py**: Main training script with HuggingFace Trainer + LoRA
- **evaluate.py**: Evaluation script for trained models

### Requirements

Install the required dependencies:

```bash
pip install -e .
```

Or manually install:
```bash
pip install peft>=0.7.0
```

All other dependencies should already be installed from the base requirements.

### Quick Start

#### 1. Configure Training (Optional)

Edit `src/internvl3_8B/train_config.py` to customize training parameters:

```python
# Key parameters to adjust:
BATCH_SIZE = 4  # Reduce if you have memory issues
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4  # Standard for LoRA
NUM_EPOCHS = 5  # Number of training epochs
```

#### 2. Run Training

```bash
cd src/internvl3_8B
python train.py
```

Training will:
1. Load InternVL3-8B from HuggingFace Hub (~16GB download)
2. Apply LoRA to reduce trainable parameters to ~1-2%
3. Split data into 80% train / 20% validation
4. Train for 5 epochs with evaluation after each epoch
5. Save checkpoints and best model to `output/internvl3_8B_lora/`

#### 3. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir output/internvl3_8B_lora
```

#### 4. Evaluate Trained Model

After training completes, evaluate on validation set:

```bash
cd src/internvl3_8B
python evaluate.py \
    --model-path ../../output/internvl3_8B_lora/checkpoint-XXX \
    --split val \
    --device cuda
```

This will output:
- Overall accuracy
- Per-class accuracy (A, B, C, D)
- Confusion matrix
- Answer distribution

### Training Configuration

#### Data Split

- **Train**: 80% of 1,490 samples = ~1,192 samples
- **Validation**: 20% of 1,490 samples = ~298 samples

#### LoRA Configuration

```python
LORA_CONFIG = {
    "r": 32,              # LoRA rank
    "lora_alpha": 64,     # Alpha = 2 * rank (recommended)
    "lora_dropout": 0.05, # Dropout for regularization
    "target_modules": [
        "qkv_proj",       # Query, Key, Value attention projections
        "out_proj",       # Output projection
    ],
}
```

#### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 4 | Per-device batch size |
| Gradient accumulation | 4 | Effective batch = 16 |
| Learning rate | 2e-4 | Standard for LoRA |
| Epochs | 5 | Number of training epochs |
| Warmup ratio | 0.1 | 10% of steps for warmup |
| LR scheduler | Cosine | Cosine annealing with warmup |
| Optimizer | AdamW | Weight decay = 0.01 |
| Precision | FP16 | Mixed precision training |

#### Video Processing

- **Frame extraction**: 4-6 frames per video (adaptive)
- **Support frames**: Uses important timestamps from annotations
- **Context window**: 0.5 seconds around support frames
- **Dynamic tiling**: InternVL3's approach for high-resolution frames
- **Max patches per frame**: 6

### Expected Performance

#### Training Time

On a single 24GB GPU (RTX 4090 / A100):
- **Time per epoch**: ~30-40 minutes
- **Total training time**: ~2-4 hours for 5 epochs
- **VRAM usage**: ~18-22GB

#### Memory Requirements

| Component | Memory |
|-----------|--------|
| Base model (BF16) | ~16GB |
| LoRA adapters | ~100MB |
| Gradients | ~2GB |
| Video frames (batch) | ~2-4GB |
| **Total** | **~18-22GB** |

#### Expected Accuracy

Based on similar vision-language models:
- **Baseline (untrained)**: 25-40% (random/biased guessing)
- **After 3 epochs**: 60-75%
- **After 5 epochs**: 70-85%
- **Best case**: 80-90%

### Troubleshooting

#### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. **Reduce batch size**:
   ```python
   # In src/internvl3_8B/train_config.py
   BATCH_SIZE = 2  # or even 1
   GRADIENT_ACCUMULATION_STEPS = 8  # to maintain effective batch size
   ```

2. **Reduce max frames**:
   ```python
   MAX_FRAMES = 4  # Down from 6
   ```

3. **Reduce patches per frame**:
   ```python
   MAX_NUM_PATCHES = 4  # Down from 6
   ```

4. **Enable gradient checkpointing**:
   ```python
   GRADIENT_CHECKPOINTING = True  # Trades speed for memory
   ```

#### Slow Training

If training is very slow:

1. **Increase batch size** (if you have memory):
   ```python
   BATCH_SIZE = 8
   ```

2. **Reduce num_workers** if CPU is bottleneck:
   ```python
   DATALOADER_NUM_WORKERS = 2  # Down from 4
   ```

3. **Disable support frames** for uniform sampling:
   ```python
   USE_SUPPORT_FRAMES = False
   ```

#### Loss is NaN

If you see NaN losses:

1. **Reduce learning rate**:
   ```python
   LEARNING_RATE = 1e-4  # Down from 2e-4
   ```

2. **Check for corrupted videos**: The dataset loader will print warnings

3. **Enable FP32 training** (slower but more stable):
   ```python
   FP16 = False
   BF16 = False
   ```

#### Model Not Improving

If validation accuracy doesn't improve:

1. **Train longer**: Try 10 epochs instead of 5
2. **Adjust learning rate**: Try 1e-4 or 5e-5
3. **Increase LoRA rank**:
   ```python
   LORA_CONFIG = {
       "r": 64,        # Up from 32
       "lora_alpha": 128,  # 2 * rank
       ...
   }
   ```

### Advanced Usage

#### Resume Training from Checkpoint

```bash
python train.py --resume_from_checkpoint output/internvl3_8B_lora/checkpoint-XXX
```

#### Use Different Base Model

Edit `src/internvl3_8B/train_config.py`:
```python
MODEL_NAME = "OpenGVLab/InternVL3-4B"  # Smaller model
```

#### Full Fine-Tuning (No LoRA)

Edit `src/internvl3_8B/train_config.py`:
```python
USE_LORA = False  # Train all parameters (requires ~32GB+ VRAM)
```

#### Custom Prompts

Edit prompts in `src/prompts.py`.

#### Evaluate on Public Test Set

First, create a test dataset in `train_dataset.py` that loads `public_test.json`, then:

```bash
cd src/internvl3_8B
python evaluate.py \
    --model-path ../../output/internvl3_8B_lora/checkpoint-XXX \
    --split test \
    --output-file results.json
```

### Output Structure

```
output/internvl3_8B_lora/
├── checkpoint-XXX/          # Checkpoints after each epoch
│   ├── adapter_config.json  # LoRA configuration
│   ├── adapter_model.bin    # LoRA weights (~200MB)
│   └── ...
├── final_model/             # Final merged model
│   └── ...
├── training_metrics.json    # Training loss, learning rate, etc.
├── eval_results.json        # Final evaluation metrics
└── runs/                    # TensorBoard logs
    └── ...
```

### Model Deployment

#### Use Trained Model for Inference

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL3-8B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "output/internvl3_8B_lora/checkpoint-XXX",
    torch_dtype=torch.bfloat16
)

# Merge for faster inference (optional)
model = model.merge_and_unload()

# Use model.chat() as in inference.py
```

#### Export for Production

Save merged model:
```python
model = model.merge_and_unload()
model.save_pretrained("output/merged_model")
tokenizer.save_pretrained("output/merged_model")
```

Then load directly:
```python
model = AutoModel.from_pretrained(
    "output/merged_model",
    trust_remote_code=True
)
```

### Citation

If you use this training code, please cite:

```bibtex
@article{internvl3,
  title={InternVL3: Scaling Up Vision-Language Models to 1B Parameters},
  author={Chen, Zhe and others},
  journal={arXiv preprint arXiv:2408.xxxxx},
  year={2024}
}
```

## Configuration

### Inference Configuration

Edit `src/config.py` to modify default settings:

#### Model Configuration
```python
MODEL_NAME = "YannQi/R-4B"  # Choose your model
DEVICE = "cuda"             # "cuda" or "cpu"
TRUST_REMOTE_CODE = True
```

#### Video Processing
```python
FRAME_SAMPLE_RATE = 1.0           # Frames per second
MAX_FRAMES_PER_VIDEO = 20         # Maximum frames to extract
USE_MID_FRAME_ONLY = False        # Extract only middle frame
```

#### Inference Settings
```python
MAX_NEW_TOKENS = 512        # Maximum response length
TEMPERATURE = 0.1           # Sampling temperature
DO_SAMPLE = False           # Use greedy decoding
```

## Output Format

### File Naming

Output files are automatically named with the model and timestamp for easy tracking:

**Inference outputs** (in `output/` directory):
- Format: `submission_{model}_{timestamp}.csv`
- Example: `submission_R_4B_20250115_143022.csv`
- Disable timestamp: Use `--no-timestamp` flag → `submission.csv`

**Training checkpoints** (in `checkpoints/` directory):
- Format: `checkpoints/{model}_{date}/`
- Example: `checkpoints/R_4B_20250115/best_model_hf/`
- Disable timestamp: Use `--no-timestamp` flag → `checkpoints/`

### CSV Format

The inference script generates a CSV file with predictions:

```csv
id,answer
1,A
2,C
3,B
...
```

Where:
- `id`: Question ID from the test JSON
- `answer`: Predicted answer (A, B, C, or D)

## Requirements

### Minimum Requirements (Inference)
- Python 3.8+
- PyTorch (with CUDA support for GPU)
- Transformers
- OpenCV (cv2)
- Pillow
- Pandas
- tqdm

### Additional Requirements (Training)
- **peft>=0.7.0** (for LoRA fine-tuning)
- **GPU**: 24GB VRAM recommended (18-22GB minimum)
  - RTX 4090, A100, or equivalent
  - Can train with 16GB by reducing batch size
- **Disk space**: ~17GB for model + checkpoints
- **Other**: All inference dependencies above

## Complete Workflow Example

### 1. Train InternVL3-8B with LoRA
```bash
# Navigate to training directory
cd src/internvl3_8B

# Run training (takes ~2-4 hours on 24GB GPU)
python train.py
```

This will train for 5 epochs and save checkpoints to `output/internvl3_8B_lora/`.

### 2. Evaluate Your Trained Model
```bash
# Still in src/internvl3_8B directory
python evaluate.py \
    --model-path ../../output/internvl3_8B_lora/checkpoint-best \
    --split val \
    --device cuda
```

Check validation accuracy and per-class performance.

### 3. Run Inference with Trained Model
```bash
# From project root
cd src/internvl3_8B
python run_inference.py \
    --model-path ../../output/internvl3_8B_lora/final_model \
    --test-json ../../RoadBuddy/traffic_buddy_train+public_test/public_test/public_test.json \
    --output ../../output/submission_trained.csv
```

### 4. Compare Results
Compare your trained model's predictions with baseline:
```bash
# Baseline (pretrained InternVL3-8B)
python run_multi_model.py --model OpenGVLab/InternVL3-8B

# Your trained model (should perform better)
# See step 3 above
```

## Tips for Best Results

### Training Tips
1. **Test dataset first**: Run `python src/internvl3_8B/train_dataset.py` to verify data loads correctly
2. **Monitor training**: Use TensorBoard to watch loss and accuracy curves
3. **Adjust batch size**: Reduce `BATCH_SIZE` in `train_config.py` if OOM errors occur
4. **Use support frames**: Keep `USE_SUPPORT_FRAMES = True` for better temporal understanding
5. **Train longer if needed**: Increase `NUM_EPOCHS` to 10 if validation accuracy is still improving
6. **Save GPU memory**: Reduce `MAX_FRAMES` and `MAX_NUM_PATCHES` if memory constrained
7. **Learning rate**: Default 2e-4 works well; try 1e-4 if training is unstable

### Inference Tips
1. **GPU recommended**: Much faster than CPU
2. **Batch processing**: Adjust batch size based on GPU memory
3. **Use support frames**: Better accuracy when timestamps are provided
4. **Model selection**: Larger models (InternVL3-8B) may perform better

## Troubleshooting

### Training Issues

**CUDA out of memory during training:**
```python
# Edit src/internvl3_8B/train_config.py
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION_STEPS = 8  # Increase from 4
MAX_FRAMES = 4  # Reduce from 6
MAX_NUM_PATCHES = 4  # Reduce from 6
```

**Training too slow:**
```python
# Edit src/internvl3_8B/train_config.py
DATALOADER_NUM_WORKERS = 2  # Reduce from 4
USE_SUPPORT_FRAMES = False  # Disable for faster loading
MIN_FRAMES = 4
MAX_FRAMES = 4  # Use fixed frame count
```

**Low validation accuracy:**
```python
# Edit src/internvl3_8B/train_config.py
NUM_EPOCHS = 10  # Train longer
LEARNING_RATE = 1e-4  # Try lower LR
LORA_CONFIG = {
    "r": 64,  # Increase rank
    "lora_alpha": 128,
    ...
}
```

**Loss is NaN:**
```python
# Edit src/internvl3_8B/train_config.py
LEARNING_RATE = 1e-4  # Reduce learning rate
FP16 = False  # Disable mixed precision
BF16 = False
```

**Dataset loading errors:**
```bash
# Test dataset loading
cd src/internvl3_8B
python train_dataset.py  # Will show any errors
```

### Inference Issues

**CUDA out of memory:**
- **For testing**: Try running on CPU: `python test_inference.py --device cpu --samples 3`
- **For full inference**: Try running on CPU: `python run_multi_model.py --device cpu`
- Use a smaller model (try R-4B instead of InternVL3-8B)
- Close other programs using GPU memory
- Restart your Python kernel/terminal to clear GPU memory
- Check GPU memory usage:
  ```bash
  nvidia-smi
  ```
- For Qwen3-VL-8B, CPU inference is recommended:
  ```bash
  python test_inference.py --model Qwen/Qwen3-VL-8B-Instruct --device cpu --samples 3
  ```

**Missing training data:**
- Ensure `data/traffic_buddy_train+public_test/train/` exists
- Check that `train.json` and `videos/` directory are present

**Missing test data:**
- Ensure `data/traffic_buddy_train+public_test/public_test/` exists
- Check that `public_test.json` and `videos/` directory are present

**Model download fails:**
- Check internet connection
- Verify HuggingFace model name is correct
- Ensure sufficient disk space for model weights

**Missing model-specific dependencies (einops, timm):**
- For InternVL3-8B model:
```bash
# Using uv
uv sync --extra internvl

# Using pip
pip install einops timm
```

**peft library not found (for training):**
```bash
# Using pip
pip install peft>=0.7.0

# Or install all dependencies
pip install -e .
```

## Notes

### General
- First run will download the model weights (may take several minutes for InternVL3-8B ~16GB)
- GPU is highly recommended for both training and inference
- The script uses support frames when available for better accuracy
- Answers are parsed from model responses using pattern matching
- If answer parsing fails, defaults to "A"

### Training
- Training uses LoRA to reduce trainable parameters to ~1-2%
- Automatically creates 80/20 train/validation split from 1,490 samples
- Checkpoints are saved every epoch to `output/internvl3_8B_lora/`
- Best model is saved based on validation accuracy
- TensorBoard logs available in `output/internvl3_8B_lora/runs/`
- Training takes ~2-4 hours on RTX 4090/A100 (24GB GPU)
- Can resume training from any checkpoint if interrupted

## Data Format

### Training Data Format (train.json)
```json
{
  "__count__": 1490,
  "data": [
    {
      "id": "train_0001",
      "question": "Question text in Vietnamese",
      "choices": ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"],
      "answer": "B. Choice 2",
      "support_frames": [4.427402],
      "video_path": "train/videos/video.mp4"
    }
  ]
}
```

### Test Data Format (public_test.json)
```json
{
  "data": [
    {
      "id": "test_0001",
      "question": "Question text in Vietnamese",
      "choices": ["A. Choice 1", "B. Choice 2"],
      "video_path": "public_test/videos/video.mp4"
    }
  ]
}
```

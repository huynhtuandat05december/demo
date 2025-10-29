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
│   └── internvl3_8B/
│       ├── inference.py           # InternVL3 inference
│       ├── run_inference.py       # Run full inference
│       ├── test_inference.py      # Test on samples
│       ├── train_config.py        # Training configuration
│       ├── train_dataset.py       # Training dataset class
│       ├── train.py               # Main training script
│       ├── test_train.py          # Test training pipeline
│       ├── inspect_model.py       # Inspect model architecture
│       └── evaluate.py            # Model evaluation script
├── output/
│   ├── submission.csv             # Inference predictions
│   └── internvl3_8B_lora/         # Training checkpoints
│       ├── checkpoint-XXX/        # Epoch checkpoints
│       ├── final_model/           # Final trained model
│       └── runs/                  # TensorBoard logs
├── main.py                        # Main inference runner
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
```

## Quick Testing

Test the InternVL3 inference pipeline on sample videos:

```bash
cd src/internvl3_8B

# Test with default settings (1 sample)
python test_inference.py

# Test multiple samples
python test_inference.py --samples 5

# Test on CPU (if GPU memory is limited)
python test_inference.py --device cpu --samples 3

# Verbose output for debugging
python test_inference.py --samples 3 --verbose
```

**Output**: Shows the predicted answer for each sample with detailed logging.

## Running Inference

Use the InternVL3 inference scripts to process the test dataset:

```bash
cd src/internvl3_8B

# Run inference on full test dataset
python run_inference.py

# Run with custom output path
python run_inference.py --output ../../output/my_submission.csv

# Run on CPU (slower but works without GPU)
python run_inference.py --device cpu
```

**Output**: Generates a CSV file with format:
```csv
id,answer
test_0001,A
test_0002,C
test_0003,B
...
```

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

#### 1. Test Training Pipeline (Recommended)

Before running full training, test that everything works:

```bash
cd src/internvl3_8B

# Quick test (dataset only, no model loading)
python test_train.py --skip-model

# Full test (includes model loading and forward pass)
python test_train.py --samples 3

# Test with more samples
python test_train.py --samples 10
```

This will verify:
- ✅ Dataset loads correctly
- ✅ Data paths are valid
- ✅ Model can be initialized with LoRA
- ✅ Forward pass works
- ✅ GPU/CPU configuration is correct

**Fix any errors before proceeding to full training!**

#### 2. Configure Training (Optional)

Edit `src/internvl3_8B/train_config.py` to customize training parameters:

```python
# Key parameters to adjust:
BATCH_SIZE = 4  # Reduce if you have memory issues
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4  # Standard for LoRA
NUM_EPOCHS = 5  # Number of training epochs
```

#### 3. Run Training

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

#### 4. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir output/internvl3_8B_lora
```

#### 5. Evaluate Trained Model

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

Edit `src/config.py` or `src/internvl3_8B/train_config.py` to modify settings:

### Inference Configuration

```python
# In src/config.py
MODEL_NAME = "OpenGVLab/InternVL3-8B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.1
```

### Training Configuration

```python
# In src/internvl3_8B/train_config.py
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
MIN_FRAMES = 4
MAX_FRAMES = 6
```

## Output Format

The inference scripts generate CSV files with predictions:

```csv
id,answer
test_0001,A
test_0002,C
test_0003,B
...
```

- `id`: Question ID from the test JSON
- `answer`: Predicted answer (A, B, C, or D)

**Training outputs** are saved to `output/internvl3_8B_lora/`

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

## Troubleshooting

### Training Issues

**Before training, always run the test script first:**
```bash
cd src/internvl3_8B
python test_train.py --samples 5
```

This will catch configuration errors early!

**Target modules error (LoRA configuration):**
```bash
# If you see "Target modules not found" error:
cd src/internvl3_8B
python inspect_model.py

# This will show you the correct layer names for your model
# Then update train_config.py with the suggested target_modules
```

**Dataset loading errors:**
```bash
# Test dataset loading without model
cd src/internvl3_8B
python test_train.py --skip-model
```

**CUDA out of memory during training:**
```python
# Edit src/internvl3_8B/train_config.py
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION_STEPS = 8  # Increase to maintain effective batch size
MAX_FRAMES = 4  # Reduce from 6
MAX_NUM_PATCHES = 4  # Reduce from 6
```

**Training too slow:**
```python
# Edit src/internvl3_8B/train_config.py
DATALOADER_NUM_WORKERS = 2  # Reduce from 4
USE_SUPPORT_FRAMES = False  # Disable for faster loading
```

**Model not improving:**
- Train longer: `NUM_EPOCHS = 10`
- Adjust learning rate: `LEARNING_RATE = 1e-4`
- Check TensorBoard logs for loss curves
- Verify dataset quality with test_train.py

### Inference Issues

**CUDA out of memory:**
```bash
# For testing
cd src/internvl3_8B
python test_inference.py --device cpu --samples 3

# For full inference
cd src/internvl3_8B
python run_inference.py --device cpu
```
- Close other programs using GPU memory
- Restart your Python kernel/terminal to clear GPU memory
- Check GPU memory usage: `nvidia-smi`

**Missing training data:**
- Ensure `RoadBuddy/traffic_buddy_train+public_test/train/` exists
- Check that `train.json` and `videos/` directory are present

**Missing test data:**
- Ensure `RoadBuddy/traffic_buddy_train+public_test/public_test/` exists
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

```
uv add flash-attn --no-build-isolation-package flash-attn
```
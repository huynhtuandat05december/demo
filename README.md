# Road Buddy - Traffic Video Question Answering

A video question answering system for traffic scenarios using vision-language models with training and inference capabilities.

## Project Structure

```
road_buddy/
├── data/
│   └── traffic_buddy_train+public_test/
│       ├── train/
│       │   ├── train.json
│       │   └── videos/
│       └── public_test/
│           ├── public_test.json
│           └── videos/
├── src/
│   ├── config.py                  # Inference configuration
│   ├── train_config.py            # Training configuration
│   ├── dataset.py                 # Training dataset
│   ├── train.py                   # Training script
│   ├── inference.py               # Single model inference
│   ├── inference_multi_model.py   # Multi-model inference
│   ├── model_adapters.py          # Model-specific adapters
│   └── video_processor.py         # Video frame extraction
├── checkpoints/                   # Training checkpoints
│   └── best_model_hf/             # Best trained model
├── output/
│   └── submission.csv             # Inference predictions
├── train.py                       # Training runner
├── run.py                         # Simple inference runner
├── run_multi_model.py             # Flexible inference runner
├── test_inference.py              # Test inference on single example
└── test_train.py                  # Test training on small subset
```

## Installation

### Basic Dependencies
```bash
pip install torch transformers pillow pandas tqdm opencv-python
```

### For Training (Additional Dependencies)
```bash
# For LoRA-based fine-tuning (recommended)
pip install peft

# Optional: For advanced features
pip install accelerate bitsandbytes
```

## Supported Models

- **YannQi/R-4B** (default)
- **OpenGVLab/InternVL3-8B**
- **Qwen/Qwen3-VL-8B-Instruct**

## Quick Testing

Before running full training or inference, test your setup with these quick test scripts:

### Test Inference (Single Example)

Test the inference pipeline on a single example to verify everything works:

```bash
# Test with default model and device
python test_inference.py

# Test with specific model
python test_inference.py --model YannQi/R-4B

# Test on CPU
python test_inference.py --device cpu

# Test with custom test file
python test_inference.py --test-json data/traffic_buddy_train+public_test/public_test/public_test.json
```

**Output**: Shows the predicted answer for one example question with detailed logging.

### Test Training (Small Subset)

Test the training pipeline on a small subset (10 samples by default) to verify everything works:

```bash
# Test with default settings (10 samples, 1 epoch)
python test_train.py

# Test with custom number of samples
python test_train.py --samples 5

# Test with specific model
python test_train.py --model YannQi/R-4B

# Test on CPU
python test_train.py --device cpu

# Test without LoRA
python test_train.py --no-lora

# Test with more epochs
python test_train.py --samples 20 --epochs 2 --batch-size 2
```

**Output**: Trains on a small subset and saves a test checkpoint to `test_checkpoints/`.

**Why test first?**
- ✅ Verifies your environment is set up correctly
- ✅ Tests data paths and file access
- ✅ Catches errors quickly without waiting for full training
- ✅ Checks GPU/CPU compatibility
- ✅ Validates model loading and inference
- ✅ Fast feedback (seconds to minutes instead of hours)

## Training

Train a model on the traffic video QA dataset.

### Quick Start Training

```bash
# Basic training with default settings
# Checkpoints will be saved to: checkpoints/{model}_{date}/
python train.py

# Training with custom hyperparameters
python train.py --batch-size 4 --learning-rate 1e-5 --epochs 5

# Training with LoRA (parameter-efficient, recommended)
python train.py --use-lora --lora-r 8 --lora-alpha 16

# Debug mode (small dataset for testing)
python train.py --debug

# Custom output directory
python train.py --output-dir my_custom_checkpoints
```

**Note**: By default, checkpoints are saved to `checkpoints/{model_name}_{date}/`. For example, training R-4B on 2025-01-15 saves to `checkpoints/R_4B_20250115/`.

### Training Options

#### Model Selection
```bash
# Train specific model
python train.py --model YannQi/R-4B
python train.py --model OpenGVLab/InternVL3-8B
```

#### Hyperparameters
```bash
python train.py \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --warmup-ratio 0.1
```

#### LoRA Configuration
```bash
# Enable LoRA for efficient fine-tuning
python train.py --use-lora --lora-r 8 --lora-alpha 16

# Full fine-tuning (not recommended, requires more GPU memory)
python train.py --no-lora
```

#### Training Features
```bash
# Mixed precision training (faster, less memory)
python train.py --fp16

# Early stopping
python train.py --early-stopping --early-stopping-patience 3

# Custom output directory
python train.py --output-dir my_checkpoints

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_2.pt
```

#### Device Selection
```bash
# Use GPU (default)
python train.py --device cuda

# Use CPU (slower)
python train.py --device cpu
```

### Training Configuration

Edit `src/train_config.py` for advanced configuration:

```python
# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# LoRA settings (parameter-efficient fine-tuning)
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16

# Data split
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation

# Checkpointing
SAVE_EVERY_N_EPOCHS = 1
SAVE_BEST_MODEL = True

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
```

### Training Output

After training, you'll find:
- **Checkpoints**: Saved in `checkpoints/` directory
- **Best model**: `checkpoints/best_model_hf/` (HuggingFace format)
- **Training logs**: Console output with loss and accuracy

Example output:
```
Epoch 1/3 - Average Loss: 0.8234
Validation Accuracy: 0.7523 (75.23%)

✅ Best model saved to: checkpoints/best_model_hf
```

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

### Minimum Requirements
- Python 3.8+
- PyTorch (with CUDA support for GPU)
- Transformers
- OpenCV (cv2)
- Pillow
- Pandas
- tqdm

### For Training
- peft (for LoRA fine-tuning)
- At least 16GB GPU RAM for training (recommended)
- 8GB GPU RAM minimum with small batch sizes

## Complete Workflow Example

### 1. Train a Model
```bash
# Train with LoRA on GPU
python train.py \
    --model YannQi/R-4B \
    --use-lora \
    --batch-size 2 \
    --epochs 3 \
    --device cuda
```

### 2. Run Inference with Trained Model
```bash
# Use the trained model for predictions
python run_multi_model.py \
    --model checkpoints/best_model_hf \
    --test-json data/traffic_buddy_train+public_test/public_test/public_test.json \
    --output output/submission.csv
```

### 3. Evaluate Results
Check `output/submission.csv` for predictions.

## Tips for Best Results

### Training Tips
1. **Use LoRA**: Faster training, less memory, often better results
2. **Start small**: Use `--debug` mode to test your setup
3. **Monitor validation**: Watch for overfitting
4. **Adjust learning rate**: Try 1e-5 to 5e-5 for fine-tuning
5. **Use mixed precision**: `--fp16` for faster training

### Inference Tips
1. **GPU recommended**: Much faster than CPU
2. **Batch processing**: Adjust batch size based on GPU memory
3. **Use support frames**: Better accuracy when timestamps are provided
4. **Model selection**: Larger models (InternVL3-8B) may perform better

## Troubleshooting

### Training Issues

**CUDA out of memory during training:**
- Reduce batch size: `--batch-size 1`
- Increase gradient accumulation: `--gradient-accumulation-steps 8`
- Use LoRA: `--use-lora`
- Enable mixed precision: `--fp16`

**Training too slow:**
- Use LoRA instead of full fine-tuning
- Enable mixed precision: `--fp16`
- Increase batch size if GPU memory allows
- Reduce number of workers if CPU bottleneck

**Low validation accuracy:**
- Train for more epochs: `--epochs 5`
- Adjust learning rate: `--learning-rate 1e-5`
- Try different models
- Check if overfitting (train vs val accuracy gap)

### Inference Issues

**CUDA out of memory:**
- Try running on CPU: `python run_multi_model.py --device cpu`
- Use a smaller model
- Reduce batch size in config.py

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

**peft library not found:**
```bash
pip install peft
```

## Notes

- First run will download the model weights (may take several minutes)
- GPU is highly recommended for both training and inference
- The script uses support frames when available for better accuracy
- Training checkpoints are automatically saved
- Best model is saved in HuggingFace format for easy loading
- Answers are parsed from model responses using pattern matching
- If answer parsing fails, defaults to "A"

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

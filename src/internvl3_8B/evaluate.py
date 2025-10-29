"""
Evaluation script for trained InternVL3-8B model on RoadBuddy dataset.
"""
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

from train_dataset import TrafficVideoDataset, collate_fn
import train_config as cfg
from inference import InternVL3Inference


def load_trained_model(
    base_model_name: str,
    lora_checkpoint_path: Path,
    device: str = "cuda"
):
    """
    Load a model with trained LoRA weights.

    Args:
        base_model_name: Base model name (e.g., "OpenGVLab/InternVL3-8B")
        lora_checkpoint_path: Path to LoRA checkpoint directory
        device: Device to load model on

    Returns:
        (model, tokenizer) tuple
    """
    print("=" * 60)
    print("Loading Trained Model")
    print("=" * 60)

    print(f"\nBase model: {base_model_name}")
    print(f"LoRA checkpoint: {lora_checkpoint_path}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    print("✓ Tokenizer loaded")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("✓ Base model loaded")

    # Load LoRA weights
    print("\nLoading LoRA weights...")
    model = PeftModel.from_pretrained(
        base_model,
        str(lora_checkpoint_path),
        torch_dtype=torch.bfloat16,
    )
    print("✓ LoRA weights loaded")

    # Merge LoRA weights into base model for faster inference (optional)
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()
    print("✓ Weights merged")

    # Move to device
    model = model.to(device)
    model.eval()

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    print("=" * 60 + "\n")

    return model, tokenizer


def evaluate_dataset(
    model,
    tokenizer,
    dataset: TrafficVideoDataset,
    device: str = "cuda",
    max_samples: int = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Dataset to evaluate on
        device: Device for inference
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Print detailed information

    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("=" * 60 + "\n")

    # Determine number of samples
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    predictions = []
    ground_truths = []
    video_ids = []

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            try:
                # Get sample
                sample = dataset[idx]

                pixel_values = sample['pixel_values'].to(device)
                num_patches_list = sample['num_patches_list']
                question = sample['question']
                correct_answer = sample['answer']
                video_id = sample['video_id']

                # Match dtype
                pixel_values = pixel_values.to(torch.bfloat16)

                # Generate answer
                generation_config = {
                    'max_new_tokens': 10,
                    'do_sample': False,
                }

                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )

                # Parse answer
                predicted_answer = parse_answer(response)

                # Store results
                predictions.append(predicted_answer)
                ground_truths.append(correct_answer)
                video_ids.append(video_id)

                if verbose:
                    print(f"\nVideo ID: {video_id}")
                    print(f"Question: {question[:100]}...")
                    print(f"Response: {response}")
                    print(f"Predicted: {predicted_answer}, Ground truth: {correct_answer}")
                    print(f"Correct: {predicted_answer == correct_answer}")

            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Use default prediction on error
                predictions.append('A')
                ground_truths.append(sample['answer'])
                video_ids.append(sample['video_id'])

    # Compute metrics
    metrics = compute_detailed_metrics(predictions, ground_truths, video_ids)

    return metrics


def parse_answer(response: str) -> str:
    """Parse answer from model response."""
    import re

    response_clean = response.strip()
    valid_answers = ["A", "B", "C", "D"]

    # Try various patterns
    patterns = [
        r'^([ABCD])$',
        r'^([ABCD])[.\s]',
        r'\b([ABCD])\b',
        r'(?:đáp án|answer|chọn)[\s:：]*([ABCD])',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_clean, re.IGNORECASE)
        if matches:
            answer = matches[0].upper()
            if answer in valid_answers:
                return answer

    # Fallback: find first valid letter
    for char in response_clean.upper():
        if char in valid_answers:
            return char

    return "A"  # Default


def compute_detailed_metrics(
    predictions: List[str],
    ground_truths: List[str],
    video_ids: List[str]
) -> Dict:
    """
    Compute detailed evaluation metrics.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        video_ids: List of video IDs

    Returns:
        Dictionary with metrics
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60 + "\n")

    # Overall accuracy
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Per-class accuracy
    print("\nPer-Class Performance:")
    print("-" * 40)

    per_class = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred, gt in zip(predictions, ground_truths):
        per_class[gt]['total'] += 1
        if pred == gt:
            per_class[gt]['correct'] += 1

    for answer in sorted(per_class.keys()):
        stats = per_class[answer]
        class_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  Answer {answer}: {class_acc:.4f} ({stats['correct']}/{stats['total']})")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    print("Pred\\True |", end="")
    for answer in ['A', 'B', 'C', 'D']:
        print(f"  {answer:>4}", end="")
    print()
    print("-" * 40)

    confusion = defaultdict(lambda: defaultdict(int))
    for pred, gt in zip(predictions, ground_truths):
        confusion[pred][gt] += 1

    for pred_answer in ['A', 'B', 'C', 'D']:
        print(f"    {pred_answer}     |", end="")
        for true_answer in ['A', 'B', 'C', 'D']:
            count = confusion[pred_answer][true_answer]
            print(f"  {count:>4}", end="")
        print()

    # Answer distribution
    print("\nAnswer Distribution:")
    print("-" * 40)
    pred_dist = Counter(predictions)
    true_dist = Counter(ground_truths)

    print("Predicted:", dict(sorted(pred_dist.items())))
    print("Ground Truth:", dict(sorted(true_dist.items())))

    # Prepare metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_class': dict(per_class),
        'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
        'prediction_distribution': dict(pred_dist),
        'ground_truth_distribution': dict(true_dist),
    }

    print("\n" + "=" * 60 + "\n")

    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained InternVL3-8B model")

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (LoRA weights)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
        help="Base model name (default: OpenGVLab/InternVL3-8B)"
    )

    # Data arguments
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    # Inference arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each sample"
    )

    # Output arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_trained_model(
        base_model_name=args.base_model,
        lora_checkpoint_path=Path(args.model_path),
        device=args.device
    )

    # Create dataset
    print("Loading dataset...")
    dataset = TrafficVideoDataset(
        json_path=cfg.TRAIN_JSON,
        video_dir=cfg.TRAIN_VIDEO_DIR,
        split=args.split,
        train_val_split=cfg.TRAIN_VAL_SPLIT,
        random_seed=cfg.RANDOM_SEED,
        min_frames=cfg.MIN_FRAMES,
        max_frames=cfg.MAX_FRAMES,
        max_num_patches=cfg.MAX_NUM_PATCHES,
        use_support_frames=cfg.USE_SUPPORT_FRAMES,
        context_window=cfg.CONTEXT_WINDOW,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # Evaluate
    metrics = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=args.device,
        max_samples=args.max_samples,
        verbose=args.verbose
    )

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

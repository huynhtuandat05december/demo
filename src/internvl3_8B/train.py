"""
Training script for InternVL3-8B with LoRA on RoadBuddy traffic dataset.
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import evaluate

# Import local modules
from train_dataset import TrafficVideoDataset, collate_fn
import train_config as cfg


class VideoQATrainer(Trainer):
    """
    Custom Trainer for Video Question Answering with InternVL3.

    Handles the specific format required for video inputs and generates
    answers using the model's chat method.
    """

    def __init__(self, *args, **kwargs):
        """Initialize trainer with answer token IDs."""
        super().__init__(*args, **kwargs)

        # Get tokenizer to encode answer options
        self.answer_tokens = {
            'A': self.tokenizer.encode('A', add_special_tokens=False)[0],
            'B': self.tokenizer.encode('B', add_special_tokens=False)[0],
            'C': self.tokenizer.encode('C', add_special_tokens=False)[0],
            'D': self.tokenizer.encode('D', add_special_tokens=False)[0],
        }

        print(f"Answer token IDs: {self.answer_tokens}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for video QA task.

        For each sample, we:
        1. Format the prompt with video frames
        2. Tokenize the prompt + correct answer
        3. Compute cross-entropy loss on the answer tokens

        Args:
            model: The model being trained
            inputs: Batch of inputs from collate_fn
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items (used for gradient accumulation)

        Returns:
            loss or (loss, outputs) tuple
        """
        # Extract batch data
        pixel_values_list = inputs['pixel_values_list']
        num_patches_lists = inputs['num_patches_lists']
        questions = inputs['questions']
        answers = inputs['answers']

        batch_size = len(questions)
        total_loss = 0.0
        num_valid = 0

        # Process each sample individually (InternVL3 doesn't support batch inference)
        for i in range(batch_size):
            try:
                # Get sample data
                pixel_values = pixel_values_list[i].to(model.device)
                num_patches_list = num_patches_lists[i]
                question = questions[i]
                correct_answer = answers[i]

                # Match dtype with model
                if hasattr(model, 'dtype'):
                    pixel_values = pixel_values.to(model.dtype)
                else:
                    # Default to bfloat16 for training
                    pixel_values = pixel_values.to(torch.bfloat16)

                # Create full prompt with answer for teacher forcing
                # Format: "{question}\nAnswer: {answer}"
                full_prompt = f"{question}\nAnswer: {correct_answer}"

                # Tokenize
                # We need to create the input_ids and labels for causal LM
                inputs_ids = self.tokenizer(
                    question,
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids.to(model.device)

                # Tokenize the full sequence (prompt + answer)
                full_ids = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids.to(model.device)

                # Create labels: -100 for prompt tokens, actual token IDs for answer
                labels = full_ids.clone()
                labels[:, :inputs_ids.shape[1]] = -100  # Ignore loss on prompt

                # Forward pass through the model
                # InternVL3 model expects: pixel_values, input_ids, labels
                outputs = model(
                    pixel_values=pixel_values.unsqueeze(0),  # Add batch dim
                    input_ids=full_ids,
                    labels=labels,
                    return_dict=True
                )

                loss = outputs.loss

                if loss is not None and not torch.isnan(loss):
                    total_loss += loss
                    num_valid += 1
                else:
                    print(f"Warning: Invalid loss for sample {i}")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Average loss over valid samples
        if num_valid > 0:
            avg_loss = total_loss / num_valid
        else:
            print("Warning: No valid samples in batch, using zero loss")
            avg_loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        return (avg_loss, outputs) if return_outputs else avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform prediction step for evaluation.

        Args:
            model: The model
            inputs: Batch of inputs
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs

        Returns:
            (loss, logits, labels) tuple
        """
        # Extract batch data
        pixel_values_list = inputs['pixel_values_list']
        num_patches_lists = inputs['num_patches_lists']
        questions = inputs['questions']
        answers = inputs['answers']

        batch_size = len(questions)

        # For prediction, we generate answers and compare with ground truth
        predictions = []
        labels = []

        model.eval()
        with torch.no_grad():
            for i in range(batch_size):
                try:
                    pixel_values = pixel_values_list[i].to(model.device)
                    num_patches_list = num_patches_lists[i]
                    question = questions[i]
                    correct_answer = answers[i]

                    # Match dtype
                    if hasattr(model, 'dtype'):
                        pixel_values = pixel_values.to(model.dtype)
                    else:
                        pixel_values = pixel_values.to(torch.bfloat16)

                    # Generate answer using chat method
                    generation_config = {
                        'max_new_tokens': 10,
                        'do_sample': False,
                    }

                    # Get base model if PEFT model
                    base_model = model.model if hasattr(model, 'model') else model

                    response = base_model.chat(
                        tokenizer=self.tokenizer,
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False
                    )

                    # Parse answer (extract A/B/C/D)
                    predicted_answer = self._parse_answer(response)

                    predictions.append(predicted_answer)
                    labels.append(correct_answer)

                except Exception as e:
                    print(f"Error in prediction for sample {i}: {e}")
                    predictions.append('A')  # Default fallback
                    labels.append(answers[i])

        # Convert to format expected by Trainer
        # We'll use simple string matching for accuracy computation
        # Return None for loss and logits (not used in evaluation)
        return (None, predictions, labels)

    def _parse_answer(self, response: str) -> str:
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


def compute_metrics(eval_pred):
    """
    Compute accuracy metric for evaluation.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids

    Returns:
        Dictionary with accuracy metric
    """
    predictions, labels = eval_pred

    # predictions and labels are lists of strings (A/B/C/D)
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    total = len(predictions)

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
    }


def setup_model_and_tokenizer(config):
    """
    Load model and tokenizer, apply LoRA.

    Args:
        config: Training configuration module

    Returns:
        (model, tokenizer) tuple
    """
    print("=" * 60)
    print("Loading Model and Tokenizer")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        cache_dir=config.CACHE_DIR,
    )
    print("✓ Tokenizer loaded")

    # Load model
    print(f"\nLoading model: {config.MODEL_NAME}")
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16 if config.BF16 else torch.float16,
    }

    model = AutoModel.from_pretrained(
        config.MODEL_NAME,
        **model_kwargs
    )
    print("✓ Model loaded")

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    # Apply LoRA
    if config.USE_LORA:
        print("\nApplying LoRA...")
        print(f"LoRA config: {config.LORA_CONFIG}")

        lora_config = LoraConfig(**config.LORA_CONFIG)

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # Apply PEFT
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        print(f"✓ LoRA applied")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"All parameters: {all_params:,} ({all_params/1e9:.2f}B)")
        print(f"Trainable: {trainable_percent:.2f}%")
    else:
        print("\nSkipping LoRA (full fine-tuning)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")

    print("=" * 60)

    return model, tokenizer


def setup_training_args(config):
    """
    Create TrainingArguments from config.

    Args:
        config: Training configuration module

    Returns:
        TrainingArguments instance
    """
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        # Training
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        # Optimizer
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        # Precision
        fp16=config.FP16,
        bf16=config.BF16,
        # Evaluation
        evaluation_strategy=config.EVALUATION_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        # Logging
        logging_strategy=config.LOGGING_STRATEGY,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
        # System
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        seed=config.SEED,
        # Advanced
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        deepspeed=config.DEEPSPEED_CONFIG,
        # Disable unused features
        remove_unused_columns=False,  # Important: we have custom data format
        # Save only model, not optimizer state (saves disk space)
        save_safetensors=True,
    )

    return training_args


def main():
    """Main training function."""
    print("\n" + "=" * 60)
    print("InternVL3-8B Training with LoRA")
    print("RoadBuddy Traffic Dataset")
    print("=" * 60 + "\n")

    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set random seeds
    if cfg.SET_SEED:
        import random
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.SEED)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)

    # Move model to device
    if not cfg.USE_LORA:  # LoRA uses device_map
        model = model.to(cfg.DEVICE)

    # Create datasets
    print("\n" + "=" * 60)
    print("Loading Datasets")
    print("=" * 60 + "\n")

    train_dataset = TrafficVideoDataset(
        json_path=cfg.TRAIN_JSON,
        data_root=cfg.DATA_ROOT,
        split='train',
        train_val_split=cfg.TRAIN_VAL_SPLIT,
        random_seed=cfg.RANDOM_SEED,
        min_frames=cfg.MIN_FRAMES,
        max_frames=cfg.MAX_FRAMES,
        max_num_patches=cfg.MAX_NUM_PATCHES,
        use_support_frames=cfg.USE_SUPPORT_FRAMES,
        context_window=cfg.CONTEXT_WINDOW,
        use_vietnamese_prompts=cfg.USE_VIETNAMESE_PROMPTS,
    )

    val_dataset = TrafficVideoDataset(
        json_path=cfg.TRAIN_JSON,
        data_root=cfg.DATA_ROOT,
        split='val',
        train_val_split=cfg.TRAIN_VAL_SPLIT,
        random_seed=cfg.RANDOM_SEED,
        min_frames=cfg.MIN_FRAMES,
        max_frames=cfg.MAX_FRAMES,
        max_num_patches=cfg.MAX_NUM_PATCHES,
        use_support_frames=cfg.USE_SUPPORT_FRAMES,
        context_window=cfg.CONTEXT_WINDOW,
        use_vietnamese_prompts=cfg.USE_VIETNAMESE_PROMPTS,
    )

    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset)} samples")

    # Setup training arguments
    training_args = setup_training_args(cfg)

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Warmup ratio: {training_args.warmup_ratio}")
    print(f"LR scheduler: {training_args.lr_scheduler_type}")
    print(f"Optimizer: {training_args.optim}")
    print(f"FP16: {training_args.fp16}, BF16: {training_args.bf16}")
    print(f"Output directory: {training_args.output_dir}")
    print("=" * 60 + "\n")

    # Create trainer
    trainer = VideoQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Train
    print("=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    try:
        train_result = trainer.train()

        # Save final model
        print("\n" + "=" * 60)
        print("Saving Final Model")
        print("=" * 60)

        final_model_path = cfg.OUTPUT_DIR / "final_model"
        trainer.save_model(str(final_model_path))
        print(f"✓ Model saved to: {final_model_path}")

        # Save training metrics
        metrics_path = cfg.OUTPUT_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")

        # Run final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60 + "\n")

        eval_results = trainer.evaluate()
        print(f"Final validation accuracy: {eval_results.get('eval_accuracy', 0.0):.4f}")

        # Save evaluation results
        eval_path = cfg.OUTPUT_DIR / "eval_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"✓ Evaluation results saved to: {eval_path}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
        print("Saving checkpoint...")
        trainer.save_model(str(cfg.OUTPUT_DIR / "interrupted_checkpoint"))
        print("✓ Checkpoint saved")

    except Exception as e:
        print("\n" + "=" * 60)
        print("Error during training")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

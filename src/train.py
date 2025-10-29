"""Training script for traffic video question answering."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    get_scheduler,
)

from src import train_config
from src.dataset import TrafficQADataset, CollatorWithProcessor


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if train_config.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_model_with_lora(model, config):
    """
    Setup LoRA for parameter-efficient fine-tuning.

    Args:
        model: The base model
        config: Training configuration

    Returns:
        Model with LoRA applied
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    except ImportError:
        print("Warning: peft library not installed. Install with: pip install peft")
        print("Continuing without LoRA...")
        return model


def freeze_model_components(model, config):
    """
    Freeze parts of the model to speed up training.

    Args:
        model: The model to freeze components of
        config: Training configuration
    """
    # Freeze vision encoder if requested
    if config.FREEZE_VISION_ENCODER and hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")

    # Freeze bottom LLM layers if requested
    if config.FREEZE_LLM_LAYERS > 0 and hasattr(model, "language_model"):
        if hasattr(model.language_model, "layers"):
            for i in range(min(config.FREEZE_LLM_LAYERS, len(model.language_model.layers))):
                for param in model.language_model.layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze {config.FREEZE_LLM_LAYERS} LLM layers")


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    Compute accuracy between predictions and labels.

    Args:
        predictions: List of predicted answers
        labels: List of ground truth answers

    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(labels) if labels else 0.0


def parse_answer_from_output(output_text: str, valid_answers: List[str] = ["A", "B", "C", "D"]) -> str:
    """
    Parse answer letter from model output.

    Args:
        output_text: Generated text from model
        valid_answers: List of valid answer letters

    Returns:
        Parsed answer letter
    """
    import re

    # Try to find answer pattern
    patterns = [
        r'\b([ABCD])\b',
        r'(?:đáp án|answer|chọn)[\s:]*([ABCD])',
        r'^([ABCD])[.\s]',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output_text, re.IGNORECASE)
        if matches:
            answer = matches[0].upper()
            if answer in valid_answers:
                return answer

    # If no clear answer found, try to extract the first valid letter
    for char in output_text.upper():
        if char in valid_answers:
            return char

    # Default to A
    return "A"


class Trainer:
    """Trainer class for fine-tuning vision-language models."""

    def __init__(
        self,
        model_name: str = train_config.MODEL_NAME,
        device: str = train_config.DEVICE,
        config=train_config,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name
            device: Device to train on
            config: Training configuration module
        """
        self.config = config
        self.device = device
        self.model_name = model_name

        # Set random seed
        set_seed(config.RANDOM_SEED)

        print(f"Initializing trainer with model: {model_name}")
        print(f"Device: {device}")

        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )

        # Load model
        print("Loading model...")
        dtype = torch.float16 if (config.USE_FP16 and device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            torch_dtype=dtype,
        )

        # Apply LoRA if requested
        if config.USE_LORA:
            print("Applying LoRA...")
            self.model = setup_model_with_lora(self.model, config)

        # Freeze components if requested
        freeze_model_components(self.model, config)

        # Move model to device
        self.model = self.model.to(device)

        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.no_improvement_count = 0

        # Create output directory
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Trainer initialized successfully!")

    def prepare_datasets(self):
        """Prepare train and validation datasets."""
        print(f"\nLoading dataset from: {self.config.TRAIN_JSON}")

        # Load full dataset
        max_samples = self.config.DEBUG_SAMPLES if self.config.DEBUG_MODE else None
        full_dataset = TrafficQADataset(
            json_path=self.config.TRAIN_JSON,
            data_dir=self.config.DATA_DIR,
            processor=self.processor,
            max_samples=max_samples,
            use_support_frames=self.config.USE_SUPPORT_FRAMES,
        )

        # Split into train and validation
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.TRAIN_VAL_SPLIT)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.RANDOM_SEED),
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        # Create data loaders
        collator = CollatorWithProcessor(self.processor, self.config.MAX_SEQ_LENGTH)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=self.config.SHUFFLE_TRAIN,
            collate_fn=collator,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Setup optimizer
        if self.config.OPTIMIZER.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.LEARNING_RATE,
                betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
                eps=self.config.ADAM_EPSILON,
                weight_decay=self.config.WEIGHT_DECAY,
            )
        else:
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
            )

        # Calculate total training steps
        num_training_steps = (
            len(self.train_loader) * self.config.NUM_EPOCHS
        ) // self.config.GRADIENT_ACCUMULATION_STEPS

        num_warmup_steps = int(num_training_steps * self.config.WARMUP_RATIO)

        # Setup scheduler
        self.scheduler = get_scheduler(
            self.config.LR_SCHEDULER_TYPE,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"\nOptimizer: {self.config.OPTIMIZER}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {num_warmup_steps}")

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device) if batch["pixel_values"] is not None else None

            # Forward pass with labels for loss computation
            # Note: This assumes the model supports teacher forcing
            # For generation-based training, we need to compute loss manually
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids,  # Teacher forcing
            )

            loss = outputs.loss / self.config.GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM,
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.LOGGING_STEPS == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1

            # Evaluation during training
            if (
                self.config.EVAL_STEPS > 0
                and self.global_step % self.config.EVAL_STEPS == 0
            ):
                val_accuracy = self.evaluate()
                self.model.train()  # Back to training mode

                # Check for improvement
                if val_accuracy > self.best_val_accuracy + self.config.EARLY_STOPPING_THRESHOLD:
                    self.best_val_accuracy = val_accuracy
                    self.no_improvement_count = 0

                    if self.config.SAVE_BEST_MODEL:
                        self.save_checkpoint(
                            epoch,
                            is_best=True,
                            val_accuracy=val_accuracy,
                        )
                else:
                    self.no_improvement_count += 1

                # Early stopping check
                if (
                    self.config.USE_EARLY_STOPPING
                    and self.no_improvement_count >= self.config.EARLY_STOPPING_PATIENCE
                ):
                    print(f"\nEarly stopping triggered after {self.no_improvement_count} evaluations without improvement")
                    return True  # Signal early stopping

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"\nEpoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        return False  # No early stopping

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Validation accuracy
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        print("\nEvaluating...")
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device) if batch["pixel_values"] is not None else None
            labels = batch["labels"]

            # Generate predictions
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=128,
                do_sample=False,
            )

            # Decode outputs
            decoded_outputs = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Parse answers
            predictions = [parse_answer_from_output(output) for output in decoded_outputs]

            all_predictions.extend(predictions)
            all_labels.extend(labels)

        # Compute accuracy
        accuracy = compute_accuracy(all_predictions, all_labels)
        print(f"Validation Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        return accuracy

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        val_accuracy: Optional[float] = None,
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            val_accuracy: Validation accuracy
        """
        checkpoint_name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = self.config.OUTPUT_DIR / checkpoint_name

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_accuracy": self.best_val_accuracy,
            "val_accuracy": val_accuracy,
            "config": {
                "model_name": self.model_name,
                "learning_rate": self.config.LEARNING_RATE,
                "batch_size": self.config.BATCH_SIZE,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

        # Also save the model in HuggingFace format
        if is_best:
            model_save_path = self.config.OUTPUT_DIR / "best_model_hf"
            self.model.save_pretrained(model_save_path)
            self.processor.save_pretrained(model_save_path)
            print(f"Model saved in HuggingFace format to: {model_save_path}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        # Prepare datasets
        self.prepare_datasets()

        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()

        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print('=' * 60)

            # Train epoch
            should_stop = self.train_epoch(epoch)

            if should_stop:
                break

            # Evaluate at end of epoch
            val_accuracy = self.evaluate()

            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(epoch, val_accuracy=val_accuracy)

            # Check if this is the best model
            if val_accuracy > self.best_val_accuracy + self.config.EARLY_STOPPING_THRESHOLD:
                self.best_val_accuracy = val_accuracy
                self.no_improvement_count = 0

                if self.config.SAVE_BEST_MODEL:
                    self.save_checkpoint(epoch, is_best=True, val_accuracy=val_accuracy)
            else:
                self.no_improvement_count += 1

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f} ({self.best_val_accuracy * 100:.2f}%)")


def main():
    """Main entry point for training."""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()

"""Dataset class for traffic video question answering."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

from src import config
from src.video_processor import VideoProcessor


class TrafficQADataset(Dataset):
    """Dataset for traffic video question answering with multiple choice."""

    def __init__(
        self,
        json_path: Path,
        data_dir: Path,
        processor=None,
        max_samples: Optional[int] = None,
        use_support_frames: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            json_path: Path to the JSON file containing questions
            data_dir: Root directory containing video files
            processor: HuggingFace processor for the model
            max_samples: Maximum number of samples to load (for debugging)
            use_support_frames: If True, extract frames at support_frames timestamps
        """
        self.data_dir = data_dir
        self.processor = processor
        self.use_support_frames = use_support_frames
        self.video_processor = VideoProcessor()

        # Load data
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = data["data"]
        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_answer_letter(self, answer_text: str) -> str:
        """
        Extract the answer letter (A, B, C, D) from answer text.

        Args:
            answer_text: Answer text like "A. Đúng" or "B. Sai"

        Returns:
            Answer letter (A, B, C, or D)
        """
        answer_text = answer_text.strip()
        if answer_text and answer_text[0] in ["A", "B", "C", "D"]:
            return answer_text[0]
        return "A"  # Default fallback

    def _format_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format the prompt for the model.

        Args:
            question: The question to ask
            choices: List of answer choices

        Returns:
            Formatted prompt string
        """
        choices_str = "\n".join(choices)
        return config.PROMPT_TEMPLATE.format(
            question=question,
            choices=choices_str,
        )

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - id: Question ID
                - image: PIL Image of video frame
                - question: Question text
                - choices: List of answer choices
                - answer: Correct answer letter (A, B, C, D)
                - prompt: Formatted prompt
        """
        sample = self.samples[idx]

        # Get video path
        video_path = self.data_dir / sample["video_path"]

        # Extract frames
        try:
            if self.use_support_frames and "support_frames" in sample:
                # Use support frames if available
                support_time = sample["support_frames"][0]
                frames = self.video_processor.extract_frames_at_timestamps(
                    video_path, [support_time]
                )
            else:
                # Extract all frames and use middle one
                frames = self.video_processor.extract_frames(video_path)

            if not frames:
                raise ValueError(f"No frames extracted from {video_path}")

            # Use the middle frame (or first if only one)
            frame = frames[len(frames) // 2]

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            # Create a black image as fallback
            frame = Image.new("RGB", (640, 480), color=(0, 0, 0))

        # Extract answer letter
        answer_letter = self._extract_answer_letter(sample["answer"])

        # Format prompt
        prompt = self._format_prompt(sample["question"], sample["choices"])

        return {
            "id": sample["id"],
            "image": frame,
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": answer_letter,
            "prompt": prompt,
        }


class CollatorWithProcessor:
    """Collator that processes samples using the model's processor."""

    def __init__(self, processor, max_length: int = 512):
        """
        Initialize the collator.

        Args:
            processor: HuggingFace processor
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples.

        Args:
            batch: List of samples from dataset

        Returns:
            Dictionary with processed tensors ready for model input
        """
        images = [sample["image"] for sample in batch]
        prompts = [sample["prompt"] for sample in batch]
        answers = [sample["answer"] for sample in batch]
        ids = [sample["id"] for sample in batch]

        # Prepare messages for each sample
        all_messages = []
        for img, prompt in zip(images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            all_messages.append(messages)

        # Process all samples
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in all_messages
        ]

        # Tokenize and prepare inputs
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "labels": answers,  # Keep as list of letters for now
            "ids": ids,
        }

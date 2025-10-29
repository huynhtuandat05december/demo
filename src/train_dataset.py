"""
Training dataset for InternVL3-8B on RoadBuddy traffic data.
"""
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image

# Handle imports from different locations
try:
    from src import prompts
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src import prompts

# Import from internvl3_8B module
try:
    from .internvl3_8B.inference import load_video, build_transform, dynamic_preprocess, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from src.internvl3_8B.inference import load_video, build_transform, dynamic_preprocess, IMAGENET_MEAN, IMAGENET_STD


class TrafficVideoDataset(Dataset):
    """
    PyTorch dataset for traffic video question answering.

    Loads videos, extracts frames using support_frames timestamps,
    and formats them for InternVL3-8B training with LoRA.
    """

    def __init__(
        self,
        json_path: Path,
        video_dir: Path,
        split: str = "train",
        train_val_split: float = 0.8,
        random_seed: int = 42,
        min_frames: int = 4,
        max_frames: int = 6,
        max_num_patches: int = 6,
        input_size: int = 448,
        use_support_frames: bool = True,
        context_window: float = 0.5,
        use_vietnamese_prompts: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            json_path: Path to train.json file
            video_dir: Directory containing video files
            split: 'train' or 'val'
            train_val_split: Ratio of train/val split (default: 0.8 = 80% train)
            random_seed: Random seed for reproducible splits
            min_frames: Minimum number of frames to extract
            max_frames: Maximum number of frames to extract
            max_num_patches: Max number of patches per frame (for dynamic preprocessing)
            input_size: Input size for image preprocessing (default: 448)
            use_support_frames: Whether to use support_frames for intelligent sampling
            context_window: Seconds of context around support frames
            use_vietnamese_prompts: Use Vietnamese prompt templates
        """
        self.video_dir = Path(video_dir)
        self.split = split
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_num_patches = max_num_patches
        self.input_size = input_size
        self.use_support_frames = use_support_frames
        self.context_window = context_window
        self.use_vietnamese_prompts = use_vietnamese_prompts

        # Load data
        print(f"Loading data from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_samples = data['data']
        print(f"Total samples in JSON: {len(all_samples)}")

        # Create train/val split
        random.seed(random_seed)
        indices = list(range(len(all_samples)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_val_split)

        if split == 'train':
            selected_indices = indices[:split_idx]
            self.samples = [all_samples[i] for i in selected_indices]
        elif split == 'val':
            selected_indices = indices[split_idx:]
            self.samples = [all_samples[i] for i in selected_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        print(f"{split.upper()} split: {len(self.samples)} samples")

        # Build image transform
        self.transform = build_transform(input_size=input_size)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def _extract_answer_label(self, answer: str) -> str:
        """
        Extract answer label (A/B/C/D) from full answer text.

        Args:
            answer: Full answer text (e.g., "B. Sai")

        Returns:
            Single letter answer (e.g., "B")
        """
        # Handle both formats:
        # "B. Sai" -> "B"
        # "B" -> "B"
        answer = answer.strip()

        if '.' in answer:
            label = answer.split('.')[0].strip().upper()
        else:
            label = answer.strip().upper()

        # Validate
        valid_answers = ['A', 'B', 'C', 'D']
        if label not in valid_answers:
            print(f"Warning: Invalid answer label '{label}' extracted from '{answer}'. Defaulting to 'A'")
            return 'A'

        return label

    def _determine_num_frames(self, support_frames: List[float]) -> int:
        """
        Determine number of frames to extract based on support frames.

        Args:
            support_frames: List of important timestamps

        Returns:
            Number of frames to extract
        """
        if support_frames and len(support_frames) > 0:
            # Use adaptive frame count based on support frames
            num_frames = min(max(len(support_frames) * 2, self.min_frames), self.max_frames)
        else:
            # Use middle value
            num_frames = (self.min_frames + self.max_frames) // 2

        return num_frames

    def _get_time_bounds(self, support_frames: List[float]) -> Optional[Tuple[float, float]]:
        """
        Calculate time bounds from support frames.

        Args:
            support_frames: List of important timestamps

        Returns:
            (start_time, end_time) tuple or None
        """
        if not self.use_support_frames or not support_frames or len(support_frames) == 0:
            return None

        start_time = max(0, min(support_frames) - self.context_window)
        end_time = max(support_frames) + self.context_window

        return (start_time, end_time)

    def _format_prompt(self, question: str, choices: List[str], num_frames: int) -> str:
        """
        Format prompt with frame markers for video input.

        Args:
            question: Question text
            choices: List of answer choices
            num_frames: Number of frames in the video

        Returns:
            Formatted prompt string
        """
        # Add frame markers (InternVL3 style)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])

        # Get base prompt from prompts.py
        base_prompt = prompts.get_prompt_template(
            question=question,
            choices=choices,
            num_frames=num_frames,
            language='vi' if self.use_vietnamese_prompts else 'en',
            simple=False
        )

        # Combine
        prompt = video_prefix + base_prompt

        return prompt

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - pixel_values: Tensor of processed frames [total_patches, 3, H, W]
                - num_patches_list: List of number of patches per frame
                - question: Formatted prompt string
                - answer: Answer label (A/B/C/D)
                - video_id: Sample ID
        """
        sample = self.samples[idx]

        # Extract data
        video_id = sample['id']
        question = sample['question']
        choices = sample['choices']
        answer_text = sample['answer']
        support_frames = sample.get('support_frames', [])
        video_rel_path = sample['video_path']

        # Construct video path
        video_path = self.video_dir / video_rel_path

        # Check if video exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Determine frame extraction parameters
        num_frames = self._determine_num_frames(support_frames)
        time_bounds = self._get_time_bounds(support_frames)

        # Load video frames using official InternVL3 approach
        try:
            pixel_values, num_patches_list = load_video(
                video_path=video_path,
                bound=time_bounds,
                input_size=self.input_size,
                max_num=self.max_num_patches,
                num_segments=num_frames
            )
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy data on error
            # Create a black frame as fallback
            dummy_frame = torch.zeros(3, self.input_size, self.input_size)
            pixel_values = dummy_frame.unsqueeze(0)
            num_patches_list = [1]

        # Format prompt
        prompt = self._format_prompt(question, choices, len(num_patches_list))

        # Extract answer label
        answer_label = self._extract_answer_label(answer_text)

        return {
            'pixel_values': pixel_values,  # [total_patches, 3, H, W]
            'num_patches_list': num_patches_list,  # List of ints
            'question': prompt,  # String with frame markers
            'answer': answer_label,  # Single letter: A/B/C/D
            'video_id': video_id,  # For tracking
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching variable-length video sequences.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary with:
            - pixel_values_list: List of tensors (one per sample)
            - num_patches_list: List of lists (patches per frame for each sample)
            - questions: List of prompt strings
            - answers: List of answer labels
            - video_ids: List of video IDs
    """
    # Note: We don't stack pixel_values because they have different lengths
    # InternVL3's chat method processes one sample at a time anyway

    pixel_values_list = [item['pixel_values'] for item in batch]
    num_patches_lists = [item['num_patches_list'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    video_ids = [item['video_id'] for item in batch]

    return {
        'pixel_values_list': pixel_values_list,
        'num_patches_lists': num_patches_lists,
        'questions': questions,
        'answers': answers,
        'video_ids': video_ids,
    }


def test_dataset():
    """Test the dataset loading."""
    from train_config import (
        TRAIN_JSON, TRAIN_VIDEO_DIR, MIN_FRAMES, MAX_FRAMES,
        USE_SUPPORT_FRAMES, CONTEXT_WINDOW, TRAIN_VAL_SPLIT, RANDOM_SEED
    )

    print("Testing TrafficVideoDataset...")
    print("=" * 60)

    # Create train dataset
    train_dataset = TrafficVideoDataset(
        json_path=TRAIN_JSON,
        video_dir=TRAIN_VIDEO_DIR,
        split='train',
        train_val_split=TRAIN_VAL_SPLIT,
        random_seed=RANDOM_SEED,
        min_frames=MIN_FRAMES,
        max_frames=MAX_FRAMES,
        use_support_frames=USE_SUPPORT_FRAMES,
        context_window=CONTEXT_WINDOW,
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")

    # Create val dataset
    val_dataset = TrafficVideoDataset(
        json_path=TRAIN_JSON,
        video_dir=TRAIN_VIDEO_DIR,
        split='val',
        train_val_split=TRAIN_VAL_SPLIT,
        random_seed=RANDOM_SEED,
        min_frames=MIN_FRAMES,
        max_frames=MAX_FRAMES,
        use_support_frames=USE_SUPPORT_FRAMES,
        context_window=CONTEXT_WINDOW,
    )

    print(f"Val dataset size: {len(val_dataset)}")

    # Test loading a sample
    print("\nTesting sample loading...")
    sample = train_dataset[0]

    print(f"Video ID: {sample['video_id']}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Num patches list: {sample['num_patches_list']}")
    print(f"Answer: {sample['answer']}")
    print(f"Question (first 200 chars): {sample['question'][:200]}...")

    # Test collate function
    print("\nTesting collate function...")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"Batch size: {len(batch['questions'])}")
    print(f"Pixel values list length: {len(batch['pixel_values_list'])}")
    print(f"Questions: {[q[:50] for q in batch['questions']]}")
    print(f"Answers: {batch['answers']}")

    print("\n" + "=" * 60)
    print("Dataset test completed successfully!")


if __name__ == "__main__":
    test_dataset()

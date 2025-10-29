"""
InternVL3.5-8B Video Inference Pipeline.

Enhanced inference for dashcam videos using InternVL3.5-8B with:
- Flash Attention for memory efficiency (~30% memory savings)
- Multi-frame processing (supports up to 512 frames)
- Optimized for 24GB VRAM
- Quantization support (8-bit/4-bit) for extra safety margin
"""

import json
import re
import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu

from src import prompts

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build image transformation pipeline with ImageNet normalization."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image into tiles for better detail.

    This is InternVL's approach to handle high-resolution images by splitting
    them into tiles while maintaining aspect ratio.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target aspect ratio grid
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    # Add thumbnail for global context
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices for video sampling."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load video frames using InternVL3.5's official approach.

    Args:
        video_path: Path to video file
        bound: Optional (start, end) time bounds in seconds
        input_size: Input size for image preprocessing
        max_num: Max number of tiles per frame
        num_segments: Number of frames to extract (max 512)

    Returns:
        pixel_values: Tensor of processed frames
        num_patches_list: List of number of patches per frame
    """
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVL35Inference:
    """
    InternVL3.5-8B inference pipeline for dashcam videos.

    This implementation uses Flash Attention for memory efficiency and is
    optimized for 24GB VRAM with conservative frame counts and quantization options.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-8B",
        device: str = "cuda",
        min_frames: int = 8,
        max_frames: int = 32,
        max_num: int = 12,
        use_support_frames: bool = True,
        context_window: float = 0.5,
        simple_prompts: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attn: bool = True,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_name: HuggingFace model name (default: OpenGVLab/InternVL3_5-8B)
            device: Device to run inference on ('cuda' or 'cpu')
            min_frames: Minimum number of frames to extract per video
            max_frames: Maximum number of frames to extract per video (max 512)
            max_num: Max number of tiles per frame
            use_support_frames: Whether to use support_frames timestamps from data
            context_window: Seconds of context around support frames (default: 0.5s)
            simple_prompts: If True, use simpler/shorter prompt templates
            load_in_8bit: Use 8-bit quantization to reduce memory (~50% reduction)
            load_in_4bit: Use 4-bit quantization to reduce memory (~75% reduction)
            use_flash_attn: Use Flash Attention for memory efficiency (~30% savings)
        """
        self.model_name = model_name
        self.device = device
        self.min_frames = min(min_frames, 512)  # InternVL3.5 max 512 frames
        self.max_frames = min(max_frames, 512)
        self.max_num = max_num
        self.use_support_frames = use_support_frames
        self.context_window = context_window
        self.simple_prompts = simple_prompts
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attn = use_flash_attn

        print(f"\n{'='*60}")
        print(f"InternVL3.5-8B Inference Pipeline")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Frame range: {self.min_frames}-{self.max_frames} frames (adaptive)")
        print(f"Max patches per frame: {max_num}")
        print(f"Support frames: {'Enabled' if use_support_frames else 'Disabled'}")
        print(f"Flash Attention: {'Enabled' if use_flash_attn else 'Disabled'}")
        if load_in_4bit:
            print(f"Quantization: 4-bit (memory optimized)")
        elif load_in_8bit:
            print(f"Quantization: 8-bit (memory optimized)")
        else:
            print(f"Quantization: None (full precision)")
        print(f"{'='*60}\n")

        # Warn about CPU usage
        if device == "cpu":
            print("⚠️  WARNING: Running on CPU")
            print("   - Multi-frame inference on CPU will be slow")
            print("   - Requires significant RAM (~8-16 GB for this model)")
            print("   - For better performance, consider using GPU\n")

        # Clear CUDA cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

            # Check available memory
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"Available GPU memory: {free_mem:.2f} GB / {total_mem:.2f} GB\n")

            # Warn if memory is low for 24GB recommendation
            if free_mem < 12:
                print("⚠️  WARNING: Low GPU memory")
                print(f"   - Only {free_mem:.2f} GB available")
                print("   - InternVL3.5 with flash attention requires 12-24 GB")
                print("   - Consider using --load-in-8bit for safety margin\n")

        # Load model
        print("Loading InternVL3.5-8B model...")
        try:
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # Configure quantization
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                print("  Using 4-bit quantization (requires bitsandbytes)")

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.bfloat16

            elif load_in_8bit:
                print("  Using 8-bit quantization (requires bitsandbytes)")
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.bfloat16

            else:
                # Full precision with flash attention
                print(f"  Using full precision (bfloat16)")
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"

            # Add flash attention (key feature for InternVL3.5)
            if use_flash_attn and device == "cuda":
                model_kwargs["use_flash_attn"] = True
                print("  Flash Attention enabled (~30% memory savings)")

            # Load model using official pattern
            self.model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs
            ).eval()

            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Load tokenizer
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False  # Official example uses use_fast=False
            )
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise

        print(f"\n{'='*60}")
        print("Pipeline initialization complete!")
        print(f"{'='*60}\n")

    def _parse_answer(self, response: str) -> str:
        """
        Parse answer from model response.

        Handles multilingual responses (Vietnamese, English, Chinese).

        Args:
            response: Raw model output

        Returns:
            Parsed answer (A, B, C, or D)
        """
        # Clean response
        response_clean = response.strip()

        # Valid answers
        valid_answers = ["A", "B", "C", "D"]

        # Try various patterns (in order of preference)
        patterns = [
            r'^([ABCD])$',  # Exact single letter (most preferred)
            r'^([ABCD])[.\s]',  # "A. " at start
            r'\b([ABCD])\b',  # Single letter with word boundary
            r'(?:đáp án|answer|chọn|答案|选择)[\s:：]*([ABCD])',  # "answer" keywords
            r'([ABCD])[\s]*[是为]',  # Chinese pattern
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[0].upper()
                if answer in valid_answers:
                    return answer

        # Try first few characters
        for i, char in enumerate(response_clean[:10].upper()):
            if char in valid_answers:
                if i == 0 or not response_clean[i-1].isalpha():
                    return char

        # Last resort: any valid letter in response
        for char in response_clean.upper():
            if char in valid_answers:
                return char

        # Default fallback
        print(f"⚠️  Warning: Could not parse answer from response: {response_clean[:200]}")
        return "A"

    def infer(
        self,
        video_path: Path,
        question: str,
        choices: List[str],
        support_frames: Optional[List[float]] = None,
        verbose: bool = False
    ) -> str:
        """
        Run inference on a single video using InternVL3.5-8B.

        Args:
            video_path: Path to video file
            question: Question text
            choices: List of answer choices
            support_frames: Optional list of important timestamps
            verbose: If True, print detailed information

        Returns:
            Predicted answer (A, B, C, or D)
        """
        try:
            # Determine number of frames to extract
            num_segments = self.min_frames
            if self.max_frames > self.min_frames:
                # Use adaptive frame count based on support frames availability
                if support_frames and len(support_frames) > 0:
                    num_segments = min(max(len(support_frames) * 2, self.min_frames), self.max_frames)
                else:
                    num_segments = (self.min_frames + self.max_frames) // 2

            # Ensure we don't exceed 512 frame limit
            num_segments = min(num_segments, 512)

            # Determine time bounds from support frames
            bound = None
            if support_frames and len(support_frames) > 0 and self.use_support_frames:
                # Use support frames to determine video segment of interest
                start_time = max(0, min(support_frames) - self.context_window)
                end_time = max(support_frames) + self.context_window
                bound = (start_time, end_time)

            if verbose:
                print(f"Loading video: {video_path}")
                print(f"  Num segments: {num_segments}")
                print(f"  Max tiles per frame: {self.max_num}")
                if bound:
                    print(f"  Time bounds: {bound[0]:.2f}s - {bound[1]:.2f}s")

            # Load video
            pixel_values, num_patches_list = load_video(
                video_path=video_path,
                bound=bound,
                input_size=448,
                max_num=self.max_num,
                num_segments=num_segments
            )

            # Move to device - official HuggingFace pattern
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

            total_patches = sum(num_patches_list)

            if verbose:
                print(f"  Loaded {len(num_patches_list)} frames")
                print(f"  Total patches: {total_patches}")
                print(f"  Pixel values shape: {pixel_values.shape}")

            # Format prompt with frame prefixes (InternVL3.5 style)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

            # Use optimized prompt template
            base_prompt = prompts.get_prompt_template(
                question=question,
                choices=choices,
                num_frames=len(num_patches_list),
                language='auto',
                simple=self.simple_prompts
            )

            # Prepend frame markers
            prompt = video_prefix + base_prompt

            if verbose:
                print(f"  Prompt length: {len(prompt)} characters")
                print(f"  Prompt preview: {prompt[:200]}...")

            # Generation config - InternVL3.5 uses do_sample=True by default
            generation_config = {
                "max_new_tokens": 1024,
                "do_sample": True,
            }

            # Generate response using InternVL3.5's chat
            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )

            if verbose:
                print(f"  Model response: '{response}'")

            # Parse answer
            answer = self._parse_answer(response)

            if verbose:
                print(f"Parsed answer: {answer}")

            # Clear CUDA cache after inference
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return answer

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return "A"

    def infer_batch(
        self,
        video_data: List[Dict],
        data_dir: Path,
        verbose: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Run inference on a batch of videos.

        Args:
            video_data: List of dictionaries with video information
            data_dir: Base directory for video paths
            verbose: If True, print detailed information for each video
            show_progress: If True, show progress bar

        Returns:
            List of dictionaries with id and answer
        """
        results = []

        iterator = tqdm(video_data, desc="Processing videos") if show_progress else video_data

        for data in iterator:
            video_id = data["id"]
            question = data["question"]
            choices = data["choices"]
            video_rel_path = data["video_path"]
            support_frames = data.get("support_frames", None)

            # Construct full video path
            video_path = data_dir / video_rel_path

            if verbose and not show_progress:
                print(f"\n{'─'*60}")
                print(f"Video ID: {video_id}")
                print(f"Question: {question[:80]}...")

            # Run inference
            answer = self.infer(
                video_path=video_path,
                question=question,
                choices=choices,
                support_frames=support_frames,
                verbose=verbose
            )

            results.append({
                "id": video_id,
                "answer": answer
            })

            if verbose and not show_progress:
                print(f"Answer: {answer}")

        return results

    def run_pipeline(
        self,
        test_json_path: Path,
        data_dir: Path,
        output_csv_path: Path,
        verbose: bool = False
    ):
        """
        Run the complete inference pipeline.

        Args:
            test_json_path: Path to test JSON file
            data_dir: Base directory for video paths
            output_csv_path: Path to save results CSV
            verbose: If True, print detailed information
        """
        print(f"\n{'='*60}")
        print("Starting Inference Pipeline")
        print(f"{'='*60}\n")

        # Load test data
        print(f"Loading test data from: {test_json_path}")
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        video_data = test_data["data"]
        print(f"Total videos: {len(video_data)}\n")

        # Run inference
        print("Running inference...")
        results = self.infer_batch(
            video_data=video_data,
            data_dir=data_dir,
            verbose=verbose,
            show_progress=True
        )

        # Save results
        print(f"\nSaving results to: {output_csv_path}")
        df = pd.DataFrame(results)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)

        print(f"✓ Results saved ({len(results)} predictions)")

        # Show answer distribution
        print("\nAnswer distribution:")
        print(df["answer"].value_counts().sort_index())

        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}\n")


def main():
    """Example usage of the inference pipeline."""
    from src import config

    # Initialize pipeline with default settings for 24GB VRAM
    pipeline = InternVL35Inference(
        model_name="OpenGVLab/InternVL3_5-8B",
        device=config.DEVICE,
        min_frames=8,
        max_frames=32,
        max_num=12,
        use_support_frames=config.USE_SUPPORT_FRAMES,
        use_flash_attn=True,  # Enable flash attention for memory efficiency
        load_in_8bit=False,  # Set to True for extra safety margin on 24GB
        load_in_4bit=False,
    )

    # Run on test data
    pipeline.run_pipeline(
        test_json_path=config.PUBLIC_TEST_JSON,
        data_dir=config.DATA_DIR,
        output_csv_path=config.INFERENCE_OUTPUT_FILE,
        verbose=config.VERBOSE
    )


if __name__ == "__main__":
    main()

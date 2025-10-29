"""
InternVL3 Traffic Video Inference Pipeline.

Enhanced inference for traffic dashcam videos using InternVL3-8B with:
- Multi-frame processing (6-12 frames adaptive)
- Traffic-optimized prompts
- Support for both CPU and CUDA
- High accuracy configuration (float32, max_num=24)
"""

import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from src.video_processor import VideoProcessor
from src import traffic_prompts


class InternVL3TrafficInference:
    """
    Enhanced InternVL3-8B inference pipeline for traffic dashcam videos.

    This implementation processes multiple frames from each video to capture
    temporal context and improve understanding of traffic scenarios.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        device: str = "cuda",
        min_frames: int = 6,
        max_frames: int = 12,
        max_num: int = 24,
        use_support_frames: bool = True,
        context_window: float = 0.5,
        simple_prompts: bool = False,
    ):
        """
        Initialize the traffic inference pipeline.

        Args:
            model_name: HuggingFace model name (default: OpenGVLab/InternVL3-8B)
            device: Device to run inference on ('cuda' or 'cpu')
            min_frames: Minimum number of frames to extract per video
            max_frames: Maximum number of frames to extract per video
            max_num: InternVL max_num parameter (image patches/tiles per frame)
            use_support_frames: Whether to use support_frames timestamps from data
            context_window: Seconds of context around support frames (default: 0.5s)
            simple_prompts: If True, use simpler/shorter prompt templates
        """
        self.model_name = model_name
        self.device = device
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_num = max_num
        self.use_support_frames = use_support_frames
        self.context_window = context_window
        self.simple_prompts = simple_prompts

        print(f"\n{'='*60}")
        print(f"InternVL3 Traffic Inference Pipeline")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Frame range: {min_frames}-{max_frames} frames (adaptive)")
        print(f"Max patches per frame: {max_num}")
        print(f"Support frames: {'Enabled' if use_support_frames else 'Disabled'}")
        print(f"{'='*60}\n")

        # Warn about CPU usage
        if device == "cpu":
            print("⚠️  WARNING: Running on CPU")
            print("   - Multi-frame inference on CPU will be slow (~2-5 min per video)")
            print("   - Requires significant RAM (~8-16 GB for this model)")
            print("   - For better performance, consider using GPU")
            print("   - Or try smaller models like YannQi/R-4B\n")

        # Clear CUDA cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

            # Check available memory
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"Available GPU memory: {free_mem:.2f} GB / {total_mem:.2f} GB\n")

            # Warn if memory is low
            if free_mem < 12:
                print("⚠️  WARNING: Low GPU memory")
                print(f"   - Only {free_mem:.2f} GB available")
                print("   - Multi-frame InternVL3 may require 12-24 GB")
                print("   - Consider reducing min_frames/max_frames or using CPU\n")

        # Load model
        print("Loading InternVL3-8B model...")
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # High accuracy
                low_cpu_mem_usage=True,
            ).to(device)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Load tokenizer
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise

        # Initialize video processor
        print("Initializing video processor...")
        self.video_processor = VideoProcessor()
        print("✓ Video processor ready")

        print(f"\n{'='*60}")
        print("Pipeline initialization complete!")
        print(f"{'='*60}\n")

    def _extract_frames_smart(
        self,
        video_path: Path,
        support_frames: Optional[List[float]] = None
    ) -> List[Image.Image]:
        """
        Intelligently extract frames from video.

        Uses support_frames timestamps when available, otherwise adaptive sampling.

        Args:
            video_path: Path to video file
            support_frames: Optional list of important timestamps

        Returns:
            List of PIL Images (6-12 frames)
        """
        if self.use_support_frames and support_frames and len(support_frames) > 0:
            # Use support frames with context
            return self.video_processor.extract_frames_with_support(
                video_path=video_path,
                support_frames=support_frames,
                context_window=self.context_window,
                min_frames=self.min_frames,
                max_frames=self.max_frames
            )
        else:
            # Use adaptive sampling
            return self.video_processor.extract_frames_adaptive(
                video_path=video_path,
                min_frames=self.min_frames,
                max_frames=self.max_frames
            )

    def _prepare_multiframe_inputs(
        self,
        frames: List[Image.Image],
        prompt: str
    ) -> Dict:
        """
        Prepare multi-frame inputs for InternVL3.

        Args:
            frames: List of PIL Images
            prompt: Formatted prompt text

        Returns:
            Dictionary with pixel_values, question, and generation_config
        """
        # Use InternVL's load_image with list of frames
        pixel_values = self.model.load_image(
            frames,  # List of PIL Images
            max_num=self.max_num
        ).to(torch.float32).to(self.device)

        return {
            "pixel_values": pixel_values,
            "question": prompt,
        }

    def _format_prompt(
        self,
        question: str,
        choices: List[str],
        num_frames: int
    ) -> str:
        """
        Format traffic-optimized prompt.

        Args:
            question: Question text
            choices: List of answer choices
            num_frames: Number of frames being analyzed

        Returns:
            Formatted prompt string
        """
        return traffic_prompts.get_prompt_template(
            question=question,
            choices=choices,
            num_frames=num_frames,
            language='auto',  # Auto-detect Vietnamese or English
            simple=self.simple_prompts
        )

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
        Run inference on a single traffic video.

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
            # Extract frames
            frames = self._extract_frames_smart(video_path, support_frames)

            if not frames:
                print(f"⚠️  Warning: No frames extracted from {video_path}")
                return "A"

            num_frames = len(frames)

            if verbose:
                print(f"Extracted {num_frames} frames")
                if support_frames:
                    print(f"Using support frames: {support_frames}")

            # Format prompt
            prompt = self._format_prompt(question, choices, num_frames)

            if verbose:
                print(f"Prompt length: {len(prompt)} characters")

            # Prepare inputs
            inputs = self._prepare_multiframe_inputs(frames, prompt)

            # Generation config
            generation_config = {
                "max_new_tokens": 50,
                "do_sample": False,  # Greedy decoding for consistency
            }

            # Generate response
            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=inputs["pixel_values"],
                    question=inputs["question"],
                    generation_config=generation_config,
                )

            if verbose:
                print(f"Model response: '{response}'")

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

        # Clear cache
        self.video_processor.clear_cache()

        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}\n")


def main():
    """Example usage of the traffic inference pipeline."""
    from src import config

    # Initialize pipeline
    pipeline = InternVL3TrafficInference(
        model_name="OpenGVLab/InternVL3-8B",
        device="cuda",  # or "cpu"
        min_frames=6,
        max_frames=12,
        max_num=24,
        use_support_frames=True,
    )

    # Run on test data
    pipeline.run_pipeline(
        test_json_path=config.PUBLIC_TEST_JSON,
        data_dir=config.DATA_DIR,
        output_csv_path=config.OUTPUT_DIR / "traffic_inference_results.csv",
        verbose=False
    )


if __name__ == "__main__":
    main()

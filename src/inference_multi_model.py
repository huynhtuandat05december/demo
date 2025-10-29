"""Multi-model inference pipeline for traffic video question answering."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
import pandas as pd

from src import config
from src.video_processor import VideoProcessor
from src.model_adapters import get_model_adapter


class MultiModelInferencePipeline:
    """Inference pipeline supporting multiple vision-language models."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_name: HuggingFace model name (supports R-4B, InternVL3, Qwen3-VL)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name

        print(f"Initializing pipeline with model: {model_name}")
        print(f"Device: {device}")

        # Load model using appropriate adapter
        self.model_adapter = get_model_adapter(
            model_name=model_name,
            device=device,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )

        # Initialize video processor
        self.video_processor = VideoProcessor()

        print("Pipeline ready!")

    def format_prompt(self, question: str, choices: List[str]) -> str:
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

    def parse_answer(self, response: str) -> str:
        """
        Parse the answer from model response.

        Args:
            response: Raw model output

        Returns:
            Parsed answer (A, B, C, or D)
        """
        # Try to find answer pattern like "A", "B.", "Answer: C", etc.
        patterns = [
            r'\b([ABCD])\b',  # Single letter
            r'(?:đáp án|answer|chọn)[\s:]*([ABCD])',  # "đáp án A" or "answer: B"
            r'^([ABCD])[.\s]',  # "A. " at start
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[0].upper()
                if answer in config.VALID_ANSWERS:
                    return answer

        # If no clear answer found, try to extract the first valid letter
        for char in response.upper():
            if char in config.VALID_ANSWERS:
                return char

        # Default to A if no answer found
        print(f"Warning: Could not parse answer from response: {response[:100]}")
        return "A"

    def run_inference(self, question_data: Dict, verbose: bool = False) -> str:
        """
        Run inference for a single question.

        Args:
            question_data: Dictionary containing question information
            verbose: If True, print detailed information

        Returns:
            Predicted answer (A, B, C, or D)
        """
        # Get video path
        video_path = config.DATA_DIR / question_data["video_path"]

        # Extract frames from video
        try:
            frames = self.video_processor.extract_frames(video_path)
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return "A"  # Default answer on error

        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return "A"

        # Use the middle frame (or first frame if only one)
        frame = frames[len(frames) // 2]

        # Format prompt
        prompt = self.format_prompt(
            question=question_data["question"],
            choices=question_data["choices"],
        )

        if verbose:
            print(f"\nQuestion: {question_data['question']}")
            print(f"Video: {video_path.name}")
            print(f"Prompt length: {len(prompt)} chars")

        # Prepare inputs using model adapter
        try:
            inputs = self.model_adapter.prepare_inputs(frame, prompt)
        except Exception as e:
            print(f"Error preparing inputs: {e}")
            return "A"

        # Generate response
        try:
            response = self.model_adapter.generate(
                inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=config.TEMPERATURE,
                thinking_mode=config.THINKING_MODE,
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return "A"

        if verbose:
            print(f"Model response: {response[:200]}...")

        # Parse answer
        answer = self.parse_answer(response)

        if verbose:
            print(f"Parsed answer: {answer}")

        return answer

    def run_pipeline(
        self,
        test_json_path: Path,
        output_csv_path: Path,
        verbose: bool = False,
    ):
        """
        Run the full inference pipeline on test data.

        Args:
            test_json_path: Path to test JSON file
            output_csv_path: Path to save submission CSV
            verbose: If True, print detailed information
        """
        # Load test data
        print(f"\nLoading test data from: {test_json_path}")
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        questions = test_data["data"]
        print(f"Total questions: {len(questions)}")

        # Run inference
        results = []
        for question_data in tqdm(questions, desc="Processing questions"):
            question_id = question_data["id"]
            answer = self.run_inference(question_data, verbose=verbose)
            results.append({"id": question_id, "answer": answer})

        # Save results
        df = pd.DataFrame(results)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to: {output_csv_path}")
        print(f"Total predictions: {len(results)}")

        # Show answer distribution
        print("\nAnswer distribution:")
        print(df["answer"].value_counts().sort_index())

        # Clear cache
        self.video_processor.clear_cache()


def main(model_name: Optional[str] = None):
    """
    Main entry point for the inference pipeline.

    Args:
        model_name: Optional model name to override config
    """
    # Use provided model name or default from config
    if model_name is None:
        model_name = config.MODEL_NAME

    print(f"\n{'='*60}")
    print(f"TRAFFIC VIDEO QUESTION ANSWERING")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {config.DEVICE}")
    print('='*60)

    # Initialize pipeline
    pipeline = MultiModelInferencePipeline(model_name=model_name)

    # Run inference on public test data
    pipeline.run_pipeline(
        test_json_path=config.PUBLIC_TEST_JSON,
        output_csv_path=config.SUBMISSION_FILE,
        verbose=False,
    )


if __name__ == "__main__":
    import sys

    # Allow passing model name as command line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Using model from command line: {model_name}")
        main(model_name=model_name)
    else:
        main()

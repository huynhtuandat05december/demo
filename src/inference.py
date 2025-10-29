"""Main inference pipeline for traffic video question answering."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModel, AutoProcessor

from src import config
from src.video_processor import VideoProcessor


class R4BInferencePipeline:
    """Inference pipeline using R-4B model for video question answering."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_name: HuggingFace model name
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        # Clear CUDA cache before loading model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")

        # Load model and processor
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=config.TRUST_REMOTE_CODE,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )

        # Initialize video processor
        self.video_processor = VideoProcessor()

        print("Model and processor loaded successfully!")

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
        Handles responses in multiple languages including Chinese.

        Args:
            response: Raw model output

        Returns:
            Parsed answer (A, B, C, or D)
        """
        # Clean response - remove extra whitespace
        response_clean = response.strip()

        # Try to find answer pattern like "A", "B.", "Answer: C", etc.
        patterns = [
            r'^([ABCD])$',  # Exact single letter (most preferred)
            r'^([ABCD])[.\s]',  # "A. " at start
            r'\b([ABCD])\b',  # Single letter with word boundary
            r'(?:đáp án|answer|chọn|答案|选择)[\s:：]*([ABCD])',  # Various "answer" keywords (including Chinese)
            r'([ABCD])[\s]*[是为]',  # Chinese pattern like "A是" or "B为"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[0].upper()
                if answer in config.VALID_ANSWERS:
                    return answer

        # Try looking at just the first few characters for a letter
        for i, char in enumerate(response_clean[:10].upper()):
            if char in config.VALID_ANSWERS:
                # Check if it's not part of a longer word
                if i == 0 or not response_clean[i-1].isalpha():
                    return char

        # If no clear answer found, try to extract ANY valid letter from the entire response
        for char in response_clean.upper():
            if char in config.VALID_ANSWERS:
                return char

        # Default to A if no answer found
        print(f"Warning: Could not parse answer from response: {response_clean[:200]}")
        return "A"

    def run_inference(self, question_data: Dict) -> str:
        """
        Run inference for a single question.

        Args:
            question_data: Dictionary containing question information

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

        # Debug: Print the prompt
        print(f"\n📝 DEBUG - Prompt sent to model:")
        print(f"{prompt}")
        print(f"-" * 60)

        # Prepare inputs with system message (some models may ignore it)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers multiple choice questions about traffic videos. Always respond with only a single letter (A, B, C, or D) without any explanation."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs - try with system message first
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            # Some models don't support system messages, fall back to user-only
            print(f"Warning: System message not supported, using user message only")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.processor(
            text=[text],
            images=[frame],
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device and convert to float32
        inputs = {
            k: v.to(self.device).to(torch.float32) if v.dtype in [torch.float32, torch.float16] else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate response
        with torch.no_grad():
            # Prepare generation kwargs
            gen_kwargs = {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": config.DO_SAMPLE,
            }

            # Only add temperature if sampling is enabled
            if config.DO_SAMPLE:
                gen_kwargs["temperature"] = config.TEMPERATURE

            # Get input length to extract only generated tokens
            input_ids = inputs.get("input_ids")
            input_length = input_ids.shape[1] if input_ids is not None else 0

            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated tokens (skip input)
        if input_length > 0:
            generated_ids = outputs[:, input_length:]
        else:
            generated_ids = outputs

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        # Debug: Print raw model response
        print(f"\n🔍 DEBUG - Raw model response (generated only):")
        print(f"'{response}'")
        print(f"Response length: {len(response)} characters")

        # Parse answer
        answer = self.parse_answer(response)
        print(f"🎯 Parsed answer: {answer}")

        # Clear cache after inference to prevent memory buildup
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return answer

    def run_pipeline(self, test_json_path: Path, output_csv_path: Path):
        """
        Run the full inference pipeline on test data.

        Args:
            test_json_path: Path to test JSON file
            output_csv_path: Path to save submission CSV
        """
        # Load test data
        print(f"Loading test data from: {test_json_path}")
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        questions = test_data["data"]
        print(f"Total questions: {len(questions)}")

        # Run inference
        results = []
        for question_data in tqdm(questions, desc="Processing questions"):
            question_id = question_data["id"]
            answer = self.run_inference(question_data)
            results.append({"id": question_id, "answer": answer})

        # Save results
        df = pd.DataFrame(results)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to: {output_csv_path}")
        print(f"Total predictions: {len(results)}")

        # Clear cache
        self.video_processor.clear_cache()


def main():
    """Main entry point for the inference pipeline."""
    # Initialize pipeline
    pipeline = R4BInferencePipeline()

    # Run inference on public test data
    pipeline.run_pipeline(
        test_json_path=config.PUBLIC_TEST_JSON,
        output_csv_path=config.SUBMISSION_FILE,
    )


if __name__ == "__main__":
    main()

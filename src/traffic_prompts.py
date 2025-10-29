"""
Traffic-optimized prompt templates for video question answering.

Provides specialized prompts for dashcam traffic video analysis with
temporal reasoning and traffic-specific context.
"""

import re
from typing import List


# English prompt for traffic video analysis
TRAFFIC_PROMPT_EN = """You are analyzing dashcam footage from a traffic camera. You are seeing {num_frames} frames extracted from a video showing a traffic scenario.

Question: {question}

Answer choices:
{choices}

Instructions:
- Carefully observe the sequence of events across all frames
- Pay attention to vehicle movements, traffic signals, road conditions, and driver actions
- Consider what happens before, during, and after key moments in the video
- Think about traffic rules, road safety, and appropriate driving behavior
- Base your answer on what you can observe in the video frames

Respond with ONLY the letter (A, B, C, or D) of the correct answer. Do not include explanations.

Your answer:"""


# Vietnamese prompt for traffic video analysis
TRAFFIC_PROMPT_VI = """Bạn đang phân tích video từ camera hành trình. Bạn đang xem {num_frames} khung hình được trích xuất từ video về một tình huống giao thông.

Câu hỏi: {question}

Các lựa chọn:
{choices}

Hướng dẫn:
- Quan sát kỹ chuỗi sự kiện qua tất cả các khung hình
- Chú ý đến di chuyển của xe, tín hiệu giao thông, điều kiện đường, và hành động của người lái
- Xem xét điều gì xảy ra trước, trong và sau các khoảnh khắc quan trọng trong video
- Suy nghĩ về luật giao thông, an toàn đường bộ và hành vi lái xe phù hợp
- Dựa câu trả lời vào những gì bạn có thể quan sát được trong các khung hình video

Chỉ trả lời bằng một chữ cái (A, B, C, hoặc D) cho đáp án đúng. Không bao gồm giải thích.

Câu trả lời của bạn:"""


# Simple English prompt (more concise)
TRAFFIC_PROMPT_EN_SIMPLE = """Watch these {num_frames} frames from a dashcam video and answer this traffic-related question:

{question}

{choices}

Consider the sequence of events, vehicle movements, traffic signals, and road conditions across all frames.

Respond with ONLY one letter (A, B, C, or D).

Answer:"""


# Simple Vietnamese prompt (more concise)
TRAFFIC_PROMPT_VI_SIMPLE = """Xem {num_frames} khung hình từ video camera hành trình và trả lời câu hỏi về giao thông:

{question}

{choices}

Xem xét chuỗi sự kiện, di chuyển xe, tín hiệu giao thông và điều kiện đường qua tất cả các khung hình.

Chỉ trả lời bằng một chữ cái (A, B, C, hoặc D).

Câu trả lời:"""


def detect_language(text: str) -> str:
    """
    Detect if text is Vietnamese or English.

    Args:
        text: Text to analyze

    Returns:
        'vi' for Vietnamese, 'en' for English
    """
    # Vietnamese-specific characters
    vietnamese_chars = r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]'

    # Check for Vietnamese characters
    if re.search(vietnamese_chars, text.lower()):
        return 'vi'

    return 'en'


def format_traffic_prompt(
    question: str,
    choices: List[str],
    num_frames: int,
    language: str = 'auto',
    simple: bool = False
) -> str:
    """
    Format a traffic-optimized prompt for video question answering.

    Args:
        question: The question to ask
        choices: List of answer choices (e.g., ["A. Option 1", "B. Option 2"])
        num_frames: Number of video frames being analyzed
        language: 'en', 'vi', or 'auto' for automatic detection
        simple: If True, use simpler/shorter prompt template

    Returns:
        Formatted prompt string
    """
    # Auto-detect language if requested
    if language == 'auto':
        language = detect_language(question)

    # Format choices as string
    choices_str = "\n".join(choices)

    # Select appropriate template
    if language == 'vi':
        template = TRAFFIC_PROMPT_VI_SIMPLE if simple else TRAFFIC_PROMPT_VI
    else:
        template = TRAFFIC_PROMPT_EN_SIMPLE if simple else TRAFFIC_PROMPT_EN

    # Format the prompt
    prompt = template.format(
        question=question,
        choices=choices_str,
        num_frames=num_frames
    )

    return prompt


def format_single_frame_prompt(
    question: str,
    choices: List[str],
    language: str = 'auto'
) -> str:
    """
    Format a prompt for single-frame analysis (fallback/compatibility).

    Args:
        question: The question to ask
        choices: List of answer choices
        language: 'en', 'vi', or 'auto' for automatic detection

    Returns:
        Formatted prompt string
    """
    # Auto-detect language if requested
    if language == 'auto':
        language = detect_language(question)

    choices_str = "\n".join(choices)

    if language == 'vi':
        prompt = f"""Dựa vào hình ảnh từ camera hành trình, trả lời câu hỏi:

{question}

{choices_str}

Chỉ trả lời bằng một chữ cái (A, B, C, hoặc D).

Câu trả lời:"""
    else:
        prompt = f"""Based on this dashcam image, answer the question:

{question}

{choices_str}

Respond with ONLY one letter (A, B, C, or D).

Answer:"""

    return prompt


# Template selection helper
def get_prompt_template(
    question: str,
    choices: List[str],
    num_frames: int = 1,
    language: str = 'auto',
    simple: bool = False
) -> str:
    """
    Get the appropriate prompt template based on parameters.

    This is the main entry point for getting prompts. It automatically
    selects between single-frame and multi-frame prompts.

    Args:
        question: The question to ask
        choices: List of answer choices
        num_frames: Number of frames (1 for single-frame, >1 for multi-frame)
        language: 'en', 'vi', or 'auto' for automatic detection
        simple: If True, use simpler/shorter prompt templates

    Returns:
        Formatted prompt string optimized for the scenario
    """
    if num_frames <= 1:
        # Single frame - use single-frame prompt
        return format_single_frame_prompt(question, choices, language)
    else:
        # Multiple frames - use traffic multi-frame prompt
        return format_traffic_prompt(question, choices, num_frames, language, simple)


# Example usage and testing
if __name__ == "__main__":
    # Example Vietnamese question
    question_vi = "Xe màu xanh có được phép rẽ trái tại đây không?"
    choices_vi = [
        "A. Có, vì không có biển cấm",
        "B. Không, vì đèn đỏ",
        "C. Có, nhưng phải nhường đường",
        "D. Không, vì có biển cấm"
    ]

    # Example English question
    question_en = "Is the blue car allowed to turn left here?"
    choices_en = [
        "A. Yes, there is no prohibition sign",
        "B. No, the light is red",
        "C. Yes, but must yield",
        "D. No, there is a prohibition sign"
    ]

    # Test Vietnamese multi-frame prompt
    print("=" * 60)
    print("Vietnamese Multi-Frame Prompt (8 frames):")
    print("=" * 60)
    print(format_traffic_prompt(question_vi, choices_vi, 8, language='vi'))
    print()

    # Test English multi-frame prompt
    print("=" * 60)
    print("English Multi-Frame Prompt (8 frames):")
    print("=" * 60)
    print(format_traffic_prompt(question_en, choices_en, 8, language='en'))
    print()

    # Test auto-detection
    print("=" * 60)
    print("Auto-Detection Test:")
    print("=" * 60)
    print(f"Vietnamese question detected as: {detect_language(question_vi)}")
    print(f"English question detected as: {detect_language(question_en)}")
    print()

    # Test simple templates
    print("=" * 60)
    print("Simple Template (Vietnamese):")
    print("=" * 60)
    print(format_traffic_prompt(question_vi, choices_vi, 10, language='vi', simple=True))

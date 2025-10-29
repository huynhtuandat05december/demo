"""Model adapters for different vision-language models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class BaseModelAdapter(ABC):
    """Base class for model adapters."""

    def __init__(self, model_name: str, device: str, trust_remote_code: bool = True):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass

    @abstractmethod
    def prepare_inputs(self, frame: Image.Image, prompt: str) -> Dict[str, Any]:
        """Prepare inputs for the model."""
        pass

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        """Generate response from the model."""
        pass


class R4BAdapter(BaseModelAdapter):
    """Adapter for YannQi/R-4B model."""

    def load_model(self):
        print(f"Loading R-4B model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        print("✓ R-4B model loaded")

    def prepare_inputs(self, frame: Image.Image, prompt: str) -> Dict[str, Any]:
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
            thinking_mode="auto"
        )

        inputs = self.processor(
            text=[text],
            images=[frame],
            return_tensors="pt",
            padding=True,
        )

        # Convert to float32
        return {
            k: v.to(self.device).to(torch.float32) if v.dtype in [torch.float32, torch.float16] else v.to(self.device)
            for k, v in inputs.items()
        }

    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 0.1)

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        # Only add temperature if sampling is enabled
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response


class InternVL3Adapter(BaseModelAdapter):
    """Adapter for OpenGVLab/InternVL3-8B model."""

    def load_model(self):
        print(f"Loading InternVL3 model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        print("✓ InternVL3 model loaded")

    def prepare_inputs(self, frame: Image.Image, prompt: str) -> Dict[str, Any]:
        # InternVL3 uses a different format with pixel values
        pixel_values = self.model.load_image(frame, max_num=6).to(
            torch.float32
        ).to(self.device)

        # Format the conversation
        generation_config = dict(max_new_tokens=512, do_sample=False)

        return {
            "pixel_values": pixel_values,
            "question": prompt,
            "generation_config": generation_config,
        }

    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 0.1)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            generation_config["temperature"] = temperature

        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=inputs["pixel_values"],
                question=inputs["question"],
                generation_config=generation_config,
            )

        return response


class Qwen3VLAdapter(BaseModelAdapter):
    """Adapter for Qwen/Qwen3-VL-8B-Instruct model."""

    def load_model(self):
        print(f"Loading Qwen3-VL model: {self.model_name}")

        # Qwen3-VL uses a specific processor
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        print("✓ Qwen3-VL model loaded")

    def prepare_inputs(self, frame: Image.Image, prompt: str) -> Dict[str, Any]:
        # Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare for chat
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

        # Convert to float32
        return {
            k: v.to(self.device).to(torch.float32) if v.dtype in [torch.float32, torch.float16] else v.to(self.device)
            for k, v in inputs.items()
        }

    def generate(self, inputs: Dict[str, Any], **kwargs) -> str:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 0.1)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response


def get_model_adapter(model_name: str, device: str, trust_remote_code: bool = True) -> BaseModelAdapter:
    """Factory function to get the appropriate model adapter."""

    # Normalize model name
    model_name_lower = model_name.lower()

    if "r-4b" in model_name_lower or "yannqi" in model_name_lower:
        adapter = R4BAdapter(model_name, device, trust_remote_code)
    elif "internvl" in model_name_lower:
        adapter = InternVL3Adapter(model_name, device, trust_remote_code)
    elif "qwen" in model_name_lower and "vl" in model_name_lower:
        adapter = Qwen3VLAdapter(model_name, device, trust_remote_code)
    else:
        raise ValueError(
            f"Unknown model type: {model_name}. "
            "Supported models: YannQi/R-4B, OpenGVLab/InternVL3-8B, Qwen/Qwen3-VL-8B-Instruct"
        )

    adapter.load_model()
    return adapter

"""Unified Qwen3-VL / Qwen3.5 load and interleaved text-image inference.

Default checkpoint is Qwen3-VL-8B. Pass a Qwen3.5 path via ``model_path`` /
``--model-path``; ``load_model`` picks the class from ``config.json``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def _read_model_config(model_path: str) -> Optional[Dict[str, Any]]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_qwen3_5(model_path: str) -> bool:
    cfg = _read_model_config(model_path)
    if cfg is None:
        return False
    if cfg.get("model_type") == "qwen3_5":
        return True
    arches = cfg.get("architectures") or []
    return any("Qwen3_5" in str(a) for a in arches)


def _is_qwen3(model_path: str) -> bool:
    cfg = _read_model_config(model_path)
    if cfg is None:
        return False
    if cfg.get("model_type") == "qwen3_vl":
        return True
    arches = cfg.get("architectures") or []
    return any("Qwen3VL" in str(a) for a in arches)


def load_model(model_path: Optional[str] = None):
    """Load Qwen3-VL or Qwen3.5 model and processor."""
    if not model_path:
        raise ValueError("Provide a Qwen3-VL or Qwen3.5 checkpoint path")
    path = model_path
    if _is_qwen3_5(path):
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, torch_dtype="auto", device_map="auto"
        ).eval()
    elif _is_qwen3(path):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype="auto", device_map="auto"
        ).eval()
    else:
        raise ValueError(
            f"unsupported VLM checkpoint (expect Qwen3-VL or Qwen3.5): {path}"
        )
    processor = AutoProcessor.from_pretrained(path)
    return model, processor


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def run_qwen_vl_interleaved(
    model,
    processor,
    content: List[Dict[str, Any]],
    max_new_tokens: int = 384,
) -> str:
    """Run one-turn interleaved text/image chat; return decoded text.

    Greedy decoding only (evaluator): do_sample=False, no temperature/top_p.
    """
    messages = [{"role": "user", "content": content}]

    template_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if getattr(model.config, "model_type", None) == "qwen3_5":
        template_kwargs["enable_thinking"] = False

    inputs = processor.apply_chat_template(messages, **template_kwargs)

    images = [part["image"] for part in content if part.get("type") == "image"]
    if "pixel_values" not in inputs and images:
        image_inputs = processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = image_inputs.pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    device = model.device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_trim = generated_ids[:, input_len:]
    text = processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _strip_thinking(text)

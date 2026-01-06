# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0


import os
import base64
import numpy as np
from typing import Dict, Tuple
from openai import OpenAI
import cv2

from geniesim.evaluator.config import load_llm_config

_current_dir = os.path.dirname(os.path.abspath(__file__))
SIM_REPO_ROOT = os.environ.get("SIM_REPO_ROOT")
API_KEY, BASE_URL, MODEL = load_llm_config(config_type="vlm")


def encode_image_to_base64(image_array: np.ndarray) -> str:
    if image_array.dtype != np.uint8:
        if image_array.dtype in [np.float32, np.float64]:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    success, encoded_image = cv2.imencode(".jpg", image_array, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not success:
        raise ValueError("image encoding failed")

    jpeg_bytes = encoded_image.tobytes()
    base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")

    return base64_str


def load_prompt() -> str:
    evaluator_dir = os.path.dirname(_current_dir)
    prompt_path = os.path.join(evaluator_dir, "prompts", "PROMPT_AUTOSCORE.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def auto_score(description: str, image_history: list) -> Tuple[bool, str]:
    if not image_history:
        raise ValueError("image history is empty")

    print(f"[auto_score] total image steps: {len(image_history)}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt_template = load_prompt()
    content = []

    intro_text = prompt_template.split("{images_placeholder}")[0].format(description=description)
    content.append({"type": "text", "text": intro_text})

    for step_idx, step_images in enumerate(image_history):
        content.append({"type": "text", "text": f"\n=== Step {step_idx} ==="})

        for view_name, image_array in step_images.items():
            # debug_filename = f"DEBUG_step{step_idx}_{view_name}.png"
            # cv2.imwrite(os.path.join(SIM_REPO_ROOT, debug_filename), image_array[:, :, ::-1])
            base64_image = encode_image_to_base64(image_array[:, :, ::-1])

            content.append({"type": "text", "text": f"[{view_name} viewpoint]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    outro_text = prompt_template.split("{images_placeholder}")[1]
    content.append({"type": "text", "text": outro_text})
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=4000,
        )
    except Exception as e:
        print(f"[auto_score] call model failed: {e}")
        raise
    try:
        raw_text = None
        if hasattr(response, "choices"):
            raw_text = getattr(response.choices[0].message, "content", None) or response.choices[0].message.content
        elif isinstance(response, dict):
            raw_text = response.get("choices", [{}])[0].get("message", {}).get("content")
        else:
            raw_text = str(response)
    except Exception:
        raw_text = str(response)

    if not raw_text:
        print("[auto_score] cannot extract text from response")
        return False, "cannot extract text from response"

    raw_text = raw_text.strip()
    print(f"\n{'='*60}")
    print(f"[auto_score] VLM full output:")
    print(f"{'='*60}")
    print(raw_text)
    print(f"{'='*60}\n")

    result = False
    reasoning = ""

    lines = raw_text.split("\n")
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith("result:"):
            result_text = line_lower.replace("result:", "").strip()
            if "true" in result_text:
                result = True
            elif "false" in result_text:
                result = False
        elif line_lower.startswith("reasoning:"):
            reasoning = line.replace("reasoning:", "", 1).replace("REASONING:", "", 1).strip()
            for j in range(i + 1, len(lines)):
                if not lines[j].strip().startswith("RESULT:") and not lines[j].strip().startswith("result:"):
                    reasoning += " " + lines[j].strip()
                else:
                    break
            break

    if not reasoning:
        raw_text_lower = raw_text.lower()
        if "true" in raw_text_lower:
            result = True
        elif "false" in raw_text_lower:
            result = False
        reasoning = raw_text

    # print(f"[auto_score] score result: {'✓ satisfied' if result else '✗ not satisfied'}")
    # print(f"[auto_score] reasoning: {reasoning}")

    return result, reasoning


if __name__ == "__main__":
    pass

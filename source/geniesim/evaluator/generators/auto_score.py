# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0


import os
import base64
import numpy as np
from typing import Dict, Tuple
from openai import OpenAI
import cv2
import re
import time

from geniesim.evaluator.config import load_llm_config
from geniesim.evaluator.utils import calculate_input_tokens

_current_dir = os.path.dirname(os.path.abspath(__file__))
SIM_REPO_ROOT = os.environ.get("SIM_REPO_ROOT", ".")
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


def save_debug_image(
    image_array: np.ndarray,
    debug_dir: str,
    timestamp: int,
    step_idx: int,
    view_name: str,
    target_size: Tuple[int, int] = None,
) -> bool:
    """
    Save debug image to disk.

    Args:
        image_array: Image array (H, W, C) in RGB format
        debug_dir: Directory to save debug images
        timestamp: Timestamp for filename
        step_idx: Step index
        view_name: View name (e.g., "head", "left_hand")
        target_size: Optional tuple (width, height) indicating the resize size

    Returns:
        True if saved successfully, False otherwise
    """
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return False

    # Ensure image is uint8 format
    debug_image = image_array.copy()
    if debug_image.dtype != np.uint8:
        if debug_image.max() <= 1.0:
            debug_image = (debug_image * 255).astype(np.uint8)
        else:
            debug_image = debug_image.astype(np.uint8)

    # Convert RGB to BGR for cv2 (image_array is RGB, cv2 needs BGR)
    if len(debug_image.shape) == 3 and debug_image.shape[2] == 3:
        debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        debug_filename = os.path.join(debug_dir, f"debug_{timestamp}_step{step_idx}_{view_name}.jpg")
        success = cv2.imwrite(debug_filename, debug_image_bgr)
        if success:
            if target_size is not None:
                width, height = target_size
                size_info = f"{width}x{height}"
            else:
                size_info = "original"
            print(f"[auto_score] DEBUG: Saved {size_info} image to {debug_filename}")
        else:
            print(f"[auto_score] DEBUG: Failed to save image to {debug_filename}")
        return success
    return False


def auto_score(
    description: str,
    image_history: list,
    target_size: Tuple[int, int] = None,
    save_debug_images: bool = False,
) -> Tuple[float, str]:
    """
    Evaluate task completion using VLM based on image history.

    Args:
        description: Task description to evaluate
        image_history: List of image dictionaries, each containing multiple camera views
        target_size: Optional tuple (width, height) to resize all images before encoding.
                     If None, images are encoded at their original size.
        save_debug_images: If True, save processed images to disk for debugging.

    Returns:
        Tuple of (score: float, reasoning: str)
    """
    if not image_history:
        raise ValueError("image history is empty")

    print(f"[auto_score] total image steps: {len(image_history)}")
    if target_size:
        print(f"[auto_score] Resizing all images to {target_size[0]}x{target_size[1]}")

    print(f"[auto_score] API_KEY: {API_KEY}")
    print(f"[auto_score] BASE_URL: {BASE_URL}")
    print(f"[auto_score] MODEL: {MODEL}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt_template = load_prompt()
    content = []

    intro_text = prompt_template.split("{images_placeholder}")[0].format(description=description)
    content.append({"type": "text", "text": intro_text})

    debug_image_dir = None
    timestamp = None
    if save_debug_images:
        debug_image_dir = os.path.join(SIM_REPO_ROOT, "debug_images", "auto_score")
        os.makedirs(debug_image_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)  # milliseconds

    for step_idx, step_images in enumerate(image_history):
        content.append({"type": "text", "text": f"\n=== Step {step_idx} ==="})

        for view_name, image_array in step_images.items():

            processed_image = image_array.copy()

            if target_size is not None:
                width, height = target_size
                if width > 0 and height > 0:
                    processed_image = cv2.resize(processed_image, (width, height), interpolation=cv2.INTER_LINEAR)

            if save_debug_images and debug_image_dir is not None and timestamp is not None:
                save_debug_image(processed_image, debug_image_dir, timestamp, step_idx, view_name, target_size)

            base64_image = encode_image_to_base64(processed_image[:, :, ::-1])

            content.append({"type": "text", "text": f"[{view_name} view]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    outro_text = prompt_template.split("{images_placeholder}")[1]
    content.append({"type": "text", "text": outro_text})

    token_info = calculate_input_tokens(content)
    print(f"\n[auto_score] Input Token Consumption:")
    print(f"  Text tokens: {token_info['text_tokens']}")
    print(f"  Image count: {token_info['image_count']}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=8000,  # Increased to accommodate detailed reasoning for multiple scoring points
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
        return 0.0, "cannot extract text from response"

    raw_text = raw_text.strip()
    print(f"\n{'='*60}")
    print(f"[auto_score] VLM full output:")
    print(f"{'='*60}")
    print(raw_text)
    print(f"{'='*60}\n")

    # Initialize default values
    score = 0.0
    reasoning = ""
    satisfied_count = None
    total_count = None
    scoring_points = []

    lines = raw_text.split("\n")
    current_field = None
    reasoning_lines = []
    scoring_points_lines = []

    # Parse the response in order: SCORING_POINTS -> REASONING -> counts -> SCORE
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        line_stripped = line.strip()

        if not line_stripped:
            continue

        # Extract SCORING_POINTS field
        if line_lower.startswith("scoring_points:"):
            current_field = "scoring_points"
            # Check if points are on the same line
            remainder = line.replace("scoring_points:", "", 1).replace("SCORING_POINTS:", "", 1).strip()
            if remainder:
                scoring_points_lines.append(remainder)
            continue

        # Extract REASONING field
        elif line_lower.startswith("reasoning:"):
            current_field = "reasoning"
            # Check if reasoning starts on the same line
            remainder = line.replace("reasoning:", "", 1).replace("REASONING:", "", 1).strip()
            if remainder:
                reasoning_lines.append(remainder)
            continue

        # Extract SATISFIED_COUNT field
        elif line_lower.startswith("satisfied_count:"):
            count_match = re.search(r"(\d+)", line_lower)
            if count_match:
                try:
                    satisfied_count = int(count_match.group(1))
                except ValueError:
                    pass
            current_field = None
            continue

        # Extract TOTAL_COUNT field
        elif line_lower.startswith("total_count:"):
            count_match = re.search(r"(\d+)", line_lower)
            if count_match:
                try:
                    total_count = int(count_match.group(1))
                except ValueError:
                    pass
            current_field = None
            continue

        # Extract SCORE field
        elif line_lower.startswith("score:"):
            score_text = line_lower.replace("score:", "").strip()
            score_match = re.search(r"(\d+\.?\d*)", score_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
                except ValueError:
                    pass
            current_field = None
            continue

        # Continue collecting based on current field
        if current_field == "scoring_points":
            # Stop if we hit another section
            if line_lower.startswith(("reasoning:", "satisfied_count:", "total_count:", "score:")):
                current_field = None
            else:
                scoring_points_lines.append(line_stripped)

        elif current_field == "reasoning":
            # Stop if we hit another section
            if line_lower.startswith(("satisfied_count:", "total_count:", "score:", "scoring_points:")):
                current_field = None
            else:
                reasoning_lines.append(line_stripped)

    # Join reasoning lines
    if reasoning_lines:
        reasoning = " ".join(reasoning_lines).strip()
    else:
        reasoning = raw_text  # Fallback to full text if no reasoning found

    # Count [TRUE] tags in reasoning if counts not provided
    if reasoning:
        true_count = len(re.findall(r"\[TRUE\]", reasoning, re.IGNORECASE))
        false_count = len(re.findall(r"\[FALSE\]", reasoning, re.IGNORECASE))

        # If counts not provided by LLM, derive from [TRUE]/[FALSE] tags
        if satisfied_count is None and true_count > 0:
            satisfied_count = true_count
            print(f"[auto_score] Derived satisfied_count from [TRUE] tags: {satisfied_count}")

        if total_count is None and (true_count + false_count) > 0:
            total_count = true_count + false_count
            print(f"[auto_score] Derived total_count from [TRUE]/[FALSE] tags: {total_count}")

    # Calculate score from counts if score not explicitly provided
    if score == 0.0 and satisfied_count is not None and total_count is not None and total_count > 0:
        score = satisfied_count / total_count
        print(f"[auto_score] Calculated score from counts: {satisfied_count}/{total_count} = {score:.3f}")

    # Print extracted information
    print(f"[auto_score] Final score: {score:.3f}")
    if satisfied_count is not None and total_count is not None:
        print(f"[auto_score] Satisfied: {satisfied_count}/{total_count}")

    return score, reasoning


if __name__ == "__main__":
    pass

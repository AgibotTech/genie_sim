# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# New: Function to analyze a single simulation asset thumbnail and return JSON, with CLI usage

import os
import re
import json
import base64
from openai import OpenAI
import time

from geniesim.evaluator.templates import INSTRUCTION_TEMPLATE
from geniesim.evaluator.config import load_llm_config

# Load LLM configuration from centralized config file
API_KEY, BASE_URL, MODEL = load_llm_config(config_type="llm")


_current_dir = os.path.dirname(os.path.abspath(__file__))


def collect_scene_info(scene_info_path):
    """
    Read scene_info and return json fields
    """
    with open(scene_info_path, "r", encoding="utf-8") as f:
        scene_info = json.load(f)
    return scene_info


def collect_prompt():
    """Read PROMPT.txt file"""
    # Go up one level to evaluator, then enter prompts directory
    evaluator_dir = os.path.dirname(_current_dir)
    prompt_path = os.path.join(evaluator_dir, "prompts", "PROMPT_INSTRUCTION.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


def generate_instructions(
    scene_dir,
    output_path=None,
    instruction_category=None,
):

    # Collect input data
    scene_info_path = os.path.join(scene_dir, "scene_info.json")
    scene_info = collect_scene_info(scene_info_path)
    prompt = collect_prompt()

    sys_prompt = prompt + "\n\n"
    sys_prompt += f"INSTRUCTION_TEMPLATE:\n{json.dumps(INSTRUCTION_TEMPLATE, ensure_ascii=False)}\n\n"
    sys_prompt += f"scene_info.json:\n{json.dumps(scene_info, ensure_ascii=False)}\n\n"

    user_prompt = f"instruction_category: {instruction_category}\n"
    user_prompt += (
        "Please generate the instructions according to the requirements above and output only the JSON object."
    )

    # Create client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Call model
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
    except Exception as e:
        print(f"[generate_instructions] Exception occurred when calling model: {e}")
        return None

    # Extract and parse model output
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
        print("[generate_instructions] Failed to extract text content from response")
        return None

    # Parse first JSON object
    try:
        m = re.search(r"(\{[\s\S]*\})", raw_text)
        json_text = m.group(1) if m else raw_text.strip()
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        # Try other parsing methods
        start = None
        for ch in ["{", "["]:
            idx = raw_text.find(ch)
            if idx != -1:
                start = idx
                break
        if start is not None:
            candidate = raw_text[start:]
            last_brace = max(candidate.rfind("}"), candidate.rfind("]"))
            if last_brace != -1:
                candidate = candidate[: last_brace + 1]
            try:
                parsed = json.loads(candidate)
            except Exception as e:
                print(f"[generate_instructions] Parsing attempt failed: {e}")
                print("Model raw output (first 500 chars):", raw_text[:500])
                return None
        else:
            print("[generate_instructions] JSON start marker not found, returning raw text for manual inspection")
            print("Model raw output (first 1000 chars):", raw_text[:1000])
            return None

    # Save result
    if output_path is None:
        # Auto-generate output path
        category = instruction_category or parsed.get("instruction_category", "unknown")
        output_path = os.path.join(scene_dir, f"instructions.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"[generate_instructions] Result saved to: {output_path}")
    print("=" * 100)

    if parsed.get("eval_type") == "VLM":
        print(f"[VLM] {parsed.get('instructions', '')}")
    else:
        instructions = parsed.get("instructions", [])
        if isinstance(instructions, list):
            for intr in instructions:
                if isinstance(intr, dict):
                    print(intr.get("instruction", ""))
                else:
                    print(intr)
        else:
            print(instructions)

    print("=" * 100)
    return parsed


if __name__ == "__main__":
    # CLI example: python instruction_gen.py --scene_info_path /path/to/scene_info.json --instruction_category color
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description="Analyze scene and generate instructions")
    parser.add_argument("--scene_dir", required=True, help="Scene directory path")
    parser.add_argument("--category", default="color", help="Instruction category")
    parser.add_argument("--output_path", help="Output file path")
    args = parser.parse_args()
    # fmt: on

    result = generate_instructions(
        scene_dir=args.scene_dir,
        output_path=args.output_path,
        instruction_category=args.category,
    )

    if result is None:
        print("Failed to generate instructions")
    else:
        print("Successfully generated instructions")

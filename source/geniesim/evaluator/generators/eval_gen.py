# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Rewrite: Read instruction_*.json and ADER_SYSTEM_PROMPT.md/PROMPT.txt from scene directory,
# call LLM and save results as problem{i}.json

import os
import re
import json
import glob
from openai import OpenAI

from geniesim.evaluator.config import load_llm_config

# Load LLM configuration from centralized config file
API_KEY, BASE_URL, MODEL = load_llm_config(config_type="llm")


_current_dir = os.path.dirname(os.path.abspath(__file__))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_first_json_from_text(text):
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # If the above fails, try a more robust extraction method: find the first { or [, then find the matching end
    start_idx = None
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx != -1:
            start_idx = idx
            break
    if start_idx is None:
        return None

    candidate = text[start_idx:]
    # Try to find the most likely end marker going forward
    last_brace = max(candidate.rfind("}"), candidate.rfind("]"))
    if last_brace != -1:
        candidate = candidate[: last_brace + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def generate_problems(
    scene_dir,
    temperature=0.0,
    max_tokens=10000,
):
    if not os.path.isdir(scene_dir):
        raise ValueError(f"{scene_dir} is not a valid directory")

    # Go up one level to evaluator, then enter prompts directory
    evaluator_dir = os.path.dirname(_current_dir)
    info_prompt_path = os.path.join(evaluator_dir, "prompts", "ADER_SYSTEM_PROMPT.md")
    sys_prompt_path = os.path.join(evaluator_dir, "prompts", "PROMPT_EVAL.txt")

    sys_prompt = read_text(sys_prompt_path).strip()
    info_prompt = read_text(info_prompt_path).strip()
    instructions = read_text(os.path.join(scene_dir, "instructions.json"))

    full_sys_prompt = sys_prompt + "\nADER SYSTEM PROMPT:\n" + info_prompt + "\ninstructions.json:\n" + instructions

    # scene_info_path = os.path.join(scene_dir, "scene_info.json")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    saved_files = []

    # Create message sequence: system uses ADER_SYSTEM_PROMPT.md content
    messages = [
        {"role": "system", "content": full_sys_prompt},
        {"role": "user", "content": "Generate evaluation config for the task"},
    ]

    print(f"[process_scene_dir] Calling model to generate problems.json")

    # Call model
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[process_scene_dir] Failed to call model: {e}")
        return []

    # Extract text
    raw_text = None
    try:
        if hasattr(response, "choices"):
            # Adapt to different return types
            choice = response.choices[0]
            # In new interface, choice may be object or dict
            if hasattr(choice, "message"):
                raw_text = getattr(choice.message, "content", None) or choice.message.content
            else:
                # dict form
                raw_text = (choice.get("message") or {}).get("content")
        elif isinstance(response, dict):
            raw_text = response.get("choices", [{}])[0].get("message", {}).get("content")
        else:
            raw_text = str(response)
    except Exception:
        raw_text = str(response)

    if not raw_text:
        print(f"[process_scene_dir] Model did not return text, saving raw response to file for debugging")
        raw_out_path = os.path.join(scene_dir, f"problem_raw.txt")
        with open(raw_out_path, "w", encoding="utf-8") as f:
            f.write(str(response))
        print(f"[process_scene_dir] Raw response saved: {raw_out_path}")
        return []

        # Parse JSON
    parsed = extract_first_json_from_text(raw_text)
    if parsed is None:
        # Parsing failed: save raw text for manual inspection
        raw_out_path = os.path.join(scene_dir, f"problem_raw.txt")
        with open(raw_out_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[process_scene_dir] Failed to parse JSON from model output. Raw text written to: {raw_out_path}")
        return []

    # Save as problem{i}.json
    out_path = os.path.join(scene_dir, f"problems.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        saved_files.append(out_path)
        print(f"[process_scene_dir] Result saved: {out_path}")
    except Exception as e:
        print(f"[process_scene_dir] Failed to save file: {out_path} -> {e}")

    return saved_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Call model to generate problems.json")
    parser.add_argument("--scene_dir", required=True, help="Scene directory path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Maximum token count")
    args = parser.parse_args()

    try:
        results = generate_problems(
            scene_dir=args.scene_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if results:
            print("Generation completed, saved files:")
            for p in results:
                print(" -", p)
        else:
            print(
                "Run completed but no problem files were generated (possibly all parsing failed or no instruction files)"
            )
    except Exception as e:
        print("Exception occurred during processing:", e)

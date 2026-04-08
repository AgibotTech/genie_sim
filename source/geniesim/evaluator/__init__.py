# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Evaluator module for task instruction generation, evaluation, and auto scoring
"""

from .config import load_llm_config
from .generators.eval_gen import generate_problems
from .generators.instruction_gen import generate_instructions
from .templates import INSTRUCTION_TEMPLATE

try:
    from .generators.auto_score import auto_score
except ModuleNotFoundError:  # Optional dependency (e.g. cv2)
    auto_score = None  # type: ignore[assignment]

__all__ = [
    "auto_score",
    "generate_problems",
    "generate_instructions",
    "INSTRUCTION_TEMPLATE",
    "load_llm_config",
]

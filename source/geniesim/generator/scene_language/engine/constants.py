import os
from typing import Literal
from pathlib import Path

PROJ_DIR: str = str(Path(__file__).parent.parent.absolute())
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

try:
    from .key import ANTHROPIC_API_KEY
except:
    print("Warning: No Anthropic keys found.")
    ANTHROPIC_API_KEY = ""
try:
    from .key import OPENAI_API_KEY
except:
    # print("Warning: No OpenAI keys found.")
    OPENAI_API_KEY = ""

try:
    import torch

    if torch.cuda.is_available():  # hack
        os.environ["MI_DEFAULT_VARIANT"] = "cuda_ad_rgb"
    else:
        os.environ["MI_DEFAULT_VARIANT"] = "scalar_rgb"
except ModuleNotFoundError:
    print(f"[INFO] torch not found, setting default variant to scalar_rgb")
    os.environ["MI_DEFAULT_VARIANT"] = "scalar_rgb"

import mitsuba as mi

mi.set_variant(os.environ["MI_DEFAULT_VARIANT"])

ENGINE_MODE: Literal["neural", "mi", "minecraft", "lmd", "mi_material", "exposed"] = os.getenv("ENGINE_MODE", "exposed")
# print(f'{ENGINE_MODE=}')
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"

PROMPT_MODE: Literal["default", "calc", "assert", "sketch"] = os.environ.get(
    "PROMPT_MODE", "default" if ENGINE_MODE == "minecraft" else "calc"
)
if ENGINE_MODE == "minecraft" and PROMPT_MODE != "default":
    print(f"WARNING {PROMPT_MODE=}")
if ENGINE_MODE == "mi" and PROMPT_MODE != "calc":
    print(f"WARNING {PROMPT_MODE=}")

ONLY_RENDER_ROOT = True

if "DRY_RUN" in os.environ:
    DRY_RUN = bool(os.environ["DRY_RUN"])
else:
    DRY_RUN = False
# print(f'DRY_RUN={DRY_RUN}')

# LLM configs
LLM_PROVIDER: Literal["gpt", "claude", "llama"] = "claude"
TEMPERATURE: float = 0.05
NUM_COMPLETIONS: int = 1
# MAX_TOKENS: int = 8192
MAX_TOKENS: int = 16384

assert 0 <= TEMPERATURE <= 1, TEMPERATURE
if NUM_COMPLETIONS > 1:
    assert TEMPERATURE > 0, TEMPERATURE

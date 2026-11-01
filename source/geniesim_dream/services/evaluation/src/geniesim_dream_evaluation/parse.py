"""Parse VLM JSON / fallback thinking-answer into success (0/1)."""

import json
import re
from typing import Any, Dict, Optional, Tuple


_ANSWER_RE = re.compile(r"answer\s*:\s*\[?\s*([01])\s*\]?", re.IGNORECASE)
_THINKING_RE = re.compile(
    r"thinking\s*:\s*(.*?)(?=\n\s*answer\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_LOOSE_01_RE = re.compile(r"(?<!\d)([01])(?!\d)")
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output, or None."""
    text = (raw_text or "").strip()
    if not text:
        return None

    fenced = _FENCE_RE.search(text)
    if fenced:
        blob = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        blob = text[start : end + 1]

    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_success(raw_text: str) -> Tuple[Optional[int], str, Optional[str]]:
    """
    Parse model output.

    Returns:
        success: 0, 1, or None if unparsable
        thinking: extracted thinking block (may be empty)
        error: error message when success is None
    """
    text = (raw_text or "").strip()
    if not text:
        return None, "", "empty_response"

    thinking = ""
    m_think = _THINKING_RE.search(text)
    if m_think:
        thinking = m_think.group(1).strip()

    m_ans = _ANSWER_RE.search(text)
    if m_ans:
        return int(m_ans.group(1)), thinking, None

    matches = list(_LOOSE_01_RE.finditer(text))
    if matches:
        return int(matches[-1].group(1)), thinking or text, None

    return None, thinking or text, "parse_failed"


def _as_01(value: Any) -> Optional[int]:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    if value is True:
        return 1
    if value is False:
        return 0
    return None


def parse_stage3_success(
    raw_text: str,
    verify: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], str, Optional[str]]:
    """Prefer JSON ``answer``; fall back to thinking/answer parsing."""
    thinking = ""
    if isinstance(verify, dict):
        mid = verify.get("mid_description")
        if isinstance(mid, str) and mid.strip():
            thinking = mid.strip()
        else:
            thinking = json.dumps(verify, ensure_ascii=False)
        ans = _as_01(verify.get("answer"))
        if ans is not None:
            return ans, thinking, None

    success, fb_thinking, err = parse_success(raw_text)
    return success, thinking or fb_thinking, err

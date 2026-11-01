"""Assemble and persist external-policy action trajectories from Dream sessions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_CHUNK_LENGTHS = (16, 32)
STEP_DIR_RE = re.compile(r"^step_(\d{4})$")


def _step_dirs(session_dir: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in session_dir.iterdir():
        if not path.is_dir():
            continue
        match = STEP_DIR_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise ValueError(f"no step_* directories under {session_dir}")
    expected = list(range(found[0][0], found[0][0] + len(found)))
    indices = [index for index, _ in found]
    if indices != expected:
        raise ValueError(f"non-contiguous step indices: {indices}")
    if found[0][0] != 0:
        raise ValueError(f"steps must start at step_0000, got step_{found[0][0]:04d}")
    return [path for _, path in found]


def assemble_normalized_actions(session_dir: Path) -> np.ndarray:
    """Stack step_*/action.normalized.json into [N, L, 29] float32."""
    session_dir = Path(session_dir)
    chunks: list[np.ndarray] = []
    for step_dir in _step_dirs(session_dir):
        path = step_dir / "action.normalized.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        payload = json.loads(path.read_text())
        array = np.asarray(payload, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 29:
            raise ValueError(f"bad normalized action shape {array.shape} in {path}")
        if array.shape[0] not in ALLOWED_CHUNK_LENGTHS:
            raise ValueError(
                f"chunk_length {array.shape[0]} not in {ALLOWED_CHUNK_LENGTHS} ({path})"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite values in {path}")
        chunks.append(array)
    stacked = np.stack(chunks, axis=0)
    lengths = {chunk.shape[0] for chunk in chunks}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent chunk lengths across steps: {sorted(lengths)}")
    return stacked


def write_session_actions(
    session_dir: Path,
    actions: np.ndarray,
    *,
    stats_sha256: str | None,
    representation: str | None = None,
    source: str = "external_policy",
) -> dict[str, Any]:
    """Write actions.npy + actions.meta.json into the session output directory."""
    session_dir = Path(session_dir)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != 29:
        raise ValueError(f"actions must be [N,L,29], got {actions.shape}")
    if actions.shape[1] not in ALLOWED_CHUNK_LENGTHS:
        raise ValueError(f"chunk_length {actions.shape[1]} not allowed")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain non-finite values")

    actions_path = session_dir / "actions.npy"
    np.save(actions_path, actions, allow_pickle=False)
    sha256 = hashlib.sha256(actions_path.read_bytes()).hexdigest()
    meta: dict[str, Any] = {
        "path": "actions.npy",
        "dtype": "float32",
        "shape": list(actions.shape),
        "normalized": True,
        "sha256": sha256,
        "stats_sha256": stats_sha256,
        "n_chunks": int(actions.shape[0]),
        "chunk_length": int(actions.shape[1]),
        "source": source,
    }
    if representation:
        meta["representation"] = representation
    (session_dir / "actions.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    return meta

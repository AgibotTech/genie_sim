# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""agent_id 的持久化 —— 进程重启后沿用同一个身份。

网关用 (job_uuid, agent_id) 识别重连:30 秒窗口内相同的二元组会复用原连接、保留已打开
的 session;换一个 agent_id 就是新 agent,parallelism=1 时旧隧道还占着名额,新拨号会被
拒绝。所以 agent_id 不能每次启动随机生成。

每张卡(CUDA_VISIBLE_DEVICES)一个文件,文件名 agent_<job_uuid>_<slot>.id。
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger("tunnel_agent")

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # 网关限制 agent_id ≤ 64 字符


def default_state_dir() -> Path:
    return Path(os.environ.get("SIMUBOTIX_STATE_DIR") or Path.home() / ".simubotix")


def _path(job_uuid: str, slot: str, state_dir: Path) -> Path:
    return state_dir / f"agent_{_UNSAFE.sub('_', job_uuid)}_{_UNSAFE.sub('_', slot)}.id"


def load_or_create_agent_id(job_uuid: str, slot: str, state_dir: Path | None = None) -> str:
    state_dir = state_dir or default_state_dir()
    path = _path(job_uuid, slot, state_dir)
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if _VALID_ID.match(existing):
        return existing

    new_id = str(uuid.uuid4())
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(new_id, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # 落盘失败不阻断启动:退化为本进程内固定,但重启后会换 id。
        logger.warning(
            "cannot persist agent_id to %s (%s); a restart will use a new id", path, exc
        )
    return new_id


def clear_agent_id(job_uuid: str, slot: str, state_dir: Path | None = None) -> None:
    state_dir = state_dir or default_state_dir()
    try:
        _path(job_uuid, slot, state_dir).unlink(missing_ok=True)
    except OSError:
        pass

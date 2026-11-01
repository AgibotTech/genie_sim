# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""User settings shared by the standalone launchers and the main CLI."""

import json
import os
from pathlib import Path
import tempfile


def settings_path():
    explicit = os.environ.get("GENIESIM_DATA_COLLECTION_CONFIG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    return base / "geniesim" / "data-collection.json"


def load_settings():
    path = settings_path()
    if not path.is_file():
        return {}
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return settings


def save_settings(values):
    path = settings_path()
    settings = load_settings()
    settings.update({key: str(value) for key, value in values.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(settings, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

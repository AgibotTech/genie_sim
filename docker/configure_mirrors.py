#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Apply image-wide Ubuntu and pip defaults; omitted or empty values inherit."""

from __future__ import annotations

import argparse
import configparser
import platform
import re
from pathlib import Path
from urllib.parse import urlsplit


def _validate_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or any(char.isspace() for char in value):
        raise ValueError(f"Expected an HTTP(S) repository URL, got {value!r}")


def _is_ubuntu(uri: str, previous: str) -> bool:
    host = urlsplit(uri).hostname or ""
    return (
        host in {"archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com"}
        or host.endswith(".archive.ubuntu.com")
        or bool(previous and uri.rstrip("/") == previous.rstrip("/"))
    )


def _target(uri: str, suites: list[str], mirror: str) -> str:
    if mirror != "upstream":
        return mirror
    if platform.machine() in {"aarch64", "arm64", "armv7l"} or urlsplit(uri).hostname == "ports.ubuntu.com":
        return "http://ports.ubuntu.com/ubuntu-ports"
    host = "security.ubuntu.com" if all(suite.endswith("-security") for suite in suites) else "archive.ubuntu.com"
    return f"http://{host}/ubuntu"


def _field(block: str, name: str) -> re.Match | None:
    return re.search(rf"^{name}:[^\n]*(?:\n[ \t]+[^\n]*)*", block, re.MULTILINE | re.IGNORECASE)


def _deb822(block: str, mirror: str, previous: str) -> tuple[str, int]:
    uris, suites = _field(block, "URIs"), _field(block, "Suites")
    if uris is None or suites is None:
        return block, 0
    uri_list = uris.group().partition(":")[2].split()
    suite_list = suites.group().partition(":")[2].split()
    signed_by = _field(block, "Signed-By")
    ubuntu_key = signed_by is not None and "ubuntu-archive-keyring.gpg" in signed_by.group()
    if not suite_list or not any(ubuntu_key or _is_ubuntu(uri, previous) for uri in uri_list):
        return block, 0
    security = [suite for suite in suite_list if suite.endswith("-security")]
    ordinary = [suite for suite in suite_list if not suite.endswith("-security")]
    if mirror == "upstream" and security and ordinary:
        parts = []
        for group in (ordinary, security):
            part = block[: suites.start()] + "Suites: " + " ".join(group) + block[suites.end() :]
            parts.append(_deb822(part, mirror, previous)[0].strip("\n"))
        return "\n\n".join(parts) + ("\n" if block.endswith("\n") else ""), 1
    replaced = [
        _target(uri, suite_list, mirror) if ubuntu_key or _is_ubuntu(uri, previous) else uri for uri in uri_list
    ]
    return block[: uris.start()] + "URIs: " + " ".join(replaced) + block[uris.end() :], 1


def _legacy(text: str, mirror: str, previous: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        if not _is_ubuntu(match["uri"], previous):
            return match.group()
        count += 1
        return match["prefix"] + _target(match["uri"], [match["suite"]], mirror) + match["suffix"]

    pattern = (
        r"^(?P<prefix>[ \t]*(?:deb|deb-src)\s+(?:\[[^\]]*\]\s+)?)(?P<uri>\S+)(?P<suffix>[ \t]+(?P<suite>\S+)[^\n]*)"
    )
    return re.sub(pattern, replace, text, flags=re.MULTILINE), count


def configure(root: Path, apt_mirror: str | None = None, pip_index: str | None = None) -> None:
    """Update managed sources under root, preserving third-party repositories and pip options."""
    if apt_mirror and apt_mirror != "upstream":
        _validate_url(apt_mirror)
    if pip_index:
        _validate_url(pip_index)
    if apt_mirror:
        state = root / "etc/geniesim/apt-mirror"
        previous = state.read_text().strip() if state.exists() else ""
        directory = root / "etc/apt/sources.list.d"
        files = [root / "etc/apt/sources.list", *sorted(directory.glob("*.list")), *sorted(directory.glob("*.sources"))]
        matches = 0
        updates = {}
        for path in files:
            if not path.is_file():
                continue
            text = path.read_text()
            if path.suffix == ".sources":
                parts = re.split(r"(\n[ \t]*\n)", text)
                for index in range(0, len(parts), 2):
                    parts[index], count = _deb822(parts[index], apt_mirror, previous)
                    matches += count
                updated = "".join(parts)
            else:
                updated, count = _legacy(text, apt_mirror, previous)
                matches += count
            if updated != text:
                updates[path] = updated
        if not matches:
            raise ValueError("No managed Ubuntu APT sources found; keep Ubuntu URIs or its archive signing key")
        for path, text in updates.items():
            path.write_text(text)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(apt_mirror + "\n")
        state.chmod(0o644)
    if pip_index:
        path = root / "etc/pip.conf"
        config = configparser.ConfigParser(interpolation=None)
        config.read(path)
        if not config.has_section("global"):
            config.add_section("global")
        config.set("global", "index-url", pip_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            config.write(stream)
        path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apt", help="Ubuntu mirror URL, or 'upstream'; empty inherits")
    parser.add_argument("--pip", help="Primary pip index URL; empty inherits")
    args = parser.parse_args()
    try:
        configure(Path("/"), args.apt, args.pip)
    except (OSError, ValueError, configparser.Error) as exc:
        parser.exit(2, f"Mirror configuration failed: {exc}\n")


if __name__ == "__main__":
    main()

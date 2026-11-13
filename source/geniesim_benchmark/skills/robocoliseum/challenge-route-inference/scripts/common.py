# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared private state and platform API helpers; no shell evaluation or token logging."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import tempfile
import urllib.error
import urllib.parse
import urllib.request


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".route-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_state(path):
    text = Path(path).expanduser().read_text()
    if text.lstrip().startswith("{"):
        return json.loads(text)
    state = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        if "=" not in line:
            raise ValueError("State must contain literal KEY=VALUE entries")
        key, value = line.split("=", 1)
        values = shlex.split(value, comments=True)
        if len(values) > 1 or value.startswith("$'"):
            raise ValueError("Unsupported shell expression in state; use JSON or shlex-quoted values")
        state[key] = values[0] if values else ""
    return state


def write_state(path, state):
    # JSON is also accepted by these helpers; .env stays compatible with existing skills.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".session-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            for key, value in state.items():
                f.write(key + "=" + shlex.quote(str(value)) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checked_url(url, schemes):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid endpoint URL or credentials embedded in URL")
    if parsed.fragment or parsed.query:
        raise ValueError("Use an endpoint without query/fragment; credentials belong in the private state")
    return url.rstrip("/")


def api(base, path, token, body=None):
    checked_url(base, ("http", "https"))
    if not token:
        raise ValueError("CHALLENGE_TOKEN missing; use challenge-login")
    request = urllib.request.Request(
        base.rstrip("/") + path,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")[:1200]
        raise RuntimeError(f"API {path}: HTTP {exc.code}: {message}") from None


def get_job(base, token, job_id, season=None):
    # /api/challenge/jobs lists one season per request and defaults to Season II, so a
    # Season I job is invisible without season=1. With no known season, search both.
    seasons = [str(season)] if season not in (None, "") else ["2", "1"]
    for current in seasons:
        page = 1
        while True:
            query = {"page": page, "per-page": 100, "season": current}
            listing = api(base, "/api/challenge/jobs?" + urllib.parse.urlencode(query), token)
            for item in listing.get("items", []):
                if int(item["id"]) == int(job_id):
                    return item
            if page >= listing.get("total_pages", 1):
                break
            page += 1
    raise ValueError("Job not found in this account/listing; verify API host, id and season")


def active_job(job):
    if job.get("detailed_status") in ("completed", "failed") or job.get("status") in (2, 3, 4, 5):
        raise ValueError("Job is terminal or stopping; do not connect/reconnect")


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        os.chmod(path, 0o600)
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def process_identity(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": fields[19],
            "cmdline": [v.decode(errors="replace") for v in (root / "cmdline").read_bytes().split(b"\0") if v],
        }
    except (OSError, IndexError):
        return None


def owns_process(record):
    current = process_identity(record["pid"])
    return bool(current and current == record.get("identity"))

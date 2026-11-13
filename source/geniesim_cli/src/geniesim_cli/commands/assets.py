# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Download version-pinned benchmark asset closures without a simulator or SDK."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gzip
import hashlib
from http.client import HTTPException
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from geniesim_cli._style import CYAN, GREEN, RED, RST, YELLOW


def digest(path):
    """Hash a file without reading the whole asset into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_name(name):
    """Reject paths that cannot be safely placed beneath a download root."""
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\0" in name
        or str(path) != name
        or name == "."
    ):
        raise ValueError(f"Unsafe asset path: {name!r}")
    return name


SOURCE_KINDS = {"modelscope", "huggingface"}


def _validate_source(name, source):
    """Validate one hosting source record; each pins its own commit and identity."""
    if not isinstance(source, dict):
        raise ValueError(f"Invalid source {name!r}")
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind for {name!r}: {kind!r}")
    endpoint = urlsplit(str(source.get("endpoint", "")))
    if (
        endpoint.scheme not in {"https", "http"}
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(f"Invalid endpoint for source {name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(source.get("dataset", ""))):
        raise ValueError(f"Invalid dataset identity for source {name!r}")
    if not re.fullmatch(r"[a-f0-9]{40}", str(source.get("revision", ""))):
        raise ValueError(f"Asset revision for source {name!r} must be a pinned commit, not a moving branch")
    return {"kind": kind, "endpoint": source["endpoint"], "dataset": source["dataset"], "revision": source["revision"]}


def load_index(path=None):
    """Load a shipped release index; future releases need a rebuilt index."""
    raw = (
        Path(path).expanduser().read_bytes()
        if path
        else files("geniesim_cli").joinpath("_asset_index.json.gz").read_bytes()
    )
    text = gzip.decompress(raw).decode("utf-8") if raw.startswith(b"\x1f\x8b") else raw.decode("utf-8")
    index = json.loads(text)
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported asset index schema")
    if not re.fullmatch(r"[a-f0-9]{40}", index["revision"]):
        raise ValueError("Asset revision must be a pinned commit, not a moving branch")
    endpoint = urlsplit(index["endpoint"])
    if (
        endpoint.scheme not in {"https", "http"}
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("Invalid dataset endpoint")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", index["dataset"]):
        raise ValueError("Invalid dataset identity")
    # Optional multi-source table; older indexes only carry the default
    # (ModelScope) endpoint/dataset/revision, which becomes the sole source.
    sources = index.get("sources")
    if not sources:
        sources = {"modelscope": {"kind": "modelscope", "endpoint": index["endpoint"],
                                  "dataset": index["dataset"], "revision": index["revision"]}}
    index["sources"] = {name: _validate_source(name, src) for name, src in sources.items()}
    default_source = index.get("default_source") or next(iter(index["sources"]))
    if default_source not in index["sources"]:
        raise ValueError("default_source must name a configured source")
    index["default_source"] = default_source
    names = set()
    for name, size, sha in index["files"]:
        safe_name(name)
        if name in names or type(size) is not int or size < 0 or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError(f"Invalid asset record: {name}")
        names.add(name)
    for task in index["tasks"].values():
        if any(type(i) is not int or not 0 <= i < len(index["files"]) for i in task["files"]):
            raise ValueError("Task references an invalid file index")
    return index


def resolve_source(index, name=None):
    """Pick a hosting source by name, defaulting to the index's default source.

    Accepts a bare index dict (e.g. from a hand-built test fixture) that has no
    ``sources`` table by deriving a single ModelScope source from its top-level
    endpoint/dataset/revision.
    """
    if not index.get("sources"):
        index["sources"] = {"modelscope": {"kind": "modelscope", "endpoint": index["endpoint"],
                                          "dataset": index["dataset"], "revision": index["revision"]}}
    name = name or index.get("default_source") or next(iter(index["sources"]))
    if name not in index["sources"]:
        raise ValueError(f"Unknown source {name!r}; choose from {sorted(index['sources'])}")
    return name, index["sources"][name]


def plan(index, selectors, source=None):
    """Union exact tasks, categories and underscore-delimited robot/category tags."""
    chosen = set()
    for selector in selectors:
        selector = selector.removesuffix(".yaml")
        matches = {
            name
            for name, task in index["tasks"].items()
            if name == selector or name.startswith(selector + "_") or task["category"] == selector
        }
        matches.update(index.get("groups", {}).get(selector, []))
        if not matches:
            raise ValueError(f"No indexed task matches {selector!r}; run geniesim assets list")
        chosen.update(matches)
    if not chosen:
        raise ValueError("At least one task selector is required")
    ids = sorted({i for name in chosen for i in index["tasks"][name]["files"]})
    rows = [{"path": index["files"][i][0], "size": index["files"][i][1], "sha256": index["files"][i][2]} for i in ids]
    contents = {r["sha256"]: r["size"] for r in rows}
    source_name, source = resolve_source(index, source)
    # Content identity, independent of which host serves the bytes.
    release_id = hashlib.sha256(
        (
            index["benchmark_commit"]
            + "\n"
            + "\n".join(f'{r["path"]}:{r["sha256"]}' for r in rows)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "source": source_name,
        "source_kind": source["kind"],
        "dataset": source["dataset"],
        "endpoint": source["endpoint"],
        "revision": source["revision"],
        "release_id": release_id,
        "benchmark_commit": index["benchmark_commit"],
        "selectors": selectors,
        "task_config_sha256": {name: index["tasks"][name].get("config_sha256") for name in sorted(chosen)},
        "tasks": sorted(chosen),
        "file_count": len(rows),
        "total_bytes": sum(r["size"] for r in rows),
        "unique_bytes": sum(contents.values()),
        "hdr_exr_count": sum(Path(r["path"]).suffix.lower() in {".hdr", ".exr"} for r in rows),
        "files": rows,
        "known_issues": [r["message"] for r in index.get("known_issues", []) if chosen.intersection(r["tasks"])],
    }


def target_path(root, name):
    """Prevent existing symlinks from redirecting asset or cache writes."""
    safe_name(name)
    target = root / name
    for path in (target, *target.parents):
        if path.is_symlink():
            raise ValueError(f"Symlink in download destination: {path}")
        if path == root:
            break
    return target


def valid(path, row):
    return path.is_file() and path.stat().st_size == row["size"] and digest(path) == row["sha256"]


def file_url(selection, row):
    """Build a fetch URL for the selected hosting source.

    ModelScope serves files through its HubApi repo endpoint with a FilePath
    query; HuggingFace serves them through a path-based resolve URL. Both are
    pinned to the same immutable revision and every byte is re-verified against
    the index SHA256 after transfer, so the source cannot alter the content.
    """
    source = selection["source"]
    endpoint = selection["endpoint"].rstrip("/")
    if source == "modelscope":
        query = urlencode({"Source": "SDK", "Revision": selection["revision"], "FilePath": row["path"], "View": False})
        return f'{endpoint}/api/v1/datasets/{selection["dataset"]}/repo?{query}'
    if source == "huggingface":
        # Path segments may contain spaces (e.g. "Color Correction.jpg"); quote
        # each segment so the path survives the resolve redirect to the CDN.
        path = "/".join(quote(segment, safe="") for segment in row["path"].split("/"))
        return f'{endpoint}/datasets/{selection["dataset"]}/resolve/{selection["revision"]}/{path}'
    raise ValueError(f"Unsupported source {source!r}")


def transfer(selection, row, partial):
    """Resume only a valid HTTP range, then verify the full expected payload."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row["size"]:
        partial.unlink()
        offset = 0
    if offset == row["size"] and partial.exists():
        if valid(partial, row):
            return
        partial.unlink()
        offset = 0
    request = Request(file_url(selection, row), headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=90) as response:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not re.fullmatch(rf'bytes {offset}-\d+/{row["size"]}', content_range):
                raise ValueError("Server returned an incompatible Content-Range")
        elif response.status == 200:
            offset = 0
        else:
            raise ValueError(f"Unexpected HTTP status: {response.status}")
        with partial.open("ab" if offset else "wb") as output:
            size = offset
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > row["size"]:
                    raise ValueError("Remote payload exceeds recorded asset size")
                output.write(block)
    if not valid(partial, row):
        partial.unlink(missing_ok=True)
        raise ValueError("Downloaded asset size or SHA256 mismatch")


@contextmanager
def destination_lock(root):
    """Only one downloader may mutate a destination at a time."""
    import fcntl

    path = target_path(root, ".geniesim-download.lock")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another asset download is using this destination") from exc
        yield


def write_json(path, data):
    temp = path.with_name(path.name + ".tmp")
    if temp.is_symlink():
        raise ValueError(f"Symlink at output temporary file: {temp}")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def download(selection, root, workers=8, overwrite=False):
    """Deduplicate locally by hash; never accept unverified bytes as an asset."""
    root = root.expanduser().absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Download root must not contain symlinks")
    root.mkdir(parents=True, exist_ok=True)
    with destination_lock(root):
        receipt = target_path(root, ".geniesim-assets.json")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            # Identify a release by content (benchmark baseline + selected
            # hashes), not by source revision: ModelScope and HuggingFace pin
            # different commits for identical bytes, so switching source into
            # the same destination must not look like a different release.
            if previous.get("release_id") and selection.get("release_id"):
                if previous["release_id"] != selection["release_id"]:
                    raise ValueError("Destination belongs to a different release; choose another directory")
            elif previous.get("revision") != selection["revision"]:
                raise ValueError("Destination belongs to a different release; choose another directory")
        cache = target_path(root, ".geniesim-download/objects")
        cache.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        for row in selection["files"]:
            path = target_path(root, row["path"])
            if path.exists() and not valid(path, row) and not overwrite:
                raise ValueError(
                    f'Existing file differs from release: {row["path"]}; choose an empty destination or use --overwrite'
                )
            groups[row["sha256"]].append(row)
        required = sum(
            rows[0]["size"]
            for sha, rows in groups.items()
            if not valid(target_path(root, f".geniesim-download/objects/{sha}"), rows[0])
            and not any(valid(target_path(root, r["path"]), r) for r in rows)
        )
        if shutil.disk_usage(root).free < required + 256 * 1024**2:
            raise ValueError(f"Insufficient disk space; need about {required / 1024**3:.2f} GiB plus 256 MiB")

        def fetch(rows):
            row = rows[0]
            blob = target_path(root, ".geniesim-download/objects/" + row["sha256"])
            if not valid(blob, row):
                candidate = next(
                    (target_path(root, r["path"]) for r in rows if valid(target_path(root, r["path"]), r)), None
                )
                if candidate is not None:
                    blob.unlink(missing_ok=True)
                    os.link(candidate, blob)
                else:
                    partial = target_path(root, ".geniesim-download/objects/" + row["sha256"] + ".part")
                    for attempt in range(3):
                        try:
                            transfer(selection, row, partial)
                            partial.replace(blob)
                            break
                        except (OSError, ValueError, HTTPException) as exc:
                            if attempt == 2:
                                label = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
                                raise ValueError(f'Download failed: {row["path"]} ({label}); rerun to resume') from None
                            time.sleep(attempt + 1)
            for alias in rows:
                path = target_path(root, alias["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                if not valid(path, alias):
                    temp = path.with_name(path.name + ".geniesim-part")
                    target_path(root, str(temp.relative_to(root)))
                    temp.unlink(missing_ok=True)
                    os.link(blob, temp)
                    temp.replace(path)
            return len(rows)

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, rows) for rows in groups.values()]
            for future in as_completed(futures):
                completed += future.result()
                if completed % 50 == 0 or completed == selection["file_count"]:
                    print(f'{CYAN}Downloaded/verified {completed}/{selection["file_count"]} files{RST}', flush=True)
        result = {**selection, "status": "downloaded_and_sha256_verified"}
        write_json(receipt, result)
        return result


def verify(selection, root):
    """Verify selected asset bytes only; this does not initialize a task."""
    root = root.expanduser().absolute()
    errors = [r["path"] for r in selection["files"] if not valid(target_path(root, r["path"]), r)]
    return {
        "revision": selection["revision"],
        "file_count": selection["file_count"],
        "errors": errors,
        "status": "failed" if errors else "sha256_verified",
    }


def run(argv):
    parser = argparse.ArgumentParser(prog="geniesim assets", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ["list", "plan", "download", "verify"]:
        sub = actions.add_parser(action)
        sub.add_argument(
            "selectors", nargs="*" if action == "list" else "+", help="Exact task, robot_category tag or category"
        )
        sub.add_argument("--index", type=Path, help="Alternate versioned dependency index")
        sub.add_argument(
            "--source",
            help="Hosting source for asset bytes (e.g. modelscope, huggingface); defaults to the index default",
        )
        if action == "plan":
            sub.add_argument("--output", type=Path, help="Save the full selected file manifest")
        if action in {"download", "verify"}:
            sub.add_argument("--dest", required=True, type=Path, help="Independent asset directory")
        if action == "download":
            sub.add_argument("--workers", type=int, choices=range(1, 33), default=8, metavar="1..32")
            sub.add_argument(
                "--overwrite", action="store_true", help="Replace existing files that differ from this release"
            )
    options = parser.parse_args(argv)
    try:
        index = load_index(options.index)
        if options.action == "list":
            names = (
                plan(index, options.selectors, options.source)["tasks"]
                if options.selectors
                else sorted(index["tasks"])
            )
            for name in names:
                task = index["tasks"][name]
                print(f'{name}  [{task["runtime"]}; {task["instances"]} instances]')
            default = resolve_source(index, options.source)
            print(
                f'{CYAN}{len(names)} indexed tasks; source {default[0]} '
                f'({default[1]["kind"]}) revision {default[1]["revision"]}{RST}'
            )
            return
        selection = plan(index, options.selectors, options.source)
        print(
            f'{CYAN}{len(selection["tasks"])} tasks; {selection["file_count"]} files; {selection["total_bytes"] / 1024**3:.2f} GiB logical / {selection["unique_bytes"] / 1024**3:.2f} GiB unique; {selection["hdr_exr_count"]} HDR/EXR{RST}'
        )
        print(
            f'{CYAN}Source: {selection["source"]} ({selection["source_kind"]}); revision: {selection["revision"]}; '
            f'benchmark: {selection["benchmark_commit"]}{RST}'
        )
        for issue in selection["known_issues"]:
            print(f"{YELLOW}Warning: {issue}{RST}")
        if options.action == "plan":
            if options.output:
                write_json(options.output.expanduser(), selection)
        elif options.action == "download":
            download(selection, options.dest, options.workers, options.overwrite)
            print(f"{GREEN}Asset bytes downloaded and verified: {options.dest.expanduser().absolute()}{RST}")
        else:
            result = verify(selection, options.dest)
            for name in result["errors"]:
                print(f"{RED}Missing or changed: {name}{RST}")
            if result["errors"]:
                raise SystemExit(1)
            print(f"{GREEN}All selected file sizes and SHA256 values match.{RST}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{RED}Asset command failed: {exc}{RST}")
        raise SystemExit(1) from None

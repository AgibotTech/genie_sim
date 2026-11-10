"""Exercise artifact acquisition with a real HTTP server and no GPU or pip."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("download_installer", ROOT / "docker/install_libuipc.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.fixture
def release(monkeypatch, tmp_path):
    payload = b"immutable fixture wheel bytes\n"
    sha = hashlib.sha256(payload).hexdigest()
    state = {
        "requests": [],
        "bad": False,
        "ignore_range": False,
        "truncate": False,
        "slow": False,
        "missing": False,
        "metadata_sha": "a" * 40,
        "metadata_requests": [],
        "modelscope_failure": None,
        "huggingface_failure": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            provider = "huggingface" if self.path.startswith(("/api/datasets/", "/datasets/")) else "modelscope"
            failure = state[provider + "_failure"]
            if self.path.startswith("/api/datasets/") or "/commits?" in self.path:
                state["metadata_requests"].append(self.path)
                if failure == "metadata":
                    self.send_error(503)
                    return
                metadata_payload = (
                    {"sha": state["metadata_sha"]}
                    if self.path.startswith("/api/datasets/")
                    else {"Code": 200, "Data": {"Commit": [{"Id": state["metadata_sha"]}]}}
                )
                if failure == "malformed":
                    metadata_payload = state["malformed_metadata"]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(metadata_payload).encode())
                return
            state["requests"].append((self.path, self.headers.get("Range")))
            if state["missing"] or failure == "missing":
                self.send_error(404)
                return
            if failure == "http":
                self.send_error(503)
                return
            content = b"x" * len(payload) if state["bad"] or failure == "hash" else payload
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            if state["ignore_range"]:
                offset = 0
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(content)-1}/{len(content)}")
            self.send_header("Content-Length", str(len(content) - offset))
            self.end_headers()
            if state["truncate"]:
                state["truncate"] = False
                self.wfile.write(content[offset : offset + 5])
            elif state["slow"] or failure == "slow":
                try:
                    for byte in content[offset:]:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        threading.Event().wait(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.wfile.write(content[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = {
        "schema_version": 1,
        "default_source": "modelscope",
        "sources": {
            name: {"endpoint": f"http://127.0.0.1:{server.server_port}", "dataset": "test/assets", "revision": "a" * 40}
            for name in ("modelscope", "huggingface")
        },
        "artifacts": {
            "89": {
                "filename": "pyuipc-sm89.whl",
                "path": "wheels/libuipc/test/pyuipc-sm89.whl",
                "size": len(payload),
                "sha256": sha,
            }
        },
    }
    monkeypatch.setattr(installer, "ARTIFACTS", manifest, raising=False)
    monkeypatch.setattr(installer, "WHEELS", {89: ("pyuipc-sm89.whl", sha)})
    for key in ("GENIESIM_ASSETS_SRC", "GENIESIM_LIBUIPC_SOURCE", "GENIESIM_LIBUIPC_REVISION"):
        monkeypatch.delenv(key, raising=False)
    cache = tmp_path / "cache"
    monkeypatch.setenv("GENIESIM_LIBUIPC_CACHE", str(cache))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    try:
        yield manifest, payload, state, cache, tmp_path
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_downloads_only_selected_wheel_and_reuses_cache(release):
    manifest, payload, state, _, _ = release
    wheel = installer.select_wheel(89)
    assert wheel.read_bytes() == payload
    assert len(state["requests"]) == 1
    assert "Revision=" + "a" * 40 in state["requests"][0][0]
    assert installer.select_wheel(89) == wheel
    assert len(state["requests"]) == 1


def test_local_assets_need_no_published_source(release, monkeypatch):
    manifest, payload, state, _, root = release
    for source in manifest["sources"].values():
        source["revision"] = None
    local = root / "assets" / manifest["artifacts"]["89"]["path"]
    local.parent.mkdir(parents=True)
    local.write_bytes(payload)
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(root / "assets"))
    assert installer.select_wheel(89).read_bytes() == payload
    assert not state["requests"]


def test_unpublished_release_reports_missing_wheel(release):
    manifest, _, state, _, _ = release
    manifest["sources"]["modelscope"].pop("revision")
    state["missing"] = True
    with pytest.raises(RuntimeError, match="not published"):
        installer.select_wheel(89)
    assert len(state["metadata_requests"]) == 1
    assert len(state["requests"]) == 2


@pytest.mark.parametrize("source", ["modelscope", "huggingface"])
def test_publication_resolves_real_commit_without_code_changes(release, monkeypatch, source):
    manifest, payload, state, _, _ = release
    manifest["sources"][source].pop("revision")
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", source)
    assert installer.select_wheel(89).read_bytes() == payload
    assert len(state["metadata_requests"]) == 1
    assert "a" * 40 in state["requests"][0][0]
    assert "/resolve/master/" not in state["requests"][0][0]
    assert "Revision=master" not in state["requests"][0][0]


def test_invalid_discovered_commit_never_fetches_payload(release, monkeypatch):
    manifest, _, state, _, _ = release
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    manifest["sources"]["huggingface"].pop("revision")
    state["metadata_sha"] = "master"
    with pytest.raises(RuntimeError, match="commit"):
        installer.select_wheel(89)
    assert not state["requests"]


@pytest.mark.parametrize("source", ["modelscope", "huggingface"])
def test_explicit_pinned_revision_skips_discovery(release, monkeypatch, source):
    manifest, payload, state, _, _ = release
    manifest["sources"][source]["revision"] = None
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", source)
    monkeypatch.setenv("GENIESIM_LIBUIPC_REVISION", "b" * 40)
    assert installer.select_wheel(89).read_bytes() == payload
    assert not state["metadata_requests"]
    assert "b" * 40 in state["requests"][0][0]
    assert ("/resolve/" in state["requests"][0][0]) == (source == "huggingface")


@pytest.mark.parametrize("ignore_range", [False, True])
def test_interrupted_transfer_resumes_or_restarts(release, ignore_range):
    manifest, payload, state, cache, _ = release
    row = manifest["artifacts"]["89"]
    partial = cache / row["sha256"] / (row["filename"] + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:5])
    state["ignore_range"] = ignore_range
    assert installer.select_wheel(89).read_bytes() == payload
    assert state["requests"][0][1] == "bytes=5-"
    assert not partial.exists()


def test_bad_payload_never_publishes_wheel(release):
    _, _, state, cache, _ = release
    state["bad"] = True
    with pytest.raises(RuntimeError, match="Download failed"):
        installer.select_wheel(89)
    assert not list(cache.rglob("*.whl"))
    assert len(state["requests"]) == 6


def test_local_hash_mismatch_is_not_silently_replaced(release, monkeypatch):
    manifest, _, state, _, root = release
    local = root / "assets" / manifest["artifacts"]["89"]["path"]
    local.parent.mkdir(parents=True)
    local.write_bytes(b"changed")
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(root / "assets"))
    with pytest.raises(RuntimeError, match="SHA256|size"):
        installer.select_wheel(89)
    assert not state["requests"]


def test_moving_revision_rejected(release, monkeypatch):
    _, _, state, _, _ = release
    monkeypatch.setenv("GENIESIM_LIBUIPC_REVISION", "main")
    with pytest.raises(RuntimeError, match="pinned"):
        installer.select_wheel(89)
    assert not state["requests"]


def test_download_only_works_without_runtime_abi_or_gpu(release, monkeypatch):
    _, _, state, _, _ = release
    monkeypatch.setattr(installer, "validate_platform", lambda: pytest.fail("runtime ABI checked"))
    monkeypatch.setattr(installer, "detect_sm", lambda _: pytest.fail("GPU queried"))
    monkeypatch.setattr(installer, "install", lambda *a, **k: pytest.fail("pip invoked"))
    installer.main(["--download-sm", "89"])
    assert len(state["requests"]) == 1


def test_verified_cache_is_read_only_and_needs_no_lock(release, monkeypatch):
    _, _, state, _, _ = release
    wheel = installer.select_wheel(89)
    monkeypatch.setattr(installer, "_download_lock", lambda _: pytest.fail("download lock opened"))
    assert installer.select_wheel(89) == wheel
    assert len(state["requests"]) == 1


def test_early_eof_keeps_partial_for_next_range_request(release):
    _, payload, state, _, _ = release
    state["truncate"] = True
    assert installer.select_wheel(89).read_bytes() == payload
    assert len(state["requests"]) == 2
    assert state["requests"][1][1] == "bytes=5-"


def test_continuous_slow_stream_hits_total_deadline(release, monkeypatch):
    _, _, state, cache, _ = release
    state["slow"] = True
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    monkeypatch.setattr(installer, "DOWNLOAD_TIMEOUT", 0.05, raising=False)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="deadline"):
        installer.select_wheel(89)
    assert time.monotonic() - started < 0.25
    assert len(state["requests"]) == 1
    assert not list(cache.rglob("*.whl"))
    assert list(cache.rglob("*.part"))


def test_slow_response_headers_are_covered_by_elapsed_deadline(release, monkeypatch):
    import socketserver

    manifest, _, _, _, _ = release

    class SlowHeaders(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(8192)
            try:
                for byte in b"HTTP/1.0 200 OK\r\nContent-Length: 29\r\n\r\n":
                    self.request.sendall(bytes([byte]))
                    threading.Event().wait(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = socketserver.TCPServer(("127.0.0.1", 0), SlowHeaders)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest["sources"]["huggingface"]["endpoint"] = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    monkeypatch.setattr(installer, "DOWNLOAD_TIMEOUT", 0.05)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="deadline"):
            installer.select_wheel(89)
        assert time.monotonic() - started < 0.2
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("failure", ["http", "missing", "hash", "metadata"])
def test_modelscope_failure_falls_back_to_huggingface(release, failure, capsys):
    manifest, payload, state, _, _ = release
    state["modelscope_failure"] = failure
    manifest["sources"]["huggingface"]["revision"] = "c" * 40
    if failure == "metadata":
        manifest["sources"]["modelscope"].pop("revision")
    assert installer.select_wheel(89).read_bytes() == payload
    hf_requests = [url for url, _ in state["requests"] if url.startswith("/datasets/")]
    assert len(hf_requests) == 1 and "/resolve/" + "c" * 40 + "/" in hf_requests[0]
    assert "huggingface" in capsys.readouterr().out


def test_source_specific_revision_is_not_reused_for_fallback(release, monkeypatch):
    manifest, payload, state, _, _ = release
    state["modelscope_failure"] = "http"
    monkeypatch.setenv("GENIESIM_LIBUIPC_REVISION", "b" * 40)
    manifest["sources"]["huggingface"]["revision"] = "c" * 40
    assert installer.select_wheel(89).read_bytes() == payload
    hf_url = next(url for url, _ in state["requests"] if url.startswith("/datasets/"))
    assert "c" * 40 in hf_url and "b" * 40 not in hf_url


def test_explicit_huggingface_failure_does_not_retry_modelscope(release, monkeypatch):
    _, _, state, _, _ = release
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    state["huggingface_failure"] = "http"
    with pytest.raises(RuntimeError, match="Download failed"):
        installer.select_wheel(89)
    assert all(url.startswith("/datasets/") for url, _ in state["requests"])


def test_modelscope_timeout_reserves_time_and_resumes_on_hf(release, monkeypatch):
    _, payload, state, _, _ = release
    state["modelscope_failure"] = "slow"
    monkeypatch.setattr(installer, "DOWNLOAD_TIMEOUT", 0.2)
    started = time.monotonic()
    assert installer.select_wheel(89).read_bytes() == payload
    assert time.monotonic() - started < 0.4
    hf = [(url, header) for url, header in state["requests"] if url.startswith("/datasets/")]
    assert len(hf) == 1 and hf[0][1].startswith("bytes=")


def test_modelscope_connection_failure_falls_back(release):
    manifest, payload, state, _, _ = release
    manifest["sources"]["modelscope"]["endpoint"] = "http://127.0.0.1:1"
    assert installer.select_wheel(89).read_bytes() == payload
    assert len(state["requests"]) == 1
    assert state["requests"][0][0].startswith("/datasets/")


@pytest.mark.parametrize("metadata", [None, [], {"Code": 500, "Data": None}, {"Code": 200, "Data": {"Commit": [None]}}])
def test_malformed_modelscope_metadata_falls_back(release, metadata):
    manifest, payload, state, _, _ = release
    manifest["sources"]["modelscope"].pop("revision")
    state["modelscope_failure"] = "malformed"
    state["malformed_metadata"] = metadata
    assert installer.select_wheel(89).read_bytes() == payload
    assert len(state["requests"]) == 1
    assert state["requests"][0][0].startswith("/datasets/")

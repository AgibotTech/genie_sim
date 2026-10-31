# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Local observer lifecycle and read-only HTTP protocol on ephemeral loopback ports."""

import json
import multiprocessing
import socket
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import pytest

from agent2act.bootstrap import ApplicationFactory, create_observer
from agent2act.configuration import parse_settings
from agent2act.service.observer.server import ObserverServer


@pytest.mark.integration
def test_packaged_observer_process_and_read_only_routes(tmp_path, config, request_payload):
    root = tmp_path / "agent.trace"
    settings = parse_settings({**config, "logging": {"trace_dir": str(root)}})
    with ApplicationFactory(settings).create_session() as session:
        session.handle(request_payload)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    observer = create_observer(root)
    try:
        base = observer.url
        with urlopen(base, timeout=5) as response:
            assert "Run Observer" in response.read().decode()
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        for asset in ("app.js", "style.css"):
            with urlopen(base + asset, timeout=5) as response:
                assert response.status == 200
        with urlopen(base + "api/catalog", timeout=5) as response:
            catalog = json.load(response)
        key = catalog["requests"][0]["id"]
        with urlopen(base + "api/request?" + urlencode({"id": key}), timeout=5) as response:
            detail = json.load(response)
        name = detail["artifacts"][0]["name"]
        with urlopen(base + "api/artifact?" + urlencode({"id": key, "name": name}), timeout=5) as response:
            assert response.headers["Content-Type"].startswith("application/json")
        invalid = [
            base + "api/artifact?" + urlencode({"id": key, "name": "../../hardware.json"}),
            base + "api/image?" + urlencode({"path": "../secret.png"}),
            base.replace(urlsplit(base).path, "/") + "api/catalog",
        ]
        for url in invalid:
            with pytest.raises(HTTPError) as error:
                urlopen(url, timeout=5)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "api/catalog", method="POST"), timeout=5)
        assert error.value.code == 405
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "api/catalog", headers={"Host": "untrusted.example"}), timeout=5)
        assert error.value.code == 404
    finally:
        observer.close()
        observer.close()
    assert before == {p: p.read_bytes() for p in before}
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", urlsplit(base).port)) != 0


@pytest.mark.integration
def test_port_failure_cleans_subprocess(tmp_path):
    children = {p.pid for p in multiprocessing.active_children()}
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError, match="could not start"):
            create_observer(tmp_path, occupied.getsockname()[1])
    assert {p.pid for p in multiprocessing.active_children()} == children


@pytest.mark.integration
def test_empty_live_server_observes_records_added_later(tmp_path, config, request_payload):
    root = tmp_path / "agent.trace"
    with ObserverServer(root) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(server.url + "api/catalog", timeout=5) as response:
                assert json.load(response)["requests"] == []
            with ApplicationFactory(
                parse_settings({**config, "logging": {"trace_dir": str(root)}})
            ).create_session() as session:
                session.handle(request_payload)
            with urlopen(server.url + "api/catalog", timeout=5) as response:
                assert len(json.load(response)["requests"]) == 1
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.integration
def test_serve_ui_end_to_end_and_shutdown(tmp_path, config, request_payload):
    import subprocess
    import sys
    import time

    import msgpack
    from websockets.sync.client import connect

    config["server"]["port"] = 0
    path, log, stdout = (tmp_path / name for name in ("config.json", "run.jsonl", "stdout.txt"))
    path.write_text(json.dumps(config))
    root = tmp_path / "run.trace"
    with stdout.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent2act",
                "serve",
                "--config",
                str(path),
                "--ui",
                "--no-browser",
                "--trace-dir",
                str(root),
                "--log-file",
                str(log),
            ],
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 15
            ready = None
            while time.monotonic() < deadline and process.poll() is None:
                lines = log.read_text().splitlines() if log.exists() else []
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("message") == "server_started":
                        ready = event
                if ready:
                    break
                time.sleep(0.02)
            assert ready, stdout.read_text()
            url = next(
                line.removeprefix("Observer: ")
                for line in stdout.read_text().splitlines()
                if line.startswith("Observer: ")
            )
            with connect(f"ws://127.0.0.1:{ready['port']}", open_timeout=5) as connection:
                connection.send(msgpack.packb(request_payload, use_bin_type=True))
                result = msgpack.unpackb(connection.recv(timeout=5), raw=False)
                assert "result" in result
            with urlopen(url + "api/catalog", timeout=5) as response:
                catalog = json.load(response)
            assert catalog["requests"][0]["status"] == "issued"
            assert len(catalog["episodes"]) == 1
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    assert process.returncode == 0
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", urlsplit(url).port)) != 0

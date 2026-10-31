# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Loopback-only, token-scoped, read-only HTTP access to one trace directory."""

import json
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .store import TraceStore, read_json

ASSETS = Path(__file__).with_name("assets")


class ObserverServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: Path, port: int = 0) -> None:
        self.store = TraceStore(root)
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)
        super().__init__(("127.0.0.1", port), ObserverHandler)
        self.url = f"http://127.0.0.1:{self.server_port}/{self.token}/"


class ObserverHandler(BaseHTTPRequestHandler):
    server: ObserverServer

    def log_message(self, format: str, *args: Any) -> None:
        # Do not print access tokens, prompt text or attachment names to stderr.
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any) -> None:
        self._send(
            200,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        prefix = f"/{self.server.token}/"
        parsed = urlsplit(self.path)
        host = self.headers.get("Host", "")
        if host != f"127.0.0.1:{self.server.server_port}" or not parsed.path.startswith(prefix):
            self._send(404, b"Not found", "text/plain")
            return
        route = parsed.path[len(prefix) :]
        query = parse_qs(parsed.query)
        try:
            if route in ("", "app.js", "style.css"):
                path = ASSETS / (route or "index.html")
                self._send(
                    200,
                    path.read_bytes(),
                    (mimetypes.guess_type(path)[0] or "text/plain") + "; charset=utf-8",
                )
                return
            with self.server.lock:
                if route == "api/catalog":
                    offset = max(0, int(query.get("offset", ["0"])[0]))
                    value = self.server.store.catalog(query.get("episode", [None])[0], offset)
                elif route == "api/request":
                    value = self.server.store.detail(query["id"][0])
                elif route == "api/artifact":
                    key, name = query["id"][0], query["name"][0]
                    if key not in self.server.store.rows or "/" in name or "\\" in name:
                        raise KeyError("artifact")
                    records, _ = self.server.store.artifacts(key)
                    if name not in {r["name"] for r in records}:
                        raise KeyError("artifact")
                    value = read_json(self.server.store.resolve(f"{key}/{name}"))
                elif route == "api/image":
                    relative = query["path"][0]
                    # Images must belong to a known request's assets; never serve arbitrary files.
                    parts = Path(relative).parts
                    if (
                        len(parts) != 4
                        or parts[2] != "assets"
                        or "/".join(parts[:2]) not in self.server.store.rows
                    ):
                        raise KeyError("image")
                    path = self.server.store.resolve(relative)
                    data = path.read_bytes()
                    if len(data) > 32 * 1024 * 1024:
                        raise ValueError("image too large")
                    kind = (
                        "image/jpeg"
                        if data.startswith(b"\xff\xd8")
                        else "image/png"
                        if data.startswith(b"\x89PNG\r\n\x1a\n")
                        else None
                    )
                    if kind is None:
                        raise ValueError("not an image")
                    self._send(200, data, kind)
                    return
                else:
                    raise KeyError("route")
            self._json(value)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            self._send(404, b"Recording unavailable or incomplete; retry on refresh", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        self._send(405, b"Read only", "text/plain")

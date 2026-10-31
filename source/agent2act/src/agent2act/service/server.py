# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded CoRobot WebSocket service with per-connection resource ownership."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any

import msgpack
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from agent2act.domain.errors import (
    ConfigurationError,
    ContractError,
    ModelIdentityError,
    PlanningExhaustedError,
    TransientBackendError,
)

from .session import AgentSession

logger = logging.getLogger(__name__)
SessionFactory = Callable[[], AgentSession]


class AgentServer:
    def __init__(
        self,
        session_factory: SessionFactory,
        host: str,
        port: int,
        max_message_bytes: int,
        max_connections: int,
    ) -> None:
        self._session_factory = session_factory
        self._capacity = threading.BoundedSemaphore(max_connections)
        self._connections: set[ServerConnection] = set()
        self._lock = threading.Lock()
        self._stopping = False
        self._server = serve(
            self._handle,
            host,
            port,
            compression=None,
            max_size=max_message_bytes,
            open_timeout=10,
            close_timeout=5,
        )

    @property
    def socket(self) -> Any:
        return self._server.socket

    def _reply(self, connection: ServerConnection, response: dict[str, Any]) -> None:
        connection.send(msgpack.packb(response, use_bin_type=True))

    def _dispatch(self, session: AgentSession, message: str | bytes) -> dict[str, Any]:
        try:
            if not isinstance(message, bytes):
                raise ContractError("binary msgpack requests required")
            try:
                request = msgpack.unpackb(message, raw=False)
            except (ValueError, TypeError, msgpack.ExtraData) as exc:
                raise ContractError("invalid msgpack request") from exc
            return session.handle(request)
        except ModelIdentityError as exc:
            logger.warning("model_identity_exhausted", extra={"event": {"session_id": session.session_id}})
            return {
                "error": str(exc),
                "error_code": "model_identity_exhausted",
                "session_id": session.session_id,
            }
        except (PlanningExhaustedError, TransientBackendError) as exc:
            code = (
                "planning_exhausted"
                if isinstance(exc, PlanningExhaustedError)
                else "transient_backend_exhausted"
            )
            logger.warning(code, extra={"event": {"session_id": session.session_id, "error": str(exc)}})
            return {"error": str(exc), "error_code": code, "session_id": session.session_id}
        except ContractError as exc:
            return {
                "error": str(exc),
                "error_code": "invalid_request_or_action",
                "session_id": session.session_id,
            }
        except Exception as exc:
            logger.exception(
                "inference_backend_failed",
                extra={"event": {"session_id": session.session_id, "exception_type": type(exc).__name__}},
            )
            return {
                "error": f"inference backend failed: {type(exc).__name__}",
                "error_code": "backend_failure",
                "session_id": session.session_id,
            }

    def _handle(self, connection: ServerConnection) -> None:
        if not self._capacity.acquire(blocking=False):
            try:
                self._reply(connection, {"error": "connection capacity reached", "error_code": "capacity"})
            except ConnectionClosed:
                pass
            finally:
                connection.close(code=1013, reason="connection capacity reached")
            return
        try:
            with self._lock:
                stopping = self._stopping
                if not stopping:
                    self._connections.add(connection)
            if stopping:
                connection.close(code=1001, reason="server stopping")
                return
            try:
                with self._session_factory() as session:
                    for message in connection:
                        self._reply(connection, self._dispatch(session, message))
            except ConfigurationError as exc:
                logger.error("session_initialization_failed", extra={"event": {"error": str(exc)}})
                self._reply(
                    connection, {"error": "session initialization failed", "error_code": "configuration"}
                )
            except Exception as exc:
                logger.error("session_failed", extra={"event": {"exception_type": type(exc).__name__}})
                self._reply(connection, {"error": "session failed", "error_code": "internal"})
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._connections.discard(connection)
            self._capacity.release()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            connections = list(self._connections)
        self._server.shutdown()
        for connection in connections:
            connection.close(code=1001, reason="server shutdown")

    def __enter__(self) -> AgentServer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.shutdown()


def create_server(
    session_factory: SessionFactory,
    host: str = "127.0.0.1",
    port: int = 8999,
    max_message_bytes: int = 33554432,
    max_connections: int = 16,
) -> AgentServer:
    return AgentServer(session_factory, host, port, max_message_bytes, max_connections)

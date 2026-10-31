# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A bounded-lifetime observer subprocess; it never owns model resources."""

import multiprocessing
from contextlib import ExitStack
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path


def _serve(root: str, port: int, ready: Connection) -> None:
    from .server import ObserverServer

    try:
        with ObserverServer(Path(root), port) as server:
            ready.send((True, server.url))
            ready.close()
            server.serve_forever(poll_interval=0.2)
    except Exception as exc:
        try:
            ready.send((False, type(exc).__name__))
        except (OSError, EOFError):
            pass
    finally:
        ready.close()


class ObserverProcess:
    def __init__(self, trace_dir: Path, port: int = 0) -> None:
        context = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._closed = False
        with ExitStack() as pipes:
            parent, child = context.Pipe(duplex=False)
            pipes.callback(parent.close)
            pipes.callback(child.close)
            try:
                self._process = context.Process(
                    target=_serve, args=(str(trace_dir.resolve()), port, child), daemon=True
                )
                self._process.start()
                child.close()
                if not parent.poll(10):
                    raise OSError("observer did not start within 10 seconds")
                success, message = parent.recv()
                if not success:
                    raise OSError(f"observer could not start: {message}")
                self.url = str(message)
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        if self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=3)
        self._process.close()

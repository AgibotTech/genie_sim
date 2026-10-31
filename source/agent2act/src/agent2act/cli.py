# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Command boundary: parse arguments, invoke services, map errors to exit codes."""

import argparse
import json
import logging
import os
import signal
import threading
import webbrowser
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import FrameType

from .bootstrap import ApplicationFactory, create_observer
from .configuration import AppSettings, load_config
from .domain.errors import ConfigurationError, ContractError
from .service.observability import configure_logging
from .service.server import create_server

logger = logging.getLogger(__name__)


def add_ui_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ui", action="store_true", help="Open the read-only local RGB/episode observer")
    parser.add_argument("--ui-port", type=int, default=0, help="Loopback UI port (default: automatic)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the UI URL without opening a browser"
    )


def _open_observer(trace_dir: Path, port: int, no_browser: bool, cleanup: ExitStack) -> None:
    observer = create_observer(trace_dir, port)
    cleanup.callback(observer.close)
    print(f"Observer: {observer.url}", flush=True)
    if not no_browser:

        def open_browser() -> None:
            try:
                if not webbrowser.open(observer.url):
                    logger.warning("observer_browser_unavailable; open the printed URL")
            except Exception:
                logger.warning("observer_browser_unavailable; open the printed URL")

        threading.Thread(target=open_browser, daemon=True).start()


def _observe(trace_dir: Path, port: int, no_browser: bool) -> None:
    if not trace_dir.is_dir():
        raise ValueError("trace directory does not exist")
    with ExitStack() as cleanup:
        _open_observer(trace_dir, port, no_browser, cleanup)
        if threading.current_thread() is threading.main_thread():

            def stop(signum: int, frame: FrameType | None) -> None:
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, stop)
            cleanup.callback(signal.signal, signal.SIGTERM, previous)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


def add_observation_arguments(parser: argparse.ArgumentParser) -> None:
    """Independent model-input ablations, shared by CLI and convenience launcher."""
    for flag, help_text in (
        ("depth", "Depth acquisition and planner access"),
        ("gripper-center", "Convert wrist EEF to configured nominal gripper center"),
        ("height-prior", "Expose explicitly calibrated table and base-frame heights"),
    ):
        parser.add_argument("--" + flag, action=argparse.BooleanOptionalAction, default=None, help=help_text)
    parser.add_argument("--height-calibration", type=Path, help="JSON height calibration; does not enable it")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent2Act CoRobot action service")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "check", "replay"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--mode", choices=("auto", "motion", "policy"))
        add_observation_arguments(command)
        if name == "serve":
            add_ui_arguments(command)
            command.add_argument("--log-file", type=Path, help="Also write structured JSON logs to this file")
            command.add_argument("--trace-dir", type=Path, help="Save per-request inputs, outputs and images")
            command.add_argument("--host")
            command.add_argument("--port", type=int)
        if name == "replay":
            command.add_argument("--request", required=True, type=Path)
    subcommands.add_parser("schema", help="print the deployment configuration JSON Schema")
    compat = subcommands.add_parser("compat", help="check the native CoRobot response parser")
    compat.add_argument("--source", type=Path, required=True)
    observe = subcommands.add_parser(
        "observe", help="Open a recorded trace without running models or actions"
    )
    observe.add_argument("--trace-dir", required=True, type=Path)
    observe.add_argument("--ui-port", type=int, default=0)
    observe.add_argument("--no-browser", action="store_true")
    return parser


def serve_application(factory: ApplicationFactory, settings: AppSettings) -> None:
    factory.describe()
    with create_server(factory.create_session, **settings.server.model_dump()) as server:

        def terminate(signum: int, frame: FrameType | None) -> None:
            raise KeyboardInterrupt

        install_handler = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, terminate) if install_handler else None
        try:
            try:
                logger.info(
                    "server_started",
                    extra={"event": {"host": settings.server.host, "port": server.socket.getsockname()[1]}},
                )
                server.serve_forever()
            except KeyboardInterrupt:
                logger.info("server_stopping")
        finally:
            if install_handler and previous is not None:
                signal.signal(signal.SIGTERM, previous)


def run(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if not 0 <= getattr(args, "ui_port", 0) <= 65535:
            raise ValueError("ui port must be between 0 and 65535")
        if args.command == "observe":
            _observe(args.trace_dir, args.ui_port, args.no_browser)
            return 0
        if args.command == "schema":
            print(json.dumps(AppSettings.model_json_schema(), indent=2))
            return 0
        if args.command == "compat":
            from .diagnostics.corobot import check_compatibility

            for result in check_compatibility(args.source):
                print(result)
            return 0
        settings = load_config(args.config, gripper_center=args.gripper_center).override(
            mode=args.mode,
            host=getattr(args, "host", None),
            port=getattr(args, "port", None),
            depth=args.depth,
            gripper_center=args.gripper_center,
            height_prior=args.height_prior,
            height_calibration=(
                json.loads(args.height_calibration.read_text()) if args.height_calibration else None
            ),
        )
        if getattr(args, "ui", False) and not args.trace_dir:
            args.trace_dir = (
                Path(settings.logging.trace_dir)
                if settings.logging.trace_dir
                else (Path("logs") / f"agent-{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}.trace")
            )
        configure_logging(settings.logging.level, settings.logging.format, getattr(args, "log_file", None))
        if getattr(args, "trace_dir", None):
            from .configuration import parse_settings

            data = settings.model_dump()
            data["logging"]["trace_dir"] = str(args.trace_dir.resolve())
            settings = parse_settings(data)
        factory = ApplicationFactory(settings)
        if args.command == "serve":
            with ExitStack() as cleanup:
                if args.ui:
                    try:
                        args.trace_dir.mkdir(parents=True, exist_ok=True)
                        _open_observer(args.trace_dir, args.ui_port, args.no_browser, cleanup)
                    except (OSError, EOFError, RuntimeError) as exc:
                        logger.warning(
                            "observer_start_failed", extra={"event": {"error_type": type(exc).__name__}}
                        )
                serve_application(factory, settings)
        elif args.command == "check":
            print(json.dumps(factory.describe(), ensure_ascii=False, indent=2))
        elif args.command == "replay":
            request = json.loads(args.request.read_text(encoding="utf-8"))
            with factory.create_session() as session:
                print(json.dumps(session.handle(request), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ConfigurationError, ContractError, OSError, ValueError, ImportError) as exc:
        import sys

        print(f"agent2act: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())

#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Private Pipe worker: frozen Newton viewer geometry -> real Isaac RGB.

Launch with the parent's current Isaac6-capable Python and pass_fds=(fd,).
Only the inherited trusted multiprocessing Connection carries the protocol;
Kit stdout/stderr remain ordinary logs. Importing this file does not start Kit.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from multiprocessing.connection import Connection
import sys
import traceback
from typing import Any, Callable

import numpy as np

from _isaac_snapshot_scene import SnapshotScene, validate_cameras


class SnapshotRenderer:
    def __init__(self, app: Any, *, stage: Any = None, replicator: Any = None) -> None:
        self.app = app
        if replicator is None:
            import omni.replicator.core as replicator
        if stage is None:
            import omni.usd

            context = omni.usd.get_context()
            context.new_stage()
            stage = context.get_stage()
        self.rep = replicator
        self.scene = SnapshotScene(stage)
        self._products: dict[str, dict[str, Any]] = {}

    def _drop_camera(self, name: str) -> None:
        product = self._products.pop(name)
        try:
            if product["annotator"] is not None:
                product["annotator"].detach([product["product"]])
        finally:
            product["product"].destroy()

    def render(self, message: dict[str, Any]) -> dict[str, Any]:
        sequence = message["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ValueError("sequence must be an integer")
        width, height = message["width"], message["height"]
        cameras = validate_cameras(message["cameras"], width, height)
        self.scene.apply(message["scene"])
        wanted = {camera["name"] for camera in cameras}
        for name in set(self._products) - wanted:
            self._drop_camera(name)
        for camera in cameras:
            name = camera["name"]
            path = self.scene.camera(camera, width, height)
            if name in self._products and self._products[name]["size"] != (
                width,
                height,
            ):
                self._drop_camera(name)
            if name not in self._products:
                product = self.rep.create.render_product(path, (width, height), force_new=True)
                # Record ownership immediately, including annotator-attach failure.
                owned = dict(product=product, annotator=None, size=(width, height))
                self._products[name] = owned
                annotator = self.rep.AnnotatorRegistry.get_annotator("rgb")
                owned["annotator"] = annotator
                annotator.attach(product)
        # Isaac6 Replicator1.13.25: wait_for_render is the annotator readback
        # barrier; delta_time=0 never advances timeline/physics. A second RTX
        # subframe lets changed mesh/camera/material state settle at that time.
        self.rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.0, wait_for_render=True)
        images: dict[str, np.ndarray] = {}
        for name in wanted:
            data = self._products[name]["annotator"].get_data(device="cpu", do_array_copy=True)
            data = np.asarray(data)
            if (
                data.dtype != np.uint8
                or data.ndim != 3
                or data.shape[:2] != (height, width)
                or data.shape[2] not in (3, 4)
            ):
                raise RuntimeError(
                    f"camera {name}: expected uint8 ({height},{width},3/4), got {data.dtype} {data.shape}"
                )
            images[name] = np.ascontiguousarray(data[..., :3]).copy()
        return dict(sequence=sequence, images=images, renderer="isaacsim")

    def close(self) -> None:
        # Continue cleanup if one annotator fails to detach; propagate the first
        # error after every render product has had its destroy call.
        error = None
        for name in list(self._products):
            try:
                self._drop_camera(name)
            except Exception as exc:
                error = error or exc
        try:
            self.scene.close()
        except Exception as exc:
            error = error or exc
        if error is not None:
            raise error


def _create_renderer(resources: ExitStack) -> SnapshotRenderer:
    from _isaac_render_common import preload_warp_before_kit

    preload_warp_before_kit()
    from isaacsim import SimulationApp

    from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config

    app = SimulationApp(prepare_launch_config({"headless": True, "renderer": "RaytracedLighting"}))
    resources.callback(app.close)
    renderer = SnapshotRenderer(app)
    resources.callback(renderer.close)
    return renderer


def serve(connection: Any, factory: Callable[[ExitStack], Any] = _create_renderer) -> int:
    """Own application resources from construction through EOF/close/failure."""
    sequence = None
    resources = ExitStack()
    result = 0
    try:
        renderer = factory(resources)
        connection.send(dict(ready=True, renderer="isaacsim"))
        while True:
            message = connection.recv()
            if not isinstance(message, dict):
                raise ValueError("worker message must be a dictionary")
            sequence = message.get("sequence")
            if message.get("op") == "close":
                break
            if message.get("op") != "render":
                raise ValueError(f"unknown worker operation: {message.get('op')!r}")
            connection.send(renderer.render(message))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        # SimulationApp.close can fast-shutdown the process. Preserve the
        # original failure BEFORE invoking any resource/Kit cleanup callback.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        try:
            connection.send(
                dict(
                    error=f"{type(exc).__name__}: {exc}",
                    sequence=sequence,
                    renderer="isaacsim",
                )
            )
        except (EOFError, BrokenPipeError, OSError):
            pass
        result = 1
    finally:
        try:
            resources.close()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            if result == 0:
                try:
                    connection.send(
                        dict(
                            error=f"cleanup {type(exc).__name__}: {exc}",
                            sequence=sequence,
                            renderer="isaacsim",
                        )
                    )
                except (EOFError, BrokenPipeError, OSError):
                    pass
            result = 1
        finally:
            connection.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ipc-fd",
        required=True,
        type=int,
        help="inherited private duplex Pipe file descriptor",
    )
    args = parser.parse_args(argv)
    # Kit parses sys.argv too; do not let it interpret the private worker CLI.
    sys.argv = [sys.argv[0]]
    return serve(Connection(args.ipc_fd))


if __name__ == "__main__":
    raise SystemExit(main())

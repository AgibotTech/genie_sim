# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Live read-only state visualization isolated from native physics callbacks."""

import multiprocessing
import os
from pathlib import Path
import tempfile
import time


class ViewerClosed(RuntimeError):
    pass


def display_worker(model_path, snapshots, ready, closed, errors):
    import mujoco
    import mujoco.viewer

    handle = None
    try:
        model = mujoco.MjModel.from_binary_path(model_path)
        data = mujoco.MjData(model)
        state = snapshots.recv()
        for key, value in state.items():
            if key == "time":
                data.time = value
            else:
                getattr(data, key)[:] = value
        mujoco.mj_forward(model, data)
        handle = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        with handle.lock():
            handle.opt.geomgroup[2] = 0
            handle.cam.lookat[:] = [0.5, 0, 1.0]
            handle.cam.distance = 1.7
            handle.cam.azimuth = 135
            handle.cam.elevation = -20
        handle.sync(state_only=True)
        ready.set()
        while handle.is_running():
            if snapshots.poll(0.02):
                state = snapshots.recv()
                if state is None:
                    break
                with handle.lock():
                    for key, value in state.items():
                        if key == "time":
                            data.time = value
                        else:
                            getattr(data, key)[:] = value
                    mujoco.mj_forward(model, data)
            handle.sync(state_only=True)
    except Exception as error:
        errors.send(f"{type(error).__name__}: {error}")
    finally:
        if handle is not None:
            handle.close()
            time.sleep(0.2)
        closed.set()
        snapshots.close()
        errors.close()


class NativeViewer:
    def __init__(self, model, data):
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("Live visualization requires DISPLAY and an X11-capable container")
        import mujoco

        self.data = data
        self.directory = tempfile.TemporaryDirectory(prefix="rubik-display-")
        model_path = str(Path(self.directory.name) / "model.mjb")
        mujoco.mj_saveModel(model, model_path)
        context = multiprocessing.get_context("spawn")
        receiver, self.sender = context.Pipe(duplex=False)
        self.errors, error_sender = context.Pipe(duplex=False)
        self.ready, self.closed = context.Event(), context.Event()
        self.process = context.Process(
            target=display_worker, args=(model_path, receiver, self.ready, self.closed, error_sender), daemon=True
        )
        self.process.start()
        receiver.close()
        error_sender.close()
        self.sender.send(self.snapshot())
        deadline = time.monotonic() + 20
        while not self.ready.is_set():
            if self.errors.poll():
                try:
                    message = self.errors.recv()
                except EOFError:
                    message = "MuJoCo display process exited during startup"
                self.close()
                raise RuntimeError(message)
            if not self.process.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("MuJoCo window startup failed; check DISPLAY and Xauthority")
            time.sleep(0.03)

    def snapshot(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
        }

    def is_running(self):
        return self.process.is_alive() and not self.closed.is_set()

    def sync(self):
        if not self.is_running():
            raise ViewerClosed("Visualization window closed")
        try:
            self.sender.send(self.snapshot())
        except (BrokenPipeError, EOFError, OSError) as error:
            raise ViewerClosed("Visualization window closed") from error

    def pace(self, started, simulation_interval):
        self.sync()
        remaining = simulation_interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    def wait_until_closed(self):
        while self.is_running():
            time.sleep(0.03)

    def close(self):
        if self.process.is_alive():
            try:
                self.sender.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.sender.close()
        self.errors.close()
        self.directory.cleanup()

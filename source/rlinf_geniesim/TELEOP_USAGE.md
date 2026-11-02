# Pico teleoperation

Teleoperation connects to an initialized ZMQ server and controls one environment
through the task's robot-specific IK configuration. Start the server and choose
its physics/render backends with the [main workflow](README.md) first.

## Connect

Run inside the teleop container described below, or from a local Python
environment with the Pico bridge and IK dependencies:

```bash
geniesim post-train teleop \
  --addr tcp://127.0.0.1:5601 --task pick_block_color --instance 0 \
  --port 8080 --viewer-port 8091
```

`--port` selects the Pico UDP port; `--host_ip` selects its bind address.
The web viewer listens on port 8091 by default. Connect the Pico application
to the host address and UDP port. `--stub --steps 20` runs a finite check
without a headset. `--debug` enables additional Pico/IK metrics.

The wrapper connects to the selected server. It does not launch a simulator
or change the physics backend. Reset and task reload are available through the
web interface. Reload pauses stepping until the server session is ready;
changing tasks also refreshes robot, instruction, and randomization state.

## Optional container

The teleop image inherits the post-training server layer with bundled Isaac Sim
6.2.x. Prepare the local GenieSim base with the
[server preparation steps](README.md#server-preparation). Then, from the
repository root on the simulation host, outside containers, build the server
and teleop layers. The server build can be skipped if its compatible image
already exists:

```bash
geniesim docker --profile post-train build
geniesim docker --profile post-train-teleop build
export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
geniesim docker --profile post-train-teleop up --headless
geniesim docker --profile post-train-teleop into
```

Inside the teleop container, run the command in [Connect](#connect). You can
connect to a separate initialized server or run the batch server launcher
inside this container using the [main workflow](README.md#start-and-initialize).
Do not start a second server on a port that is already in use.

`GENIESIM_IMAGE` and `GENIESIM_CONTAINER` override the selected profile's image
and container. To inherit a custom server image, append
`--build-arg SERVER_IMAGE=IMAGE` to the teleop build. The container uses host
networking for Pico UDP, the web viewer, and server RPC. `up` reuses the image
without building it. Lifecycle commands require matching checkout/profile
labels.

To stop and remove the container, run on the simulation host:

```bash
geniesim docker --profile post-train-teleop down
```

Alternatively, set `TELEOP_CONTAINER=post_train_teleop` when invoking
`geniesim post-train teleop` from the host checkout to run the client in that container as the
host UID/GID. `TELEOP_CONTAINER_REPO` selects the checkout path inside it
(default `/workspace`); `TELEOP_PYTHON` selects the interpreter for local runs.
The CLI delegates to `host_run_teleop.sh`, which remains available for direct
script callers. `geniesim teleop run` is the separate general GenieSim teleop
workflow; use `geniesim post-train teleop` for the RLinf ZMQ debug interface.

The container installs the repository's IK wheel. Its redistribution records
need separate review; see [DEPENDENCIES.md](DEPENDENCIES.md).

## Troubleshooting

- Confirm the endpoint is initialized with the selected task before connecting.
- Use `scripts/pico_probe.py --help` to inspect Pico packets without a simulator.
- If reload or reset fails after contacting the server, stepping stays paused
  because the cached layout or hold targets may be stale. Reload successfully
  to rebuild that state and resume.
- Non-finite state is rejected before entering IK. Reset the affected session
  and inspect the server error if invalid state persists.

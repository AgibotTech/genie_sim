# Fused Isaac PhysX backend

Select `physics: isaac_fused` and `render: isaac_fused` together. The server
constructs one `FusedIsaacBackend` that serves both physics and render roles
and owns one Isaac Sim subprocess. The stage is shared within that subprocess,
so rendering observes the physics state without a ROS pose-copy pass.

## Channels and lifecycle

`physics/fused_shm.py` carries actions, joint state, poses, and simulation time
with semaphore synchronization. `render/tiled_shm.py` carries camera frames.
These are internal backend channels, separate from the SHM/ZMQ connection to
an RLinf client. The backend owns their creation and cleanup. Each rebuilt
backend gets a new generation suffix to avoid attaching to stale resources.

The handler uses the same object as engine and bridge, so `close()` is
idempotent. Pose transfer hooks are no-ops because the physics and render
state already share a stage. Task reload refreshes backend state and schema;
the client must wait for the resulting session before stepping again.

## Reset

Reset writes selected environment tensor indices and zeroes their velocities.
Per-instance body poses, robot attachments/mimic joints, background mechanisms,
and randomization use the current scene schema. Inactive superset bodies are
parked with gravity disabled. Rest-state baking can remove post-reset settling;
without it, settling advances the shared scene, including other environments.
See [the full reset contract](INSTANCE_BODY_POSES.md).

## Validation

`tests/integration/fused_smoke.py` and `fused_integration_check.py` exercise the
real node and require the simulator runtime, GPU, and scene assets.

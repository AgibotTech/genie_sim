# Pre-release testing

Run these checks on a prepared simulation host before delivering or publishing
the post-training server: image build, server startup on real assets, and Pico
teleoperation's task switching and reset. Additional simulator integration
commands are listed in [runtime validation](../tests/README.md).

Record the checkout revision (`git rev-parse HEAD`), the host GPU and driver
(`nvidia-smi`), and the asset revision with the results. Each step lists what
must hold for it to pass. Stop at the first failure and keep the logs.

## Prerequisites

- An NVIDIA GPU and driver that meet the
  [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
  and support CUDA 13.
- Docker with the NVIDIA Container Toolkit.
- Free disk space for three images. `docker images` reports about 80 GB for
  each; they share most layers.
- The asset package at a host path, exported as `GENIESIM_ASSETS_SRC`.
- The GenieSim CLI, installed from the repository root on the host:

  ```bash
  python3 -m pip install -e source/geniesim_cli
  export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
  ```

Run host commands from the repository root, outside containers. Commands marked
*server container* or *teleop container* run inside that container from
`/workspace`.

## 1. Image build

Build all three layers without the Docker cache:

```bash
geniesim docker6.2 build --no-cache
geniesim docker --profile post-train build --no-cache
geniesim docker --profile post-train-teleop build --no-cache
```

Pass when:

- Each build exits with status 0 and prints `Built <image>` for
  `registry.agibot.com/genie-sim/geniesim4:latest`,
  `geniesim-post-train:server` and `geniesim-post-train:teleop`.
- The post-training builds use the local base. They must not need access to
  `registry.agibot.com`.
- Both profile images report Isaac Sim 6.2 and have consistent Python
  dependencies:

  ```bash
  for image in geniesim-post-train:server geniesim-post-train:teleop; do
    docker run --rm --entrypoint bash "$image" -c \
      'cat /isaac-sim/VERSION; python3 -m pip check'
  done
  ```

  The version starts with `6.2.` and `pip check` reports no broken requirements.

## 2. Server startup

Start the server container and enter it:

```bash
geniesim docker --profile post-train up --headless
geniesim docker --profile post-train into
```

*Server container:* start one server, then initialize a task with the default
backends (fused Isaac PhysX):

```bash
geniesim post-train server start --count 1 --base-port 5601 --devices 0
geniesim post-train server status --count 1 --base-port 5601
source source/rlinf_geniesim/scripts/rlinf_env.sh
python3 source/rlinf_geniesim/scripts/init_task.py \
  --addr tcp://127.0.0.1:5601 --sub pick_block_color --envs 1
PYTHONPATH=source:source/geniesim_benchmark/src \
  python3 source/rlinf_geniesim/scripts/smoke_test.py \
  --address tcp://127.0.0.1:5601 --attach pick_block_color
geniesim post-train server logs --port 5601
```

The first start builds the ROS workspace. The first initialization of a task
also bakes its rest state and compiles kernels, which takes several minutes;
later initializations of the same task reuse the cache.

Pass when:

- `start` reports `serving at tcp://127.0.0.1:5601` and `status` reports the
  server as running.
- `init_task.py` prints `num_envs=1` and a nonzero `action_dim`.
- `smoke_test.py` prints `[smoke] PASS`.
- The log contains no traceback.

Keep this server running for teleoperation.

## 3. Teleoperation: task switching and reset

Run teleoperation against the server from step 2. If the server was restarted,
initialize it with the default command above first.

In a second host terminal, start the teleop container. It uses host networking,
so it reaches the server at `127.0.0.1:5601`:

```bash
geniesim docker --profile post-train-teleop up --headless
geniesim docker --profile post-train-teleop into
```

*Teleop container:* connect without a headset. `--stub` keeps running and
serves the web console until interrupted:

```bash
geniesim post-train teleop \
  --addr tcp://127.0.0.1:5601 --task pick_block_color --instance 0 \
  --port 8080 --viewer-port 8091 --stub
```

Open `http://<host>:8091` for the console. The console's panels call the HTTP
API below; the same checks can be scripted from the host with `curl`. Every
call is a `POST` with a JSON body and returns `"ok": true` on success:

```bash
api() { curl -s -X POST "http://127.0.0.1:8091/api/$1" -d "$2"; echo; }
```

| Step | Call | Pass when |
| --- | --- | --- |
| Session state | `api state '{}'` | `session_ready` is true, `current.sub_task_name` is `pick_block_color`, and `steps` increases between calls |
| Catalog | `api catalog '{}'` | `sub_tasks` lists the server's released tasks |
| Task details | `api task '{"sub_task_name":"pick_block_color"}'` | `switchable_instance_ids` and `instructions_by_instance` are not empty |
| Reset to another instance | `api reset '{"instance_id":1,"instruction_id":<id>,"seed":-1}'` with an instruction ID listed for instance 1 | `ok` and `states_finite` are true; `api state '{}'` shows the new instance and instruction, and the HUD text after the next frames |
| Seeded reset | the same call with `"seed":7` | `ok` and `states_finite` are true |
| Invalid reset | `api reset '{"instance_id":1,"instruction_id":999}'` | `ok` is false with an error naming the unavailable instruction; stepping continues |
| Switch to a task with another background | `api reload '{"sub_task_name":"hold_pot"}'` | `ok` is true and `num_envs` is 1. Leaving `task_name` empty selects the official pairing |
| Reset in the new task | `api task` for `hold_pot`, then `api reset` with one of its instances | `ok` and `states_finite` are true; `steps` keeps increasing |
| Switch back | `api reload '{"sub_task_name":"pick_block_color"}'`, then `api reset` | Both return `ok`; `steps` keeps increasing for at least a minute |
| Camera stream | `timeout 3 curl -s http://127.0.0.1:8091/stream \| wc -c` | The output is nonzero |

After each reload and reset, the console's camera view shows the selected scene
and the HUD shows the selected instruction. Stop the client with Ctrl-C. The
container uses host networking, so check on the host that its ports are
released: `ss -ulpn | grep 8080` and `ss -ltpn | grep 8091` print nothing.

### With a headset

Run the same command without `--stub`, and connect the Pico application to the
host address and UDP port 8080. Check that:

- Without input the robot holds its pose, and the client log reports that it is
  holding.
- Holding a grip engages that arm: the arm follows the controller in direction
  and scale, and stops following when the grip is released.
- The index triggers open and close the grippers.
- The left controller's first button (`keyOne`) and the console's reset both
  return the robot to the selected instance.
- After switching tasks from the console, headset control still works.
- After the headset disconnects, the robot holds its pose; control resumes when
  it reconnects.

## 4. Cleanup

*Server container:*

```bash
geniesim post-train server stop --count 1 --base-port 5601
```

Then on the host:

```bash
geniesim docker --profile post-train-teleop down
geniesim docker --profile post-train down
```

Pass when `nvidia-smi` lists no remaining simulation processes and GPU memory
returns to its level before the tests.

## Recording results

| Area | Revision | Host GPU / driver | Result | Notes and log paths |
| --- | --- | --- | --- | --- |
| Image build | | | | |
| Server startup | | | | |
| Teleop without headset | | | | |
| Teleop with headset | | | | |
| Cleanup | | | | |

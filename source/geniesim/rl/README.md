# GenieSim RL — Reinforcement Learning with GenieSim Simulation

This module connects [GenieSim](https://github.com/AgibotTech/genie_sim) with
[RLinf](https://github.com/RLinf/RLinf) for robot reinforcement learning,
featuring **Isaac Sim + MuJoCo dual-simulator** architecture and
**SpaceMouse human-in-the-loop** training.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  RLinf (training framework)                                          │
│    Task Env (e.g. PlaceWorkpieceEnv)                                 │
│      ├── _extract_states()   52-dim SHM state → 26-dim model state   │
│      ├── _expand_actions()    7-dim model action → 14-dim SHM action │
│      └── _compute_reward()   dense reward from body poses            │
│    GenieSimBaseEnv  →  GenieSimShmClient                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  Shared Memory (SHM)
                           │  ├── Frame SHM  (camera images)
                           │  ├── Ctrl SHM   (per-env state/action/info)
                           │  └── Step SHM   (request-reply sync)
┌──────────────────────────┴───────────────────────────────────────────┐
│  sim_server.py  (GenieSim side)                                      │
│    GenieSimVectorEnv  ← manages MuJoCo lifecycle + signal-based sync │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐       ┌──────────────────────┐  │
│  │ MuJoCo env_0 │   │ MuJoCo env_1 │  ...  │ Isaac Sim renderer   │  │
│  │ 1000 Hz      │   │ 1000 Hz      │       │ 30 Hz (GridCloner)   │  │
│  └──────────────┘   └──────────────┘       └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

- **MuJoCo** handles physics at 1000 Hz per environment — one process per env, isolated by ROS 2 namespace
- **Isaac Sim** provides photo-realistic rendering via `GridCloner`
- **Shared memory (SHM)** is the only data channel between RLinf and sim (zero-copy for camera images)
- **EE control mode** — IK (damped Jacobian) runs inside MuJoCo; actions are delta EE pose targets
- **Task Env pattern** — SHM transports full-dimensional state/action; each task maps to/from a smaller model space via `_extract_states()` / `_expand_actions()`

### Reward Design

The `place_workpiece` task uses three reward components:

| Component | Description |
|-----------|-------------|
| `r_alive` | Exponentially decaying reward based on 3D distance and orientation error to target |
| `r_below` | Penalty when workpiece drops below target height |
| `r_success` | Sparse one-time reward when workpiece is placed at target and held still |

---

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 3090+, VRAM ≥ 24GB)
- Docker with NVIDIA Container Toolkit
- 3Dconnexion SpaceMouse (for data collection)

### 1. Clone Repositories

```bash
mkdir workspace && cd workspace
git clone https://github.com/AgibotTech/genie_sim.git
git clone -b dev/geniesim https://github.com/RLinf/RLinf.git
```

For GenieSim installation and asset downloads, refer to the
[GenieSim documentation](https://agibot-world.com/sim-evaluation/docs/#/v3).

### 2. Build Docker Images

**Base image** (GenieSim + Isaac Sim + MuJoCo + ROS 2):

```bash
bash genie_sim/scripts/build_geniesim_rlinf_image.sh
```

**Training image** (RLinf + PyTorch + training dependencies):

```bash
cd RLinf
docker build \
  --build-arg BUILD_TARGET=embodied-geniesim \
  -t geniesim-rlinf-train:latest \
  .
```

### 3. Download Pretrained Weights

```bash
cd RLinf/examples/embodiment/config
# For mainland China: export HF_ENDPOINT=https://hf-mirror.com
hf download RLinf/RLinf-ResNet10-pretrained --local-dir .
```

### 4. Collect Demonstrations

Connect the SpaceMouse via USB, then:

```bash
cd workspace
bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50
```

SpaceMouse controls:

| Action | Effect |
|--------|--------|
| Translate device | Move right arm end-effector (x/y/z) |
| Rotate device | Rotate right arm end-effector (roll/pitch/yaw) |
| Press left button | Save demo → environment resets |
| Press right button | Discard demo → environment resets |

Demos are saved to `genie_sim/sac_demo/`.

### 5. Convert Demonstrations

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert
```

### 6. Start Training

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train
```

During training, env_0 accepts real-time SpaceMouse intervention while remaining
environments are driven by the policy.

Override Hydra parameters:

```bash
# Adjust discount factor
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.gamma=0.97

# Adjust BC regularization
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.bc_coef=5.0
```

### 7. Monitor Training

```bash
tensorboard --logdir workspace/results/
```

Key metrics: `critic_loss`, `q_values`, `eval/success_rate`, `entropy`, `bc_loss`.

### 8. Debug Shell

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh shell
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `run.sh collect --num-demos N` | Collect N demonstrations |
| `run.sh convert` | Convert demos to replay buffer |
| `run.sh train` | Start SAC + SpaceMouse HIL training |
| `run.sh shell` | Interactive container shell |
| `run.sh help` | Show all commands |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `PermissionError` on `/dev/shm/geniesim_*` | Run `bash RLinf/rlinf/envs/geniesim/scripts/cleanup_stale.sh` |
| Stale `.geniesim_idle` causing hang | Same cleanup script above |
| Isaac Sim startup timeout | Increase `startup_timeout_sec` in env YAML |
| GPU out of memory | Reduce `env.train.total_num_envs` via Hydra override |

## License

Mozilla Public License Version 2.0 — see `LICENSE` in the repository root.

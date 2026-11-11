![Genie Sim](./docs/banner.webp)

<div align="center">
  <div align="center">
    <a href="https://agibot-world.com/videos/genieSim/modules/full-en-3840w.667868830394362f.mp4" target="_blank">
      <img src="./docs/video_preview.jpg" alt="Play Genie Sim video" />
    </a>
  </div>
  <a href="https://arxiv.org/abs/2601.02078" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2601.02078-red.svg?logo=arxiv&logoColor=white" alt="arXiv Paper: 2601.02078">
  </a>
  <a href="https://agibot-world.com/genie-sim/">
    <img src="https://img.shields.io/badge/Webpage-GenieSim-green?" alt="Webpage">
  </a>
  <a href="https://agibot-world.com/genie-sim-report/">
    <img src="https://img.shields.io/badge/TechReport-GenieSim-blue?" alt="TechReport">
  </a>
  <a href="https://github.com/AgibotTech/genie_sim">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub">
  </a>
  <a href="https://huggingface.co/datasets/agibot-world/GenieSimAssets">
    <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=HuggingFace" alt="HuggingFace">
  </a>
  <a href="https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset">
    <img src="https://img.shields.io/badge/Dataset-ModelScope-FF6B35?logo=model&logoColor=white" alt="ModelScope Dataset">
  </a>
</div>

# 1. 🌍 Genie Sim

**3.5.0 focuses on Genie Sim Benchmark**, using the `geniesim3:latest` combo
image with bundled Isaac Sim **5.1** and system-Python Isaac Sim **6.2**.
Use `geniesim docker5.1` for this combo. The default `geniesim docker` selects
the dedicated Isaac Sim 6.2 image, `geniesim4:latest`, also available via
`geniesim docker6.2`, which belongs to the separate **4.0**
line. See [release lines and runtime selection](docker/README.md#release-lines).

> **⚠️ E.O.L. / deprecated:** `geniesim docker4.5` (`geniesim2`) and
> `geniesim docker5.1` (`geniesim3`). Use `geniesim docker` / `docker6.2` for new deployments.

**Genie Sim is the simulation platform from AgiBot.** It provides developers with a complete toolchain for *environment reconstruction*, *scene generalization*, *data collection*, and *automated evaluation*. Its core module, **Genie Sim Benchmark** is a standardized tool dedicated to establishing the most accurate and authoritative evaluation for embodied intelligence.

The platform integrates **3D reconstruction** with **visual generation** to create a high-fidelity simulation environment. It pioneers **LLM-driven technology** to generate vast simulation scenes and evaluation configurations in minutes. The evaluation system covers **200+ tasks** across **100,000+ scenarios** to establish a comprehensive capability profile for models. Genie Sim also opens over **10,000 hours of synthetic data** including real-world robot operation scenarios.

The platform will significantly accelerate model development, reduce reliance on physical hardware, and empower innovation in embodied intelligence. Simulation assets, dataset, and code are **fully open source**.

# 2. ✨ Features

- 🧞 **`geniesim` CLI**: one command-line entry point for the entire platform — docker lifecycle, ROS 2 workspace builds, health diagnosis (`doctor`), pure-Python wheel deploy, asset / scene bootstrap. Packaged as a standalone PEP 517 / PEP 621 wheel.
- 🤖 **Agent-ready SKILLs**: packaged `SKILL.md` recipes for benchmarking, inference checks, scene generation, asset search, and teleop — invokable directly by Claude Code / agentic coding agents, so the simulator can be driven without a human first reading the docs.
- ⚡ **Genie Sim RT Engine (realtime, interactive)**: `geniesim_ros` brings physics, render, and the robot into ROS 2 natively — teleop, MoveIt, and ros2_control close the loop on one shared `sim_time`. Multiple physics backends (Isaac Sim PhysX, Newton-standalone) selectable per scene, with cloth + soft-body support on the Newton path.
- 🎨 **High-Fidelity Sim-Ready Assets**: 5,140 validated 3D assets covering five real-world operation fields: retail, industry, catering, home and office. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSimAssets).
- 🛰️ **3DGS-based Reconstruction Pipeline**: Integrate 3DGS-based reconstruction process with visual generative model to synthesize realistic simulation environment with high-precision meshes. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- 🌐 <b><big>Genie Sim World: A spatial world model which generates a photorealistic 3D world from an equirectangular panorama.</big></b> [GitHub](source/geniesim_world).
- 💬 **LLM-Driven Scene Generation**: Natural language-driven generation and generalization which instantly generates diverse simulation scenes through conversational interaction.
- 📦 **Large-Scale Synthetic Dataset**: Over 10,000 hours open-source synthetic data across 200+ loco-manipulation tasks with multi-sensor streams, alongside multi-dimensional variations.
- 🛠️ **Synthetic Data Generation**: Efficient toolkit for data collection with error-recovery mechanism, supporting both low-latency teleoperation and automated data programming. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- 🧪 **Robust and Diverse Benchmark**: Provide 100,000+ simulation scenarios and use LLM to autonomously generate task instructions and evaluation configurations. Discrepancy between simulation and real-world test results is less than 10%.
- 🔍 **VLM-based Auto-Evaluation System**: Full-spectrum evaluation criteria to provide model's capability profile covering manipulation skills, cognitive comprehension and task complexity.
- 🚀 **Zero-Shot Sim-to-Real Transfer**: Model trained with our synthetic data exhibits zero-shot sim-to-real transfer capability with superior task success rate compared to model trained with real data.

# 3. 🚀 Quick start

### 3.1 🧞 CLI & 🤖 SKILLs — install, build the container, get a shell

```mermaid
flowchart LR
  pip(["📦 pip install<br/>geniesim_cli<br/><i>(host)</i>"]) ==> docker(["🐳 geniesim<br/>docker up<br/><i>(host)</i>"]) ==> auto(["⚙️ tier-1 peers<br/>auto-install<br/><i>(in-container)</i>"]) ==> shell(["💻 geniesim<br/>docker into<br/><i>(host)</i>"])
  classDef step fill:#dbeafe,stroke:#1e40af,color:#1e3a8a,font-weight:bold
  class pip,docker,auto,shell step
  linkStyle 0,1,2 stroke:#1e40af,stroke-width:2px
```

```bash
# === on the host ===
pip install -e source/geniesim_cli/      # standalone CLI wheel
geniesim docker build                    # build the Isaac Sim 6.2 + Jazzy image
geniesim docker up                       # boot the container; the entrypoint
                                         # editable-installs every tier-1 from
                                         # the bind-mounted workspace
geniesim docker into                     # interactive shell inside the container

# === inside the container — sanity-check the stack ===
geniesim status                          # per-distribution health probe
geniesim doctor                          # diagnose & repair (rosdep, env, …)
geniesim bootstrap                       # install or repair required Tier-1 peers
```

#### 🤖 With an AI assistant — just ask

The repo ships `AGENTS.md` + per-package `skills/<name>/SKILL.md` as machine-readable signposts. Claude Code, Cursor, etc. follow them on their own — say _"run the pnp benchmark against localhost:8999"_ or _"launch the WBC scene with MoveIt"_ and the agent walks the trail.

#### 📖 Without an AI assistant — follow the signposts

[`AGENTS.md`](AGENTS.md) (root, boot sequence) → [`source/AGENTS.md`](source/AGENTS.md) (module map) → `source/<pkg>/AGENTS.md` (architecture + commands) → `source/<pkg>/skills/<name>/SKILL.md` (copy-paste recipes). Full CLI verb table: [`.agent/geniesim_cli.md`](.agent/geniesim_cli.md).

> [!IMPORTANT]
> 🎨 **Tier-1 peers come up automatically; tier-2 peers are opt-in.** `geniesim docker up`'s entrypoint editable-installs every tier-1 peer
> declared in `source/geniesim/pyproject.toml`. The asset library `geniesim_assets` is a tier-1 peer distributed **out of band**
> (sourced from [HuggingFace](https://huggingface.co/datasets/agibot-world/GenieSimAssets) / [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)),
> and editable-installed by the entrypoint so every other peer can find it.
>
> Tier-2 peers (`geniesim_teleop` · `geniesim_scene_agent` · `geniesim_world`) have heavy stacks (VR / LLM / CUDA-ML) and are **not** auto-installed.
> Install only the module you need, following its [teleop](source/geniesim_teleop/README.md),
> [scene-agent](source/geniesim_scene_agent/README.md), or [World](source/geniesim_world/README.md)
> setup guide. World uses a dedicated environment. `geniesim bootstrap` installs or repairs
> required peers and skips these optional packages.
>
> `geniesim_dream` is a separately installed first-class package with its own environments;
> it is outside the umbrella's Tier-1/Tier-2 extras. See the [module tiers](source/README.md#installation-tiers).

### 3.2 🧪 Benchmark — debug locally, compete globally on RoboColiseum

Isaac benchmark tasks use the explicit `geniesim docker5.1` combo; see the
[benchmark runtime guide](source/geniesim_benchmark/README.md).

```mermaid
flowchart LR
  list(["📋 benchmark<br/>list / categories"]) ==> probe(["🔌 check-<br/>inference"]) ==> run(["🧪 benchmark<br/>run · batch"]) ==> submit(["🏆 RoboColiseum<br/>leaderboard"])
  classDef step fill:#ede9fe,stroke:#5b21b6,color:#3b0764,font-weight:bold
  class list,probe,run,submit step
  linkStyle 0,1,2 stroke:#5b21b6,stroke-width:2px
```

The Genie Sim Benchmark is the engine behind [**RoboColiseum**](https://robocoliseum.ai/) — and the same tool you use to get ready for it:

- **🏆 It's a competition.** RoboColiseum ranks embodied-AI policies on four boards (`instruction` / `robust` / `manip` / `spatial`).
- **🧪 Debug locally → submit remotely.** Run the benchmark tasks on your own machine to debug your policy against your own inference server, then launch a model evaluation on the official leaderboard with the _same_ server — no blind submissions.
- **🤖 Agent-friendly one-click SKILLs.** The whole competition pipeline — download datasets, fetch the π<sub>0.5</sub> baseline weights, run an example leaderboard service, submit & track results — ships under [`source/geniesim_benchmark/skills/robocoliseum/`](source/geniesim_benchmark/skills/robocoliseum/); just ask an AI agent (Claude Code, Cursor, …) to drive them.

**Run it locally** — `cat source/geniesim_benchmark/USAGE.md` for how to run a benchmark task on your own machine against your inference server (CLI verbs, common commands, and overrides).

### 3.3 ⚡ RT Engine — realtime, interactive ROS 2 simulation

```mermaid
flowchart LR
  build(["🔨 ros build<br/>dev"]) ==> scene(["🎬 launch<br/>scene × launcher"]) ==> sim(["⚡ RT Engine<br/>PhysX · Newton · mjwarp"]) ==> moveit(["🦾 MoveIt 2<br/>+ WBC RViz"])
  classDef step fill:#dcfce7,stroke:#15803d,color:#14532d,font-weight:bold
  class build,scene,sim,moveit step
  linkStyle 0,1,2 stroke:#15803d,stroke-width:2px
```

The RT Engine ships its own SKILLs that walk you through every step — workspace build → scene launch → MoveIt + WBC RViz → bringing your own robot → teleop → recording → debugging. Don't memorize `ros2 launch` flags — follow the skill:

```bash
# inside the container:
geniesim ros build dev
cat source/geniesim_ros/skills/build-workspace/SKILL.md    # `geniesim ros build dev`
cat source/geniesim_ros/skills/launch-scene/SKILL.md       # scene × launcher matrix
cat source/geniesim_ros/skills/moveit-wbc/SKILL.md         # MoveIt 2 + WBC RViz for Genie G2
cat source/geniesim_ros/skills/add-robot/SKILL.md          # bring your own URDF / xacro
cat source/geniesim_ros/skills/*/SKILL.md                  # and more
```

### 3.4 🌐 Generate a 3D world — `geniesim_world`

```mermaid
flowchart LR
  pano(["🖼️ Panorama<br/>(.png)"]) ==> da(["🔭 DA360<br/>depth"]) ==> sharp(["✨ SHARP<br/>Gaussians"]) ==> world(["🌍 3D world<br/>(PLY · Gaussians)"])
  classDef step fill:#ffedd5,stroke:#c2410c,color:#7c2d12,font-weight:bold
  class pano,da,sharp,world step
  linkStyle 0,1,2 stroke:#c2410c,stroke-width:2px
```

Turn an equirectangular panorama into an explorable photorealistic 3D world. The generated PLY / Gaussians are consumed by `geniesim_world`'s own tooling. RT Engine scene loading is not supported; see the [platform roadmap](docs/ROADMAP.md) for proposed integration.

```bash
cat source/geniesim_world/skills/generate-world/SKILL.md
```

### 3.5 💬 Generate a scene from a prompt — `geniesim_scene_agent`

```mermaid
flowchart LR
  nl(["💬 NL prompt<br/>'an apple on the table'"]) ==> assets(["🔍 search<br/>asset library"]) ==> dsl(["📝 Scene recipe<br/>DSL / direct / hybrid"]) ==> usda(["🏗️ scene.usda<br/>+ scene metadata"])
  classDef step fill:#ede9fe,stroke:#6d28d9,color:#3b0764,font-weight:bold
  class nl,assets,dsl,usda step
  linkStyle 0,1,2 stroke:#6d28d9,stroke-width:2px
```

Describe a scene in words, search the external asset library, and generate a foreground scene using the bundled Python tools and agent skills. Scene authoring is a standalone workflow; robot task execution uses the independent collection runtime. Optional GPU embeddings support asset retrieval, and rendering uses a configured Isaac Sim environment or the scene-agent image.

```bash
python -m pip install -e 'source/geniesim_scene_agent[scene]'
geniesim-scene-agent resources
geniesim-scene-agent assets-root

cat source/geniesim_scene_agent/skills/geniesim-asset-search/SKILL.md
cat source/geniesim_scene_agent/skills/geniesim-scene-generation/SKILL.md
cat source/geniesim_scene_agent/skills/geniesim-scene-to-data/SKILL.md
```

See [the scene-agent guide](source/geniesim_scene_agent/README.md) for scene generation and optional task orchestration, [the container guide](source/geniesim_scene_agent/docker/README.md) for rendering, and [the example library](source/geniesim_scene_agent/examples/README.md) for task and scene references. The [module map](source/README.md) lists each package's docs and skills.

### 3.6 🎮 Teleoperation — `geniesim_teleop`

```mermaid
flowchart LR
  up(["🐳 docker5.1 up"]) ==> collect(["🎮 autoteleop.sh --isaac-version 5.1<br/>VR collection"]) ==> keep(["💾 keep / discard<br/>(y / n)"]) ==> post(["📦 post-process<br/>→ HDF5 episodes"])
  classDef step fill:#fce7f3,stroke:#be185d,color:#831843,font-weight:bold
  class up,collect,keep,post step
  linkStyle 0,1,2 stroke:#be185d,stroke-width:2px
```

Collect demonstration data by teleoperating the robot with a Pico VR headset, then turn the recorded rosbags into HDF5 episodes. For automatic simulator startup, build with `geniesim docker5.1 build`. Once the GUI container is up (`geniesim docker5.1 up` succeeds), the whole flow is two host-side scripts:

```bash
# 1. VR collection — press y (keep) / n (discard) when the collection is done:
./source/geniesim_teleop/scripts/autoteleop.sh --isaac-version 5.1

# 2. Post-process the recording into HDF5 episodes:
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 5.1 <task_name>
```

Isaac 6.2 mode (`--isaac-version 6.2`, the default) uses `geniesim4` and starts only control processes; provide the simulator separately.

See [`source/geniesim_teleop/README.md`](source/geniesim_teleop/README.md) for the controller button map, Pico setup, and task configuration.

### 3.7 📥 Data collection — `data_collection`

```mermaid
flowchart LR
  build(["🔨 data-collection build"]) ==> list(["📋 data-collection llm list<br/>pick a task"]) ==> run(["📥 data-collection run<br/>cuRobo auto-planning"]) ==> ep(["💾 episodes<br/>recording_data/"])
  classDef step fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,font-weight:bold
  class build,list,run,ep step
  linkStyle 0,1,2 stroke:#0369a1,stroke-width:2px
```

Collect automated tasks with cuRobo v2 trajectory planning in an independent Isaac Sim 5.1 image. The host CLI registers the module and locates external assets and task libraries; runtime dependencies are installed in the collection image.

```bash
# 1. Register the module and build its complete runtime image:
geniesim data-collection install
geniesim data-collection build    # optional: --china or --mirrors NAME

# 2. Discover LLM tasks, then collect one:
geniesim data-collection llm list --robot g2
geniesim data-collection run <TASK> --scene <SCENE_ID> --run-times 1 --render

# Declarative JSON jobs use a separate entry:
geniesim data-collection run --task-json tasks/my_job.json --headless --standalone
```

The image defaults to `registry.agibot.com/genie-sim/geniesim4-data-collection:latest` and builds directly from official Isaac Sim 5.1. Configure external assets with `GENIESIM_ASSETS_PATH` and task libraries with `GENIESIM_LLM_TASK_ROOT` or the install command's `--assets-root` / `--task-root` options.

Episodes land in `source/data_collection/recording_data/`. See [the collection guide](source/data_collection/README.md), [task configuration](source/data_collection/TASK_CONFIG_GUIDE.md), and [replay](source/data_collection/replay/README.md) for the runtime and output contracts.

### 3.8 🧠 Agent2Act — LLM-to-action runtime

Agent2Act is a closed-loop framework that connects multimodal LLMs to robot actions in
Genie Sim. It supports motion generation, policy backends such as π0.5 / OpenPI, and automatic
routing between execution modes, using visual observations and robot feedback to guide each step.

See [`source/agent2act/README.md`](source/agent2act/README.md) for installation, configuration,
and usage instructions.

### 3.9 🌙 World-model inference — `geniesim_dream`

Generate G1/G2 robot videos from recorded actions or a pi0.5 policy loop, with optional VLM evaluation. Dream provides a browser UI and 20 example bundles. It uses separate world-model, policy and evaluator environments, outside the default `geniesim bootstrap` flow.

See [the Dream quick start](source/geniesim_dream/README.md), [installation](source/geniesim_dream/docs/setup.md), and [model assets](source/geniesim_dream/docs/assets.md). Its pinned Cosmos Framework and OpenPI dependencies are Git submodules; checkpoints are downloaded separately.

### 3.10 Post-training server

```mermaid
flowchart LR
  build(["🔨 Build the Docker images"]) ==> shell(["💻 Enter the container"]) ==> server(["⚙️ Start the server"])
  classDef step fill:#ccfbf1,stroke:#0f766e,color:#134e4a,font-weight:bold
  class build,shell,server step
  linkStyle 0,1 stroke:#0f766e,stroke-width:2px
```

Start a GenieSim simulation server for policy training and evaluation. Prepare the host CLI and an editable `geniesim_assets` installation in the same Python environment (see § 3.1). The asset directory is detected and mounted automatically.

Build the GenieSim base from NVIDIA's Isaac Sim 6.2 image, then add the post-training layer:

```bash
# On the simulation host, outside containers, from the repository root:
geniesim docker6.2 build
geniesim docker --profile post-train build
geniesim docker --profile post-train up --headless
geniesim docker --profile post-train into
```

```bash
# Inside the post_train_server container, from /workspace:
geniesim post-train server start --count 1 --base-port 5601 --devices 0
```

This starts one server on GPU 0, listening on TCP port 5601. A client selects the task and initializes the simulation.

For training and evaluation with RLinf, see the [RLinf GenieSim guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/geniesim.html).

# 4. 📜 Roadmap & Updates

The arc of the platform — what shipped when, and what's coming next.

```mermaid
flowchart LR
  v2["<b>v2.x · 2025</b><br/>Challenge tasks<br/>2× sim speed"]
  v30["<b>v3.0 · Jan 2026</b><br/>Isaac Sim 5.1<br/>Genie G2 · 3DGS"]
  v31["<b>v3.1 · Apr 2026</b><br/>Genie Sim World<br/>RLinf"]
  v32["<b>v3.2 · Jun 2026</b><br/>🧞 CLI · 🤖 SKILLs<br/>⚡ RT Engine"]
  v35["<b>v3.5.0 · Nov 2, 2026</b><br/>Benchmark-focused release<br/>Isaac Sim 5.1 + 6.2 combo"]
  next["<b>v4.0 line</b><br/>Dedicated Isaac Sim 6.2<br/>RT Engine qualification"]
  v2 --> v30 --> v31 --> v32 --> v35 --> next

  classDef done fill:#d1f0d8,stroke:#1f6f3d,color:#0b3b1f
  classDef current fill:#cfe2ff,stroke:#084298,color:#084298,stroke-width:2px
  classDef future fill:#fff3cd,stroke:#856404,color:#5c4400
  class v2,v30,v31,v32 done
  class v35 current
  class next future
```

## 📜 Release log

<details open><summary><b>[11/2/2026] v3.5.0 — Benchmark release with the Isaac Sim 5.1 + 6.2 combo</b></summary>

- 🧪 **Extended benchmark suites** — fine manipulation, dexterous manipulation, whole-body coordination, and cross-embodiment tasks, plus 25 unscored open-world sandbox scenarios.
- ⚙️ **Task-selected runtimes** — direct Isaac evaluation, managed Newton wok evaluation, and native MuJoCo Rubik execution with separate observation rendering.
- 🐳 **Benchmark combo runtime** — `geniesim3:latest` retains bundled Isaac Sim 5.1 and the system-Python Isaac Sim 6.2 co-install. Each benchmark entry selects its runtime. The dedicated `geniesim4:latest` image remains on the separate 4.0 line.
- 🌙 **Genie Sim Dream** — G1/G2 world-model inference, policy integration, visual evaluation, and a browser UI in dedicated environments.
- 📥 **Collection and post-training workflows** — independent Isaac Sim 5.1/cuRobo v2 collection and replay, scene-agent authoring workflows, and RLinf server/client integration.
- 🔖 **Checkout package version** — all eight first-class Genie Sim distributions and owned ROS package metadata in this release checkout share `VERSION=3.5.0`. This package version does not relabel the dedicated 4.0 image line. Dependencies, recording formats, and independently maintained modules keep their own versions.

See the [module index](source/README.md) and [benchmark task commands](source/geniesim_benchmark/benchmark_test_docs.md) for workflow details.

</details>

<details><summary><b>[6/25/2026] v3.2.0 — Genie Sim RT Engine + Agentic CLI/SKILLs</b></summary>

- 🏆 **RoboColiseum** — opened [RoboColiseum](https://robocoliseum.ai/): submit a policy and get it scored on the official leaderboard. Added a new `spatial` board (alongside `instruction` / `robust` / `manip`), and the `robust` board now aggregates and scores tasks per perturbation type (instruction, robot pose, background, image quality, camera position). The full pipeline — download datasets, fetch baseline weights, run an example leaderboard service, submit & track — ships as agent-friendly one-click SKILLs under [`source/geniesim_benchmark/skills/robocoliseum/`](source/geniesim_benchmark/skills/robocoliseum/).
- 🧞 **`geniesim` CLI** — one command for docker, ROS 2 builds, `bootstrap`, `status`, `doctor`, `deploy`; bash/zsh completion; standalone PEP 517 / PEP 621 wheel.
- 🤖 **Agent SKILLs** — self-contained `SKILL.md` recipes (run-benchmark, check-inference, geniesim-scene-generation, geniesim-asset-search, geniesim-scene-to-data, run-teleop) that Claude Code and other agents invoke to drive the simulator end-to-end.
- ⚡ **Genie Sim RT Engine (realtime, interactive)** — `geniesim_ros` ships physics + render + robot as a first-class ROS 2 node sharing one `sim_time`. Multiple physics backends (Isaac Sim PhysX, Isaac Sim Newton, Kit-free Newton-standalone) with cloth / soft-body on the Newton path.
- 🦾 **Customizable robots** — Genie G2 family (Tier 1, continuously maintained) with `arm × gripper` matrix, drop-in MoveIt 2 + WBC RViz, three IK plugins; reference URDF for Franka, UR5, Aloha, ARX, Agilex; bring-your-own via xacro + offline mesh-prep tools.
- 🎬 **New demos** — pick-and-place (`scene_pnp_g2_op`) and whole-body control (`scene_wbc_g2_sp`).
- 🛠️ **Engine internals** — AS3 asset layout (per-robot `robot.usda` + `payloads/Physics/{physics,physx,mujoco}.usda`); IsaacSim 5.1 + 6.0 share one CLI surface; fisheye camera + de-skewed rotary lidar.
- 🔗 **Module dependency DAG** — auto-generated from each peer's `pyproject.toml`, rendered in [`source/README.md`](source/README.md); enforced by `geniesim tool deps-dag` in CI.

</details>

<details><summary><b>[4/8/2026] v3.1</b></summary>

- 🌐 **Release Genie Sim World** — a multimodal spatial world model for 3D world generation.
- 🧪 Update new benchmarks for instruction following, spatial understanding, manipulation skills, robustness, and sim2real.
- 🎓 Support human-in-the-loop and distributed reinforcement learning pipeline of RLinf.

</details>

<details><summary><b>[1/7/2026] v3.0</b></summary>

- 🆙 Update Isaac Sim to v5.1.0 and support RTX 50series graphic card.
- 🦾 Provide USD and URDF files of Agibot Genie G2 robot and support whole body control.
- 🛰️ Support 3DGS-based scene reconstruction and convert output to USD format for application in Isaac Sim.
- 📦 Release synthetic dataset and corresponding data collection pipeline.
- 💬 Add LLM-based features to generate scenarios, task instructions and evaluation configurations.

</details>

<details><summary><b>[7/14/2025] v2.2</b></summary>

- 📊 Provide detailed evaluation metrics for all Agibot World Challenge tasks.
- ⚙️ Add automatic evaluation script to run each task multiple times and record score of all steps.

</details>

<details><summary><b>[6/25/2025] v2.1</b></summary>

- ➕ Add 10 more manipulation tasks for Agibot World Challenge 2025 including all simulation assets.
- 📦 Open-source synthetic datasets for 10 manipulation tasks on [Huggingface](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData).
- 🔌 Integrate UniVLA policy and support model inference simulation evaluation.
- 🧮 Update IK solver sdk which supports cross-embodiment IK solving for other robots.
- ⚡ Optimize communication framework and improve simulation running speed by 2×.
- 🧪 Update automatic evaluation framework for more complicated long-range tasks.

</details>

# 5. 📚 Documentation

## 5.1 📖 Documentations

Please refer to these links to download assets and dataset:

- [Assets](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)
- [Dataset](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset) (New Benchmark Dataset coming soon)

## 5.2 💬 Support

<img src="./docs/feishu.png" width="30%"/>

## 5.3 📄 License and Citation

All the data and code within `source/geniesim_*` and `source/data_collection` are under `Mozilla Public License 2.0` unless licensed specifically. The `source/scene_reconstruction` project contains code under multiple licenses, for complete and updated licensing details, please see the LICENSE files

Please consider citing our work either way below if it helps your research.

```
@misc{yin2026geniesim30,
  title={Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot},
  author={Chenghao Yin and Da Huang and Di Yang and Jichao Wang and Nanshu Zhao and Chen Xu and Wenjun Sun and Linjie Hou and Zhijun Li and Junhui Wu and Zhaobo Liu and Zhen Xiao and Sheng Zhang and Lei Bao and Rui Feng and Zhenquan Pang and Jiayu Li and Qian Wang and Maoqing Yao},
  year={2026},
  eprint={2601.02078},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.02078},
}
```

## 5.4 🔗 References

1. PDDL Parser (2020). Version 1.1. [Source code]. https://github.com/pucrs-automated-planning/pddl-parser.
2. BDDL. Version 1.x.x [Source code]. https://github.com/StanfordVL/bddl
3. CUROBO [Source code]. https://github.com/NVlabs/curobo
4. Isaac Lab [Source code]. https://github.com/isaac-sim/IsaacLab
5. Omni Gibson [Source code]. https://github.com/StanfordVL/OmniGibson
6. The Scene Language [Source code]. https://github.com/zzyunzhi/scene-language
7. COAL [Source code]. https://github.com/coal-library/coal
8. OCTOMAP [Source code]. https://github.com/OctoMap/octomap
9. PINOCCHIO [Source code]. https://github.com/stack-of-tasks/pinocchio
10. URDFDOM [Source code]. https://github.com/ros/urdfdom
11. LIBCCD [Source code]. https://github.com/danfis/libccd
12. LIBMINIZIP [Source code]. https://github.com/switch-st/libminizip
13. LIBODE [Source code]. https://github.com/markmbaum/libode
14. LIBURING [Source code]. https://github.com/axboe/liburing
15. MuJoCo [Source code]. https://github.com/google-deepmind/mujoco
16. Wuji SDK [Source code]. https://github.com/wuji-technology/wuji-sdk — [MIT license](THIRD_PARTY_LICENSES/LICENSE.WUJI_SDK)

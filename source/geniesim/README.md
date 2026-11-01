# geniesim — Umbrella meta-package 🧞

`geniesim` is the **umbrella distribution** for the Genie Sim platform.
It ships no code of its own — installing it pulls in every
content-bearing peer in one shot, modelled on Isaac Sim's `isaacsim`
umbrella.

License: [Mozilla Public License Version 2.0](LICENSE)

---

## 📦 What's in the box

**Tier 1** — always pulled in by `pip install -e source/geniesim/`:

| Peer | Role |
|---|---|
| 🧞 [`geniesim_cli`](../geniesim_cli/) | CLI dispatcher — the `geniesim` console script |
| 🎨 `geniesim_assets` | Sim-ready 3D asset pack (downloaded separately) |
| 🧪 [`geniesim_benchmark`](../geniesim_benchmark/) | Benchmark tasks + scoring |
| ⚡ [`geniesim_ros`](../geniesim_ros/) | Genie Sim RT Engine (ROS 2) |

**Tier 2** — opt-in via extras (`pip install -e "source/geniesim/[<extra>]"`):

| Peer | Extra | Why opt-in |
|---|---|---|
| 🎮 [`geniesim_teleop`](../geniesim_teleop/) | `[teleop]` · `[all]` | VR / Pico device stack — only needed for teleoperation |
| 🏗️ [`geniesim_scene_agent`](../geniesim_scene_agent/) | `[scene_agent]` · `[all]` | Scene authoring with USD; optional GPU embeddings |
| 🌍 [`geniesim_world`](../geniesim_world/) | `[world]` · `[all]` | PanoRecon: CUDA + PyTorch + SHARP + DA360 — own conda env recommended |

---

## 🛠️ Install

The umbrella is **not on PyPI** — install from source via the CLI's
bootstrap flow:

```bash
pip install -e source/geniesim_cli/
geniesim bootstrap                 # installs every tier-1 peer in topological order
```

Or, if you already have the peers installed and just want the umbrella
itself for its dep pin:

```bash
pip install -e source/geniesim/          # tier-1 only
pip install -e "source/geniesim/[teleop]"     # + VR / Pico teleop
pip install -e "source/geniesim/[scene_agent]"  # + scene authoring tools
pip install -e "source/geniesim/[world]"      # + panorama → 3D world (heavy CUDA / ML)
pip install -e "source/geniesim/[all]"        # everything
```

---

## 💡 Why a meta-package

So that `pip install geniesim` (once on PyPI) drops the full SDK in one
step, and so that downstream tooling can declare `geniesim` as a single
dependency line. Heavy runtime deps (Isaac Sim, MuJoCo, open3d, …) live
in the relevant peer, never here.

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🧞 CLI fresh-machine setup: [`../geniesim_cli/AGENTS.md` § 0](../geniesim_cli/AGENTS.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 🤖 Agent guide: [`../../AGENTS.md`](../../AGENTS.md)
- 🔧 Umbrella deep-dive: [`../../.agent/geniesim.md`](../../.agent/geniesim.md)

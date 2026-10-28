# geniesim_cli — `geniesim` command-line dispatcher 🧞

Lightweight command-line front-end for the Genie Sim stack. Published as
the `geniesim_cli` distribution; sole owner of the `geniesim` console
script.

License: [Mozilla Public License Version 2.0](LICENSE)
Agent doc: [`AGENTS.md`](AGENTS.md) · Deep-dive: [`../../.agent/geniesim_cli.md`](../../.agent/geniesim_cli.md)

---

## 📦 Install

```bash
pip install -e source/geniesim_cli/
geniesim version
```

The CLI is a **standalone PEP 517 / PEP 621 wheel** with no heavy
runtime deps (no USD, no Isaac Sim, no MuJoCo at import time), so it
runs even on a control node where the rest of the SDK isn't installed.

---

## 🛠️ Command surface

| Command | Purpose |
|---|---|
| `geniesim version` | Print versions of all distributions + Python |
| `geniesim status` | Per-distribution health probe |
| `geniesim doctor` | Diagnose & repair (status + rosdep + env) |
| `geniesim bootstrap` | Install required peers from local sources in dependency order |
| `geniesim env` | Show `GENIESIM_*` env vars |
| `geniesim completion bash\|zsh` | Generate shell completion |
| `geniesim docker {build,up,down,into,logs}` | Manage the Genie Sim container |
| `geniesim docker5.1 …` / `docker6.2 …` / `docker4.5 …` | Per–Isaac-Sim variants |
| `geniesim ros build {dev,release,cleanup}` | `colcon build` the ROS 2 workspace |
| `geniesim ros doctor` | Repair rosdep |
| `geniesim tool {deps-dag,ros-dag,docs} [--fix]` | Repo-maintenance audits (DAG + doc-coverage) |
| `geniesim deploy [MODULE]` | Build pure-Python wheel(s) into `./deploy/` |
| `geniesim benchmark {run,batch,list,…}` | Drive `geniesim_benchmark` tasks |
| `geniesim teleop run` | Launch the teleop loop |

Run `geniesim -h` for the full list.

---

## 🚀 Fresh-machine setup

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/   # 1. CLI first (no heavy deps)
geniesim bootstrap                    # 2. installs required local peers; reports missing asset sources
geniesim status                       # 3. all-green check
geniesim doctor                       # 4. repair if not
```

See [`AGENTS.md` § 0 — Fresh-machine setup](AGENTS.md) for the install-order contract.

---

## 🏗️ Architecture (in one line)

`cli.py` dispatches public commands: every subcommand lives in
`commands/<name>.py` and is loaded lazily so the import cost stays near
zero. Heavy deps (USD, MuJoCo, anything `geniesim.*`) must NEVER be
imported at module top-level — see [`AGENTS.md`](AGENTS.md) §4.

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 📚 Full command surface + style rules: [`../../.agent/geniesim_cli.md`](../../.agent/geniesim_cli.md)
- 🤖 Agent guide: [`AGENTS.md`](AGENTS.md)

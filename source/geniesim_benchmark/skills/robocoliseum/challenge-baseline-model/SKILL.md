---
name: challenge-baseline-model
description: >
  Provision and launch the Simulation Challenge baseline inference model end to end: clone the
  inference code from a given git repo/branch, download the checkpoints from ModelScope into the
  repo's local checkpoints path, install deps, and start the inference agent. Trigger: When the user
  asks to "拉取 baseline 模型", "下载推理代码/权重", "搭一个 baseline", "clone the inference repo",
  "download ckpts", "set up the baseline model", "跑起来 baseline", "provision the baseline", or
  hands over a repo URL to stand up an inference server. Load even on light
  phrasings: "部署一下 demo", "部署 demo", "跑个 demo", "deploy a/the demo", "下载一下 baseline",
  "下载 baseline", "拉一下 baseline".
metadata:
  author: zy
  version: "1.0"
---

# challenge-baseline-model — Provision & run the baseline inference model

End-to-end setup of a contestant inference model: **clone code → download checkpoints → install
deps → run agent**. Once running, the agent serves a Simulation Challenge job — hand off to
`challenge-run-agent` for the launch contract and `challenge-inference-protocol` for the obs/action
wire format.

> ⚠️ **The CONFIG block below holds DEBUG/example values** (provided for bring-up). Replace the
> repo URL, branch, and checkpoint specs with the official ones when ready — everything downstream
> reads from these variables, so it's a single-place edit.

---

## CONFIG — edit these (current = debug example)

```bash
# --- code ---
REPO_URL="https://github.com/Anonymous-694/ACoT-VLA.git"   # clone URL (strip GitHub /tree/<branch>)
REPO_BRANCH="agibot_world_challenge"
# Local clone target (= inference repo root). If the user doesn't specify a path, default to the
# current directory — i.e. assume you're already inside (or want the repo placed in) the cwd.
REPO_DIR="${REPO_DIR:-$(pwd)}"

# --- checkpoints ---
# Downloaded from ModelScope (agibot_world/GenieSim3.0-Dataset, checkpoints/<name>) via the bundled
# scripts/download_checkpoint.sh — NO rsync / SSH needed, only `pip install modelscope`.
# From the Genie Sim checkout root; for a copied skill, use its installation directory.
SKILL_DIR="$(pwd)/source/geniesim_benchmark/skills/robocoliseum/challenge-baseline-model"
DL_CKPT="$SKILL_DIR/scripts/download_checkpoint.sh"

# Which checkpoints to fetch. Each name maps to a board (see table below). Pull only what the job
# needs — list one or more of: instruction_and_robust_pi05 / manipulation_pi05 / spatial_pi05
CKPT_NAMES=(
  "instruction_and_robust_pi05"
  # "manipulation_pi05"
  # "spatial_pi05"
)
```

**Checkpoint ↔ board map** (`config.board` → ModelScope checkpoint name):

| Job board | Checkpoint name (`checkpoints/<name>`) |
|-----------|----------------------------------------|
| `instruction` | `instruction_and_robust_pi05` |
| `robust`      | `instruction_and_robust_pi05` |
| `manip`       | `manipulation_pi05` |
| `spatial`     | `spatial_pi05` |

---

## Step 1 — Clone the inference code

```bash
if [ -d "$REPO_DIR/.git" ]; then
  echo "repo exists at $REPO_DIR — fetching $REPO_BRANCH"
  git -C "$REPO_DIR" fetch origin "$REPO_BRANCH" && git -C "$REPO_DIR" checkout "$REPO_BRANCH" && git -C "$REPO_DIR" pull --ff-only
else
  git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi
```

(`/tree/<branch>` in a GitHub web URL just means that branch — clone the `.git` URL with `-b`.)

## Step 2 — Download checkpoints (ModelScope)

Fetch each needed checkpoint into `$REPO_DIR/checkpoints/<name>` with the bundled
`download_checkpoint.sh`. It pulls from ModelScope — **no SSH/rsync, no password** — only the
`modelscope` CLI (`pip install modelscope` if missing).

```bash
for name in "${CKPT_NAMES[@]}"; do
  "$DL_CKPT" "$name" "$REPO_DIR/checkpoints"
done
```

Each lands at `$REPO_DIR/checkpoints/<name>/` (e.g. `checkpoints/instruction_and_robust_pi05/`).
Single-board example:

```bash
"$DL_CKPT" instruction_and_robust_pi05 "$REPO_DIR/checkpoints"
```

Script signature: `download_checkpoint.sh [CHECKPOINT_NAME] [LOCAL_DIR]` — valid names are
`instruction_and_robust_pi05` / `manipulation_pi05` / `spatial_pi05`; output goes to
`<LOCAL_DIR>/<name>/`. `download_checkpoint.sh -h` lists them.

> **Layout note:** there's **no step subdir** — `params/` (and `assets/`, `_CHECKPOINT_METADATA`)
> sit **directly** under `checkpoints/<name>_pi05/`. `tunnel.sh` selects the ckpt by board via
> `PI05_BOARD` and points at the `_pi05`-suffixed dirs directly (`instruction`/`robust` →
> `checkpoints/instruction_and_robust_pi05`, `spatial` → `checkpoints/spatial_pi05`, `manip` →
> `checkpoints/manipulation_pi05`), so just download into `checkpoints/` and no symlink is needed.
> Override the resolved path with `PI05_CKPT_DIR` if your layout differs.
> See `challenge-inference-protocol` §3 for the per-board ckpt layout.

## Step 3 — Install dependencies

```bash
cd "$REPO_DIR"
uv sync            # or: pip install -e packages/openpi-client && pip install -r requirements
```

## Step 4 — Run the inference agent

Needs a live job (`challenge-submit-job` → `JOB_UUID`, `TUNNEL_ENDPOINT`) and `CHALLENGE_TOKEN`.
Pick the board to match the job's `config.board`:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
cd "$REPO_DIR"
PI05_BOARD=instruction ./scripts/tunnel.sh 0 "$JOB_UUID" "$TUNNEL_ENDPOINT"
```

Verify it reaches `state -> RUNNING`, then cases stream in. For concurrency, GPU pinning, and the
drain/exit lifecycle, follow `challenge-run-agent`. For decoding what flows over the wire, see
`challenge-inference-protocol`.

### VRAM footprint & packing multiple instances on one GPU

A single pi05 inference service uses **< 8 GB** of VRAM at runtime, so a 24 GB card (e.g. RTX 4090)
fits **at most 3 services concurrently** (3 is the practical max for pi05 — more will OOM under load).
Pass the *same* GPU index as the first arg to each `tunnel.sh` and launch them as separate processes
(different `JOB_UUID`s):

```bash
PI05_BOARD=instruction ./scripts/tunnel.sh 0 "$JOB_UUID_A" "$TUNNEL_ENDPOINT" &
PI05_BOARD=instruction ./scripts/tunnel.sh 0 "$JOB_UUID_B" "$TUNNEL_ENDPOINT" &
PI05_BOARD=instruction ./scripts/tunnel.sh 0 "$JOB_UUID_C" "$TUNNEL_ENDPOINT" &
```

The reference `tunnel.sh` already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` (on-demand allocation, good for sharing), **but it hardcodes
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.48`** (≈11.5 GB per process) — too high to pack three.

**Derive the per-process cap from the concurrency the user asks for**, instead of a fixed number.
The user tells you how many services they want on one GPU (call it `N`, **max 3 for pi05 on 24 GB**);
split the GPU evenly, keeping ~10% headroom:

```
mem_fraction = 0.9 / N      # clamped to [0.10, 0.90], then leave PREALLOCATE=false
```

| User-requested N (per GPU) | `XLA_PYTHON_CLIENT_MEM_FRACTION` | per-process cap on 24 GB |
|----------------------------|----------------------------------|--------------------------|
| 1 | `0.90` | ~21.6 GB (effectively uncapped) |
| 2 | `0.45` | ~10.8 GB |
| 3 (max) | `0.30` | ~7.2 GB |

`N > 3` on a single 24 GB card is not supported for pi05 — runtime usage (~8 GB each) overruns the
card even though the fraction math would compute a smaller ceiling.

Since one service only touches < 8 GB at runtime and `PREALLOCATE=false`, the fraction is just a
*ceiling* — sizing it to `0.9/N` reserves an even slice per neighbor without wasting memory.

Make `tunnel.sh` compute this from a concurrency env var (honor an explicit override if set):

```bash
# in scripts/tunnel.sh, replacing the hardcoded `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.48`
N=${PI05_PARALLELISM:-1}                       # services sharing this GPU (from the user's request)
frac=$(awk -v n="$N" 'BEGIN{f=0.9/n; if(f>0.9)f=0.9; if(f<0.1)f=0.1; printf "%.2f", f}')
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$frac}
```

Then launch N services on the same GPU index, telling each how many share the card:

```bash
N=3
for i in $(seq 1 "$N"); do
  PI05_PARALLELISM=$N PI05_BOARD=instruction \
    ./scripts/tunnel.sh 0 "${JOB_UUIDS[$i]}" "$TUNNEL_ENDPOINT" &
done
```

(`PI05_PARALLELISM` is the divisor; an explicit `XLA_PYTHON_CLIENT_MEM_FRACTION=...` still wins if
you want to override the computed value for a specific launch.)

---

## Notes

- **Repo path default:** if the user doesn't give an inference-code path, `REPO_DIR` defaults to the
  current directory (`$(pwd)`) — assume you're already in (or want the repo in) the cwd. Override by
  exporting `REPO_DIR` or editing the CONFIG line.
- **Swapping to the official model:** edit only the CONFIG block (repo URL/branch + `CKPT_NAMES`).
  Add `manipulation_pi05` / `spatial_pi05` to `CKPT_NAMES` to fetch those boards.
- **Idempotent:** Step 1 re-fetches if the repo already exists; Step 2's `modelscope download`
  caches/resumes, so re-running continues rather than restarting.
- **No SSH needed:** checkpoints come from ModelScope over HTTP — the only prerequisite is
  `pip install modelscope`. Downloads can be large; run Step 2 with `run_in_background` if driving
  from the assistant.
- **One service < 8 GB VRAM:** a 24 GB GPU (RTX 4090) hosts **at most 3** pi05 services at once. Don't
  hardcode the mem fraction — **derive it from the concurrency the user asks for**:
  `XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9 / N` (N = services per GPU, max 3; 2→0.45, 3→0.30), keep
  `PREALLOCATE=false`. `tunnel.sh` reads this from `PI05_PARALLELISM`. See Step 4's *VRAM footprint* note.

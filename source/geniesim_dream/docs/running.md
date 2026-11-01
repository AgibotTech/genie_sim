# Running Dream

Prepare the [three environments](setup.md) and [model assets](assets.md) first.
Dream and the evaluator must share an output directory on the same machine or
mounted at the same path.

## Start the stack

From `source/geniesim_dream/`, start the pi0.5 server, policy bridge, VLM
evaluator and Dream UI together. Set `DREAM_ASSET_ROOT` first if the assets are
outside `checkpoints/`:

```bash
./tools/start_services.sh
```

Keep the terminal open; Ctrl+C stops the stack. In another terminal, run
`./tools/start_services.sh status` from the same directory to check the
four ports and the Dream/VLM readiness endpoints. Logs are in `.local/logs/`.

## Run a session

Open **http://127.0.0.1:8971/**. Choose a run mode first; the example list
then shows the entries available for that mode:

| Mode in the UI | Action source |
| --- | --- |
| **Open loop · recorded action replay** | Actions saved in the example bundle or supplied through the API |
| **Closed loop · pi0.5 policy** | New actions predicted from generated observations |

For a policy loop, run **20 chunks**; the UI selects 20 by default in this
mode. Recorded action replay starts at two chunks and can run only as many as
the selected example contains. Other task policies load on first use.

The UI defaults to 32 action rows per chunk for both robots and 10 UniPC
sampling steps. Choose these settings before creating the session; they stay
fixed for that session. Select **Create session**, then **Run continuously**
for the chosen chunk count or **Step once** for a single chunk. When the run
finishes,
select **Generate full video**, then optionally **Run VLM evaluation**. The
VLM report estimates completion from generated video. The session output under
`outputs/<session-id>/` contains `vision.mp4` and `actions.npy`; a report appears
under `evaluations/` if you ran evaluation. If you set `DREAM_OUTPUT_DIR`, use
that directory in place of `outputs/`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Launcher reports a missing Python environment | Finish the matching step in the [installation guide](setup.md), or set `DREAM_WORLD_PYTHON`, `DREAM_POLICY_PYTHON` or `DREAM_EVAL_PYTHON` to the intended interpreter. |
| Launcher reports a missing checkpoint directory | Check `DREAM_ASSET_ROOT` and the [asset layout](assets.md#file-layout). |
| Port 8712, 8972, 8981 or 8971 is occupied | Run `./tools/start_services.sh status` and stop the existing stack from its launcher terminal before starting another one. |
| A service starts but is not ready | Inspect `.local/logs/policy.log`, `bridge.log`, `evaluator.log` or `world.log` for that service. |
| Policy startup needs network access | Put OpenPI's PaliGemma tokenizer in the [local cache](assets.md#policy-tokenizer-for-offline-runs) for offline use. |
| VLM evaluation cannot load a model | Check the task's Qwen checkpoint under `vlm/` and the paths in `services/evaluation/configs/models.example.json`. |

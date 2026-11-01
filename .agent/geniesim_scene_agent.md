# geniesim_scene_agent — repository-level dispatcher

The canonical guide is [source/geniesim_scene_agent/AGENTS.md](../source/geniesim_scene_agent/AGENTS.md).
Read it for package entry points, skill routing, scene contracts, and validation.

This optional peer contains English skills and Python tools for asset search,
scene generation, task reference retrieval, and automatic collection task authoring.

| Topic | Local reference |
| --- | --- |
| Installation and usage | [README](../source/geniesim_scene_agent/README.md) |
| Skill workflows | [skill index](../source/geniesim_scene_agent/skills/README.md) |
| Isaac Sim 5.1 image | [container guide](../source/geniesim_scene_agent/docker/README.md) |
| Task library | [examples](../source/geniesim_scene_agent/examples/README.md) |

The umbrella extra is `[scene_agent]`. Install the local peer first; default
bootstrap installs tier-1 peers only. The CLI supports `status` and
`deploy geniesim_scene_agent`; scene tools have their own Python/console entries.
Assets and optional model weights remain external runtime data.

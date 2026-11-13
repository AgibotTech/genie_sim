# Use the prebuilt Isaac Sim 6.2 image

The locally built images are published under
[`yonhdee/genie_sim` on Docker Hub](https://hub.docker.com/r/yonhdee/genie_sim/tags).
To save build time, pull the image's family tag and retag it locally with the
name used by the CLI and downstream recipes.

## Current runtime: geniesim4

```bash
docker pull yonhdee/genie_sim:geniesim4
docker tag yonhdee/genie_sim:geniesim4 registry.agibot.com/genie-sim/geniesim4:latest
geniesim docker up
geniesim docker into
```

The image provides dedicated Isaac Sim 6.2, system Python 3.12 and ROS 2 Jazzy.
Keep the checkout mounted: the entrypoint installs its packages when the
container starts. Image tags and the checkout's package version are independent.

Derived post-training profiles still require their own builds. See the
[Docker guide](README.md#post-training-profiles) for those workflows and
[runtime acceptance](../source/geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md) for
qualification scope.

## Older runtime: geniesim3 (E.O.L.)

For existing deployments using the deprecated Isaac Sim 5.1 + 6.2 combo:

```bash
docker pull yonhdee/genie_sim:geniesim3
docker tag yonhdee/genie_sim:geniesim3 registry.agibot.com/genie-sim/geniesim3:latest
```

Use Docker image names as shown, without an `https://` prefix. Retagging uses
the downloaded image locally; it does not push it to the AgiBot registry.

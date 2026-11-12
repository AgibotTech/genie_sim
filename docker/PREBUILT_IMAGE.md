# Use the prebuilt Isaac Sim 6.2 image

To skip the local Docker build, pull `yonhdee/geniesim4:latest` from Docker Hub
and tag it with the image name used by the CLI and downstream recipes:

```bash
docker pull yonhdee/geniesim4:latest
docker tag yonhdee/geniesim4:latest registry.agibot.com/genie-sim/geniesim4:latest
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

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single-world capabilities of the upstream Isaac Kit engines."""

from typing import Any, Optional, Sequence

from ._upstream_base import _UpstreamSubprocessBackend


class _IsaacKitSubprocessBackend(_UpstreamSubprocessBackend):
    _supports_batching = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._num_envs != 1:
            raise ValueError(
                f"{self.BACKEND_NAME} requires num_envs=1; got {self._num_envs}. "
                "Use isaac_fused for multiple environments."
            )

    def reset(
        self,
        env_ids: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        instances: Optional[Sequence[int]] = None,
    ) -> None:
        """Validate capabilities before publishing a reset without an error reply."""
        if env_ids is not None:
            if len(env_ids) == 0:
                return
            if any(int(env_id) != 0 for env_id in env_ids):
                raise ValueError(f"{self.BACKEND_NAME} contains only env 0; got {list(env_ids)}")
        for values, operation in ((seeds, "seeded randomization"), (instances, "instance switching")):
            if values is not None and any(int(value) >= 0 for value in values):
                raise NotImplementedError(f"{self.BACKEND_NAME} reset does not support {operation}; use isaac_fused.")
        super().reset(env_ids=env_ids, seeds=seeds, instances=instances)

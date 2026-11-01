"""OpenPI policy client.

Talks to an openpi websocket policy server (``scripts/serve_policy.py`` in the
``third_party/openpi`` submodule). Requires the ``openpi-client`` package:

    pip install -e third_party/openpi/packages/openpi-client
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .profiles import CHECKPOINT_KEY, PROFILE_KEY, get_profile
from .types import ACTION_DIM, Observation

# Default observation keys for pi05-style configs; override per checkpoint.
DEFAULT_IMAGE_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.hand_left",
    "right_wrist": "observation.images.hand_right",
}
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PROMPT_KEY = "prompt"

# Optional hint telling a visualizing policy server where this episode's pictures
# belong. Ignored by every normal server -- the pi05 input transform keeps only
# the keys it names -- so sending it costs a short string per round and nothing
# else. It exists because the server outlives a batch run while the output
# directory changes per episode, and only the client knows that directory.
VIZ_DIR_KEY = "dream.viz_dir"


def build_openpi_payload(
    obs: Observation,
    *,
    image_keys: dict[str, str] | None = None,
    state_key: str = DEFAULT_STATE_KEY,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    viz_dir: str | None = None,
) -> dict[str, Any]:
    """Map a Dream Observation to the flat dict an openpi policy server expects.

    ``obs.state`` is forwarded as-is (policy layout ``[L7_arm, R7_arm, L_grip, R_grip]``).

    Args:
        viz_dir: where a visualizing server should write this episode's pictures.
            Omitted from the payload when None, so the default rollout sends
            exactly what it sent before.

    Raises:
        KeyError: if a required camera view is missing from ``obs.images``.
    """
    image_keys = image_keys or DEFAULT_IMAGE_KEYS
    payload: dict[str, Any] = {
        state_key: np.asarray(obs.state, dtype=np.float32).reshape(-1),
        prompt_key: obs.task,
    }
    if viz_dir is not None:
        payload[VIZ_DIR_KEY] = str(viz_dir)
    for view, key in image_keys.items():
        if view not in obs.images:
            raise KeyError(f"observation is missing view {view!r}; has {sorted(obs.images)}")
        payload[key] = np.ascontiguousarray(obs.images[view])
    return payload


def validate_actions(actions: np.ndarray) -> np.ndarray:
    """Normalize a policy response to ``(horizon, 16)`` float32."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected actions (horizon, {ACTION_DIM}), got {actions.shape}")
    return actions


class OpenPIPolicy:
    """Policy backed by an openpi websocket policy server.

    Sends ``Observation`` images/state/prompt to the server and returns action
    chunks ``(horizon, 16)`` float32 in WM layout ``[L7_arm, L_grip, R7_arm, R_grip]``.
    """

    def __init__(
        self,
        url: str,
        *,
        action_horizon: int | None = None,
        image_keys: dict[str, str] | None = None,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        viz_dir: str | None = None,
    ):
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError as exc:
            raise ImportError(
                "openpi-client is not installed. Run: "
                "pip install -e third_party/openpi/packages/openpi-client"
            ) from exc
        self._client = WebsocketClientPolicy(host=url)
        self._client_type = WebsocketClientPolicy
        self._url = url
        self._action_horizon = action_horizon
        self._image_keys = image_keys
        self._state_key = state_key
        self._prompt_key = prompt_key
        self._viz_dir = str(viz_dir) if viz_dir is not None else None

    def reset(self) -> None:
        if self._client is not None:
            self._client.reset()

    def infer(self, obs: Observation, *, profile: str | None = None) -> np.ndarray:
        """Return an action chunk ``(horizon, 16)`` float32 in WM layout.

        Layout is ``[L7_arm, L_grip, R7_arm, R_grip]``.
        """
        payload = build_openpi_payload(
            obs,
            image_keys=self._image_keys,
            state_key=self._state_key,
            prompt_key=self._prompt_key,
            viz_dir=self._viz_dir,
        )
        if self._client is None:
            self._client = self._client_type(host=self._url)
        if profile is not None:
            selected = get_profile(profile)
            metadata = self._client.get_server_metadata()
            if not metadata.get("task_routing"):
                self._client = None
                raise RuntimeError(
                    "pi05 server does not support task switching; "
                    "restart the policy terminal with serve.sh policy"
                )
            payload[PROFILE_KEY] = selected.id
        try:
            started = time.monotonic()
            result = self._client.infer(payload)
            logging.getLogger(__name__).info(
                "pi05 response task=%s round_trip=%.3fs server=%s policy=%s",
                profile,
                time.monotonic() - started,
                result.get("server_timing"),
                result.get("policy_timing"),
            )
        except Exception:
            # The upstream server closes the socket on model-load/inference
            # errors. Surface this attempt's error, reconnect on a later request.
            self._client = None
            raise
        if profile is not None and (
            result.get(PROFILE_KEY) != selected.id
            or result.get(CHECKPOINT_KEY) != selected.checkpoint_name
        ):
            raise RuntimeError(f"pi05 response does not confirm checkpoint for task {selected.id}")
        if "actions" not in result:
            raise KeyError(
                f"policy server response is missing 'actions'; got keys {sorted(result)}"
            )
        actions = validate_actions(result["actions"])
        if self._action_horizon is not None:
            actions = actions[: self._action_horizon]
        return actions

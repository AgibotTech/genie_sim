"""Single resident pi05 checkpoint, switched explicitly by each request's task."""

from __future__ import annotations

import logging
import threading
import time

from .profiles import CHECKPOINT_KEY, PROFILE_KEY, PROFILES, get_profile


class TaskPolicy:
    def __init__(self, loader, release, default_profile="cloth"):
        self._loader = loader
        self._release = release
        self._lock = threading.Lock()
        self._policy = None
        self._active = None
        self.metadata = {
            "task_routing": True,
            "task_profiles": {key: p.checkpoint_name for key, p in PROFILES.items()},
        }
        with self._lock:
            self._select(get_profile(default_profile))

    def _select(self, profile):
        if self._active == profile.id:
            return
        # Check assets before evicting the working model. Loading all five at
        # once can exhaust VRAM; release the previous model before a real load.
        path = profile.checkpoint()
        for relative in ("model.safetensors", "assets/dream/norm_stats.json"):
            if not (path / relative).is_file():
                raise FileNotFoundError(f"Missing pi05 asset: {path / relative}")
        logging.info("pi05 switching %s -> %s (%s)", self._active, profile.id, path)
        started = time.monotonic()
        self._policy = None
        self._active = None
        self._release()
        # The loader constructs transforms and loads norm_stats from this exact
        # checkpoint. A failed load leaves no active policy; never serve stale
        # weights under the requested task's name.
        policy = self._loader(str(path))
        self._policy = policy
        self._active = profile.id
        logging.info("pi05 task ready: %s (switch %.3fs)", profile.id, time.monotonic() - started)

    def infer(self, observation):
        profile = get_profile(observation.get(PROFILE_KEY))
        with self._lock:
            switching = self._active != profile.id
            self._select(profile)
            payload = {key: value for key, value in observation.items() if key != PROFILE_KEY}
            started = time.monotonic()
            result = dict(self._policy.infer(payload))
            if switching:
                logging.info(
                    "pi05 first prediction ready: %s (prediction %.3fs)",
                    profile.id,
                    time.monotonic() - started,
                )
            result[PROFILE_KEY] = profile.id
            result[CHECKPOINT_KEY] = profile.checkpoint_name
            return result

    def reset(self):
        with self._lock:
            if self._policy is not None:
                self._policy.reset()
